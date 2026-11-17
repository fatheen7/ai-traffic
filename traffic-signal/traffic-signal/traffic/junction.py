"""Run the whole system on recorded video: one video per approach, plus optional siren audio.

Every frame:
  1. each camera's frame goes through the detector and the line counter;
  2. each new count is added to that road's tally for the current cycle;
  3. the siren detector's state at this moment can start or end an emergency;
  4. the signal controller advances by one frame's time.

Note: the videos are recordings, so the signal plan is computed live but cannot change the traffic in them.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict

import cv2
import numpy as np

from .config import APPROACHES, Config
from .detector import VehicleDetector
from .signal import SignalController
from .siren import SirenDetector, read_wav
from .tracker import LineCounter

GREEN = (98, 185, 63)
YELLOW = (48, 194, 242)
RED = (77, 72, 229)
AMBER = (58, 163, 242)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
INK = (22, 20, 18)
LIGHT_BGR = {"green": GREEN, "yellow": AMBER, "red": RED}


def make_counter(frame_shape, cfg: Config) -> LineCounter:
    h, w = frame_shape[:2]
    c = cfg.counter
    return LineCounter(line_y=h * c.line, max_jump=w * c.max_jump, direction=c.direction, keep_for=c.keep_for)


def count_video(path: str, cfg: Config | None = None, max_seconds: float | None = None, keyframes: int = 0):
    """Count vehicles crossing the line in one video. Returns a summary dict (used by the CLI and the web API)."""
    cfg = cfg or Config()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    det = VehicleDetector(cfg.detector)
    counter = None
    timeline = []
    shots = []
    limit = int(max_seconds * fps) if max_seconds else None
    wanted = set()
    if keyframes and total_frames:
        last = min(total_frames, limit or total_frames)
        wanted = {int(last * (k + 1) / (keyframes + 1)) for k in range(keyframes)}
    i = 0
    size = None
    while True:
        ok, frame = cap.read()
        if not ok or (limit and i >= limit):
            break
        frame = det.resize(frame)
        size = [int(frame.shape[1]), int(frame.shape[0])]
        found = det.detect(frame)
        if counter is None:
            counter = make_counter(frame.shape, cfg)
        counter.update([d.centre for d in found])
        if i % max(1, int(fps)) == 0:
            timeline.append({"t": round(i / fps, 2), "count": counter.total, "on_screen": len(found)})
        if i in wanted:
            shots.append(annotate_tile(frame, found, counter, det.mask, None, label=None))
        i += 1
    cap.release()
    if counter is None:
        raise ValueError("The video has no readable frames.")
    return {
        "frames": i,
        "fps": round(fps, 2),
        "seconds": round(i / fps, 2),
        "count": counter.total,
        "down": counter.down,
        "up": counter.up,
        "timeline": timeline,
        "keyframes": shots,
        "size": size,
    }


def annotate_tile(frame, found, counter: LineCounter, mask, light: str | None, label: str | None, count_text: str | None = None):
    """Draw boxes, the counting line, a mask inset and the road's light on a copy of the frame."""
    img = frame.copy()
    h, w = img.shape[:2]
    y = int(counter.line_y)
    for x0 in range(0, w, 16):
        cv2.line(img, (x0, y), (min(w, x0 + 9), y), YELLOW, 2)
    for d in found:
        cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h), GREEN, 2)
    for tr in counter.tracks:
        if len(tr.history) > 1:
            pts = np.array(tr.history[-12:], dtype=np.int32)
            cv2.polylines(img, [pts], False, YELLOW if tr.counted else WHITE, 1)
    if mask is not None:
        iw = w // 4
        ih = int(mask.shape[0] * iw / mask.shape[1])
        inset = cv2.cvtColor(cv2.resize(mask, (iw, ih)), cv2.COLOR_GRAY2BGR)
        img[h - ih - 8:h - 8, w - iw - 8:w - 8] = inset
        cv2.rectangle(img, (w - iw - 9, h - ih - 9), (w - 8, h - 8), GREY, 1)
        cv2.putText(img, "mask", (w - iw - 4, h - ih - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    if label:
        cv2.rectangle(img, (0, 0), (w, 30), INK, -1)
        cv2.putText(img, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        if count_text:
            (tw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(img, count_text, (w - tw - 40, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        if light:
            cv2.circle(img, (w - 18, 15), 9, LIGHT_BGR[light], -1)
    return img


def draw_hud(width, ctl: SignalController, names, siren_on: bool, siren_ratio: float, height=96):
    hud = np.full((height, width, 3), INK, np.uint8)
    mins, secs = divmod(int(ctl.t), 60)
    cv2.putText(hud, f"{mins:02d}:{secs:02d}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(hud, f"Cycle {ctl.cycle}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    mode = "adaptive" if ctl.adaptive else "fixed timer"
    cv2.putText(hud, mode, (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    col = (width - 360) // len(names)
    for k, name in enumerate(names):
        x = 130 + k * col
        light = ctl.light(k)
        cv2.circle(hud, (x + 8, 24), 8, LIGHT_BGR[light], -1)
        cv2.putText(hud, name.capitalize(), (x + 24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(hud, f"counted {ctl.counts[k]}", (x, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        green = ctl.greens[k]
        cv2.putText(hud, f"green {green:.0f}s", (x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        if ctl.phase == k and ctl.stage == "green":
            left = "SOS" if ctl.emergency_road == k else f"{max(0, ctl.left):.0f}s left"
            cv2.putText(hud, left, (x + 90, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    sx = width - 220
    cv2.putText(hud, "Siren (FFT)", (sx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    bar = int(min(1.0, siren_ratio / 10) * 190)
    cv2.rectangle(hud, (sx, 46), (sx + 190, 60), (60, 60, 60), -1)
    cv2.rectangle(hud, (sx, 46), (sx + bar, 60), RED if siren_on else GREY, -1)
    thr = int(0.4 * 190)
    cv2.line(hud, (sx + thr, 42), (sx + thr, 64), WHITE, 1)
    cv2.putText(hud, "DETECTED" if siren_on else f"band ratio {siren_ratio:.1f}", (sx, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED if siren_on else GREY, 1, cv2.LINE_AA)
    return hud


def to_h264(src: str, dst: str) -> bool:
    """Re-encode with ffmpeg so browsers can play the file. Returns False if ffmpeg is missing."""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-movflags", "+faststart", dst]
    return subprocess.run(cmd).returncode == 0


def run_junction(videos: dict, out_video: str | None = None, out_json: str | None = None, siren_wav: str | None = None,
                 siren_road: str | None = None, cfg: Config | None = None, adaptive: bool = True, tile_width: int = 480):
    """videos: {"north": path, "east": path, ...} in signal order. Returns the results dict."""
    cfg = cfg or Config()
    cfg.detector.width = tile_width
    names = [n for n in APPROACHES if n in videos] + [n for n in videos if n not in APPROACHES]
    caps = [cv2.VideoCapture(videos[n]) for n in names]
    for n, c in zip(names, caps):
        if not c.isOpened():
            raise ValueError(f"Could not open video for {n}: {videos[n]}")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25
    dt = 1 / fps
    dets = [VehicleDetector(cfg.detector) for _ in names]
    counters = [None] * len(names)
    totals = [0] * len(names)
    ctl = SignalController(roads=len(names), names=names, cfg=cfg.signal, adaptive=adaptive)

    siren_frames, siren_idx, siren_intervals = [], None, []
    if siren_wav:
        samples, rate = read_wav(siren_wav)
        sd = SirenDetector(cfg.siren)
        siren_frames = sd.scan(samples, rate)
        siren_intervals = sd.intervals(siren_frames)
        siren_idx = names.index(siren_road) if siren_road in names else 0

    writer = None
    tmp = None
    frame_no = 0
    was_on = False
    while True:
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                frames = None
                break
            frames.append(f)
        if frames is None:
            break
        tiles = []
        for k, f in enumerate(frames):
            f = dets[k].resize(f)
            found = dets[k].detect(f)
            if counters[k] is None:
                counters[k] = make_counter(f.shape, cfg)
            for _ in counters[k].update([d.centre for d in found]):
                ctl.add_count(k)
                totals[k] += 1
            if out_video:
                tiles.append(annotate_tile(f, found, counters[k], dets[k].mask, ctl.light(k), f"{names[k].capitalize()} camera",
                                           f"{ctl.counts[k]} this cycle"))
        # siren
        on, ratio = False, 0.0
        if siren_frames:
            j = min(len(siren_frames) - 1, int(ctl.t / cfg.siren.window))
            on, ratio = siren_frames[j].siren, siren_frames[j].ratio
            if on and not was_on:
                ctl.emergency(siren_idx)
            elif was_on and not on:
                ctl.release_emergency()
            was_on = on
        if out_video:
            th, tw = tiles[0].shape[:2]
            tiles = [cv2.resize(t, (tw, th)) for t in tiles]
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)])
            grid = np.vstack([grid, draw_hud(grid.shape[1], ctl, names, on, ratio)])
            if writer is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid.shape[1], grid.shape[0]))
            writer.write(grid)
        ctl.tick(dt)
        frame_no += 1
    for c in caps:
        c.release()
    if writer is not None:
        writer.release()
        os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
        if not to_h264(tmp, out_video):
            shutil.move(tmp, out_video)
        elif os.path.exists(tmp):
            os.remove(tmp)

    ctl.cycles[-1].counts = list(ctl.counts)
    results = {
        "roads": names,
        "fps": round(fps, 2),
        "seconds": round(frame_no / fps, 2),
        "mode": "adaptive" if adaptive else "fixed",
        "vehicles_counted": dict(zip(names, totals)),
        "cycles": [asdict(c) for c in ctl.cycles],
        "events": [asdict(e) for e in ctl.events],
        "siren": {"road": siren_road, "intervals": [[round(a, 2), round(b, 2)] for a, b in siren_intervals]},
        "config": asdict(cfg),
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=2)
    return results
