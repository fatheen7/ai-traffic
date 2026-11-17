"""Command line.

    python -m traffic plan 24 14 10 2
    python -m traffic count samples/north.mp4 --show
    python -m traffic siren samples/siren_east.wav
    python -m traffic junction --north n.mp4 --east e.mp4 --south s.mp4 --west w.mp4 \
        --siren samples/siren_east.wav --siren-road east --out output/junction.mp4 --json output/results.json
"""

import argparse
import json

from .config import Config
from .junction import count_video, run_junction
from .signal import green_times
from .siren import SirenDetector, read_wav


def _signal_args(p):
    p.add_argument("--cycle", type=float, default=120, help="seconds of green shared per cycle (default 120)")
    p.add_argument("--min-green", type=float, default=10)
    p.add_argument("--max-green", type=float, default=60)


def _counter_args(p):
    p.add_argument("--line", type=float, default=0.6, help="counting line as a fraction of frame height")
    p.add_argument("--min-area", type=int, default=120, help="smallest shape counted as a vehicle, in pixels")
    p.add_argument("--direction", choices=["down", "up", "both"], default="down")


def _config(a) -> Config:
    cfg = Config()
    if hasattr(a, "line"):
        cfg.counter.line = a.line
        cfg.counter.direction = a.direction
        cfg.detector.min_area = a.min_area
    if hasattr(a, "cycle"):
        cfg.signal.cycle, cfg.signal.min_green, cfg.signal.max_green = a.cycle, a.min_green, a.max_green
        cfg.signal.first_green = getattr(a, "first_green", None) or round(a.cycle / 4)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m traffic", description="AI Traffic Management System")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="green times from vehicle counts")
    p.add_argument("counts", type=int, nargs="+")
    _signal_args(p)

    p = sub.add_parser("count", help="count vehicles crossing the line in one video")
    p.add_argument("video")
    _counter_args(p)
    p.add_argument("--show", action="store_true", help="show the annotated video while counting")

    p = sub.add_parser("siren", help="find sirens in a WAV recording")
    p.add_argument("wav")

    p = sub.add_parser("junction", help="run the full system on one video per approach")
    for road in ("north", "east", "south", "west"):
        p.add_argument(f"--{road}", help=f"video for the {road} approach")
    p.add_argument("--siren", help="WAV recording from the roadside microphone")
    p.add_argument("--siren-road", default="east", help="road the siren recording belongs to")
    p.add_argument("--out", help="annotated output video (.mp4)")
    p.add_argument("--json", help="results file (.json)")
    p.add_argument("--fixed", action="store_true", help="use a fixed timer instead of adaptive timing")
    p.add_argument("--first-green", type=float)
    _counter_args(p)
    _signal_args(p)

    a = ap.parse_args(argv)

    if a.cmd == "plan":
        greens = green_times(a.counts, a.cycle, a.min_green, a.max_green)
        for k, (n, g) in enumerate(zip(a.counts, greens)):
            print(f"road {k + 1}: {n:>4} vehicles -> {g:>3} s green")
        return 0

    if a.cmd == "count":
        if a.show:
            return _show(a)
        r = count_video(a.video, _config(a))
        print(f"{a.video}: {r['count']} vehicles in {r['seconds']} s ({r['down']} down, {r['up']} up)")
        return 0

    if a.cmd == "siren":
        x, rate = read_wav(a.wav)
        sd = SirenDetector()
        spans = sd.intervals(sd.scan(x, rate))
        if not spans:
            print("No siren detected.")
        for s, e in spans:
            print(f"Siren from {s:.1f} s to {e:.1f} s")
        return 0

    if a.cmd == "junction":
        videos = {r: getattr(a, r) for r in ("north", "east", "south", "west") if getattr(a, r)}
        if not videos:
            ap.error("give at least one approach video, for example --north north.mp4")
        res = run_junction(videos, a.out, a.json, a.siren, a.siren_road, _config(a), adaptive=not a.fixed)
        print(json.dumps({"vehicles_counted": res["vehicles_counted"], "cycles": [(c["counts"], c["greens"]) for c in res["cycles"]],
                          "siren": res["siren"]["intervals"]}, indent=1))
        for e in res["events"]:
            print(f"{e['t']:>7.1f}s  {e['text']}")
        return 0


def _show(a):
    import cv2

    from .detector import VehicleDetector
    from .junction import annotate_tile, make_counter

    cfg = _config(a)
    cap = cv2.VideoCapture(a.video)
    det, counter = VehicleDetector(cfg.detector), None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = det.resize(f)
        found = det.detect(f)
        counter = counter or make_counter(f.shape, cfg)
        counter.update([d.centre for d in found])
        cv2.imshow("traffic", annotate_tile(f, found, counter, det.mask, None, "Counting", f"{counter.total} vehicles"))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{a.video}: {counter.total if counter else 0} vehicles")
    return 0
