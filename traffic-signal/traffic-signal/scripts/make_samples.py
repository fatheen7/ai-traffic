"""Make synthetic sample inputs so the project runs out of the box.

Writes, into samples/:
  north.mp4, east.mp4, south.mp4, west.mp4  one overhead camera per approach (synthetic, labelled SAMPLE)
  siren_east.wav                            roadside microphone: traffic noise, with a siren passing on the east road

Replace these with real recordings for real results: any fixed camera looking along the road works,
with traffic heading towards the junction moving down the frame.
"""

import argparse
import json
import os
import wave

import cv2
import numpy as np

W, H, FPS = 480, 270, 15
# vehicles per minute heading to the junction on each approach (deliberately uneven)
RATES = {"north": 26, "east": 16, "south": 10, "west": 4}
COLOURS = [(182, 111, 61), (58, 80, 199), (60, 179, 226), (98, 143, 79), (176, 95, 138), (128, 116, 107), (204, 212, 216), (54, 47, 43), (60, 106, 180), (168, 163, 90)]


def road_background(rng):
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (64, 75, 70)
    cv2.rectangle(bg, (140, 0), (340, H), (88, 92, 96), -1)          # footpaths
    cv2.rectangle(bg, (150, 0), (330, H), (62, 60, 58), -1)          # carriageway
    noise = rng.normal(0, 4, (H, W, 1))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    for y in range(-10, H, 34):                                       # centre line
        cv2.line(bg, (240, y), (240, y + 18), (90, 200, 225), 2)
    cv2.putText(bg, "SAMPLE (synthetic)", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return bg


def new_vehicle(rng, lane_x, direction, y):
    kind = rng.random()
    if kind < 0.15:
        L, Wd = 58, 26   # bus
    elif kind < 0.35:
        L, Wd = 18, 9    # two-wheeler
    else:
        L, Wd = 36, 20   # car
    return {"x": lane_x + rng.uniform(-6, 6), "y": y, "L": L, "W": Wd, "dir": direction,
            "v": rng.uniform(70, 95), "colour": COLOURS[rng.integers(len(COLOURS))]}


def draw_vehicle(img, v):
    x, y, L, Wd = v["x"], v["y"], v["L"], v["W"]
    x0, y0, x1, y1 = int(x - Wd / 2), int(y - L / 2), int(x + Wd / 2), int(y + L / 2)
    # soft shadow, offset to the right (MOG2 marks shadows grey; the detector drops them)
    sh = img[max(0, y0 + 4):max(0, y1 + 4), max(0, x0 + 6):max(0, x1 + 8)]
    if sh.size:
        sh[:] = (sh * 0.62).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1, y1), v["colour"], -1)
    if L > 20:
        front = y1 - 9 if v["dir"] > 0 else y0 + 3
        cv2.rectangle(img, (x0 + 3, front), (x1 - 3, front + 6), (40, 36, 34), -1)
    else:
        cv2.circle(img, (int(x), int(y)), 4, (30, 30, 30), -1)


def make_clip(path, rate_in, seconds, seed):
    rng = np.random.default_rng(seed)
    bg = road_background(rng)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    lanes = {1: 285, -1: 195}  # traffic towards the junction drives down the frame, in the right-hand half
    rates = {1: rate_in / 60, -1: 10 / 60}
    next_t = {d: rng.exponential(1 / rates[d]) for d in lanes}
    vehicles = []
    truth = []
    line_y = H * 0.6
    for f in range(int(seconds * FPS)):
        t = f / FPS
        for d in lanes:
            if t >= next_t[d]:
                start = -40 if d > 0 else H + 40
                clear = all(abs(v["y"] - start) > 80 for v in vehicles if v["dir"] == d)
                if clear:
                    vehicles.append(new_vehicle(rng, lanes[d], d, start))
                    next_t[d] = t + rng.exponential(1 / rates[d])
        # car-following: never closer than a fixed gap to the vehicle ahead
        for v in sorted(vehicles, key=lambda v: -v["y"] * v["dir"]):
            ahead = [o for o in vehicles if o is not v and o["dir"] == v["dir"] and (o["y"] - v["y"]) * v["dir"] > 0]
            speed = v["v"]
            if ahead:
                o = min(ahead, key=lambda o: abs(o["y"] - v["y"]))
                gap = abs(o["y"] - v["y"]) - (o["L"] + v["L"]) / 2
                speed = min(speed, max(0.0, (gap - 14) * 3), o["v"] + 5)
            prev = v["y"]
            v["y"] += v["dir"] * speed / FPS
            if v["dir"] > 0 and prev < line_y <= v["y"]:
                truth.append(round(t, 2))
        vehicles = [v for v in vehicles if -80 < v["y"] < H + 80]
        frame = bg.copy()
        for v in vehicles:
            draw_vehicle(frame, v)
        # light camera noise and a slow change in brightness, like a real sensor
        gain = 1 + 0.03 * np.sin(t / 7)
        noise = rng.normal(0, 2.5, frame.shape)
        frame = np.clip(frame * gain + noise, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    return truth


def make_siren_wav(path, seconds, start=70.0, end=88.0, rate=16000, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    # traffic rumble: low-passed noise plus engine hum
    white = rng.normal(0, 1, n)
    rumble = np.convolve(white, np.ones(40) / 40, mode="same") * 2.0
    hum = 0.08 * np.sin(2 * np.pi * 95 * t) + 0.05 * np.sin(2 * np.pi * 180 * t)
    hiss = rng.normal(0, 0.01, n)
    # wail siren: pitch sweeps 650-1450 Hz about every 2.2 s
    freq = 1050 + 400 * np.sin(2 * np.pi * t / 2.2)
    phase = 2 * np.pi * np.cumsum(freq) / rate
    env = np.clip(np.minimum((t - start) / 4, (end - t) / 4), 0, 1)  # approaching, passing, leaving
    siren = 0.5 * env * np.sin(phase)
    # two short horn blasts that should not trigger the detector
    horn = np.zeros(n)
    for hs in (25.0, 110.0):
        m = (t > hs) & (t < hs + 0.5)
        horn[m] = 0.3 * np.sign(np.sin(2 * np.pi * 420 * t[m]))
    x = rumble + hum + hiss + siren + horn
    x = x / np.max(np.abs(x)) * 0.9
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seconds", type=float, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    truth = {}
    for k, (road, rate) in enumerate(RATES.items()):
        truth[road] = make_clip(os.path.join(a.out, f"{road}.mp4"), rate, a.seconds, seed=10 + k)
    make_siren_wav(os.path.join(a.out, "siren_east.wav"), a.seconds)
    with open(os.path.join(a.out, "ground_truth.json"), "w") as fh:
        json.dump({"note": "Times (s) when each vehicle heading to the junction crossed the counting line at 60% of frame height.",
                   "crossings": truth, "counts": {r: len(v) for r, v in truth.items()},
                   "siren": {"road": "east", "start": 70.0, "end": 88.0}}, fh, indent=1)
    print("ground truth:", {r: len(v) for r, v in truth.items()})


if __name__ == "__main__":
    main()
