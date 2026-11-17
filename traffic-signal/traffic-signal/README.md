# AI Traffic Management System

Traffic signals that share green time by demand. One camera per road counts vehicles with OpenCV; each cycle, busier roads get longer greens; when a microphone hears an ambulance siren, the junction clears the way.

A working rebuild of our 2025 competition project (team of 4), presented at Universiti Sains Islam Malaysia, Kuala Lumpur. My part in the original was the signal logic and the pitch.

**Stack:** Python · OpenCV (MOG2 background subtraction, contours) · NumPy (FFT) · FastAPI · Vercel

## What it does

| Step | Module | How |
| --- | --- | --- |
| Find vehicles | `traffic/detector.py` | Grayscale, blur, MOG2 background subtraction, drop shadows, open + dilate, contours, area filter |
| Count them | `traffic/tracker.py` | Nearest-centroid tracking; a vehicle counts once when it crosses the line towards the junction |
| Time the signals | `traffic/signal.py` | `green = clamp(count / total × cycle, min, max)`; green → yellow → all-red per road |
| Hear sirens | `traffic/siren.py` | FFT every 0.25 s; siren band (600–1,600 Hz) power ≥ 4× the rest for 0.8 s |
| Run a junction | `traffic/junction.py` | Four videos in lockstep + microphone audio → annotated video and `results.json` |

On the synthetic sample footage the cameras counted 128 of 142 vehicles (90%). Misses are dark vehicles on dark asphalt and vehicles too close to separate.

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/make_samples.py              # sample clips + siren recording in samples/
python -m traffic plan 24 14 10 2           # green times from counts
python -m traffic count samples/north.mp4   # add --show to watch it
python -m traffic siren samples/siren_east.wav
python -m traffic junction \
  --north samples/north.mp4 --east samples/east.mp4 \
  --south samples/south.mp4 --west samples/west.mp4 \
  --siren samples/siren_east.wav --siren-road east \
  --cycle 40 --min-green 5 --max-green 20 --first-green 10 \
  --out output/junction.mp4 --json output/results.json
pytest                                       # 21 tests
python dev.py                                # site + API at http://localhost:8000
```

### Using real footage

Any fixed camera looking along a road works. Point `--north/--east/--south/--west` at your clips. Traffic heading to the junction should move down the frame (or pass `--direction up`). Tune `--line` (counting line, fraction of frame height) and `--min-area` (smallest vehicle in pixels at 480 px wide). The siren recording must be a PCM WAV file.

To refresh the website's demo after a new run, copy `output/junction.mp4` and `output/results.json` into `public/media/`, and update `public/media/ground_truth.json` if you have true counts.

## Deploy to Vercel

1. Push this folder to a GitHub repository.
2. In Vercel: **Add New → Project**, import the repository, leave the framework preset as **Other**, deploy. `public/` becomes the site and `api/index.py` becomes the Python API at `/api/*`.
3. Put the repository link in `public/index.html` (`<body data-repo="https://github.com/...">`) so the page shows a "Code on GitHub" link.
4. Optional: **Settings → Domains** → add `traffic.muhammadfatheen.com`.

Uploads to the API are limited to 4 MB (Vercel's request limit is 4.5 MB), and videos are processed for their first 30 seconds.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | – | versions |
| POST | `/api/plan` | `{"counts": [24, 14, 10, 2]}` | green times |
| POST | `/api/count` | form: `video`, `line`, `min_area`, `direction` | count, timeline, 3 annotated frames |
| POST | `/api/siren` | form: `audio` (WAV) | siren intervals and the band-ratio timeline |

## Limits

- Background subtraction struggles at night, in rain and with moving shadows.
- Vehicles bumper to bumper merge into one shape, so jams are undercounted.
- Passing counts measure flow, not queue length or waiting time.
- One junction on its own; no coordination with neighbours.

Next steps: a trained detector (YOLO) for jams and night, queue length as the input, linked junctions, and a pilot comparing average wait and ambulance clearance time against a fixed timer.
