"""Web API (FastAPI). On Vercel this file becomes a Python serverless function serving /api/*.

    GET  /api/health          versions
    POST /api/plan            {"counts": [24, 14, 10, 2]} -> green times
    POST /api/count           a short video (form field "video") -> vehicle count, timeline, annotated frames
    POST /api/siren           a WAV recording (form field "audio") -> when a siren was heard
"""

import base64
import os
import sys
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic import __version__  # noqa: E402
from traffic.config import Config  # noqa: E402
from traffic.junction import count_video  # noqa: E402
from traffic.signal import green_times  # noqa: E402
from traffic.siren import SirenDetector, read_wav  # noqa: E402

MAX_BYTES = 4 * 1024 * 1024  # Vercel accepts request bodies up to 4.5 MB
MAX_SECONDS = 30

app = FastAPI(title="AI Traffic Management System", version=__version__)


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "opencv": cv2.__version__, "numpy": np.__version__}


class PlanIn(BaseModel):
    counts: list[int] = Field(..., min_length=1, max_length=8)
    cycle: float = Field(120, gt=0, le=600)
    min_green: float = Field(10, ge=0)
    max_green: float = Field(60, gt=0)


@app.post("/api/plan")
def plan(body: PlanIn):
    if any(c < 0 for c in body.counts):
        raise HTTPException(400, "Counts cannot be negative.")
    if body.min_green > body.max_green:
        raise HTTPException(400, "The minimum green must not be longer than the maximum.")
    greens = green_times(body.counts, body.cycle, body.min_green, body.max_green)
    total = sum(body.counts)
    return {
        "counts": body.counts,
        "greens": greens,
        "shares": [round(c / total, 3) if total else round(1 / len(body.counts), 3) for c in body.counts],
        "rule": f"green = clamp(count / total x {body.cycle:g} s, {body.min_green:g} s, {body.max_green:g} s)",
    }


async def _save(upload: UploadFile, suffix: str) -> str:
    data = await upload.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That file is over 4 MB. Trim it to a short clip and try again.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _jpeg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


@app.post("/api/count")
async def count(
    video: UploadFile = File(...),
    line: float = Form(0.6),
    min_area: int = Form(120),
    direction: str = Form("down"),
):
    if not 0.05 <= line <= 0.95:
        raise HTTPException(400, "The counting line must be between 5% and 95% of the frame height.")
    if direction not in ("down", "up", "both"):
        raise HTTPException(400, "Direction must be down, up or both.")
    ext = os.path.splitext(video.filename or "")[1].lower() or ".mp4"
    path = await _save(video, ext)
    try:
        cfg = Config()
        cfg.counter.line = line
        cfg.counter.direction = direction
        cfg.detector.min_area = max(20, min_area)
        result = count_video(path, cfg, max_seconds=MAX_SECONDS, keyframes=3)
    except ValueError as e:
        raise HTTPException(400, f"{e} Try an MP4 (H.264) clip.")
    finally:
        os.remove(path)
    result["keyframes"] = [_jpeg(k) for k in result["keyframes"]]
    result["limited_to_seconds"] = MAX_SECONDS
    return result


@app.post("/api/siren")
async def siren(audio: UploadFile = File(...)):
    path = await _save(audio, ".wav")
    try:
        samples, rate = read_wav(path)
    except Exception:
        raise HTTPException(400, "Could not read that file. Upload a PCM WAV recording.")
    finally:
        os.remove(path)
    sd = SirenDetector()
    frames = sd.scan(samples, rate)
    step = max(1, len(frames) // 240)
    return {
        "seconds": round(len(samples) / rate, 2),
        "band_hz": list(sd.cfg.band),
        "threshold": sd.cfg.ratio,
        "intervals": [[round(a, 2), round(b, 2)] for a, b in sd.intervals(frames)],
        "timeline": [{"t": round(f.t, 2), "ratio": round(f.ratio, 2), "siren": f.siren} for f in frames[::step]],
    }
