"""Detect an emergency siren in audio with an FFT.

A wail siren sweeps roughly 600-1,600 Hz. For each short window we compare the average power inside
that band with the average power outside it. When the band stays several times stronger for long
enough, we call it a siren. Traffic noise is mostly low-frequency rumble, so it rarely passes.
"""

import wave
from dataclasses import dataclass

import numpy as np

from .config import SirenConfig


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Mono float samples in [-1, 1] and the sample rate. Supports 8/16/32-bit PCM WAV."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if width == 1:
        x = (x - 128) / 128
    else:
        x /= float(np.iinfo(dtype).max)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


@dataclass
class SirenFrame:
    t: float          # window start, seconds
    ratio: float      # band power / out-of-band power
    peak_hz: float    # strongest frequency in the window
    siren: bool       # detector state after this window


class SirenDetector:
    def __init__(self, cfg: SirenConfig | None = None):
        self.cfg = cfg or SirenConfig()

    def band_ratio(self, samples: np.ndarray, rate: int) -> tuple[float, float]:
        win = samples * np.hanning(len(samples))
        power = np.abs(np.fft.rfft(win)) ** 2
        freqs = np.fft.rfftfreq(len(samples), 1 / rate)
        lo, hi = self.cfg.band
        keep = (freqs > 50) & (freqs < 4000)
        inside = keep & (freqs >= lo) & (freqs <= hi)
        outside = keep & ~inside
        ratio = power[inside].mean() / (power[outside].mean() + 1e-12)
        peak = float(freqs[keep][np.argmax(power[keep])])
        return float(ratio), peak

    def scan(self, samples: np.ndarray, rate: int) -> list[SirenFrame]:
        """Run over a whole recording. The detector turns on after `hold` seconds above the threshold
        and off after `hold` seconds below it."""
        step = int(self.cfg.window * rate)
        frames, above, below, on = [], 0.0, 0.0, False
        for start in range(0, len(samples) - step + 1, step):
            ratio, peak = self.band_ratio(samples[start:start + step], rate)
            if ratio >= self.cfg.ratio:
                above += self.cfg.window
                below = 0.0
            else:
                below += self.cfg.window
                above = 0.0
            if not on and above >= self.cfg.hold:
                on = True
            elif on and below >= self.cfg.hold * 3:
                on = False
            frames.append(SirenFrame(start / rate, ratio, peak, on))
        return frames

    def intervals(self, frames: list[SirenFrame]) -> list[tuple[float, float]]:
        """(start, end) seconds where the siren was detected."""
        out, start = [], None
        for f in frames:
            if f.siren and start is None:
                start = f.t
            elif not f.siren and start is not None:
                out.append((start, f.t))
                start = None
        if start is not None:
            out.append((start, frames[-1].t + self.cfg.window))
        return out
