"""All tunable numbers in one place."""

from dataclasses import dataclass, field

APPROACHES = ("north", "east", "south", "west")


@dataclass
class DetectorConfig:
    width: int = 480            # frames are resized to this width before processing
    blur: int = 5               # Gaussian blur kernel (odd)
    history: int = 300          # MOG2: frames used to learn the background
    var_threshold: float = 24   # MOG2: how different a pixel must be to count as foreground
    detect_shadows: bool = True # MOG2 marks shadows as grey (127); we drop them
    min_area: int = 120         # smallest shape (pixels, after resize) counted as a vehicle
    open_kernel: int = 3
    dilate_kernel: int = 5


@dataclass
class CounterConfig:
    line: float = 0.6           # counting line, as a fraction of frame height
    direction: str = "down"     # count vehicles moving "down", "up" or "both" across the line
    max_jump: float = 0.12      # max centroid movement between frames, as a fraction of width
    keep_for: int = 8           # frames a track survives without a match


@dataclass
class SignalConfig:
    cycle: float = 120          # seconds of green shared out per cycle
    min_green: float = 10
    max_green: float = 60
    yellow: float = 3
    all_red: float = 2
    first_green: float = 30     # green per road in the first cycle, before any counts exist


@dataclass
class SirenConfig:
    band: tuple = (600.0, 1600.0)  # Hz, where a wail siren sweeps
    window: float = 0.25           # seconds of audio per FFT
    ratio: float = 4.0             # band energy must be this many times the rest of the spectrum
    hold: float = 0.8              # ...for this many seconds before we call it a siren


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    siren: SirenConfig = field(default_factory=SirenConfig)
