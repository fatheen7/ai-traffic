"""AI Traffic Management System: count vehicles with OpenCV, time the signals by demand, clear the way for ambulances."""

from .config import Config
from .detector import VehicleDetector
from .tracker import LineCounter
from .signal import green_times, SignalController
from .siren import SirenDetector

__all__ = ["Config", "VehicleDetector", "LineCounter", "green_times", "SignalController", "SirenDetector"]
__version__ = "1.0.0"
