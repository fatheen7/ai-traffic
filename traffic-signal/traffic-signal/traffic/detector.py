"""Find moving vehicles in a video frame with OpenCV background subtraction (no deep learning).

Per frame:
    resize -> grayscale -> Gaussian blur -> MOG2 -> drop shadows -> open -> dilate -> contours -> area filter
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float

    @property
    def centre(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


class VehicleDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        c = self.cfg
        self.mog = cv2.createBackgroundSubtractorMOG2(history=c.history, varThreshold=c.var_threshold, detectShadows=c.detect_shadows)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.open_kernel, c.open_kernel))
        self.k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.dilate_kernel, c.dilate_kernel))
        self.mask = None

    def resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == self.cfg.width:
            return frame
        return cv2.resize(frame, (self.cfg.width, round(h * self.cfg.width / w)), interpolation=cv2.INTER_AREA)

    def detect(self, frame: np.ndarray, learning_rate: float = -1) -> list[Detection]:
        """frame must already be resized (see resize). Returns vehicles found in this frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.cfg.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.cfg.blur, self.cfg.blur), 0)
        fg = self.mog.apply(gray, learningRate=learning_rate)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # shadows are 127, real foreground 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        if self.cfg.dilate_kernel > 1:
            fg = cv2.dilate(fg, self.k_dilate)
        self.mask = fg
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            found.append(Detection(x, y, w, h, area))
        return found
