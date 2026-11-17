import cv2
import numpy as np

from traffic.config import Config
from traffic.detector import VehicleDetector
from traffic.junction import make_counter


def frames(n=90, w=480, h=270, speed=6):
    """A grey road with one car driving down the frame."""
    rng = np.random.default_rng(1)
    for i in range(n):
        f = np.full((h, w, 3), 60, np.uint8)
        f = np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
        y = -30 + i * speed
        cv2.rectangle(f, (270, y), (292, y + 38), (40, 160, 220), -1)
        yield f


def test_moving_car_is_detected_and_counted_once():
    cfg = Config()
    det = VehicleDetector(cfg.detector)
    counter = None
    seen = 0
    for f in frames():
        found = det.detect(det.resize(f))
        seen += bool(found)
        counter = counter or make_counter(f.shape, cfg)
        counter.update([d.centre for d in found])
    assert seen > 30
    assert counter.down == 1


def test_static_scene_has_no_detections_once_learned():
    cfg = Config()
    det = VehicleDetector(cfg.detector)
    f = np.full((270, 480, 3), 60, np.uint8)
    for _ in range(60):
        found = det.detect(f)
    assert found == []
