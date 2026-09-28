"""YOLO band detector: an alternative to the U-Net + connected-components path.

Boxes come straight from the detector. Lanes still come from detect_lanes,
fed a map painted from the boxes; intensity is still measured on the
background-corrected signal, so quantification is identical between detectors.
"""
import os
import threading

import numpy as np

from app.config import BASE_DIR
from app.detection.bands import DetectedBand, _assign_percent_of_lane
from app.detection.lanes import LaneBoundary

YOLO_PATH = os.environ.get("GEL_YOLO_PATH") or os.path.join(BASE_DIR, "app", "training", "artifacts", "band_detector_yolo.pt")
LOW_CONFIDENCE_CUTOFF = 0.5
MIN_CONFIDENCE = 0.05

_lock = threading.Lock()
_model = None


def _load():
    global _model
    with _lock:
        if _model is None:
            from ultralytics import YOLO

            _model = YOLO(YOLO_PATH)
    return _model


def available() -> bool:
    return os.path.exists(YOLO_PATH)


def sensitivity_to_confidence(sensitivity: float) -> float:
    """Higher sensitivity -> lower confidence threshold -> more boxes kept."""
    return max(MIN_CONFIDENCE, 0.5 - 0.45 * min(1.0, max(0.0, sensitivity)))


def predict_boxes(gray: np.ndarray) -> np.ndarray:
    """(N, 5) array of x0, y0, x1, y1, confidence in image pixels."""
    model = _load()
    rgb = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    imgsz = int(model.overrides.get("imgsz", 640))
    result = model.predict(rgb, imgsz=imgsz, conf=MIN_CONFIDENCE, iou=0.5, max_det=2000, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 5), np.float32)
    return np.hstack([result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)


def box_map(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    prob = np.zeros(shape, np.float32)
    for x0, y0, x1, y1, conf in boxes:
        region = prob[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        np.maximum(region, conf, out=region)
    return prob


def bands_in_lane(signal: np.ndarray, boxes: np.ndarray, lane: LaneBoundary, sensitivity: float) -> list[DetectedBand]:
    threshold = sensitivity_to_confidence(sensitivity)
    h, w = signal.shape
    bands = []
    for x0, y0, x1, y1, conf in boxes:
        cx = (x0 + x1) / 2
        if conf < threshold or not (lane.x_start <= cx < lane.x_end):
            continue
        gx0, gy0 = max(0, int(x0)), max(0, int(y0))
        gx1, gy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        box_signal = signal[gy0:gy1, gx0:gx1]
        bands.append(
            DetectedBand(
                x=float(gx0),
                y=float(gy0),
                width=float(max(1, gx1 - gx0)),
                height=float(max(1, gy1 - gy0)),
                intensity=float(box_signal.sum()) if box_signal.size else 0.0,
                confidence=float(conf),
                low_confidence=bool(conf < LOW_CONFIDENCE_CUTOFF),
            )
        )
    _assign_percent_of_lane(bands)
    bands.sort(key=lambda b: b.y)
    return bands
