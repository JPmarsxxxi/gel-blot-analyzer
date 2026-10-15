"""Small per-process cache for the grayscale image, its background-corrected
signal and its band polarity, keyed by path + mtime. Avoids re-running rolling-ball background
correction (the slow step) on every single band edit request.
"""
import os
import threading

import numpy as np

from app.detection.imaging import load_rgb, to_grayscale
from app.detection.background import background_subtract, detect_polarity

_lock = threading.Lock()
_cache: dict[str, tuple[float, np.ndarray, np.ndarray, str]] = {}
_MAX_ENTRIES = 32


def _polarity(gray: np.ndarray) -> str:
    from app.detection import pipeline  # imported lazily: pipeline imports this module

    if pipeline.DETECTOR == "yolo":
        from app.detection import yolo_infer

        return yolo_infer.detect_polarity(gray)
    return detect_polarity(gray)


def get_gray_and_signal(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2], cached[3]

    rgb = load_rgb(path)
    gray = to_grayscale(rgb)
    polarity = _polarity(gray)
    signal = background_subtract(gray, polarity=polarity).signal

    with _lock:
        if len(_cache) >= _MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        _cache[path] = (mtime, gray, signal, polarity)

    return gray, signal, polarity


def invalidate(path: str) -> None:
    with _lock:
        _cache.pop(path, None)
