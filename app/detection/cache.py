"""Small per-process cache for the (grayscale, background-corrected signal) pair
of an image, keyed by path + mtime. Avoids re-running rolling-ball background
correction (the slow step) on every single band edit request.
"""
import os
import threading

import numpy as np

from app.detection.imaging import load_rgb, to_grayscale
from app.detection.background import background_subtract

_lock = threading.Lock()
_cache: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
_MAX_ENTRIES = 32


def get_gray_and_signal(path: str) -> tuple[np.ndarray, np.ndarray]:
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]

    rgb = load_rgb(path)
    gray = to_grayscale(rgb)
    signal = background_subtract(gray).signal

    with _lock:
        if len(_cache) >= _MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        _cache[path] = (mtime, gray, signal)

    return gray, signal


def invalidate(path: str) -> None:
    with _lock:
        _cache.pop(path, None)
