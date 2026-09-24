"""Best-effort "does this look like a gel/blot?" check.

Not a classifier, not a blocker -- per SPEC this only drives a warning banner,
uploads always proceed regardless of the result. Three cheap signals: gels are
low-saturation, background-dominated (foreground pixels are a minority), and
have detectable lane structure.
"""
import numpy as np
from skimage.color import rgb2hsv

from app.detection.lanes import LaneBoundary


def looks_like_gel(rgb: np.ndarray, gray: np.ndarray, lanes: list[LaneBoundary], signal: np.ndarray) -> bool:
    hsv = rgb2hsv(rgb)
    mean_saturation = float(hsv[:, :, 1].mean())
    if mean_saturation > 0.4:
        return False

    if signal.max() > 0:
        fg_frac = float(np.count_nonzero(signal > signal.max() * 0.15)) / signal.size
        if fg_frac > 0.55:
            return False

    # detect_lanes() always returns >=1 lane (falls back to a single full-width
    # lane when it finds no column structure); that fallback case itself is a
    # signal of "doesn't look like a gel" when there's also no band-like signal.
    no_lane_structure = len(lanes) == 1 and lanes[0].x_start == 0.0 and lanes[0].x_end == gray.shape[1]
    if no_lane_structure and signal.max() <= 0:
        return False

    return True
