"""Find a blot membrane sitting in a large, uniform camera frame (raw imager
exports such as a ChemiDoc "merge" image), so it can be cropped before
detection. Deliberately strict: every threshold below was set so that 114 raw
ChemiDoc exports trigger and 685 ordinary gels and blots don't
(app/training/membrane_check.py); a miss just leaves the image as uploaded.
"""
import numpy as np
from scipy import ndimage

WORK_SIZE = 600
MIN_SIDE = 64


def find_membrane(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the crop in full-resolution pixels, or None."""
    h, w = gray.shape
    if min(h, w) < MIN_SIDE:
        return None
    step = max(1, int(round(max(h, w) / WORK_SIZE)))
    g = gray[::step, ::step].astype(np.float32)
    gh, gw = g.shape

    b = max(2, int(0.04 * min(gh, gw)))
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    background = np.median(border)
    mad = np.median(np.abs(border - background))
    if mad > 3:
        return None
    tolerance = max(25.0, 6 * mad)

    mask = ndimage.binary_fill_holes(ndimage.binary_opening(np.abs(g - background) > tolerance, iterations=2))
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    order = np.argsort(areas)[::-1]
    k = int(order[0]) + 1
    if n > 1 and areas[order[1]] > 0.75 * areas[k - 1]:
        return None
    rows, cols = ndimage.find_objects(labels)[k - 1]
    bh, bw = rows.stop - rows.start, cols.stop - cols.start

    if rows.start <= 1 or cols.start <= 1 or rows.stop >= gh - 1 or cols.stop >= gw - 1:
        return None
    if areas[k - 1] < 0.8 * bw * bh:
        return None
    if bw * bh > 0.15 * gw * gh or min(bw / gw, bh / gh) < 0.07:
        return None
    outside = np.ones(mask.shape, bool)
    outside[rows, cols] = False
    if np.mean(np.abs(g[outside] - background) <= tolerance) < 0.97:
        return None
    inner = g[rows, cols][labels[rows, cols] == k]
    if np.median(inner) - background < 25 or np.percentile(inner, 95) - np.percentile(inner, 5) > 80:
        return None

    pad_y, pad_x = 0.02 * bh, 0.02 * bw
    x0 = max(0, int((cols.start - pad_x) * step))
    y0 = max(0, int((rows.start - pad_y) * step))
    x1 = min(w, int(np.ceil((cols.stop + pad_x) * step)))
    y1 = min(h, int(np.ceil((rows.stop + pad_y) * step)))
    return x0, y0, x1 - x0, y1 - y0
