"""Automatic vertical lane detection from the band-probability map.

Lanes are found from where the band model sees bands, not from raw brightness:
a glowing gel slab, smears and well edges all make brightness-based column
profiles plateau, which merged neighboring lanes on most real gels. Lane
centers are peaks of the column profile of band pixels; boundaries sit at the
emptiest column between neighbors, and the outer lanes extend half a lane
pitch past their center instead of running to the image edge.

Parameters were tuned on the GelGenie validation gels with
app/training/evaluate_lanes.py.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

BAND_THRESHOLD = 0.4
MIN_PROMINENCE_FRAC = 0.02
MIN_DISTANCE_FRAC = 0.012
SMOOTH_FRAC = 0.004


@dataclass
class LaneBoundary:
    x_start: float
    x_end: float


def detect_lanes(band_prob: np.ndarray) -> list[LaneBoundary]:
    width = band_prob.shape[1]
    profile = (band_prob > BAND_THRESHOLD).sum(axis=0).astype(np.float64)

    if profile.max() <= 0:
        # Nothing band-like anywhere: one full-width lane, so the user still has
        # somewhere to add bands by hand rather than a dead editor.
        return [LaneBoundary(x_start=0.0, x_end=float(width))]

    profile = gaussian_filter1d(profile, sigma=max(1.0, width * SMOOTH_FRAC))
    peaks, _ = find_peaks(
        profile,
        distance=max(3, int(width * MIN_DISTANCE_FRAC)),
        prominence=MIN_PROMINENCE_FRAC * profile.max(),
    )
    if len(peaks) == 0:
        return [LaneBoundary(x_start=0.0, x_end=float(width))]

    pitch = float(np.median(np.diff(peaks))) if len(peaks) > 1 else width * 0.1
    boundaries = []
    for i, peak in enumerate(peaks):
        left = peak - pitch / 2 if i == 0 else peaks[i - 1] + np.argmin(profile[peaks[i - 1]:peak])
        right = peak + pitch / 2 if i == len(peaks) - 1 else peak + np.argmin(profile[peak:peaks[i + 1]])
        boundaries.append(LaneBoundary(x_start=float(max(0.0, left)), x_end=float(min(width, right))))
    return boundaries
