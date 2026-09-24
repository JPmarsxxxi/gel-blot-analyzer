"""Automatic vertical lane detection from a background-corrected signal image.

A lane shows up as a column of elevated signal running the height of the gel
(bands + smear); gaps between lanes are consistently low. We find lane centers
as peaks of the column-sum profile and place boundaries at the midpoints
between neighboring peaks.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class LaneBoundary:
    x_start: float
    x_end: float


def detect_lanes(signal: np.ndarray, expected_lane_width_frac: float = 0.06) -> list[LaneBoundary]:
    height, width = signal.shape
    col_profile = signal.sum(axis=0)

    if col_profile.max() <= 0:
        return [LaneBoundary(x_start=0.0, x_end=float(width))]

    smooth_sigma = max(2.0, width * 0.005)
    smoothed = gaussian_filter1d(col_profile, sigma=smooth_sigma)

    min_distance = max(5, int(width * expected_lane_width_frac))
    prominence = 0.05 * (smoothed.max() - smoothed.min() + 1e-9)

    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)

    if len(peaks) == 0:
        # No detectable lane structure (e.g. blank/near-empty gel) -- fall back to
        # a single full-width lane so the user still has somewhere to add bands
        # manually, rather than a dead editor with nothing to click on.
        return [LaneBoundary(x_start=0.0, x_end=float(width))]

    boundaries: list[LaneBoundary] = []
    for i, peak in enumerate(peaks):
        left = 0.0 if i == 0 else (peaks[i - 1] + peak) / 2.0
        right = float(width) if i == len(peaks) - 1 else (peak + peaks[i + 1]) / 2.0
        boundaries.append(LaneBoundary(x_start=left, x_end=right))

    return boundaries
