"""Per-lane band detection: ML segmentation mask -> connected components ->
classical peak-boundary refinement -> background-subtracted intensity.

The ML model (app/detection/ml_infer.py) gives per-pixel band probability.
Thresholding + connected components locates candidate bands; the classical
intensity-profile step then snaps each box's vertical extent to the actual
peak in the background-subtracted signal, which is sharper and less blurry
than the (256x256-resized) ML mask alone.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths
from skimage.measure import label, regionprops

from app.detection.lanes import LaneBoundary

LOW_CONFIDENCE_CUTOFF = 0.6


@dataclass
class DetectedBand:
    x: float
    y: float
    width: float
    height: float
    intensity: float
    confidence: float
    low_confidence: bool
    percent_of_lane: float = 0.0


def sensitivity_to_threshold(sensitivity: float) -> float:
    """Higher sensitivity -> lower probability threshold -> more bands kept."""
    sensitivity = min(1.0, max(0.0, sensitivity))
    return max(0.12, min(0.8, 0.78 - sensitivity * 0.58))


def _refine_y_range(signal: np.ndarray, x0: int, x1: int, y0: int, y1: int, full_height: int) -> tuple[int, int]:
    center = (y0 + y1) // 2
    win_half = max(int((y1 - y0) * 1.5), 10)
    wy0 = max(0, center - win_half)
    wy1 = min(full_height, center + win_half)
    if x1 <= x0 or wy1 <= wy0:
        return y0, y1

    profile = signal[wy0:wy1, x0:x1].sum(axis=1)
    if profile.max() <= 0:
        return y0, y1

    peaks, _ = find_peaks(profile)
    if len(peaks) == 0:
        return y0, y1

    target_idx = int(np.argmin(np.abs(peaks - (center - wy0))))
    peak = peaks[target_idx]

    _, _, left_ips, right_ips = peak_widths(profile, [peak], rel_height=0.7)
    ry0 = wy0 + int(round(left_ips[0]))
    ry1 = wy0 + int(round(right_ips[0]))
    if ry1 <= ry0:
        return y0, y1
    return ry0, ry1


def detect_bands_in_lane(
    signal: np.ndarray,
    prob_mask: np.ndarray,
    lane: LaneBoundary,
    sensitivity: float,
) -> list[DetectedBand]:
    height, width = signal.shape
    lx0 = max(0, int(round(lane.x_start)))
    lx1 = min(width, int(round(lane.x_end)))
    if lx1 <= lx0:
        return []

    threshold = sensitivity_to_threshold(sensitivity)
    lane_prob = prob_mask[:, lx0:lx1]
    binary = lane_prob >= threshold

    labeled = label(binary)
    min_area = max(4, int((lx1 - lx0) * height * 0.0008))

    bands: list[DetectedBand] = []
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        ry0, rx0, ry1, rx1 = region.bbox  # local to lane_prob: (min_row, min_col, max_row, max_col)
        gx0, gx1 = lx0 + rx0, lx0 + rx1
        gy0, gy1 = _refine_y_range(signal, gx0, gx1, ry0, ry1, height)

        region_mask = labeled[ry0:ry1, rx0:rx1] == region.label
        confidence = float(lane_prob[ry0:ry1, rx0:rx1][region_mask].mean())

        box_signal = signal[gy0:gy1, gx0:gx1]
        intensity = float(box_signal.sum()) if box_signal.size else 0.0

        bands.append(
            DetectedBand(
                x=float(gx0),
                y=float(gy0),
                width=float(max(1, gx1 - gx0)),
                height=float(max(1, gy1 - gy0)),
                intensity=intensity,
                confidence=confidence,
                low_confidence=confidence < LOW_CONFIDENCE_CUTOFF,
            )
        )

    _assign_percent_of_lane(bands)
    bands.sort(key=lambda b: b.y)
    return bands


def _assign_percent_of_lane(bands: list[DetectedBand]) -> None:
    total = sum(b.intensity for b in bands)
    if total > 0:
        for b in bands:
            b.percent_of_lane = (b.intensity / total) * 100.0
    else:
        for b in bands:
            b.percent_of_lane = 0.0


def recompute_percent_of_lane(intensities: list[float]) -> list[float]:
    """Used when bands are manually edited/added and percentages need a refresh."""
    total = sum(intensities)
    if total <= 0:
        return [0.0 for _ in intensities]
    return [(v / total) * 100.0 for v in intensities]
