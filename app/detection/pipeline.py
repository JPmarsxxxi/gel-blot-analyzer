"""Top-level detection pipeline tying together imaging, background correction,
ML inference, lane detection, and band detection. This is what the server
routes call; nothing here touches the database.
"""
from dataclasses import dataclass

import numpy as np

from app.detection import imaging, ml_infer
from app.detection.bands import DetectedBand, detect_bands_in_lane
from app.detection.cache import get_gray_and_signal, invalidate
from app.detection.gel_heuristic import looks_like_gel
from app.detection.lanes import LaneBoundary, detect_lanes


@dataclass
class DetectionResult:
    is_gel_like: bool
    lanes: list[LaneBoundary]
    bands_by_lane: list[list[DetectedBand]]


def run_full_detection(path: str, sensitivity: float = 0.5) -> DetectionResult:
    rgb = imaging.load_rgb(path)
    gray, signal = get_gray_and_signal(path)
    prob_mask = ml_infer.predict_band_probability(gray)
    lanes = detect_lanes(prob_mask)

    bands_by_lane = [detect_bands_in_lane(signal, prob_mask, lane, sensitivity) for lane in lanes]
    is_gel = looks_like_gel(rgb, gray, lanes, signal)

    return DetectionResult(is_gel_like=is_gel, lanes=lanes, bands_by_lane=bands_by_lane)


def run_band_redetection(
    path: str, lanes: list[LaneBoundary], sensitivity: float
) -> list[list[DetectedBand]]:
    """Re-runs only band detection (e.g. sensitivity slider change), keeping
    the given (possibly user-adjusted) lane boundaries."""
    gray, signal = get_gray_and_signal(path)
    prob_mask = ml_infer.predict_band_probability(gray)
    return [detect_bands_in_lane(signal, prob_mask, lane, sensitivity) for lane in lanes]


def get_signal(path: str) -> np.ndarray:
    _, signal = get_gray_and_signal(path)
    return signal


def apply_image_adjustments(
    source_path: str,
    dest_path: str,
    crop_x: float,
    crop_y: float,
    crop_w: float | None,
    crop_h: float | None,
    rotate_deg: float,
    brightness: float,
    contrast: float,
) -> tuple[int, int]:
    """Applies adjustments and writes the result to dest_path. Returns (width, height)."""
    rgb = imaging.load_rgb(source_path)
    adjusted = imaging.apply_adjustments(
        rgb,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        rotate_deg=rotate_deg,
        brightness=brightness,
        contrast=contrast,
    )
    imaging.save_rgb(adjusted, dest_path)
    invalidate(dest_path)
    h, w = adjusted.shape[:2]
    return w, h
