"""Automatic background correction (rolling-ball) with band-polarity detection.

Gels/blots come in two flavors: dark bands on a light background (e.g. Coomassie
stain under white light) or bright bands on a dark background (e.g. chemiluminescent
western blot). We auto-detect which one we're looking at via Otsu thresholding
(bands are assumed to be the minority-area class) and orient the rolling-ball
background estimate accordingly, so downstream intensity math always operates on
a "higher = stronger band" signal image.
"""
from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu
from skimage.restoration import rolling_ball


@dataclass
class BackgroundResult:
    signal: np.ndarray  # background-subtracted, oriented so higher = stronger band
    background: np.ndarray  # estimated background, in the same orientation as `signal`
    polarity: str  # "dark" (bands darker than bg) or "bright" (bands brighter than bg)


def detect_polarity(gray: np.ndarray) -> str:
    try:
        t = threshold_otsu(gray)
    except ValueError:
        return "dark"
    dark_area = np.count_nonzero(gray < t)
    bright_area = np.count_nonzero(gray >= t)
    return "dark" if dark_area < bright_area else "bright"


def _rolling_ball_radius(shape: tuple[int, int]) -> float:
    return max(20.0, min(shape) / 8.0)


def background_subtract(gray: np.ndarray, radius: float | None = None) -> BackgroundResult:
    polarity = detect_polarity(gray)
    oriented = (255.0 - gray) if polarity == "dark" else gray

    r = radius if radius is not None else _rolling_ball_radius(gray.shape)
    background = rolling_ball(oriented, radius=r)
    signal = np.clip(oriented - background, 0, None)

    return BackgroundResult(signal=signal, background=background, polarity=polarity)
