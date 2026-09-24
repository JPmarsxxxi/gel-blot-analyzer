"""Image I/O and manual adjustment (crop/rotate/brightness/contrast).

All functions operate on numpy arrays in [0, 255] float space unless noted.
"""
import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # Perceptual luma weights; gels are usually single-channel anyway but
    # some scans/exports come in as RGB.
    weights = np.array([0.2126, 0.7152, 0.0722])
    return (rgb.astype(np.float64) * weights).sum(axis=2)


def save_rgb(rgb: np.ndarray, path: str) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def apply_adjustments(
    rgb: np.ndarray,
    crop_x: float = 0.0,
    crop_y: float = 0.0,
    crop_w: float | None = None,
    crop_h: float | None = None,
    rotate_deg: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Apply crop -> rotate -> brightness/contrast, in that order.

    crop_* are in pixel coordinates of the *source* image. brightness is
    additive (-100..100), contrast is multiplicative (0.1..3.0), matching
    the sliders exposed in the UI.
    """
    img = Image.fromarray(rgb)

    if crop_w and crop_h:
        left = int(round(crop_x))
        top = int(round(crop_y))
        right = int(round(crop_x + crop_w))
        bottom = int(round(crop_y + crop_h))
        img = img.crop((left, top, right, bottom))

    if rotate_deg:
        img = img.rotate(-rotate_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

    arr = np.asarray(img, dtype=np.float64)
    if contrast != 1.0:
        arr = (arr - 128.0) * contrast + 128.0
    if brightness:
        arr = arr + brightness
    arr = np.clip(arr, 0, 255)

    return arr.astype(np.uint8)
