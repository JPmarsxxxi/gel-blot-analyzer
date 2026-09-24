"""Synthetic gel/blot image generator with pixel-accurate ground truth.

Produces (image, mask) pairs matching the format of the real GelGenie dataset
(grayscale image, binary foreground/background mask) so both can feed the same
training pipeline, plus richer metadata (lane boundaries, band boxes) that the
classical detection code can be validated against independently.
"""
import random
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SynthBand:
    lane_index: int
    x: float
    y: float
    width: float
    height: float


@dataclass
class SynthLane:
    x_start: float
    x_end: float


@dataclass
class SynthSample:
    image: np.ndarray  # uint8 HxW
    mask: np.ndarray  # uint8 HxW, {0,1}
    lanes: list[SynthLane] = field(default_factory=list)
    bands: list[SynthBand] = field(default_factory=list)
    polarity: str = "dark"


def _smooth_noise(shape, sigma, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 1, size=shape)
    return gaussian_filter(noise, sigma=sigma)


def generate_sample(
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> SynthSample:
    rng = np.random.default_rng(seed)
    width = width or int(rng.integers(420, 820))
    height = height or int(rng.integers(320, 620))

    polarity = "dark" if rng.random() < 0.75 else "bright"
    base_bg = rng.uniform(190, 235) if polarity == "dark" else rng.uniform(15, 55)

    # Smoothly varying illumination field (uneven lighting is common in real scans).
    illum = _smooth_noise((height, width), sigma=max(width, height) / 6, rng=rng)
    illum = (illum - illum.min()) / (illum.max() - illum.min() + 1e-9)
    illum = (illum - 0.5) * rng.uniform(5, 25)

    fine_noise = rng.normal(0, rng.uniform(1.5, 5.0), size=(height, width))

    img = np.full((height, width), base_bg, dtype=np.float64) + illum + fine_noise

    mask = np.zeros((height, width), dtype=np.uint8)
    signal = np.zeros((height, width), dtype=np.float64)  # additive band signal, positive

    n_lanes = int(rng.integers(3, 13))
    margin = width * rng.uniform(0.02, 0.06)
    usable = width - 2 * margin
    lane_pitch = usable / n_lanes

    lanes: list[SynthLane] = []
    bands: list[SynthBand] = []

    for li in range(n_lanes):
        center = margin + lane_pitch * (li + 0.5) + rng.uniform(-lane_pitch * 0.08, lane_pitch * 0.08)
        lane_w = lane_pitch * rng.uniform(0.55, 0.85)
        x_start = center - lane_w / 2
        x_end = center + lane_w / 2
        lanes.append(SynthLane(x_start=max(0, x_start), x_end=min(width, x_end)))

        n_bands = int(rng.integers(0, 9))
        top_margin = height * rng.uniform(0.04, 0.1)
        bottom_margin = height * rng.uniform(0.04, 0.1)
        usable_h = height - top_margin - bottom_margin

        # Occasional vertical smear/streak under one or more bands.
        has_smear = rng.random() < 0.15

        placed_y: list[float] = []
        for _bi in range(n_bands):
            attempts = 0
            while attempts < 8:
                y = top_margin + rng.uniform(0, usable_h)
                if all(abs(y - py) > height * 0.02 for py in placed_y):
                    break
                attempts += 1
            placed_y.append(y)

            band_h = rng.uniform(height * 0.012, height * 0.045)
            band_w = lane_w * rng.uniform(0.55, 1.0)
            intensity = rng.uniform(25, 140) if rng.random() > 0.15 else rng.uniform(8, 25)

            bx0 = int(max(0, center - band_w / 2))
            bx1 = int(min(width, center + band_w / 2))
            by0 = int(max(0, y - band_h * 2))
            by1 = int(min(height, y + band_h * 2))
            if bx1 <= bx0 or by1 <= by0:
                continue

            yy, xx = np.mgrid[by0:by1, bx0:bx1]
            sigma_y = band_h / 2.2
            sigma_x = band_w / 2.6
            gauss = intensity * np.exp(
                -(((yy - y) ** 2) / (2 * sigma_y**2) + ((xx - center) ** 2) / (2 * sigma_x**2))
            )
            signal[by0:by1, bx0:bx1] += gauss

            fg = gauss > (intensity * 0.28)
            mask[by0:by1, bx0:bx1] |= fg.astype(np.uint8)

            bands.append(
                SynthBand(
                    lane_index=li,
                    x=float(bx0),
                    y=float(max(0, y - sigma_y * 1.4)),
                    width=float(bx1 - bx0),
                    height=float(min(height, sigma_y * 2.8)),
                )
            )

        if has_smear and placed_y:
            smear_y0 = int(top_margin)
            smear_y1 = int(min(placed_y) if placed_y else height * 0.5)
            if smear_y1 > smear_y0:
                sx0 = int(max(0, center - lane_w * 0.3))
                sx1 = int(min(width, center + lane_w * 0.3))
                smear_strength = rng.uniform(5, 20)
                fade = np.linspace(0.2, 1.0, smear_y1 - smear_y0)[:, None]
                signal[smear_y0:smear_y1, sx0:sx1] += smear_strength * fade

    if polarity == "dark":
        img = img - signal
    else:
        img = img + signal

    img = np.clip(img + rng.normal(0, 1.0, size=img.shape), 0, 255).astype(np.uint8)

    return SynthSample(image=img, mask=mask, lanes=lanes, bands=bands, polarity=polarity)


def generate_batch(n: int, seed: int | None = None) -> list[SynthSample]:
    base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    return [generate_sample(seed=base_seed + i) for i in range(n)]
