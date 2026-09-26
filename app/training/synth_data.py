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


def _add_artifacts(img: np.ndarray, rng: np.random.Generator, dark_bands: bool) -> None:
    """Dust, scratches and well marks: band-like distractors that are NOT bands
    (left out of the mask), so the model learns to ignore them."""
    h, w = img.shape
    sign = -1.0 if dark_bands else 1.0
    for _ in range(int(rng.integers(0, 40))):
        y, x, r = rng.integers(0, h), rng.integers(0, w), rng.uniform(0.6, 2.5)
        y0, y1, x0, x1 = max(0, int(y - r)), min(h, int(y + r) + 1), max(0, int(x - r)), min(w, int(x + r) + 1)
        img[y0:y1, x0:x1] += sign * rng.uniform(20, 90) * rng.choice([-1.0, 1.0], p=[0.3, 0.7])
    for _ in range(int(rng.integers(0, 3))):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        length, angle = rng.uniform(0.1, 0.5) * w, rng.uniform(0, np.pi)
        for t in np.linspace(0, 1, int(length)):
            xx, yy = int(x0 + t * length * np.cos(angle)), int(y0 + t * length * np.sin(angle))
            if 0 <= xx < w and 0 <= yy < h:
                img[yy, xx] += sign * rng.uniform(15, 50)


def generate_blot_sample(seed: int | None = None, style: str | None = None) -> SynthSample:
    """Western-blot and protein-gel styles, unlike generate_sample's DNA gels:
    the same protein sits at the same height across lanes with varying amount,
    bands are flat-topped and may saturate, rows can "smile", and blots are
    often cropped to strips."""
    rng = np.random.default_rng(seed)
    style = style or str(rng.choice(["western", "coomassie", "strip"]))
    width = int(rng.integers(360, 900))
    height = int(rng.integers(60, 180)) if style == "strip" else int(rng.integers(260, 640))
    dark_bands = style != "western" or rng.random() < 0.7

    base_bg = rng.uniform(170, 240) if dark_bands else rng.uniform(10, 60)
    illum = _smooth_noise((height, width), sigma=max(width, height) / 5, rng=rng)
    illum = (illum - illum.min()) / (illum.max() - illum.min() + 1e-9) * rng.uniform(5, 40)
    blotch = _smooth_noise((height, width), sigma=rng.uniform(4, 15), rng=rng) * rng.uniform(0, 8)
    img = np.full((height, width), base_bg) + illum + blotch

    n_lanes = int(rng.integers(2, 16))
    margin = width * rng.uniform(0.01, 0.08)
    pitch = (width - 2 * margin) / n_lanes
    lane_w = pitch * rng.uniform(0.6, 0.9)
    n_rows = int(rng.integers(1, 3)) if style in ("western", "strip") else int(rng.integers(4, 14))
    rows = np.sort(rng.uniform(0.08, 0.92, n_rows)) * height
    band_h = height * (rng.uniform(0.08, 0.25) if style == "strip" else rng.uniform(0.012, 0.04))
    smile = rng.uniform(-0.15, 0.25) * band_h
    saturation = rng.uniform(60, 200)

    signal = np.zeros((height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    lanes, bands = [], []
    yy, xx = np.mgrid[0:height, 0:width]
    for li in range(n_lanes):
        center = margin + pitch * (li + 0.5) + rng.uniform(-0.05, 0.05) * pitch
        lanes.append(SynthLane(x_start=max(0.0, center - lane_w / 2), x_end=min(float(width), center + lane_w / 2)))
        for y in rows:
            if rng.random() < 0.15:
                continue
            amount = rng.uniform(10, 260) * (rng.uniform(0.05, 0.4) if rng.random() < 0.2 else 1.0)
            half_w = lane_w / 2 * rng.uniform(0.8, 1.0)
            bx0, bx1 = int(max(0, center - half_w)), int(min(width, center + half_w))
            by0, by1 = int(max(0, y - band_h * 2)), int(min(height, y + band_h * 2))
            if bx1 <= bx0 or by1 <= by0:
                continue
            ly, lx = yy[by0:by1, bx0:bx1], xx[by0:by1, bx0:bx1]
            u = (lx - center) / half_w
            y_c = y + smile * (1 - u**2)
            profile_x = np.clip(1.4 - np.abs(u) ** 4 * 1.4, 0, 1)
            profile_y = np.exp(-((ly - y_c) ** 2) / (2 * (band_h / 2.4) ** 2))
            band = amount * profile_x * profile_y
            band = saturation * np.tanh(band / saturation)
            signal[by0:by1, bx0:bx1] += band
            fg = band > max(4.0, band.max() * 0.3)
            if fg.any():
                mask[by0:by1, bx0:bx1] |= fg.astype(np.uint8)
                bands.append(SynthBand(lane_index=li, x=float(bx0), y=float(by0), width=float(bx1 - bx0), height=float(by1 - by0)))
        if rng.random() < 0.2:
            sx0, sx1 = int(max(0, center - lane_w * 0.3)), int(min(width, center + lane_w * 0.3))
            signal[: int(height * 0.8), sx0:sx1] += rng.uniform(3, 15) * np.linspace(1, 0.1, int(height * 0.8))[:, None]

    if style != "strip" and rng.random() < 0.5:
        well_y = int(height * rng.uniform(0.01, 0.05))
        for lane in lanes:
            img[well_y:well_y + 3, int(lane.x_start):int(lane.x_end)] += (-1 if dark_bands else 1) * rng.uniform(20, 60)

    img = img - signal if dark_bands else img + signal
    _add_artifacts(img, rng, dark_bands)
    img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    return SynthSample(image=img, mask=mask, lanes=lanes, bands=bands, polarity="dark" if dark_bands else "bright")


def generate_varied_sample(seed: int | None = None) -> SynthSample:
    """Half DNA-gel style (with distractor artifacts added), half blot styles."""
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        return generate_blot_sample(seed=None if seed is None else seed + 7_000_001)
    sample = generate_sample(seed=None if seed is None else seed + 9_000_001)
    img = sample.image.astype(np.float64)
    _add_artifacts(img, rng, sample.polarity == "dark")
    sample.image = np.clip(img, 0, 255).astype(np.uint8)
    return sample
