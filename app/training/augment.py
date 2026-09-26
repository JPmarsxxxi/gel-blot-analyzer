"""Training-time augmentation for the band model.

Gels found in the wild differ from GelGenie's lab scans in exposure, gamma,
polarity, blur, noise, compression and slight rotation. Each is applied
randomly to training images (never to validation or the benchmark). The mask
only changes under geometric transforms.
"""
import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    img = image.astype(np.float64)

    if rng.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[:, ::-1]

    if rng.random() < 0.3:
        angle = rng.uniform(-4, 4)
        fill = float(np.median(img))
        img = np.asarray(Image.fromarray(img.astype(np.float32)).rotate(angle, resample=Image.BILINEAR, fillcolor=fill))
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).rotate(angle, resample=Image.NEAREST)) > 127

    if rng.random() < 0.4:
        h, w = img.shape
        s = rng.uniform(0.6, 0.95)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]

    if rng.random() < 0.3:
        img = 255.0 - img

    lo, hi = np.percentile(img, (1, 99))
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    if rng.random() < 0.7:
        norm = norm ** rng.uniform(0.5, 1.8)
    if rng.random() < 0.7:
        a, b = rng.uniform(0.0, 0.25), rng.uniform(0.75, 1.0)
        norm = a + norm * (b - a)
    img = norm * 255.0

    if rng.random() < 0.3:
        img = gaussian_filter(img, sigma=rng.uniform(0.5, 2.0))
    if rng.random() < 0.4:
        img = img + rng.normal(0, rng.uniform(2, 14), size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(rng.integers(15, 85)))
        img = np.asarray(Image.open(buf).convert("L"))

    return np.ascontiguousarray(img), np.ascontiguousarray(mask).astype(np.uint8)
