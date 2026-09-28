"""Builds a YOLO-format band-detection dataset from the same data as the U-Net.

Run: python -m app.training.yolo_data [--synth 600]

Each connected component of a GelGenie mask becomes one "band" box. Synthetic
gels (synth_data.py) are added to the training split, matching the U-Net's
data mix so the comparison is about architecture, not data. Images are
written as 8-bit grayscale PNGs (16-bit TIFFs rescaled by load_rgb).
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label

from app.detection.imaging import load_rgb
from app.training.dataset import _load_real_pairs
from app.training.synth_data import generate_sample

OUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "experiments", "yolo_ds")
MIN_BAND_AREA = 20


def mask_to_boxes(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(cx, cy, w, h) normalized to [0, 1], one per connected band."""
    h, w = mask.shape
    labeled, _ = label(mask > 0)
    sizes = np.bincount(labeled.ravel())
    boxes = []
    for i, sl in enumerate(find_objects(labeled), start=1):
        if sl is None or sizes[i] < MIN_BAND_AREA:
            continue
        ys, xs = sl
        boxes.append(((xs.start + xs.stop) / 2 / w, (ys.start + ys.stop) / 2 / h, (xs.stop - xs.start) / w, (ys.stop - ys.start) / h))
    return boxes


def _write(split: str, name: str, gray: np.ndarray, mask: np.ndarray) -> None:
    Image.fromarray(gray).save(os.path.join(OUT_DIR, "images", split, name + ".png"))
    with open(os.path.join(OUT_DIR, "labels", split, name + ".txt"), "w") as f:
        for cx, cy, bw, bh in mask_to_boxes(mask):
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def build(n_synth: int) -> str:
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(OUT_DIR, kind, split), exist_ok=True)

    for split in ("train", "val"):
        for img_path, mask_path in _load_real_pairs(split):
            subset = img_path.split("external_data" + os.sep)[-1].split(os.sep)[0]
            name = f"{subset}_{os.path.splitext(os.path.basename(img_path))[0]}".replace(" ", "_")
            gray = np.asarray(Image.fromarray(load_rgb(img_path)).convert("L"))
            _write(split, name, gray, np.asarray(Image.open(mask_path)))

    for i in range(n_synth):
        sample = generate_sample(seed=500_000 + i)
        _write("train", f"synth_{i:04d}", sample.image, sample.mask)

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: band\n")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=int, default=600)
    args = parser.parse_args()
    print(build(args.synth))
