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

from app.detection.imaging import enhance_contrast, load_rgb
from app.training.dataset import REAL_SUBSETS, _load_real_pairs
from app.training.synth_data import generate_sample, generate_varied_sample

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


def _write(out_dir: str, split: str, name: str, gray: np.ndarray, mask: np.ndarray, contrast: str | None) -> None:
    if contrast:
        gray = np.clip(enhance_contrast(gray.astype(np.float64), contrast), 0, 255).astype(np.uint8)
    Image.fromarray(gray).save(os.path.join(out_dir, "images", split, name + ".png"))
    with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
        for cx, cy, bw, bh in mask_to_boxes(mask):
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def build(n_synth: int, out_dir: str = OUT_DIR, extra_real: bool = False, varied_synth: bool = False, contrast: str | None = None) -> str:
    """extra_real adds GelGenie's lsdb_gels (images from the RGP-caps archive,
    CC BY-SA 2.1 JP); the benchmark keeps using the original five subsets."""
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, kind, split), exist_ok=True)

    subsets = REAL_SUBSETS + (("lsdb_gels",) if extra_real else ())
    for split in ("train", "val"):
        for img_path, mask_path in _load_real_pairs(split, subsets):
            subset = img_path.split("external_data" + os.sep)[-1].split(os.sep)[0]
            name = f"{subset}__{os.path.splitext(os.path.basename(img_path))[0]}".replace(" ", "_")
            gray = np.asarray(Image.fromarray(load_rgb(img_path)).convert("L"))
            _write(out_dir, split, name, gray, np.asarray(Image.open(mask_path)), contrast)

    for i in range(n_synth):
        sample = (generate_varied_sample if varied_synth else generate_sample)(seed=500_000 + i)
        _write(out_dir, "train", f"synth_{i:04d}", sample.image, sample.mask, contrast)

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: band\n")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=int, default=600)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--extra-real", action="store_true", help="add lsdb_gels to training")
    parser.add_argument("--varied-synth", action="store_true", help="western-blot, protein-gel and strip styles plus distractors")
    parser.add_argument("--contrast", choices=["stretch", "clahe"], default=None, help="bake input contrast normalization into the images")
    args = parser.parse_args()
    print(build(args.synth, args.out_dir, args.extra_real, args.varied_synth, args.contrast))
