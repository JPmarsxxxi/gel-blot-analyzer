"""PyTorch Dataset combining on-the-fly synthetic samples with the real
GelGenie (Dunn Lab / University of Edinburgh, CC-BY-4.0, Zenodo 13218469)
gel-image + hand-labeled segmentation-mask pairs.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.detection.imaging import enhance_contrast, load_rgb
from app.training.synth_data import generate_sample

EXTERNAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "external_data")
# The GelGenie subsets with images included (lsdb_gels ships masks only).
REAL_SUBSETS = ("nathan_gels", "matthew_gels", "matthew_gels_2", "quantitation_ladder_gels", "stella_gels_for_finetuning")

TARGET_SIZE = (256, 256)  # (H, W), must be divisible by 16 for the 4-level U-Net


def _resize_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_im = Image.fromarray(image).resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
    mask_im = Image.fromarray(mask * 255).resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.NEAREST)
    return np.array(img_im, dtype=np.uint8), (np.array(mask_im) > 127).astype(np.uint8)


def _load_real_pairs(split: str) -> list[tuple[str, str]]:
    subdir = {"train": ("images", "masks"), "val": ("val_images", "val_masks"), "test": ("test_images", "test_masks")}[split]
    pairs = []
    for subset in REAL_SUBSETS:
        img_dir = os.path.join(EXTERNAL_DATA_DIR, subset, subset, subdir[0])
        mask_dir = os.path.join(EXTERNAL_DATA_DIR, subset, subset, subdir[1])
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.startswith("."):
                continue
            stem = os.path.splitext(fname)[0]
            for ext in (".tif", ".tiff", ".png"):
                candidate = os.path.join(mask_dir, stem + ext)
                if os.path.exists(candidate):
                    pairs.append((os.path.join(img_dir, fname), candidate))
                    break
    return pairs


_real_cache: dict[tuple[str, str, str | None], tuple[np.ndarray, np.ndarray]] = {}


def _to_model_input(img: np.ndarray, contrast: str | None) -> np.ndarray:
    # Applied at native resolution, before resizing, to match ml_infer.
    return np.clip(enhance_contrast(img.astype(np.float64), contrast), 0, 255).astype(np.uint8) if contrast else img


def _load_real_sample(img_path: str, mask_path: str, contrast: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns the pair already resized to TARGET_SIZE, cached: full-size
    16-bit TIFFs take far longer to decode than an epoch's compute."""
    key = (img_path, mask_path, contrast)
    if key not in _real_cache:
        img = np.asarray(Image.fromarray(load_rgb(img_path)).convert("L"), dtype=np.uint8)
        img = _to_model_input(img, contrast)
        mask = np.asarray(Image.open(mask_path))
        mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
        _real_cache[key] = _resize_pair(img, mask)
    return _real_cache[key]


class GelSegmentationDataset(Dataset):
    """split: "train" | "val" | "test".

    For "train", synthetic samples are generated on the fly (effectively
    unbounded) and mixed with every real training image once per epoch.
    """

    def __init__(self, split: str = "train", synth_per_epoch: int = 400, seed: int | None = None, contrast: str | None = None):
        self.split = split
        self.contrast = contrast
        self.synth_per_epoch = synth_per_epoch if split == "train" else max(20, synth_per_epoch // 10)
        self.real_pairs = _load_real_pairs(split)
        self.rng_seed = seed

    def __len__(self) -> int:
        return self.synth_per_epoch + len(self.real_pairs)

    def __getitem__(self, idx: int):
        if idx < self.synth_per_epoch:
            seed = None if self.rng_seed is None else self.rng_seed * 100003 + idx
            sample = generate_sample(seed=seed)
            image, mask = _to_model_input(sample.image, self.contrast), sample.mask
        else:
            img_path, mask_path = self.real_pairs[idx - self.synth_per_epoch]
            image, mask = _load_real_sample(img_path, mask_path, self.contrast)
        if image.shape != TARGET_SIZE:
            image, mask = _resize_pair(image, mask)

        if self.split == "train" and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        img_t = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t
