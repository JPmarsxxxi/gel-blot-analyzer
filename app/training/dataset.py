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

from app.training.synth_data import generate_sample

EXTERNAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "external_data", "nathan_gels", "nathan_gels")

TARGET_SIZE = (256, 256)  # (H, W), must be divisible by 16 for the 4-level U-Net


def _resize_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_im = Image.fromarray(image).resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
    mask_im = Image.fromarray(mask * 255).resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.NEAREST)
    return np.array(img_im, dtype=np.uint8), (np.array(mask_im) > 127).astype(np.uint8)


def _load_real_pairs(split: str) -> list[tuple[str, str]]:
    subdir = {"train": ("images", "masks"), "val": ("val_images", "val_masks"), "test": ("test_images", "test_masks")}[split]
    img_dir = os.path.join(EXTERNAL_DATA_DIR, subdir[0])
    mask_dir = os.path.join(EXTERNAL_DATA_DIR, subdir[1])

    pairs = []
    if not os.path.isdir(img_dir):
        return pairs
    for fname in sorted(os.listdir(img_dir)):
        if fname.startswith("."):
            continue
        stem = os.path.splitext(fname)[0]
        mask_path = None
        for ext in (".tif", ".tiff", ".png"):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path:
            pairs.append((os.path.join(img_dir, fname), mask_path))
    return pairs


def _load_real_sample(img_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path))
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    return img, mask


class GelSegmentationDataset(Dataset):
    """split: "train" | "val" | "test".

    For "train", synthetic samples are generated on the fly (effectively
    unbounded) and mixed with the real training images (oversampled to give
    real data meaningful weight despite being a small fraction of the epoch).
    """

    def __init__(self, split: str = "train", synth_per_epoch: int = 400, seed: int | None = None):
        self.split = split
        self.synth_per_epoch = synth_per_epoch if split == "train" else max(20, synth_per_epoch // 10)
        self.real_pairs = _load_real_pairs(split)
        self.rng_seed = seed
        # Oversample real pairs so they aren't drowned out by synthetic volume.
        self.real_repeat = 6 if split == "train" else 1

    def __len__(self) -> int:
        return self.synth_per_epoch + len(self.real_pairs) * self.real_repeat

    def __getitem__(self, idx: int):
        if idx < self.synth_per_epoch:
            seed = None if self.rng_seed is None else self.rng_seed * 100003 + idx
            sample = generate_sample(seed=seed)
            image, mask = sample.image, sample.mask
        else:
            real_idx = (idx - self.synth_per_epoch) % len(self.real_pairs)
            img_path, mask_path = self.real_pairs[real_idx]
            image, mask = _load_real_sample(img_path, mask_path)

        image, mask = _resize_pair(image, mask)

        if self.split == "train" and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        img_t = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t
