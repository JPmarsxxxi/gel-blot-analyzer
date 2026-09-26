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

from app.detection.imaging import load_rgb
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


_real_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _load_real_sample(img_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the pair already resized to TARGET_SIZE, cached: full-size
    16-bit TIFFs take far longer to decode than an epoch's compute."""
    key = (img_path, mask_path)
    if key not in _real_cache:
        img = np.asarray(Image.fromarray(load_rgb(img_path)).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path))
        mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
        _real_cache[key] = _resize_pair(img, mask)
    return _real_cache[key]


class GelSegmentationDataset(Dataset):
    """split: "train" | "val" | "test".

    For "train", synthetic samples are generated on the fly (effectively
    unbounded) and mixed with every real training image once per epoch.
    """

    def __init__(self, split: str = "train", synth_per_epoch: int = 400, seed: int | None = None):
        self.split = split
        self.synth_per_epoch = synth_per_epoch if split == "train" else max(20, synth_per_epoch // 10)
        self.real_pairs = _load_real_pairs(split)
        self.rng_seed = seed

    def __len__(self) -> int:
        return self.synth_per_epoch + len(self.real_pairs)

    def __getitem__(self, idx: int):
        if idx < self.synth_per_epoch:
            seed = None if self.rng_seed is None else self.rng_seed * 100003 + idx
            sample = generate_sample(seed=seed)
            image, mask = sample.image, sample.mask
        else:
            img_path, mask_path = self.real_pairs[idx - self.synth_per_epoch]
            image, mask = _load_real_sample(img_path, mask_path)
        if image.shape != TARGET_SIZE:
            image, mask = _resize_pair(image, mask)

        if self.split == "train" and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        img_t = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t


WORK_LONG = 1024  # working resolution for tiled training and inference: long side, never upscaled
CROP = 256


def to_working_resolution(image: np.ndarray, mask: np.ndarray | None = None):
    h, w = image.shape
    scale = min(1.0, WORK_LONG / max(h, w))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR), dtype=np.uint8)
        if mask is not None:
            mask = (np.asarray(Image.fromarray(mask * 255).resize(size, Image.NEAREST)) > 127).astype(np.uint8)
    return image, mask


def pad_to_multiple(image: np.ndarray, multiple: int = 16, fill: float | None = None) -> np.ndarray:
    h, w = image.shape
    ph, pw = (-h) % multiple, (-w) % multiple
    if not (ph or pw):
        return image
    value = float(np.median(image)) if fill is None else fill
    return np.pad(image, ((0, ph), (0, pw)), constant_values=value)


_work_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _load_working_pair(img_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    key = (img_path, mask_path)
    if key not in _work_cache:
        img = np.asarray(Image.fromarray(load_rgb(img_path)).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path))
        mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
        _work_cache[key] = to_working_resolution(img, mask)
    return _work_cache[key]


class TileDataset(Dataset):
    """Training on CROP x CROP crops at working resolution instead of whole
    images squashed to 256x256, which shrank thin bands to a pixel or two.

    train: crops_per_image random crops per real gel plus synthetic crops; half
    are centred on a band pixel so crops aren't mostly empty background.
    val/test: whole gels at working resolution, padded to a multiple of 16
    (use batch size 1, sizes vary).
    """

    def __init__(self, split: str = "train", synth_per_epoch: int = 300, seed: int | None = None, crops_per_image: int = 2):
        self.split = split
        self.synth_per_epoch = synth_per_epoch if split == "train" else max(20, synth_per_epoch // 10)
        self.real_pairs = _load_real_pairs(split)
        self.rng_seed = seed
        self.crops = crops_per_image if split == "train" else 1

    def __len__(self) -> int:
        return self.synth_per_epoch + len(self.real_pairs) * self.crops

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = pad_to_multiple(image, CROP) if min(image.shape) < CROP else image
        mask = np.pad(mask, ((0, image.shape[0] - mask.shape[0]), (0, image.shape[1] - mask.shape[1])))
        h, w = image.shape
        ys, xs = np.nonzero(mask)
        if len(ys) and random.random() < 0.5:
            i = random.randrange(len(ys))
            y0 = min(max(0, ys[i] - CROP // 2), h - CROP)
            x0 = min(max(0, xs[i] - CROP // 2), w - CROP)
        else:
            y0, x0 = random.randint(0, h - CROP), random.randint(0, w - CROP)
        return image[y0:y0 + CROP, x0:x0 + CROP], mask[y0:y0 + CROP, x0:x0 + CROP]

    def __getitem__(self, idx: int):
        if idx < self.synth_per_epoch:
            seed = None if self.rng_seed is None else self.rng_seed * 100003 + idx
            sample = generate_sample(seed=seed)
            image, mask = to_working_resolution(sample.image, sample.mask)
        else:
            image, mask = _load_working_pair(*self.real_pairs[(idx - self.synth_per_epoch) // self.crops])

        if self.split == "train":
            image, mask = self._crop(image, mask)
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
        else:
            h, w = image.shape
            image = pad_to_multiple(image)
            mask = np.pad(mask, ((0, image.shape[0] - h), (0, image.shape[1] - w)))

        return torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0, torch.from_numpy(mask.copy()).float().unsqueeze(0)
