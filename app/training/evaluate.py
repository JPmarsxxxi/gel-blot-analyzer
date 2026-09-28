"""Scores a band-detection checkpoint separately on synthetic and real (GelGenie) data.

Run: python -m app.training.evaluate [--checkpoint PATH]

The mixed val_dice stored by train.py is dominated by synthetic samples, which are
far easier than real gels, so it overstates real-world accuracy. Report these instead.
"""
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from app.training.dataset import GelSegmentationDataset
from app.training.model import BandUNet
from app.training.train import ARTIFACT_PATH, dice_score


def _mean_dice(model: BandUNet, dataset, device: torch.device) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, masks in DataLoader(dataset, batch_size=4, shuffle=False):
            imgs, masks = imgs.to(device), masks.to(device)
            total += dice_score(model(imgs), masks) * imgs.size(0)
            n += imgs.size(0)
    return total / max(1, n)


def evaluate(checkpoint_path: str, device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BandUNet(base_channels=checkpoint.get("base_channels", 16)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results = {}
    val_ds = GelSegmentationDataset(split="val", synth_per_epoch=300, seed=1234)
    results["synthetic_val"] = _mean_dice(model, Subset(val_ds, range(val_ds.synth_per_epoch)), device)
    for split in ("val", "test"):
        ds = GelSegmentationDataset(split=split, synth_per_epoch=0)
        real_only = Subset(ds, range(ds.synth_per_epoch, len(ds)))
        results[f"real_{split}"] = _mean_dice(model, real_only, device) if len(real_only) else float("nan")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=ARTIFACT_PATH)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"checkpoint: {args.checkpoint}")
    print(f"stored metadata: { {k: v for k, v in checkpoint.items() if k != 'state_dict'} }")
    for name, score in evaluate(args.checkpoint, device).items():
        print(f"{name:14s} dice={score:.4f}")
