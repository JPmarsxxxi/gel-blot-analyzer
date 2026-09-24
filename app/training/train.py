"""Offline training script for the band-segmentation model.

Run: python -m app.training.train [--epochs N] [--batch-size N]

Not part of the runtime server — produces the model artifact the server loads
for inference (see app/detection/ml_infer.py).

Training data: synthetic gel images generated on the fly (app/training/synth_data.py)
combined with the real GelGenie dataset (Dunn Lab / University of Edinburgh,
CC-BY-4.0, https://zenodo.org/records/13218469), a subset of which is checked out
under app/training/external_data/.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Subset

from app.training.dataset import GelSegmentationDataset
from app.training.model import BandUNet, combined_loss

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "band_detector.pt")
RECORD_PATH = os.path.join(ARTIFACT_DIR, "training_record.json")


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float()
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def _val_dice(model: BandUNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            total += dice_score(model(imgs), masks) * imgs.size(0)
            n += imgs.size(0)
    return total / max(1, n)


def run(epochs: int, batch_size: int, synth_per_epoch: int, lr: float, resume: bool) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = GelSegmentationDataset(split="train", synth_per_epoch=synth_per_epoch, seed=42)
    val_ds = GelSegmentationDataset(split="val", synth_per_epoch=synth_per_epoch, seed=1234)
    # Select checkpoints on real gels only when available: synthetic samples are
    # easy enough that a mixed score mostly tracks them.
    if val_ds.real_pairs:
        val_ds = Subset(val_ds, range(val_ds.synth_per_epoch, len(val_ds)))
    print(f"train samples/epoch: {len(train_ds)}  val samples/epoch: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BandUNet(base_channels=16).to(device)

    best_val_dice = -1.0
    if resume and os.path.exists(ARTIFACT_PATH):
        checkpoint = torch.load(ARTIFACT_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        best_val_dice = _val_dice(model, val_loader, device)
        print(f"resumed from {ARTIFACT_PATH} (val_dice={best_val_dice:.4f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        val_dice = _val_dice(model, val_loader, device)

        scheduler.step()
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}  ({dt:.1f}s)")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "base_channels": 16,
                    "val_dice": val_dice,
                    "val_dice_on": "real" if isinstance(val_ds, Subset) else "mixed",
                    "epoch": epoch,
                    "epochs": epochs,
                    "resumed": resume,
                },
                ARTIFACT_PATH,
            )

    print(f"best val_dice={best_val_dice:.4f}, saved to {ARTIFACT_PATH}")

    # Imported here to avoid a circular import (evaluate imports from this module).
    from app.training.evaluate import evaluate

    checkpoint = torch.load(ARTIFACT_PATH, map_location=device)
    record = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_completed": True,
        "args": {"epochs": epochs, "batch_size": batch_size, "synth_per_epoch": synth_per_epoch, "lr": lr, "resume": resume},
        "best_epoch": checkpoint.get("epoch"),
        "val_dice": checkpoint["val_dice"],
        "val_dice_on": checkpoint["val_dice_on"],
        "dice": evaluate(ARTIFACT_PATH, device),
    }
    with open(RECORD_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synth-per-epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="warm-start from the existing checkpoint")
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.synth_per_epoch, args.lr, args.resume)
