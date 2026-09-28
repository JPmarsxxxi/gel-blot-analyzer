"""Trains the YOLO band detector on the dataset from yolo_data.py.

Run: python -m app.training.train_yolo [--epochs 150 --patience 20 --imgsz 640]

Stops early once validation mAP has not improved for --patience epochs, then
copies the best weights to --out. Ultralytics is AGPL-3.0: fine for this
experiment, but shipping it in a hosted product needs a licence decision.
"""
import argparse
import os
import shutil

from app.training.yolo_data import OUT_DIR

EXPERIMENTS_DIR = os.path.dirname(OUT_DIR)


def run(epochs: int, patience: int, imgsz: int, batch: int, base: str, out: str) -> None:
    from ultralytics import YOLO

    model = YOLO(base)
    results = model.train(
        data=os.path.join(OUT_DIR, "data.yaml"),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=2,
        project=EXPERIMENTS_DIR,
        name="yolo_run",
        exist_ok=True,
        single_cls=True,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        max_det=2000,
        plots=True,
        verbose=False,
    )
    best = os.path.join(results.save_dir, "weights", "best.pt")
    shutil.copy(best, out)
    print(f"best weights -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--out", default=os.path.join(EXPERIMENTS_DIR, "yolo_best.pt"))
    args = parser.parse_args()
    run(args.epochs, args.patience, args.imgsz, args.batch, args.base, args.out)
