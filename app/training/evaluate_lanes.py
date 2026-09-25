"""Scores lane detection against lanes derived from GelGenie band masks.

Run: python -m app.training.evaluate_lanes [--split val|test|both] [--show N]

The masks mark band pixels, not lanes, so a reference lane is a run of columns
that contain band pixels (runs closer than a small gap are merged). A detected
lane is spurious if it holds no reference band; a reference lane is missed if
no detected lane's center falls inside it; two reference lanes whose centers
share one detected lane count as merged.
"""
import argparse

import numpy as np
from PIL import Image

from app.detection.imaging import load_rgb, to_grayscale
from app.detection.lanes import detect_lanes
from app.detection import ml_infer
from app.training.dataset import _load_real_pairs


def reference_lanes(mask: np.ndarray) -> list[tuple[int, int]]:
    occupied = (mask > 0).sum(axis=0) > 0
    width = len(occupied)
    max_gap = max(2, int(width * 0.005))
    runs, start, last = [], None, None
    for x in np.flatnonzero(occupied):
        if start is None:
            start = last = x
        elif x - last > max_gap:
            runs.append((start, last + 1))
            start = x
        last = x
    if start is not None:
        runs.append((start, last + 1))
    min_width = max(3, int(width * 0.01))
    return [r for r in runs if r[1] - r[0] >= min_width]


def score_image(img_path: str, mask_path: str) -> dict:
    gray = to_grayscale(load_rgb(img_path))
    detected = detect_lanes(ml_infer.predict_band_probability(gray))
    mask = np.asarray(Image.open(mask_path))
    ref = reference_lanes(mask)

    band_cols = (mask > 0).any(axis=0)
    spurious = sum(1 for l in detected if not band_cols[int(l.x_start):int(np.ceil(l.x_end))].any())
    centers = [(l.x_start + l.x_end) / 2 for l in detected]
    missed = sum(1 for a, b in ref if not any(a <= c < b for c in centers))
    owner = [next((i for i, l in enumerate(detected) if l.x_start <= (a + b) / 2 < l.x_end), None) for a, b in ref]
    merged = sum(owner.count(i) - 1 for i in set(owner) if i is not None and owner.count(i) > 1)
    return {"ref": len(ref), "detected": len(detected), "spurious": spurious, "missed": missed, "merged": merged}


def main(splits: list[str], show: int) -> None:
    rows = []
    for split in splits:
        for img_path, mask_path in _load_real_pairs(split):
            rows.append((img_path, score_image(img_path, mask_path)))

    total = {k: sum(r[k] for _, r in rows) for k in ("ref", "detected", "spurious", "missed", "merged")}
    exact = sum(1 for _, r in rows if r["spurious"] == r["missed"] == r["merged"] == 0)
    print(f"images: {len(rows)}  all lanes right: {exact} ({exact / len(rows):.0%})")
    print(f"reference lanes: {total['ref']}  detected: {total['detected']}")
    print(f"spurious: {total['spurious']}  missed: {total['missed']}  merged: {total['merged']}")
    worst = sorted(rows, key=lambda r: -(r[1]["spurious"] + r[1]["missed"] + r[1]["merged"]))[:show]
    for path, r in worst:
        print(f"  {path.split('external_data/')[-1]}: {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()
    main(["val", "test"] if args.split == "both" else [args.split], args.show)
