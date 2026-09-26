"""End-to-end benchmark of the detection pipeline on real, full-resolution gels.

Run: python -m app.training.benchmark --name NAME [--split val|test]

Scores exactly what the app does (pipeline.run_full_detection at default
sensitivity), so any model or pre-processing variant is comparable, whether
it produces a probability map or boxes. Ground truth is the GelGenie band
masks; each connected component is one reference band.

- recall: reference bands whose centroid lies inside a detected box.
- precision: detected boxes containing a reference centroid, counting only
  boxes in labelled lanes (annotators skipped some lanes entirely, so boxes
  there would be unfairly counted as false).
- merged: extra reference bands sharing one box (two bands in one box = 1).

Writes app/training/artifacts/benchmarks/NAME_SPLIT.json.
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, label

from app.detection import pipeline
from app.training.dataset import _load_real_pairs
from app.training.evaluate_lanes import lane_errors, reference_lanes

OUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "benchmarks")
MIN_BAND_AREA = 20


def score_image(img_path: str, mask_path: str) -> dict:
    t0 = time.time()
    result = pipeline.run_full_detection(img_path)
    seconds = time.time() - t0

    mask = np.asarray(Image.open(mask_path)) > 0
    labeled, n = label(mask)
    sizes = np.bincount(labeled.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_BAND_AREA]
    centroids = [(cy, cx) for cy, cx in center_of_mass(mask, labeled, keep)] if keep else []

    boxes = [b for lane in result.bands_by_lane for b in lane]
    inside = [
        [i for i, (cy, cx) in enumerate(centroids) if b.x <= cx < b.x + b.width and b.y <= cy < b.y + b.height]
        for b in boxes
    ]
    found = set(i for hits in inside for i in hits)
    labelled_cols = np.zeros(mask.shape[1], bool)
    for a, b in reference_lanes(mask):
        labelled_cols[a:b] = True
    scored = [hits for b, hits in zip(boxes, inside) if labelled_cols[min(mask.shape[1] - 1, int(b.x + b.width / 2))]]

    return {
        "ref_bands": len(centroids),
        "found": len(found),
        "boxes_scored": len(scored),
        "boxes_true": sum(1 for hits in scored if hits),
        "merged": sum(len(hits) - 1 for hits in inside if len(hits) > 1),
        "lanes": lane_errors(result.lanes, mask),
        "seconds": seconds,
    }


def summarize(rows: list[dict]) -> dict:
    ref = sum(r["ref_bands"] for r in rows)
    recall = sum(r["found"] for r in rows) / max(1, ref)
    precision = sum(r["boxes_true"] for r in rows) / max(1, sum(r["boxes_scored"] for r in rows))
    lanes = {k: sum(r["lanes"][k] for r in rows) for k in ("ref", "spurious", "missed", "merged")}
    lanes["all_right"] = sum(1 for r in rows if r["lanes"]["spurious"] == r["lanes"]["missed"] == r["lanes"]["merged"] == 0)
    return {
        "images": len(rows),
        "ref_bands": ref,
        "band_recall": round(recall, 4),
        "band_precision": round(precision, 4),
        "band_f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "merged_bands": sum(r["merged"] for r in rows),
        "lanes": lanes,
        "seconds_per_image": round(float(np.mean([r["seconds"] for r in rows])), 2),
    }


def main(name: str, split: str) -> None:
    per_subset: dict[str, list[dict]] = {}
    for img_path, mask_path in _load_real_pairs(split):
        subset = img_path.split("external_data" + os.sep)[-1].split(os.sep)[0]
        per_subset.setdefault(subset, []).append(score_image(img_path, mask_path))

    report = {
        "name": name,
        "split": split,
        "overall": summarize([r for rows in per_subset.values() for r in rows]),
        "by_subset": {k: summarize(v) for k, v in sorted(per_subset.items())},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{split}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["overall"], indent=2))
    for subset, s in report["by_subset"].items():
        print(f"  {subset:28s} n={s['images']:3d}  recall={s['band_recall']:.3f}  precision={s['band_precision']:.3f}  f1={s['band_f1']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    main(args.name, args.split)
