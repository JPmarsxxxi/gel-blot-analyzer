"""Reference-free check of quantification on GelGenie's ladder gels.

Run: python -m app.training.quant_consistency

A ladder's composition is fixed: each band is always the same fraction of the
ladder's total DNA. These gels load different amounts in different lanes, so
a trustworthy measurement gives each band the same % of lane in every lane.
The spread of that value across lanes (coefficient of variation) measures how
reliable "% of lane" is, without needing manufacturer band masses.

Every method measures the same background-corrected signal (the app's); only
where the band is drawn differs:
- human: the hand-drawn GelGenie mask component (best case);
- ours_yolo: the app's YOLO box around the band;
- gelgenie_sharp: GelGenie's segmentation component.
A band a method merges with a neighbour, or misses, counts as not measured.
Bands are identified by top-to-bottom order within a lane, using only lanes
with the gel's modal number of annotated bands. % of lane is computed over the
bands all methods measured in that lane, so methods are compared like for like.

Note: these gels are in GelGenie's training split (and so in ours), so this
tests measurement, not detection generalization.

Writes app/training/artifacts/benchmarks/quant_consistency.json.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, find_objects, label

from app.detection.cache import get_gray_and_signal
from app.training.benchmark import MIN_BAND_AREA, OUT_DIR
from app.training.compare_gelgenie import GELGENIE_MODELS, MODEL_DIR, _gelgenie_mask
from app.training.dataset import EXTERNAL_DATA_DIR

LADDER_DIR = os.path.join(EXTERNAL_DATA_DIR, "quantitation_ladder_gels", "quantitation_ladder_gels")
METHODS = ("human", "ours_yolo", "gelgenie_sharp")


def _ladder_pairs() -> list[tuple[str, str]]:
    pairs = []
    for img_dir, mask_dir in (("images", "masks"), ("val_images", "val_masks"), ("test_images", "test_masks")):
        for name in sorted(os.listdir(os.path.join(LADDER_DIR, img_dir))):
            if not name.startswith("."):
                pairs.append((os.path.join(LADDER_DIR, img_dir, name), os.path.join(LADDER_DIR, mask_dir, os.path.splitext(name)[0] + ".tif")))
    return pairs


def _gel_bands(gt: np.ndarray):
    """[(lane_index, [(component_label, cy, cx), ...] top to bottom)] for lanes
    with the gel's modal band count. Lanes on ladder gels touch, so bands are
    grouped by centroid x (a new lane starts where the gap between sorted
    centroids exceeds half the median band width), not by empty columns."""
    lab, n = label(gt)
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_BAND_AREA]
    if not keep:
        return lab, []
    cents = dict(zip(keep, center_of_mass(gt, lab, keep)))
    widths = [sl[1].stop - sl[1].start for i, sl in enumerate(find_objects(lab), 1) if i in cents]
    gap = 0.5 * float(np.median(widths))
    order = sorted(cents, key=lambda k: cents[k][1])
    groups, current = [], [order[0]]
    for prev, k in zip(order, order[1:]):
        if cents[k][1] - cents[prev][1] > gap:
            groups.append(current)
            current = []
        current.append(k)
    groups.append(current)
    lanes = [(li, sorted(((k, *cents[k]) for k in g), key=lambda t: t[1])) for li, g in enumerate(groups)]
    modal = Counter(len(b) for _, b in lanes).most_common(1)[0][0]
    return lab, [(li, b) for li, b in lanes if len(b) == modal and modal >= 5]


def _measure_gel(img_path: str, mask_path: str, yolo_boxes, gg_mask: np.ndarray) -> dict:
    gt = np.asarray(Image.open(mask_path)) > 0
    _, signal = get_gray_and_signal(img_path)
    lab, lanes = _gel_bands(gt)
    gg_lab, _ = label(gg_mask)
    all_cents = [(cy, cx) for _, bands in lanes for _, cy, cx in bands]

    out = {}
    for li, bands in lanes:
        values: dict[str, list[float | None]] = {m: [] for m in METHODS}
        for comp, cy, cx in bands:
            values["human"].append(float(signal[lab == comp].sum()))

            hits = [bx for bx in yolo_boxes if bx[0] <= cx < bx[2] and bx[1] <= cy < bx[3]]
            box = hits[0] if len(hits) == 1 else None
            shared = box is not None and sum(1 for y, x in all_cents if box[0] <= x < box[2] and box[1] <= y < box[3]) > 1
            values["ours_yolo"].append(None if box is None or shared else float(signal[int(box[1]):int(box[3]), int(box[0]):int(box[2])].sum()))

            g = gg_lab[int(round(cy)), int(round(cx))]
            merged = g and sum(1 for y, x in all_cents if gg_lab[int(round(y)), int(round(x))] == g) > 1
            values["gelgenie_sharp"].append(None if not g or merged else float(signal[gg_lab == g].sum()))
        out[li] = values
    return out


def _cv(xs: list[float]) -> float:
    return float(np.std(xs) / np.mean(xs)) if len(xs) >= 3 and np.mean(xs) > 0 else float("nan")


def main() -> None:
    import torch

    os.environ["GEL_DETECTOR"] = "yolo"
    from app.detection import pipeline

    pipeline.DETECTOR = "yolo"
    if not os.path.exists(os.path.join(MODEL_DIR, "gelgenie_sharp.pt")):
        raise SystemExit(f"run compare_gelgenie first to download {GELGENIE_MODELS['gelgenie_sharp']}")
    gg = torch.jit.load(os.path.join(MODEL_DIR, "gelgenie_sharp.pt"), map_location="cpu").eval()

    per_ladder: dict[str, dict] = defaultdict(lambda: {"cv": defaultdict(list), "coverage": defaultdict(list), "abs_diff_pp": defaultdict(list)})
    for img_path, mask_path in _ladder_pairs():
        ladder = "NEB" if "NEB" in os.path.basename(img_path) else "Thermo"
        result = pipeline.run_full_detection(img_path)
        boxes = [(b.x, b.y, b.x + b.width, b.y + b.height) for lane in result.bands_by_lane for b in lane]
        lanes = _measure_gel(img_path, mask_path, boxes, _gelgenie_mask(gg, img_path))

        rows: dict[str, dict[int, list[float]]] = {m: defaultdict(list) for m in METHODS}
        for values in lanes.values():
            n = len(values["human"])
            for m in METHODS:
                per_ladder[ladder]["coverage"][m].append(sum(v is not None for v in values[m]) / n)
            common = [i for i in range(n) if all(values[m][i] is not None for m in METHODS)]
            if len(common) < 5:
                continue
            for m in METHODS:
                total = sum(values[m][i] for i in common)
                pct = {i: 100 * values[m][i] / total for i in common}
                for i in common:
                    rows[m][i].append(pct[i])
                if m != "human":
                    human_total = sum(values["human"][i] for i in common)
                    per_ladder[ladder]["abs_diff_pp"][m].extend(abs(pct[i] - 100 * values["human"][i] / human_total) for i in common)
        for m in METHODS:
            per_ladder[ladder]["cv"][m].extend(c for c in (_cv(v) for v in rows[m].values()) if not np.isnan(c))
        print(os.path.basename(img_path), f"lanes used: {len(lanes)}", f"band rows: {len(rows['human'])}", flush=True)

    report = {}
    for ladder, d in per_ladder.items():
        report[ladder] = {
            m: {
                "median_cv_pct_of_lane": round(float(np.median(d["cv"][m])) * 100, 1) if d["cv"][m] else None,
                "band_rows": len(d["cv"][m]),
                "coverage": round(float(np.mean(d["coverage"][m])), 3),
                "mean_abs_diff_vs_human_pp": round(float(np.mean(d["abs_diff_pp"][m])), 2) if d["abs_diff_pp"][m] else None,
            }
            for m in METHODS
        }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "quant_consistency.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
