"""Head-to-head with GelGenie (Nature Communications 2025), the published open
model for gel band segmentation, on the same gels with the same band metric.

Run: python -m app.training.compare_gelgenie

Downloads GelGenie's released TorchScript models (Universal and its Sharp Band
fine-tune; the "Extended" variants were trained on the test split, so they are
excluded) and reproduces its inference: raw image / dtype max, grayscale,
symmetric zero-padding to a multiple of 32, argmax over 2 classes, bands =
connected components. Our YOLO path runs through pipeline.run_full_detection.

Two recall rules are reported:
- lenient (benchmark.py's): a band is found if its centroid is in any box, so
  one blob covering two touching bands finds both;
- one-per-box: each box credits at most one band, so merges cost recall.
Both detectors were trained on GelGenie's train split; neither saw the test
split or the external gels. Also reports pixel Dice (GelGenie's metric;
their paper gives ~0.82 on the test split) for the segmentation models.

Writes app/training/artifacts/benchmarks/gelgenie_comparison.json.
"""
import json
import os
import time
import urllib.request

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, find_objects, label

from app.training.benchmark import MIN_BAND_AREA, OUT_DIR, _external_pairs
from app.training.dataset import _load_real_pairs
from app.training.evaluate_lanes import reference_lanes
from app.training.yolo_data import OUT_DIR as YOLO_DS_DIR

MODEL_DIR = os.path.join(os.path.dirname(YOLO_DS_DIR), "gelgenie")
GELGENIE_MODELS = {
    "gelgenie_universal": "https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/resolve/main/torchscript_checkpoints/unet_dec_21_epoch_579.pt",
    "gelgenie_sharp": "https://huggingface.co/mattaq/GelGenie-Universal-FineTune-May-2024/resolve/main/torchscript_checkpoints/unet_dec_21_finetune_epoch_590.pt",
}


def _gelgenie_input(path: str) -> np.ndarray:
    import cv2
    import imageio.v2 as imageio  # reads 16-bit and LZW TIFFs the way GelGenie does (needs imagecodecs)

    img = imageio.imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32) / (65535 if img.dtype == np.uint16 else 255)


def _gelgenie_mask(model, path: str) -> np.ndarray:
    import torch

    x = _gelgenie_input(path)
    h, w = x.shape
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    t, l = (H - h) // 2, (W - w) // 2
    with torch.no_grad():
        out = model(torch.from_numpy(np.pad(x, ((t, H - h - t), (l, W - w - l))))[None, None])
    return out.argmax(dim=1)[0, t:t + h, l:l + w].numpy().astype(bool)


def _boxes_from_mask(pred: np.ndarray) -> list[tuple[float, float, float, float]]:
    lab, _ = label(pred)
    sizes = np.bincount(lab.ravel())
    return [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop) for i, sl in enumerate(find_objects(lab), 1) if sl and sizes[i] >= MIN_BAND_AREA]


def _score(boxes, gt: np.ndarray) -> dict:
    lab, n = label(gt)
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_BAND_AREA]
    cents = center_of_mass(gt, lab, keep) if keep else []
    inside = [[i for i, (cy, cx) in enumerate(cents) if x0 <= cx < x1 and y0 <= cy < y1] for x0, y0, x1, y1 in boxes]
    credited: set[int] = set()
    for hits in inside:
        free = [i for i in hits if i not in credited]
        if free:
            credited.add(free[0])
    cols = np.zeros(gt.shape[1], bool)
    for a, b in reference_lanes(gt.astype(np.uint8)):
        cols[a:b] = True
    scored = [hits for (x0, _, x1, _), hits in zip(boxes, inside) if cols[min(gt.shape[1] - 1, int((x0 + x1) / 2))]]
    return {
        "ref": len(cents),
        "found": len({i for hits in inside for i in hits}),
        "found_one_per_box": len(credited),
        "scored": len(scored),
        "true": sum(1 for hits in scored if hits),
    }


def _summarize(rows: list[dict]) -> dict:
    ref = sum(r["ref"] for r in rows)
    precision = sum(r["true"] for r in rows) / max(1, sum(r["scored"] for r in rows))
    out = {"images": len(rows), "ref_bands": ref, "precision": round(precision, 4)}
    for rule, key in (("lenient", "found"), ("one_per_box", "found_one_per_box")):
        recall = sum(r[key] for r in rows) / max(1, ref)
        out[f"recall_{rule}"] = round(recall, 4)
        out[f"f1_{rule}"] = round(2 * precision * recall / max(1e-9, precision + recall), 4)
    if all("dice" in r for r in rows):
        out["pixel_dice"] = round(float(np.mean([r["dice"] for r in rows])), 4)
    out["seconds_per_image"] = round(float(np.mean([r["seconds"] for r in rows])), 2)
    return out


def main() -> None:
    import torch

    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {}
    for name, url in GELGENIE_MODELS.items():
        path = os.path.join(MODEL_DIR, name + ".pt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        models[name] = torch.jit.load(path, map_location="cpu").eval()

    os.environ["GEL_DETECTOR"] = "yolo"
    from app.detection import pipeline

    pipeline.DETECTOR = "yolo"
    report = {}
    for split, pairs in {"test": _load_real_pairs("test"), "external": _external_pairs()}.items():
        rows: dict[str, list[dict]] = {k: [] for k in (*GELGENIE_MODELS, "ours_yolo")}
        for img_path, mask_path in pairs:
            gt = np.asarray(Image.open(mask_path)) > 0
            for name, model in models.items():
                t0 = time.time()
                pred = _gelgenie_mask(model, img_path)
                r = _score(_boxes_from_mask(pred), gt)
                r.update(seconds=time.time() - t0, dice=2 * (pred & gt).sum() / max(1, pred.sum() + gt.sum()))
                rows[name].append(r)
            t0 = time.time()
            result = pipeline.run_full_detection(img_path)
            boxes = [(b.x, b.y, b.x + b.width, b.y + b.height) for lane in result.bands_by_lane for b in lane]
            r = _score(boxes, gt)
            r["seconds"] = time.time() - t0
            rows["ours_yolo"].append(r)
        report[split] = {k: _summarize(v) for k, v in rows.items()}
        for k, s in report[split].items():
            print(f"{split:8s} {k:20s} P={s['precision']:.3f}  R={s['recall_one_per_box']:.3f}  F1={s['f1_one_per_box']:.3f}  (lenient F1={s['f1_lenient']:.3f})  dice={s.get('pixel_dice', '-')}  {s['seconds_per_image']}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gelgenie_comparison.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
