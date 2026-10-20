"""Check membrane auto-crop: raw imager exports should be cropped, ordinary
gels and blots never.

Run: python -m app.training.membrane_check

Positives are the raw ChemiDoc exports of Zenodo record 21330132 (CC BY 4.0),
unzipped under WESTERN_DIR (default C:/Users/User/western_blots, or
GEL_WESTERN_DIR). Negatives are every GelGenie image, the external gels and
the paper strips from Zenodo record 14291079.
"""
import glob
import os

from app.detection.imaging import load_rgb, to_grayscale
from app.detection.membrane import find_membrane
from app.training.dataset import EXTERNAL_DATA_DIR, _load_real_pairs

WESTERN_DIR = os.environ.get("GEL_WESTERN_DIR") or os.path.join("C:/", "Users", "User", "western_blots")


def _count(paths: list[str]) -> tuple[int, list[str]]:
    hits = [p for p in paths if find_membrane(to_grayscale(load_rgb(p))) is not None]
    return len(hits), hits


def main() -> None:
    raw = os.path.join(WESTERN_DIR, "zenodo_21330132", "data", "*", "*")
    positives = {
        "chemidoc merge": sorted(glob.glob(os.path.join(raw, "*(merge).tif"))),
        "chemidoc colorimetric": sorted(glob.glob(os.path.join(raw, "*(Colorimetric).tif"))),
    }
    negatives = {
        "gelgenie train/val/test": [p for split in ("train", "val", "test") for p, _ in _load_real_pairs(split)],
        "external gels": sorted(glob.glob(os.path.join(EXTERNAL_DATA_DIR, "external_gels", "external_gels", "images", "*"))),
        "paper strips": sorted(glob.glob(os.path.join(WESTERN_DIR, "strips_14291079", "*.png"))),
    }
    for name, paths in positives.items():
        n, _ = _count(paths)
        print(f"{name:26s} cropped {n}/{len(paths)}")
    for name, paths in negatives.items():
        n, hits = _count(paths)
        print(f"{name:26s} cropped {n}/{len(paths)} (should be 0)")
        for p in hits:
            print(f"    {p}")


if __name__ == "__main__":
    main()
