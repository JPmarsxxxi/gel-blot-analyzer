"""Inference wrapper for the trained band-segmentation model.

Lazily loads the model artifact once per process. If no artifact is present
(training hasn't been run yet), falls back to a classical-only signal so the
server still functions end to end.
"""
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from app.config import MODEL_PATH
from app.training.model import BandUNet

_lock = threading.Lock()
_model: BandUNet | None = None
_device: torch.device | None = None
_loaded = False


def _ensure_loaded() -> None:
    global _model, _device, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=_device)
            model = BandUNet(base_channels=checkpoint.get("base_channels", 16))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(_device)
            model.eval()
            _model = model
        else:
            _model = None
        _loaded = True


def model_available() -> bool:
    _ensure_loaded()
    return _model is not None


def predict_band_probability(gray: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Returns a float32 probability map, same shape as `gray`, values in [0, 1].

    `gray` is expected in [0, 255] range (any polarity; the model was trained
    on both dark-band-on-light and light-band-on-dark examples).
    """
    _ensure_loaded()
    h, w = gray.shape

    if _model is None:
        # Classical-only fallback: normalize local contrast as a crude proxy.
        norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)
        return np.abs(norm - float(np.median(norm))).astype(np.float32)

    with torch.no_grad():
        t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        t = t.to(_device)
        logits = _model(t)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
        return probs.squeeze(0).squeeze(0).cpu().numpy()
