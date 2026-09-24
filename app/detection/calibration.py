"""Ladder / molecular-weight calibration.

Standard log-linear fit: log10(kDa) is approximately linear in migration
distance (band center y-position) for a given gel. Requires >= 2 ladder
bands with a known size to fit; fewer than that, no estimate is produced
(size is simply omitted, per spec).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationCurve:
    slope: float
    intercept: float
    r_squared: float

    def estimate_kda(self, center_y: float) -> float:
        log_kda = self.slope * center_y + self.intercept
        return float(10**log_kda)


def fit_calibration(ladder_points: list[tuple[float, float]]) -> CalibrationCurve | None:
    """ladder_points: list of (center_y, known_kda) for ladder bands with a known size."""
    pts = [(y, k) for y, k in ladder_points if k is not None and k > 0]
    if len(pts) < 2:
        return None

    ys = np.array([p[0] for p in pts], dtype=np.float64)
    log_kda = np.log10(np.array([p[1] for p in pts], dtype=np.float64))

    slope, intercept = np.polyfit(ys, log_kda, 1)

    predicted = slope * ys + intercept
    ss_res = float(np.sum((log_kda - predicted) ** 2))
    ss_tot = float(np.sum((log_kda - log_kda.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return CalibrationCurve(slope=float(slope), intercept=float(intercept), r_squared=r_squared)
