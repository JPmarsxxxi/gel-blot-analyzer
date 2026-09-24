import numpy as np

from app.detection.bands import sensitivity_to_threshold, detect_bands_in_lane
from app.detection.calibration import fit_calibration
from app.detection.lanes import LaneBoundary, detect_lanes


def test_sensitivity_to_threshold_is_monotonically_decreasing():
    thresholds = [sensitivity_to_threshold(s) for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert all(thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1))


def test_higher_sensitivity_detects_at_least_as_many_bands():
    h, w = 200, 100
    prob = np.zeros((h, w), dtype=np.float32)
    signal = np.zeros((h, w), dtype=np.float64)
    for y, peak in [(40, 0.9), (100, 0.55), (160, 0.35)]:
        prob[y - 5 : y + 5, 20:80] = peak
        signal[y - 5 : y + 5, 20:80] = 50.0

    lane = LaneBoundary(x_start=0, x_end=100)
    counts = [len(detect_bands_in_lane(signal, prob, lane, s)) for s in [0.1, 0.5, 0.9]]
    assert counts[0] <= counts[1] <= counts[2]


def test_detect_lanes_falls_back_to_single_lane_when_blank():
    blank = np.zeros((100, 100))
    lanes = detect_lanes(blank)
    assert len(lanes) == 1
    assert lanes[0].x_start == 0.0
    assert lanes[0].x_end == 100.0


def test_calibration_needs_at_least_two_points():
    assert fit_calibration([(10.0, 100.0)]) is None
    curve = fit_calibration([(10.0, 100.0), (50.0, 25.0)])
    assert curve is not None
    assert curve.estimate_kda(10.0) > curve.estimate_kda(50.0)


def test_percent_of_lane_sums_to_100_when_bands_present():
    h, w = 200, 100
    prob = np.zeros((h, w), dtype=np.float32)
    signal = np.zeros((h, w), dtype=np.float64)
    for y, peak in [(40, 0.9), (100, 0.8)]:
        prob[y - 5 : y + 5, 20:80] = peak
        signal[y - 5 : y + 5, 20:80] = 50.0

    lane = LaneBoundary(x_start=0, x_end=100)
    bands = detect_bands_in_lane(signal, prob, lane, 0.5)
    assert len(bands) >= 1
    total_pct = sum(b.percent_of_lane for b in bands)
    assert abs(total_pct - 100.0) < 1e-6
