import numpy as np
import pytest

from conftest import synthetic_skin_samples
from inputs.heart_ble import BLEHeartRate, rmssd
from inputs.rppg import MIN_QUALITY, RPPGDetector, bpm_to_va


@pytest.mark.parametrize("true_bpm", [58, 72, 95, 120])
def test_rppg_recovers_known_heart_rate(true_bpm):
    d = RPPGDetector()
    d.add_samples(synthetic_skin_samples(true_bpm, seconds=12, seed=true_bpm))
    est = d.estimate()
    assert est is not None
    assert abs(est["bpm"] - true_bpm) <= 3
    assert est["reliable"]


def test_rppg_flags_pure_noise_as_unreliable():
    rng = np.random.default_rng(1)
    d = RPPGDetector()
    t = np.cumsum(np.full(360, 33.3))
    d.add_samples([[ti, *(150 + rng.normal(0, 2, 3))] for ti in t])
    est = d.estimate()
    assert est is not None and est["quality"] < MIN_QUALITY and not est["reliable"]


def test_rppg_needs_enough_data_and_ignores_bad_rows():
    d = RPPGDetector()
    d.add_samples(synthetic_skin_samples(70, seconds=3))
    assert d.estimate() is None
    d.add_samples([[1, 2, 3], "junk"[:0]])  # malformed rows are skipped
    assert d.estimate() is None


def test_rppg_keeps_only_recent_window():
    d = RPPGDetector()
    d.add_samples(synthetic_skin_samples(60, seconds=12, seed=3))
    d.add_samples(synthetic_skin_samples(100, seconds=12, t0_ms=12_500, seed=4))
    assert abs(d.estimate()["bpm"] - 100) <= 3


def test_heart_rate_maps_to_arousal_relative_to_resting():
    _, a_rest = bpm_to_va(62, resting_bpm=62)
    _, a_high = bpm_to_va(90, resting_bpm=62)
    assert a_rest == 0 and a_high > 0.5


def test_rmssd():
    assert rmssd([800, 810, 800, 810]) == pytest.approx(10.0)
    assert rmssd([800, 100, 5000]) is None  # artefacts removed -> too few beats


def test_ble_latest():
    b = BLEHeartRate()
    for i in range(5):
        b.add(float(i), 70, [850, 860])
    out = b.latest()
    assert out["bpm"] == 70 and out["rmssd"] is not None
