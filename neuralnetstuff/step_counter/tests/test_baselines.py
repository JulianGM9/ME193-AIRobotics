"""Event matching, summary metrics, and the two baselines on synthetic signals."""

import numpy as np
import pytest

from stepnet.baselines import Detector, grid
from stepnet.data.blocks import Block, SessionData
from stepnet.eval.metrics import block_rows, bootstrap, match, summarize


def test_match_within_tolerance_one_to_one():
    assert match([100, 200, 305], [100, 200, 300]) == (3, 0, 0)
    assert match([100, 200, 320], [100, 200, 300]) == (2, 1, 1)  # 20 samples off: a miss and a false alarm
    assert match([99, 101], [100]) == (1, 1, 0)  # two detections can't share one stride
    assert match([], [1, 2]) == (0, 0, 2) and match([5], []) == (0, 1, 0)


def fake_session(swing_with_impact, swing_without_impact):
    """L-gyro dips once per second; only some are followed by a heel-strike impact."""
    n = 1200
    t = np.arange(n) / 100
    gyr = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    acc[:, 0] = 1.0
    starts = list(swing_with_impact) + list(swing_without_impact)
    for s in starts:
        sel = (t >= s) & (t < s + 1)
        gyr[sel, 2] = -250 * np.sin(np.pi * (t[sel] - s)) ** 8  # dip at s + 0.5
    for s in swing_with_impact:
        acc[int((s + 0.75) * 100), 0] = 3.0  # foot lands 0.25 s after mid-swing
    return SessionData("S001", {}, np.arange(n), acc, gyr)


P = {"cutoff_hz": 8.0, "height_dps": 100, "prominence_dps": 50, "min_gap_s": 0.4}


def test_b1_finds_every_dip_b2_only_the_landed_ones():
    s = fake_session(swing_with_impact=[1, 2, 3], swing_without_impact=[6, 7])
    det = Detector(s)
    b1 = det.detect(P)
    b2 = det.detect({**P, "impact_g": 2.0, "impact_window_s": 0.6})
    assert np.allclose(b1 / 100, [1.5, 2.5, 3.5, 6.5, 7.5], atol=0.03)
    assert np.allclose(b2 / 100, [1.5, 2.5, 3.5], atol=0.03)


def test_rows_and_summary():
    s = fake_session([1, 2, 3], [6, 7])
    walk = Block("S001", 3, "walk_normal", "walk_normal", "bout", 50, 450, 6, np.array([150, 250, 350]))
    swing = Block("S001", 7, "leg_swing_seated", "leg_swing_seated", "hold", 550, 850, 0, np.array([], int))
    rows = block_rows([walk, swing], {"S001": Detector(s).detect(P)})
    assert [r["count_err"] for r in rows] == [0, 4]  # 3 strides = 6 steps; 2 swings miscounted as 4 steps
    summary = summarize(rows)
    assert summary["recall"] == 1.0 and summary["precision"] == pytest.approx(3 / 5)
    assert summary["false_per_min"] == pytest.approx(2 / (3 / 60))
    ci = bootstrap(rows, n=200)
    assert ci["recall"] == (1.0, 1.0)


def test_grid_is_the_full_product():
    assert len(grid({"a": [1, 2], "b": [3, 4, 5]})) == 6


def test_test_sessions_are_locked_by_default(tmp_path):
    from stepnet.data.blocks import TEST_SESSIONS, real_sessions
    for sid in ("S001", "S008", *TEST_SESSIONS):
        (tmp_path / sid).mkdir()
    assert real_sessions(tmp_path) == ["S001", "S008"]
    assert real_sessions(tmp_path, include_test=True) == ["S001", "S008", *TEST_SESSIONS]
