"""Unit tests for gaitlib on synthetic signals with known answers (PLAN.md D17).

Run from the gait_identification folder:
    python3 -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import gaitlib as g  # noqa: E402

T = np.arange(g.WINDOW) / g.FS_HZ


def test_to_body_applies_axis_map():
    xyz = np.array([[1.0, 2.0, 3.0]])
    assert g.to_body(xyz, {"S": "-x", "F": "-y", "L": "+z"}).tolist() == [[-1.0, -2.0, 3.0]]
    assert g.to_body(xyz, {"S": "+z", "F": "-x", "L": "+y"}).tolist() == [[3.0, -1.0, 2.0]]


def test_model1_removes_gravity():
    # Standing still: constant 1 g on S -> no peak once the mean is removed.
    acc = np.tile([1.0, 0.0, 0.0], (g.WINDOW, 1))
    assert g.model1_feature(acc)[0] == pytest.approx(0.0)
    # Same 2 g spike with gravity along S (standing) or not (lying) -> same feature.
    spike = np.zeros(g.WINDOW)
    spike[50] = 2.0
    upright = np.column_stack([1.0 + spike, np.zeros(g.WINDOW), np.zeros(g.WINDOW)])
    lying = np.column_stack([spike, np.ones(g.WINDOW), np.zeros(g.WINDOW)])
    assert g.model1_feature(upright)[0] == pytest.approx(g.model1_feature(lying)[0])
    assert g.model1_feature(upright)[0] == pytest.approx(2.0 - 2.0 / g.WINDOW)


def test_model1_uses_requested_axis():
    acc = np.zeros((g.WINDOW, 3))
    acc[10, 1] = 3.0  # spike only on F
    assert g.model1_feature(acc, axis="S")[0] == pytest.approx(0.0)
    assert g.model1_feature(acc, axis="F")[0] > 2.9


@pytest.mark.parametrize("freq", [0.9, 1.0, 1.4, 2.5])
def test_cadence_finds_sine_frequency(freq):
    sig = 200 * np.sin(2 * np.pi * freq * T)
    assert g.cadence_hz(sig) == pytest.approx(freq, abs=0.1)


def test_cadence_is_zero_when_still():
    assert g.cadence_hz(np.random.default_rng(0).normal(0, 1, g.WINDOW)) == 0.0


def test_model2_has_18_named_features_and_ignores_posture():
    rng = np.random.default_rng(1)
    motion = rng.normal(0, 0.3, (g.WINDOW, 3))
    gyr = rng.normal(0, 50, (g.WINDOW, 3))
    f = g.model2_features(motion + [1.0, 0, 0], gyr)
    assert len(f) == len(g.MODEL2_FEATURE_NAMES) == 18
    # Rotating which axis carries gravity must not change the per-channel features
    # (only the acceleration magnitude features can see gravity, and those are tilt-invariant
    # for a pure rotation, which this isn't exactly - so check just the per-channel ones).
    f_lying = g.model2_features(motion + [0, 1.0, 0], gyr)
    per_channel = slice(0, 15)
    assert np.allclose(f[per_channel], f_lying[per_channel])


def test_windows_trim_edges_and_never_cross_labels():
    t = np.arange(0, 30000, 10.0)  # 30 s at 100 Hz
    segs = [["walking", 0, 15000, True], ["running", 15000, 30000, True]]
    wins = g.session_windows(t, segs)
    for label, i in wins:
        start, end = t[i], t[i + g.WINDOW - 1]
        seg = next(s for s in segs if s[0] == label)
        assert start >= seg[1] + g.TRIM_S * 1000
        assert end < seg[2] - g.TRIM_S * 1000
    # 15 s segment - 6 s trim = 9 s -> windows starting every 1 s that fit 2 s -> 8 each
    assert sum(1 for l, _ in wins if l == "walking") == 8


def test_windows_skip_gaps_and_bad_segments():
    t = np.arange(0, 20000, 10.0)
    t[1000:] += 500  # a 0.5 s dropout at t = 10 s
    wins = g.session_windows(t, [["walking", 0, 20500, True]])
    for _, i in wins:
        assert np.diff(t[i:i + g.WINDOW]).max() <= g.MAX_GAP_MS
    assert g.session_windows(t, [["walking", 0, 20500, False]]) == []
    assert g.session_windows(t, [["transition", 0, 20500, True]]) == []


def has_clear_winner(neighbor_labels):
    counts = sorted((list(neighbor_labels).count(c) for c in set(neighbor_labels)), reverse=True)
    return len(counts) == 1 or counts[0] > counts[1]


def test_knn_matches_sklearn_when_no_ties():
    sklearn_neighbors = pytest.importorskip("sklearn.neighbors")
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c * 3, 1, (40, 5)) for c in range(4)]) * [1, 10, 100, 0.1, 1]
    y = np.repeat(g.CLASSES, 40)
    Xq = rng.normal(4, 3, (200, 5)) * [1, 10, 100, 0.1, 1]
    for k in (1, 3, 5, 7):
        ours = g.KNN(k).fit(X, y)
        mean, std = X.mean(axis=0), X.std(axis=0)
        ref = sklearn_neighbors.KNeighborsClassifier(k).fit((X - mean) / std, y)
        ref_pred = ref.predict((Xq - mean) / std)
        # Compare only queries without a vote tie (tie-breaking rules differ on purpose).
        _, idx = ref.kneighbors((Xq - mean) / std)
        no_tie = np.array([has_clear_winner(y[r]) for r in idx])
        assert no_tie.sum() > 100
        assert (ours.predict(Xq)[no_tie] == ref_pred[no_tie]).all()


def test_knn_tie_goes_to_closer_class():
    X = np.array([[0.0], [1.0], [10.0], [12.0]])
    y = np.array(["walking", "walking", "running", "running"])
    m = g.KNN(4).fit(X, y)
    assert m.predict([[2.0]])[0] == "walking"  # 2-2 tie, walking neighbors are closer
    assert m.predict([[10.0]])[0] == "running"


def test_knn_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(20, 3)), np.repeat(["standing", "biking"], 10)
    m = g.KNN(3).fit(X, y)
    m.save(tmp_path / "m.npz", model=2)
    m2 = g.KNN.load(tmp_path / "m.npz")
    assert m2.meta == {"model": 2}
    assert (m.predict(X) == m2.predict(X)).all()


def test_smooth_majority_of_three():
    raw = ["walking", "walking", "running", "walking", "running", "running", "running"]
    assert g.smooth(raw).tolist() == [
        "walking", "walking", "walking", "walking", "running", "running", "running"
    ]
