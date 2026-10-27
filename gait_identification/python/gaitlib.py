"""Shared gait-classifier code: session loading, windowing, features and KNN.

Used by the laptop tools (tools/train.py, tools/evaluate.py) AND by the
live app on the board, so training and live classification compute
features identically (PLAN.md D1). Depends only on numpy.
"""
import json
from pathlib import Path

import numpy as np

FS_HZ = 100  # sample rate (PLAN.md D7)
WINDOW = 200  # 2 s windows (D8)
HOP = 100  # 50% overlap -> one prediction per second
TRIM_S = 3.0  # drop this much from both ends of each labeled segment (D8)
MAX_GAP_MS = 15  # a window may not span a gap longer than this (1.5 sample periods)
SMOOTH_N = 3  # majority vote over the last 3 predictions (D8)

CLASSES = ["standing", "walking", "running", "biking"]
POSTURE_CHECK = "lying_still"  # recorded but never trained on (D10)

AXES = "SFL"  # shin, forward, lateral (D6)
CADENCE_BAND_HZ = (0.3, 4.0)
CADENCE_MIN_STD_DPS = 10.0  # below this the leg isn't swinging -> cadence 0

MODEL2_FEATURE_NAMES = (
    [f"std_acc_{a}" for a in AXES] + [f"std_gyr_{a}" for a in AXES]
    + [f"range_acc_{a}" for a in AXES] + [f"range_gyr_{a}" for a in AXES]
    + [f"mean_gyr_{a}" for a in AXES]
    + ["peak_acc_mag", "std_acc_mag", "cadence_hz"]
)
# Groups for the "drop one group at a time" check (D10).
MODEL2_GROUPS = {
    "per-channel std": [n for n in MODEL2_FEATURE_NAMES if n.startswith("std_") and "mag" not in n],
    "per-channel range": [n for n in MODEL2_FEATURE_NAMES if n.startswith("range_")],
    "gyro means": [n for n in MODEL2_FEATURE_NAMES if n.startswith("mean_gyr")],
    "accel magnitude": ["peak_acc_mag", "std_acc_mag"],
    "cadence": ["cadence_hz"],
}


# ---------------------------------------------------------------- sessions

def to_body(xyz, axis_map):
    """Reorder/flip sensor x/y/z columns into S, F, L using e.g. {"S": "-x", "F": "-y", "L": "+z"}."""
    cols = {"x": 0, "y": 1, "z": 2}
    return np.column_stack([
        (-1.0 if axis_map[a][0] == "-" else 1.0) * xyz[:, cols[axis_map[a][1]]] for a in AXES
    ])


def raw_to_body(raw_xyz_acc, raw_xyz_gyr, meta):
    """Raw int16 sensor readings -> (accel in g, bias-corrected gyro in deg/s), both in S/F/L."""
    units = meta["units"]
    bias = meta.get("gyro_bias_dps") or {"x": 0.0, "y": 0.0, "z": 0.0}
    acc = np.asarray(raw_xyz_acc, float) / units["accel_lsb_per_g"]
    gyr = np.asarray(raw_xyz_gyr, float) / units["gyro_lsb_per_dps"] - np.array([bias["x"], bias["y"], bias["z"]])
    return to_body(acc, meta["axis_map"]), to_body(gyr, meta["axis_map"])


def load_session(json_path):
    """Returns (meta, t_ms, acc_SFL_g, gyr_SFL_dps)."""
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text())
    raw = np.loadtxt(json_path.with_suffix(".csv"), delimiter=",", skiprows=1, ndmin=2)
    acc, gyr = raw_to_body(raw[:, 2:5], raw[:, 5:8], meta)
    return meta, raw[:, 1], acc, gyr


def session_windows(t_ms, labels_segments, labels=CLASSES + [POSTURE_CHECK]):
    """Start indices of usable windows, as (label, start_idx) pairs.

    A window is usable if it lies entirely inside one ok segment after trimming
    TRIM_S from both ends (so it never crosses a label change) and contains no
    gap in the data.
    """
    out = []
    gap_after = np.diff(t_ms) > MAX_GAP_MS  # gap between sample i and i+1
    gap_count = np.concatenate([[0], np.cumsum(gap_after)])
    for label, start_ms, end_ms, ok in labels_segments:
        if not ok or label not in labels:
            continue
        lo = np.searchsorted(t_ms, start_ms + TRIM_S * 1000)
        hi = np.searchsorted(t_ms, end_ms - TRIM_S * 1000)  # exclusive
        for i in range(lo, hi - WINDOW + 1, HOP):
            if gap_count[i + WINDOW - 1] == gap_count[i]:
                out.append((label, i))
    return out


# ---------------------------------------------------------------- features

def model1_feature(acc, axis="S", stat="max"):
    """Model 1: peak acceleration in one direction, gravity removed (D11).

    acc: (WINDOW, 3) accel in g, S/F/L columns. Returns a 1-element array.
    """
    a = acc[:, AXES.index(axis)]
    dyn = a - a.mean()
    return np.array([dyn.max() if stat == "max" else np.percentile(dyn, 95)])


def cadence_hz(signal, fs=FS_HZ):
    """Dominant repetition rate of a signal within CADENCE_BAND_HZ, 0 if it barely moves."""
    x = signal - signal.mean()
    if x.std() < CADENCE_MIN_STD_DPS:
        return 0.0
    n_fft = 2048  # zero-padding: finer peak location than the 0.5 Hz bin of a 2 s window
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    band = (freqs >= CADENCE_BAND_HZ[0]) & (freqs <= CADENCE_BAND_HZ[1])
    return float(freqs[band][np.argmax(spec[band])])


def model2_features(acc, gyr):
    """Model 2: 18 motion-only features from all six channels (D10).

    No accelerometer means - they mostly encode which way gravity points (posture, D9).
    """
    mag = np.linalg.norm(acc, axis=1)
    return np.concatenate([
        acc.std(axis=0), gyr.std(axis=0),
        np.ptp(acc, axis=0), np.ptp(gyr, axis=0),
        gyr.mean(axis=0),
        [mag.max(), mag.std(), cadence_hz(gyr[:, AXES.index("L")])],
    ])


def window_features(acc, gyr, model, axis="S", stat="max"):
    if model == 1:
        return model1_feature(acc, axis, stat)
    return model2_features(acc, gyr)


# ---------------------------------------------------------------- KNN

class KNN:
    """Hand-written k-nearest-neighbors with standardization (D12).

    - features standardized with the TRAINING mean/std only
    - Euclidean distance
    - majority vote of the k nearest; a tie goes to the tied class whose
      neighbors have the smallest total distance
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant feature: leave it unscaled instead of dividing by 0
        self.X_ = (X - self.mean_) / self.std_
        self.y_ = np.asarray(y)
        self.classes_ = np.array(sorted(set(self.y_.tolist()), key=lambda c: CLASSES.index(c) if c in CLASSES else 99))
        return self

    def predict(self, X):
        X = (np.atleast_2d(np.asarray(X, float)) - self.mean_) / self.std_
        d = np.sqrt(((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        k = min(self.k, len(self.y_))
        nearest = np.argsort(d, axis=1, kind="stable")[:, :k]
        preds = []
        for row, idx in enumerate(nearest):
            votes = {}
            for j in idx:
                c = self.y_[j]
                count, dist = votes.get(c, (0, 0.0))
                votes[c] = (count + 1, dist + d[row, j])
            # most votes first, then smallest total distance
            preds.append(min(votes.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[0])
        return np.array(preds)

    def save(self, path, **extra):
        np.savez(path, k=self.k, mean=self.mean_, std=self.std_, X=self.X_, y=self.y_,
                 meta=json.dumps(extra))

    @classmethod
    def load(cls, path):
        f = np.load(path, allow_pickle=False)
        m = cls(int(f["k"]))
        m.mean_, m.std_, m.X_, m.y_ = f["mean"], f["std"], f["X"], f["y"]
        m.classes_ = np.array(sorted(set(m.y_.tolist()), key=lambda c: CLASSES.index(c) if c in CLASSES else 99))
        m.meta = json.loads(str(f["meta"]))
        return m


class Smoother:
    """Majority vote over the last n raw predictions (D8). The shown class only
    changes when a class has a strict majority; otherwise it stays."""

    def __init__(self, n=SMOOTH_N):
        self.n, self.recent, self.shown = n, [], None

    def update(self, pred):
        self.recent = (self.recent + [pred])[-self.n:]
        best = max(set(self.recent), key=self.recent.count)
        if self.recent.count(best) * 2 > len(self.recent):
            self.shown = best
        elif self.shown is None:
            self.shown = pred
        return self.shown


def smooth(preds, n=SMOOTH_N):
    """Smooth a whole sequence of raw predictions (offline evaluation)."""
    sm = Smoother(n)
    return np.array([sm.update(p) for p in preds])
