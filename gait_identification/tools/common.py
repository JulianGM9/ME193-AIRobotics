"""Dataset helpers shared by train.py and evaluate.py (laptop side)."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import gaitlib as g  # noqa: E402

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "python" / "models"  # inside python/ so it ships with the board app
REPORTS_DIR = ROOT / "reports"

# Every feature variant the tools need: the deployed models plus the evaluation comparisons.
MODEL1_VARIANTS = [(axis, stat) for axis in g.AXES for stat in ("max", "p95")]


@dataclass
class Windows:
    session: np.ndarray  # session number per window
    label: np.ndarray
    order: np.ndarray  # window start time (ms) - for smoothing in time order
    m2: np.ndarray  # (n, 18) Model 2 features
    m1: dict  # (axis, stat) -> (n, 1) Model 1 features

    def subset(self, mask):
        return Windows(self.session[mask], self.label[mask], self.order[mask], self.m2[mask],
                       {k: v[mask] for k, v in self.m1.items()})


def find_sessions(data_dir=None, include_test=False):
    """Session JSON paths, sorted by session number. Test-protocol sessions are skipped."""
    import json
    paths = sorted(Path(data_dir or DATA_DIR).glob("session_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [p for p in paths if include_test or json.loads(p.read_text()).get("protocol") != "test"]


def extract(session_paths):
    """Window every session and compute all features."""
    rows = {"session": [], "label": [], "order": [], "m2": [], "m1": {v: [] for v in MODEL1_VARIANTS}}
    for path in session_paths:
        meta, t_ms, acc, gyr = g.load_session(path)
        for label, i in g.session_windows(t_ms, meta["segments"]):
            a, w = acc[i:i + g.WINDOW], gyr[i:i + g.WINDOW]
            rows["session"].append(meta["session"])
            rows["label"].append(label)
            rows["order"].append(t_ms[i])
            rows["m2"].append(g.model2_features(a, w))
            for axis, stat in MODEL1_VARIANTS:
                rows["m1"][(axis, stat)].append(g.model1_feature(a, axis, stat))
    return Windows(np.array(rows["session"]), np.array(rows["label"]), np.array(rows["order"]),
                   np.array(rows["m2"]).reshape(-1, len(g.MODEL2_FEATURE_NAMES)),
                   {k: np.array(v).reshape(-1, 1) for k, v in rows["m1"].items()})


def features_for(win, model, axis="S", stat="max", drop_group=None):
    if model == 1:
        return win.m1[(axis, stat)]
    if drop_group is None:
        return win.m2
    keep = [i for i, n in enumerate(g.MODEL2_FEATURE_NAMES) if n not in g.MODEL2_GROUPS[drop_group]]
    return win.m2[:, keep]


def choose_k(win, model, axis="S", stat="max", drop_group=None, ks=range(1, 16, 2)):
    """Pick k by leave-one-session-out accuracy (D12). Falls back to 5 with fewer than 2 sessions.
    Returns (k, {k: accuracy})."""
    sessions = sorted(set(win.session.tolist()))
    if len(sessions) < 2:
        return 5, {}
    X, y = features_for(win, model, axis, stat, drop_group), win.label
    scores = {}
    for k in ks:
        correct = total = 0
        for s in sessions:
            tr, te = win.session != s, win.session == s
            pred = g.KNN(k).fit(X[tr], y[tr]).predict(X[te])
            correct += (pred == y[te]).sum()
            total += te.sum()
        scores[k] = correct / total
    best = max(scores.values())
    # Prefer 5 if it ties the best (the planned default), otherwise the smallest best k.
    k = 5 if scores.get(5) == best else min(k for k, s in scores.items() if s == best)
    return k, scores
