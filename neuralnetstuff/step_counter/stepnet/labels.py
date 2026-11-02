"""Stride labels: candidate detection, the bout count check, and versioned label files (PLAN.md D6, D7, D18).

What we label: one event per right-foot stride, at the mid-swing minimum of the L-axis
(sagittal) gyro. On this mount it dips to about -150..-300 deg/s once per stride.

  data/labels/<session>/v001.json, v002.json, ...   (a new version on every save, never edited)

The count check (D7): inside every bout or no-step block that is still marked ok,
|2 * strides - counted steps| <= 1. Labels can only be saved when every check passes.

A bout's counted range starts at "Go", not at the start of its instruction: moving into
position while the instruction plays (turning at the top of the stairs, walking back to the
hallway) is not part of the bout, and Julian doesn't count it (found in Guided session 1).
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from stepnet.preprocess import to_body, to_physical

LABELS_DIR = Path(__file__).resolve().parents[1] / "data" / "labels"
EVENT = "mid_swing_L_gyro_min"

# Deliberately loose (high recall): it proposes too many candidates and the human rejects
# the extras. Labels must never be raw detector output, or a model trained on them could only
# learn to copy this detector (D7).
CANDIDATE_PARAMS = {"height_dps": 60.0, "prominence_dps": 40.0, "min_gap_s": 0.35}


def body_signals(samples, meta):
    """Calibrated accel (g) and bias-corrected gyro (deg/s) in the body frame, each (n, 3) S/F/L."""
    acc, gyr = to_physical(samples, meta.get("accel_calibration"))
    return to_body(acc), to_body(gyr - np.asarray(meta["gyro_bias_dps"]))


def find_candidates(samples, meta, params=CANDIDATE_PARAMS) -> list[int]:
    """seq numbers of candidate strides: local minima of the L-axis gyro."""
    _, gyr = body_signals(samples, meta)
    idx, _ = find_peaks(-gyr[:, 2], height=params["height_dps"], prominence=params["prominence_dps"],
                        distance=max(1, int(params["min_gap_s"] * 100)))
    return [int(s) for s in samples[idx, 0]]


def counted_range(seg) -> tuple[int, int]:
    """[first, last) seq whose strides count toward the block's steps (and that training may use)."""
    return seg.get("go_seq") or seg["start_seq"], seg["end_seq"]


def check_segments(meta, strides, segment_ok=None) -> list[dict]:
    """The count check for every bout and no-step block (D7)."""
    strides = np.asarray(sorted(strides), dtype=np.int64)
    segment_ok = {int(k): v for k, v in (segment_ok or {}).items()}
    checks = []
    for seg in meta["segments"]:
        if seg["kind"] not in ("bout", "hold"):
            continue
        ok = segment_ok.get(seg["i"], seg.get("ok", True))
        first, last = counted_range(seg)
        n = int(((strides >= first) & (strides < last)).sum())
        steps = seg.get("steps")
        if not ok:
            status = "excluded"
        elif steps is None:
            status = "no count"
        else:
            status = "pass" if abs(2 * n - steps) <= 1 else "fail"
        checks.append({"i": seg["i"], "label": seg["label"], "kind": seg["kind"], "steps": steps,
                       "strides": n, "ok": ok, "status": status})
    return checks


def all_pass(checks) -> bool:
    return all(c["status"] in ("pass", "excluded") for c in checks)


def label_versions(session, labels_dir=LABELS_DIR) -> list[Path]:
    return sorted((Path(labels_dir) / session).glob("v[0-9][0-9][0-9].json"))


def load_latest_labels(session, labels_dir=LABELS_DIR):
    versions = label_versions(session, labels_dir)
    return json.loads(versions[-1].read_text()) if versions else None


def save_labels(session, meta, strides, segment_ok=None, labeler="julian", labels_dir=LABELS_DIR) -> dict:
    """Write the next version. Raises ValueError (with the failing checks) unless every check passes."""
    checks = check_segments(meta, strides, segment_ok)
    if not all_pass(checks):
        failing = [f"#{c['i']} {c['label']}: {c['strides']} strides vs {c['steps']} steps ({c['status']})"
                   for c in checks if c["status"] not in ("pass", "excluded")]
        raise ValueError("count check failed: " + "; ".join(failing))
    version = len(label_versions(session, labels_dir)) + 1
    labels = {
        "session": session,
        "version": version,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labeler": labeler,
        "event": EVENT,
        "candidate_params": CANDIDATE_PARAMS,
        "strides": sorted(int(s) for s in strides),
        "segment_ok": {str(k): bool(v) for k, v in (segment_ok or {}).items()},
        "checks": checks,
    }
    out = Path(labels_dir) / session / f"v{version:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2) + "\n")
    return labels
