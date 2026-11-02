"""Recorded sessions on disk (PLAN.md D18).

  data/raw/<session>/imu.csv    raw samples  seq,t_ms,ax,ay,az,gx,gy,gz  (written once)
  data/raw/<session>/meta.json  who/when/mount/calibration + labeled segments

Segments are seq ranges [start_seq, end_seq), so they stay valid when gaps are filled
from the board's backup log. Session ids: S001, S002, ... for real sessions and
T001, ... for dry runs (never used for training).
"""
import json
from pathlib import Path

import numpy as np

from stepnet.io.protocol import COLUMNS

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def next_session_id(prefix="S", raw_dir=RAW_DIR) -> str:
    nums = [int(p.name[1:]) for p in Path(raw_dir).glob(f"{prefix}[0-9][0-9][0-9]") if p.is_dir()]
    return f"{prefix}{max(nums, default=0) + 1:03d}"


def write_session(session_dir, samples, meta):
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a recording
    np.savetxt(session_dir / "imu.csv", np.asarray(samples, dtype=np.int64), fmt="%d",
               delimiter=",", header=",".join(COLUMNS), comments="")
    write_meta(session_dir, meta)


def write_meta(session_dir, meta):
    (Path(session_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def read_session(session_dir):
    """-> (samples (n, 8) int64, meta dict)."""
    session_dir = Path(session_dir)
    samples = np.loadtxt(session_dir / "imu.csv", dtype=np.int64, delimiter=",", skiprows=1, ndmin=2)
    return samples, json.loads((session_dir / "meta.json").read_text())


def missing_seqs(samples) -> np.ndarray:
    """seq numbers absent between the first and last sample."""
    seqs = np.asarray(samples)[:, 0]
    return np.setdiff1d(np.arange(seqs[0], seqs[-1] + 1), seqs)


def fill_gaps(session_dir, board_csv) -> int:
    """Fill missing samples from the board's backup log (same run). Returns how many were filled."""
    samples, meta = read_session(session_dir)
    board = np.loadtxt(board_csv, dtype=np.int64, delimiter=",", skiprows=1, ndmin=2)
    missing = missing_seqs(samples)
    found = board[np.isin(board[:, 0], missing)]
    if len(found) == 0:
        return 0
    merged = np.concatenate([samples, found])
    merged = merged[np.argsort(merged[:, 0], kind="stable")]
    np.savetxt(Path(session_dir) / "imu.csv", merged, fmt="%d", delimiter=",",
               header=",".join(COLUMNS), comments="")
    meta["missing"] = int(len(missing_seqs(merged)))
    meta["filled_from_board"] = meta.get("filled_from_board", 0) + int(len(found))
    write_meta(session_dir, meta)
    return int(len(found))
