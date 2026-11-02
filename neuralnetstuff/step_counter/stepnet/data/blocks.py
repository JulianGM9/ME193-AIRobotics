"""Sessions + their usable labeled blocks, ready for detectors and evaluation (PLAN.md D9, D14).

A block is usable when it is a bout or no-step block that is still in use in the latest label
version. Only its counted range is used ([go_seq, end_seq) for a bout): strides there are
labeled and checked against the step count. Everything else in a session is context only.
"""
from dataclasses import dataclass, field

import numpy as np

from stepnet import labels as lab
from stepnet.io.session import RAW_DIR, read_session


@dataclass
class Block:
    session: str
    i: int
    label: str
    activity: str  # label without the repeat suffix, e.g. walk_normal_2 -> walk_normal
    kind: str  # "bout", "counted" (timed, may contain steps) or "hold" (no steps)
    first: int  # counted range [first, last) in seq
    last: int
    steps: int  # counted steps (both feet)
    strides: np.ndarray  # labeled stride seqs inside the range

    @property
    def seconds(self):
        return (self.last - self.first) / 100


@dataclass
class SessionData:
    session: str
    meta: dict
    seq: np.ndarray  # (n,)
    acc: np.ndarray  # (n, 3) body frame S, F, L in g, calibrated
    gyr: np.ndarray  # (n, 3) body frame in deg/s, bias removed
    blocks: list[Block] = field(default_factory=list)


def activity_of(label: str) -> str:
    return label[:-2] if label[-2:] in ("_2", "_3") else label


def load_session(session, raw_dir=RAW_DIR, labels_dir=lab.LABELS_DIR) -> SessionData:
    samples, meta = read_session(raw_dir / session)
    labels = lab.load_latest_labels(session, labels_dir)
    if labels is None:
        raise ValueError(f"{session} is not labeled")
    acc, gyr = lab.body_signals(samples, meta)
    strides = np.asarray(labels["strides"], dtype=np.int64)
    checks = {c["i"]: c for c in lab.check_segments(meta, strides, labels["segment_ok"])}
    data = SessionData(session, meta, samples[:, 0].copy(), acc, gyr)
    for seg in meta["segments"]:
        c = checks.get(seg["i"])
        if c is None or c["status"] != "pass":
            continue
        first, last = lab.counted_range(seg)
        data.blocks.append(Block(session, seg["i"], seg["label"], activity_of(seg["label"]), seg["kind"],
                                 first, last, seg["steps"], strides[(strides >= first) & (strides < last)]))
    return data


def real_sessions(raw_dir=RAW_DIR) -> list[str]:
    return sorted(p.name for p in raw_dir.glob("S[0-9][0-9][0-9]"))
