"""Classical baselines the network has to beat (PLAN.md D11, M4).

B1  peak detection: low-pass the L-axis (sagittal) gyro, then find its minima (mid-swing) with
    a minimum depth, prominence and spacing.
B2  B1 + a heel-strike gate: a real stride ends with the foot landing, so the swing must be
    followed by an acceleration impact. Seated leg swings never land (peak |a| ~1.3 g, versus
    >= 2.2 g after real strides in Guided session 1), so B2 should reject them.

    (The plan's first idea for B2, "the shin is still between strides", was dropped: on an
    ankle mount the shin keeps rotating at ~35 deg/s over the planted foot, and a seated leg
    swing is *stiller* than that, so the gate would have done the opposite of what we want.)

Both are tuned by grid search on training sessions only (see scripts/evaluate_baselines.py).
"""
import itertools
import time

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from stepnet.eval.metrics import block_rows, tuning_loss
from stepnet.io.protocol import SAMPLE_RATE_HZ

B1_GRID = {
    "cutoff_hz": [3.0, 5.0, 8.0],
    "height_dps": [40, 60, 90, 130, 180],
    "prominence_dps": [20, 40, 80, 120],
    "min_gap_s": [0.3, 0.4, 0.5],
}
B2_GRID = {**B1_GRID, "impact_g": [1.4, 1.6, 1.8, 2.0, 2.4], "impact_window_s": [0.4, 0.6]}
IMPACT_DELAY_S = 0.05  # the heel strike comes after mid-swing, not at it


def lowpass(x, cutoff_hz):
    """Zero-phase 4th-order Butterworth (offline; live use adds latency, like the CNN's window)."""
    b, a = butter(4, cutoff_hz / (SAMPLE_RATE_HZ / 2))
    return filtfilt(b, a, x)


def grid(spec):
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(spec[k] for k in keys))]


class Detector:
    """Runs B1/B2 on one session, caching the filtered signals so grid search is fast."""

    def __init__(self, session_data):
        self.s = session_data
        self.acc_mag = np.linalg.norm(session_data.acc, axis=1)
        self._filtered = {}
        self._peaks = {}

    def _gyro_l(self, cutoff_hz):
        if cutoff_hz not in self._filtered:
            self._filtered[cutoff_hz] = lowpass(self.s.gyr[:, 2], cutoff_hz)
        return self._filtered[cutoff_hz]

    def peaks_b1(self, p) -> np.ndarray:
        """Indexes (not seqs) of B1 detections."""
        key = (p["cutoff_hz"], p["height_dps"], p["prominence_dps"], p["min_gap_s"])
        if key not in self._peaks:
            idx, _ = find_peaks(-self._gyro_l(p["cutoff_hz"]), height=p["height_dps"],
                                prominence=p["prominence_dps"], distance=int(p["min_gap_s"] * SAMPLE_RATE_HZ))
            self._peaks[key] = idx
        return self._peaks[key]

    def detect(self, p) -> np.ndarray:
        """Stride seqs. Parameters with impact_g -> B2, otherwise B1."""
        idx = self.peaks_b1(p)
        if "impact_g" in p:
            lo = int(IMPACT_DELAY_S * SAMPLE_RATE_HZ)
            hi = int(p["impact_window_s"] * SAMPLE_RATE_HZ)
            n = len(self.acc_mag)
            keep = [i for i in idx if i + lo < n and self.acc_mag[i + lo:min(n, i + hi)].max() >= p["impact_g"]]
            idx = np.asarray(keep, dtype=np.int64)
        return self.s.seq[idx]


def loso(name, spec, data, detectors):
    """Leave-one-session-out: tune on the other sessions, run on the held-out one.

    Returns (held-out rows of all folds, {session: params}, params tuned on every session).
    """
    sessions = list(data)
    candidates = grid(spec)
    t0 = time.time()
    loss = np.array([[tuning_loss(block_rows(data[s].blocks, {s: detectors[s].detect(p)})) for s in sessions]
                     for p in candidates])
    print(f"{name}: {len(candidates)} parameter sets x {len(sessions)} sessions in {time.time() - t0:.0f} s")
    rows, folds = [], {}
    for k, held in enumerate(sessions):
        train_loss = loss.sum(axis=1) - loss[:, k]
        best = candidates[int(np.argmin(train_loss))]
        folds[held] = best
        rows += block_rows(data[held].blocks, {held: detectors[held].detect(best)})
    all_data = candidates[int(np.argmin(loss.sum(axis=1)))]
    return rows, folds, all_data
