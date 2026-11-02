"""From probability curves to a step count with an interval (PLAN.md D15, D16).

For a stretch of time (a block, or "the session so far" in the live view):

  point estimate   N_hat = 2 x (peaks of the ensemble-mean curve above the threshold)
  Poisson-binomial each candidate peak k has probability p_k (its height on the mean curve):
                   Var_PB = sum p_k (1 - p_k)          (independent yes/no events)
  ensemble         Var_ens = variance of the members' own stride counts (model disagreement)
  doubling         Var_dbl = B x sigma_delta^2          (+/-1 per bout from steps = 2 x strides)
  combined         sigma^2 = 4 (Var_PB + Var_ens) + Var_dbl      (x4: strides -> steps)

These pieces assume independent errors, which real errors are not (a model that undercounts a
shuffle undercounts the whole shuffle), so the final interval is N_hat +/- q x sigma, where the
conformal factor q is measured on held-out data: the quantile of |N_true - N_hat| / sigma.
"""
import numpy as np
from scipy.signal import find_peaks

CANDIDATE_MIN = 0.05  # peaks this small still count toward Var_PB
SIGMA_FLOOR = 0.5  # steps: never claim more certainty than this
BOUT_GAP_S = 2.0  # strides further apart than this start a new bout


def peaks_in(prob, lo, hi, height, min_gap):
    idx, props = find_peaks(prob[lo:hi], height=height, distance=min_gap)
    return idx + lo, props["peak_heights"]


def count_bouts(stride_idx) -> int:
    if len(stride_idx) == 0:
        return 0
    return 1 + int((np.diff(np.sort(stride_idx)) > BOUT_GAP_S * 100).sum())


def estimate(member_curves, lo, hi, sigma_delta2, threshold=0.5, min_gap_s=0.3) -> dict:
    """Count + uncertainty for samples [lo, hi) given each ensemble member's probability curve."""
    curves = np.atleast_2d(member_curves)
    mean = curves.mean(axis=0)
    gap = max(1, int(min_gap_s * 100))
    hard, _ = peaks_in(mean, lo, hi, threshold, gap)
    _, p = peaks_in(mean, lo, hi, CANDIDATE_MIN, gap)
    p = np.clip(p, 0, 1)
    member_counts = [len(peaks_in(c, lo, hi, threshold, gap)[0]) for c in curves]
    var_pb = float((p * (1 - p)).sum())
    var_ens = float(np.var(member_counts, ddof=1)) if len(curves) > 1 else 0.0
    bouts = count_bouts(hard)
    var_dbl = bouts * sigma_delta2
    sigma = max(SIGMA_FLOOR, float(np.sqrt(4 * (var_pb + var_ens) + var_dbl)))
    return {"n_hat": 2 * len(hard), "strides": len(hard), "expected_strides": float(p.sum()),
            "var_pb": var_pb, "var_ens": var_ens, "var_dbl": var_dbl, "bouts": bouts, "sigma": sigma,
            "member_counts": member_counts}


def doubling_variance(blocks) -> float:
    """sigma_delta^2: mean squared (counted steps - 2 x labeled strides) over blocks with steps."""
    deltas = [b.steps - 2 * len(b.strides) for b in blocks if b.kind != "hold" and b.steps]
    return float(np.mean(np.square(deltas))) if deltas else 0.25


def conformal_q(scores, level) -> float:
    """Split-conformal quantile: the ceil((n + 1) level)-th smallest score (finite-sample valid)."""
    s = np.sort(np.asarray(scores))
    k = int(np.ceil((len(s) + 1) * level)) - 1
    return float(s[min(k, len(s) - 1)])


def reliability(p, hit, bins=10) -> tuple[list, float]:
    """Per bin (mean predicted p, observed hit rate, count), plus the expected calibration error."""
    p, hit = np.asarray(p), np.asarray(hit, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (p >= a) & (p < b) if b < 1 else (p >= a)
        if sel.sum():
            rows.append((float(p[sel].mean()), float(hit[sel].mean()), int(sel.sum())))
            ece += sel.sum() / len(p) * abs(p[sel].mean() - hit[sel].mean())
    return rows, float(ece)
