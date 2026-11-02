"""Evaluation metrics and bootstrap confidence intervals (PLAN.md D14).

Per block (a usable bout or no-step block):
  events     detections matched one-to-one to labeled strides within +/-100 ms -> tp, fp, fn
  count      predicted steps = 2 x detected strides; count error = predicted - counted steps
Summary over blocks:
  stride F1, mean |count error| per bout (steps), net count error of all bouts (%),
  false strides per minute in no-step blocks
Confidence intervals: bootstrap over blocks (resample blocks with replacement).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

TOLERANCE_SAMPLES = 10  # +/-100 ms


def match(pred, true, tol=TOLERANCE_SAMPLES) -> tuple[int, int, int]:
    """(tp, fp, fn): each labeled stride can be matched by at most one detection."""
    pred, true = np.asarray(pred), np.asarray(true)
    if len(pred) == 0 or len(true) == 0:
        return 0, len(pred), len(true)
    dist = np.abs(pred[:, None] - true[None, :]).astype(float)
    cost = np.where(dist <= tol, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    tp = int((cost[rows, cols] <= tol).sum())
    return tp, len(pred) - tp, len(true) - tp


def block_rows(blocks, detections) -> list[dict]:
    """One result row per block. detections: session -> array of detected stride seqs."""
    rows = []
    for b in blocks:
        det = np.asarray(detections[b.session])
        pred = det[(det >= b.first) & (det < b.last)]
        tp, fp, fn = match(pred, b.strides)
        rows.append({"session": b.session, "i": b.i, "label": b.label, "activity": b.activity, "kind": b.kind,
                     "seconds": b.seconds, "steps": b.steps, "true_strides": len(b.strides),
                     "pred_strides": len(pred), "tp": tp, "fp": fp, "fn": fn,
                     "count_err": 2 * len(pred) - b.steps})
    return rows


def summarize(rows) -> dict:
    rows = list(rows)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    bouts = [r for r in rows if r["kind"] != "hold"]  # bouts and counted blocks: they have steps
    holds = [r for r in rows if r["kind"] == "hold"]
    steps = sum(r["steps"] for r in bouts)
    hold_min = sum(r["seconds"] for r in holds) / 60
    return {
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "bout_abs_err": float(np.mean([abs(r["count_err"]) for r in bouts])) if bouts else 0.0,
        "net_err_pct": 100 * sum(r["count_err"] for r in bouts) / steps if steps else 0.0,
        "false_per_min": sum(r["fp"] for r in holds) / hold_min if hold_min else 0.0,
        "blocks": len(rows),
    }


def bootstrap(rows, n=2000, seed=0, level=0.95) -> dict:
    """{metric: (low, high)} percentile intervals from resampling blocks with replacement."""
    rows = list(rows)
    rng = np.random.default_rng(seed)
    draws = [summarize([rows[k] for k in rng.integers(0, len(rows), len(rows))]) for _ in range(n)]
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {key: (float(np.percentile([d[key] for d in draws], lo)), float(np.percentile([d[key] for d in draws], hi)))
            for key in draws[0] if key != "blocks"}


def tuning_loss(rows) -> float:
    """What grid search minimizes on training sessions: total |count error| in steps, over all blocks."""
    return float(sum(abs(r["count_err"]) for r in rows))


HARD_ACTIVITIES = ("shuffle", "foot_tap_seated")  # + every counted block (PLAN.md D14)


def is_hard(row) -> bool:
    return row["kind"] == "counted" or row["activity"] in HARD_ACTIVITIES


def paired_bootstrap(rows_a, rows_b, key, n=2000, seed=0, level=0.95) -> tuple[float, float, float]:
    """(difference, low, high) of summarize(a)[key] - summarize(b)[key], resampling the SAME blocks.

    Both models are scored on identical blocks, so resampling them together cancels the
    block-to-block variation they share: a much sharper test than comparing two separate CIs.
    """
    rows_a, rows_b = list(rows_a), list(rows_b)
    assert [(r["session"], r["i"]) for r in rows_a] == [(r["session"], r["i"]) for r in rows_b]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        k = rng.integers(0, len(rows_a), len(rows_a))
        diffs.append(summarize([rows_a[i] for i in k])[key] - summarize([rows_b[i] for i in k])[key])
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return summarize(rows_a)[key] - summarize(rows_b)[key], float(lo), float(hi)
