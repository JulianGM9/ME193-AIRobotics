"""The counting + uncertainty maths (PLAN.md testing strategy: PB vs brute force, conformal coverage)."""

import itertools

import numpy as np
import pytest

from stepnet.uncertainty.counting import conformal_q, count_bouts, estimate, reliability


def bump(n, centres, height):
    t = np.arange(n)
    return np.max([height * np.exp(-0.5 * ((t - c) / 5) ** 2) for c in centres], axis=0)


def test_poisson_binomial_variance_matches_brute_force():
    p = np.array([0.9, 0.6, 0.3, 0.95])
    # enumerate all 2^4 outcomes: the exact variance of the number of successes
    outcomes = list(itertools.product([0, 1], repeat=len(p)))
    probs = [np.prod([pi if o else 1 - pi for pi, o in zip(p, out)]) for out in outcomes]
    counts = [sum(out) for out in outcomes]
    mean = np.dot(probs, counts)
    exact = np.dot(probs, (np.array(counts) - mean) ** 2)
    assert exact == pytest.approx((p * (1 - p)).sum())


def test_confident_curves_give_a_tight_count():
    curve = bump(1000, [100, 200, 300, 400], 0.99)
    est = estimate([curve, curve, curve], 0, 1000, sigma_delta2=0.0)
    assert est["n_hat"] == 8 and est["var_ens"] == 0
    assert est["sigma"] == pytest.approx(0.5)  # the floor: nothing is ever certain


def test_doubt_and_disagreement_widen_the_interval():
    sure = bump(1000, [100, 200, 300, 400], 0.99)
    unsure = np.maximum(sure, bump(1000, [500], 0.5))  # one coin-flip stride
    a = estimate([sure, sure], 0, 1000, 0.0)
    b = estimate([unsure, unsure], 0, 1000, 0.0)
    c = estimate([sure, unsure], 0, 1000, 0.0)
    assert b["var_pb"] > a["var_pb"]
    assert c["var_ens"] > 0  # the members disagree on the fifth stride


def test_bouts_split_at_gaps():
    assert count_bouts(np.array([0, 100, 200, 700, 800])) == 2
    assert count_bouts(np.array([])) == 0


def test_conformal_quantile_gives_the_promised_coverage():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(300):
        cal, new = rng.standard_t(3, 60), rng.standard_t(3)  # heavy tails: a Gaussian z would undercover
        hits.append(abs(new) <= conformal_q(np.abs(cal), 0.9))
    assert 0.86 <= np.mean(hits) <= 0.96


def test_reliability_of_a_calibrated_predictor():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    rows, ece = reliability(p, rng.uniform(0, 1, 5000) < p)
    assert ece < 0.03 and len(rows) == 10
