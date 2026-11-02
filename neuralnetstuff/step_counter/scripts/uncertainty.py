"""Ensemble uncertainty + conformal calibration (milestone M6, PLAN.md D15, D16).

Run from neuralnetstuff/step_counter/:
  ../../.venv/bin/python -u scripts/uncertainty.py --cv            # 4-fold CV -> reports/m6_uncertainty.md
  ../../.venv/bin/python -u scripts/uncertainty.py --train-final   # 5-member ensemble on all 8 training sessions

--cv: for each fold (2 sessions held out), train 5 CNNs with different seeds on the other 6,
then for every held-out block compare four ways of putting an interval around the count:
  PB only        Poisson-binomial + doubling,          Gaussian z
  MC Dropout     20 passes of one CNN with dropout on,  Gaussian z
  ensemble       PB + ensemble spread + doubling,      Gaussian z
  ens+conformal  the same sigma, times a conformal q measured on the OTHER folds (cross-conformal)
Coverage = how often the counted steps land inside the interval; it should match the level.

--train-final saves the ensemble the final test (M10) and the live view (M7) use, with its
calibration (q per level, sigma_delta^2) measured by --cv: models/cnn_ensemble_v1/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.config import load_config, to_dict
from stepnet.data.blocks import load_session, real_sessions
from stepnet.eval.metrics import TOLERANCE_SAMPLES, is_hard, match
from stepnet.tracking import ROOT, Run, dataset_manifest
from stepnet.train import predict_session, train_model
from stepnet.uncertainty.counting import (SIGMA_FLOOR, conformal_q, doubling_variance, estimate, peaks_in,
                                          reliability, CANDIDATE_MIN)

FOLDS = [("S001", "S006"), ("S002", "S008"), ("S003", "S004"), ("S005", "S007")]  # each mixes protocols
MEMBERS = 5
MC_PASSES = 20
LEVELS = [0.5, 0.8, 0.9, 0.95]
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "cnn_ensemble_v1"
CV_RESULTS = REPORTS / "m6_cv_blocks.json"


def train_members(sessions, cfg, tag, log):
    members = []
    for k in range(MEMBERS):
        cfg.train.seed = k
        t0 = time.time()
        model, stats = train_model(sessions, cfg)
        members.append((model, stats))
        log(f"    {tag} member {k + 1}/{MEMBERS} trained in {time.time() - t0:.0f} s")
    return members


def run_cv(cfg):
    data = {s: load_session(s) for s in real_sessions()}
    assert sorted(s for f in FOLDS for s in f) == sorted(data), "folds must cover every training session once"
    records, candidates = [], []
    for fold, held in enumerate(FOLDS):
        train = [data[s] for s in data if s not in held]
        sd2 = doubling_variance([b for s in train for b in s.blocks])
        members = train_members(train, cfg, f"fold {fold + 1}/{len(FOLDS)} (held out {', '.join(held)})", print)
        for sid in held:
            s = data[sid]
            curves = np.array([predict_session(m, st, s, cfg.data.window, cfg.detect.hop) for m, st in members])
            mc = np.array([predict_session(members[0][0], members[0][1], s, cfg.data.window, cfg.detect.hop, mc_dropout=True)
                           for _ in range(MC_PASSES)])
            for b in s.blocks:
                lo, hi = np.searchsorted(s.seq, b.first), np.searchsorted(s.seq, b.last)
                ens = estimate(curves, lo, hi, sd2, cfg.detect.threshold, cfg.detect.min_gap_s)
                single = estimate(curves[:1], lo, hi, sd2, cfg.detect.threshold, cfg.detect.min_gap_s)
                mcd = estimate(mc, lo, hi, sd2, cfg.detect.threshold, cfg.detect.min_gap_s)
                rec = {"fold": fold, "session": sid, "i": b.i, "label": b.label, "activity": b.activity, "kind": b.kind,
                       "steps": b.steps, "seconds": b.seconds, "sigma_delta2": sd2,
                       "ens": ens, "single": single,
                       # MC Dropout: centre on member 1 without dropout, spread from the dropout passes
                       "mc": {**single, "sigma": max(SIGMA_FLOOR, float(np.sqrt(4 * np.var(mcd["member_counts"], ddof=1)
                                                                                     + single["var_dbl"])))},
                       "pb": {**ens, "sigma": max(SIGMA_FLOOR, float(np.sqrt(4 * ens["var_pb"] + ens["var_dbl"])))}}
                records.append(rec)
                # reliability: every candidate peak of the mean curve, and whether it is a real stride
                idx, p = peaks_in(curves.mean(axis=0), lo, hi, CANDIDATE_MIN, max(1, int(cfg.detect.min_gap_s * 100)))
                for q, pk in zip(s.seq[idx], p):
                    hit = len(b.strides) > 0 and np.abs(b.strides - q).min() <= TOLERANCE_SAMPLES
                    candidates.append({"p": float(pk), "hit": bool(hit), "hard": b.kind == "counted"})
            print(f"    scored {sid}: {len(s.blocks)} blocks", flush=True)
    CV_RESULTS.write_text(json.dumps({"records": records, "candidates": candidates}))
    return records, candidates


def coverage_table(records):
    """{method: {level: (coverage, mean width)}} with cross-conformal q for ens+conformal."""
    out = {}
    for method in ("pb", "mc", "ens"):
        out[method] = {}
        for lv in LEVELS:
            z = norm.ppf((1 + lv) / 2)
            hits = [abs(r["steps"] - r[method]["n_hat"]) <= z * r[method]["sigma"] for r in records]
            out[method][lv] = (float(np.mean(hits)), float(np.mean([2 * z * r[method]["sigma"] for r in records])))
    for name, floor in (("ens+conformal", False), ("ens+conformal, q>=z", True)):
        out[name] = {}
        for lv in LEVELS:
            hits, widths = [], []
            for r in records:
                q = cross_q(records, r, lv, floor)
                hits.append(abs(r["steps"] - r["ens"]["n_hat"]) <= q * r["ens"]["sigma"])
                widths.append(2 * q * r["ens"]["sigma"])
            out[name][lv] = (float(np.mean(hits)), float(np.mean(widths)))
    return out


def scores_of(records):
    return [abs(r["steps"] - r["ens"]["n_hat"]) / r["ens"]["sigma"] for r in records]


def calibrated_q(scores, level, floor=True):
    """The conformal factor, but never below the Gaussian z of the model's own sigma.

    With most blocks counted exactly right, the plain conformal quantile collapses to 0 (a
    zero-width interval). That still covers ~level of blocks ON AVERAGE, but only by being
    right on the easy ones: in the M6 cross-validation it covered just 60% of the hard blocks.
    Conformal may widen the model's interval, never shrink it.
    """
    q = conformal_q(scores, level)
    return max(q, float(norm.ppf((1 + level) / 2))) if floor else q


def cross_q(records, r, level, floor):
    """Cross-conformal: q for a block comes only from the OTHER folds' blocks."""
    return calibrated_q(scores_of([o for o in records if o["fold"] != r["fold"]]), level, floor)


def plots(records, candidates, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    text, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"pb": "#2a78d6", "mc": "#eb6834", "ens": "#1baf7a", "ens+conformal": "#eda100", "ens+conformal, q>=z": "#e87ba4"}  # slots 1-5
    names = {"pb": "Poisson-binomial only", "mc": "MC Dropout", "ens": "ensemble", "ens+conformal": "ensemble + conformal",
             "ens+conformal, q>=z": "ensemble + conformal, q >= z (used)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    rows, ece = reliability([c["p"] for c in candidates], [c["hit"] for c in candidates])
    ax1.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="perfect calibration")
    ax1.plot([r[0] for r in rows], [r[1] for r in rows], color=colors["ens"], lw=2, marker="o", ms=8,
             label=f"ensemble candidates (ECE {ece:.3f})")
    for x, y, n in rows:
        ax1.annotate(str(n), (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, color=muted)
    ax1.set_xlabel("predicted probability that a peak is a stride", color=muted)
    ax1.set_ylabel("fraction that really were strides", color=muted)
    ax1.set_title("Reliability (numbers: candidates per bin)", loc="left", fontsize=10, color=text)
    for m, vals in table.items():
        ax2.plot(LEVELS, [vals[lv][0] for lv in LEVELS], color=colors[m], lw=2, marker="o", ms=8, label=names[m])
    ax2.plot([0.45, 1], [0.45, 1], color=muted, lw=1, ls="--", label="nominal = actual")
    ax2.set_xlabel("nominal interval level", color=muted)
    ax2.set_ylabel("actual coverage on held-out blocks", color=muted)
    ax2.set_title(f"Interval coverage ({len(records)} held-out blocks)", loc="left", fontsize=10, color=text)
    for ax in (ax1, ax2):
        ax.grid(color=grid_c, lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(REPORTS / "m6_calibration.png", dpi=130)
    plt.close(fig)
    return ece


def write_report(records, candidates, cfg):
    table = coverage_table(records)
    ece = plots(records, candidates, table)
    names = {"pb": "Poisson-binomial only", "mc": "MC Dropout", "ens": "ensemble (Gaussian z)", "ens+conformal": "ensemble + conformal",
             "ens+conformal, q>=z": "**ensemble + conformal, q ≥ z** (used)"}
    lines = ["# M6: Uncertainty and calibration (4-fold cross-validation, training sessions only)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/uncertainty.py --cv`. {MEMBERS} CNNs per fold "
             f"(seeds 0-{MEMBERS - 1}), {len(records)} held-out blocks. Folds: "
             + ", ".join("+".join(f) for f in FOLDS) + ".", "",
             "## Coverage: how often the counted steps fall inside the interval (and mean width in steps)", "",
             "| method | " + " | ".join(f"{int(lv * 100)}%" for lv in LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for m, vals in table.items():
        lines.append(f"| {names[m]} | " + " | ".join(f"{vals[lv][0] * 100:.0f}% (±{vals[lv][1] / 2:.1f})" for lv in LEVELS) + " |")
    lines += ["", "## Coverage at 90%, by kind of block", "",
              "| interval | scripted | hard subset |", "|---|---|---|"]
    z90 = norm.ppf(0.95)
    variants = (("ensemble (Gaussian z)", lambda r: z90),
                ("ensemble + conformal", lambda r: cross_q(records, r, 0.9, False)),
                ("**ensemble + conformal, q ≥ z** (used)", lambda r: cross_q(records, r, 0.9, True)))
    for name, qf in variants:
        cells = []
        for sel in (lambda r: not is_hard(r), is_hard):
            rs = [r for r in records if sel(r)]
            hits = [abs(r["steps"] - r["ens"]["n_hat"]) <= qf(r) * r["ens"]["sigma"] for r in rs]
            cells.append(f"{np.mean(hits) * 100:.0f}% of {len(rs)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    err_e = np.mean([abs(r["steps"] - r["ens"]["n_hat"]) for r in records if r["kind"] != "hold"])
    err_s = np.mean([abs(r["steps"] - r["single"]["n_hat"]) for r in records if r["kind"] != "hold"])
    fp_e = sum(r["ens"]["strides"] for r in records if r["kind"] == "hold")
    fp_s = sum(r["single"]["strides"] for r in records if r["kind"] == "hold")
    lines += ["", "## Ensemble vs one CNN (same folds)", "",
              f"- mean |count error| per block with steps: ensemble {err_e:.2f}, single CNN {err_s:.2f}",
              f"- false strides in no-step blocks: ensemble {fp_e}, single CNN {fp_s}",
              "", f"## Reliability of candidate probabilities: ECE {ece:.3f} over {len(candidates)} candidates", "",
              "![calibration](m6_calibration.png)", ""]
    scores = scores_of(records)
    q = {str(lv): calibrated_q(scores, lv) for lv in LEVELS}
    lines += ["## Interval factors used by the final ensemble (all CV blocks)", "",
              ", ".join(f"q{int(lv * 100)}: conformal {conformal_q(scores, lv):.2f} -> used {calibrated_q(scores, lv):.2f}"
                        for lv in LEVELS) + " (used = max(conformal, Gaussian z))", ""]
    (REPORTS / "m6_uncertainty.md").write_text("\n".join(lines))
    return q, lines


def train_final(cfg):
    if not CV_RESULTS.exists():
        sys.exit("run --cv first: the final ensemble's calibration comes from the cross-validation")
    records = json.loads(CV_RESULTS.read_text())["records"]
    scores = scores_of(records)
    sessions = real_sessions()
    data = [load_session(s) for s in sessions]
    dataset_hash, _ = dataset_manifest(sessions)
    run = Run(cfg, dataset_hash, tag="ensemble")
    members = train_members(data, cfg, "final", print)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, (model, (mean, std)) in enumerate(members):
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "seed": k}, MODEL_DIR / f"member{k}.pt")
    calibration = {"q": {str(lv): calibrated_q(scores, lv) for lv in LEVELS},
                   "q_conformal": {str(lv): conformal_q(scores, lv) for lv in LEVELS},
                   "sigma_delta2": doubling_variance([b for s in data for b in s.blocks]),
                   "threshold": cfg.detect.threshold, "min_gap_s": cfg.detect.min_gap_s,
                   "members": MEMBERS, "trained_on": sessions, "dataset": dataset_hash, "run": run.dir.name,
                   "config": to_dict(cfg), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (MODEL_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    registry_path = ROOT / "models" / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["cnn_ensemble_v1"] = {"path": "models/cnn_ensemble_v1", "kind": "ensemble", "members": MEMBERS,
                                   "dataset": dataset_hash, "run": run.dir.name, "created_at": calibration["created_at"]}
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    run.finish({"model_dir": str(MODEL_DIR), "q": calibration["q"]})
    print(f"Saved {MODEL_DIR} and registered cnn_ensemble_v1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--train-final", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from the saved CV results")
    ap.add_argument("--recalibrate", action="store_true", help="rewrite the final ensemble's q from the saved CV results")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    cfg = load_config(ROOT / "configs" / "cnn.yaml")
    REPORTS.mkdir(exist_ok=True)
    if args.cv or args.report_only:
        if args.cv:
            records, candidates = run_cv(cfg)
        else:
            saved = json.loads(CV_RESULTS.read_text())
            records, candidates = saved["records"], saved["candidates"]
        _, lines = write_report(records, candidates, cfg)
        print("\n".join(lines))
    if args.recalibrate:
        cal_path = MODEL_DIR / "calibration.json"
        cal = json.loads(cal_path.read_text())
        scores = scores_of(json.loads(CV_RESULTS.read_text())["records"])
        cal["q"] = {str(lv): calibrated_q(scores, lv) for lv in LEVELS}
        cal["q_conformal"] = {str(lv): conformal_q(scores, lv) for lv in LEVELS}
        cal_path.write_text(json.dumps(cal, indent=2) + "\n")
        print("q:", cal["q"])
    if args.train_final:
        train_final(load_config(ROOT / "configs" / "cnn.yaml"))


if __name__ == "__main__":
    main()
