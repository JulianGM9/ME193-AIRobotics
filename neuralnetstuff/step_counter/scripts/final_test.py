"""The one-time final evaluation on the locked test sessions (milestone M10, PLAN.md D9, D14).

  ../../.venv/bin/python scripts/final_test.py --final

Everything here was fixed before this runs: the ensemble and its conformal q come from the
training sessions only (scripts/uncertainty.py), and B2 is tuned on all training sessions.
The test sessions S009 (protocol A) and S010 (protocol C) are scored once -> reports/m10_final_test.md.

D14, the success criterion (revised after M4):
  1. no worse than B2 on scripted blocks: the paired 95% CI of (ensemble - B2) mean |count error|
     and false strides/min must not lie entirely above 0
  2. better than B2 on the hard subset: the paired 95% CI of (ensemble - B2) mean |count error|
     must lie entirely below 0
  3. calibrated: 90% intervals cover the counted steps about 90% of the time
     (90% inside the 95% Wilson interval of the observed coverage)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.baselines import B2_GRID, Detector, loso
from stepnet.data.blocks import TEST_SESSIONS, load_session, real_sessions
from stepnet.eval.metrics import block_rows, bootstrap, is_hard, paired_bootstrap, summarize
from stepnet.tracking import ROOT
from stepnet.train import predict_session
from stepnet.uncertainty.counting import estimate, peaks_in
from stepnet.uncertainty.ensemble import load_ensemble

REPORT = ROOT / "reports" / "m10_final_test.md"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="required: this looks at the locked test sessions")
    args = ap.parse_args()
    if not args.final:
        sys.exit("This is the one-time evaluation on the locked test sessions. Run it with --final.")

    members, cal = load_ensemble()
    q90 = cal["q"]["0.9"]
    train = {s: load_session(s) for s in real_sessions()}
    test = {s: load_session(s) for s in TEST_SESSIONS}
    assert not set(test) & set(cal["trained_on"]), "a test session was used for training"

    # B2: the parameters tuned on every training session (what we'd ship)
    _, _, b2_params = loso("B2", B2_GRID, train, {s: Detector(d) for s, d in train.items()})

    ens_rows, b2_rows, intervals = [], [], []
    for sid, s in test.items():
        curves = np.array([predict_session(m, st, s, cal["config"]["data"]["window"], cal["config"]["detect"]["hop"])
                           for m, st in members])
        mean = curves.mean(axis=0)
        gap = max(1, int(cal["min_gap_s"] * 100))
        idx, _ = peaks_in(mean, 0, len(mean), cal["threshold"], gap)
        ens_rows += block_rows(s.blocks, {sid: s.seq[idx]})
        b2_rows += block_rows(s.blocks, {sid: Detector(s).detect(b2_params)})
        for b in s.blocks:
            lo, hi = np.searchsorted(s.seq, b.first), np.searchsorted(s.seq, b.last)
            est = estimate(curves, lo, hi, cal["sigma_delta2"], cal["threshold"], cal["min_gap_s"])
            half = q90 * est["sigma"]
            intervals.append({"session": sid, "i": b.i, "label": b.label, "kind": b.kind, "activity": b.activity,
                              "steps": b.steps, "n_hat": est["n_hat"], "half": half,
                              "covered": abs(b.steps - est["n_hat"]) <= half})

    lines = ["# M10: Final test on the locked sessions (run once)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/final_test.py --final`. Test sessions: "
             f"{', '.join(test)} ({len(ens_rows)} usable blocks). Ensemble `cnn_ensemble_v1` ({cal['members']} CNNs, "
             f"trained on {', '.join(cal['trained_on'])}), conformal q90 = {q90:.2f}. B2 tuned on the same training sessions.", ""]
    verdicts = []
    for title, sel in (("Scripted blocks", lambda r: not is_hard(r)), ("Hard subset", is_hard), ("All blocks", lambda r: True)):
        a = [r for r in ens_rows if sel(r)]
        b = [r for r in b2_rows if sel(r)]
        if not a:
            continue
        sa, sb, ca, cb = summarize(a), summarize(b), bootstrap(a), bootstrap(b)
        lines += [f"## {title} ({len(a)} blocks)", "", "| metric | ensemble | B2 | ensemble − B2 (paired 95% CI) |", "|---|---|---|---|"]
        diffs = {}
        for key, label, f in (("f1", "stride F1", "{:.3f}"), ("bout_abs_err", "mean abs count error per block with steps", "{:.2f}"),
                              ("net_err_pct", "net count error (%)", "{:+.1f}"), ("false_per_min", "false strides / min", "{:.1f}")):
            d = paired_bootstrap(a, b, key)
            diffs[key] = d
            lines.append(f"| {label} | {f.format(sa[key])} [{f.format(ca[key][0])}, {f.format(ca[key][1])}] | "
                         f"{f.format(sb[key])} [{f.format(cb[key][0])}, {f.format(cb[key][1])}] | "
                         f"{f.format(d[0])} [{f.format(d[1])}, {f.format(d[2])}] |")
        lines.append("")
        if title == "Scripted blocks":
            ok = diffs["bout_abs_err"][1] <= 0 and diffs["false_per_min"][1] <= 0
            verdicts.append(("1. no worse than B2 on scripted blocks", ok,
                             f"count error diff CI [{diffs['bout_abs_err'][1]:.2f}, {diffs['bout_abs_err'][2]:.2f}], "
                             f"false/min diff CI [{diffs['false_per_min'][1]:.2f}, {diffs['false_per_min'][2]:.2f}]"))
        if title == "Hard subset":
            ok = diffs["bout_abs_err"][2] < 0
            verdicts.append(("2. better than B2 on the hard subset", ok,
                             f"count error diff {diffs['bout_abs_err'][0]:.2f} CI [{diffs['bout_abs_err'][1]:.2f}, {diffs['bout_abs_err'][2]:.2f}]"))
    k, n = sum(r["covered"] for r in intervals), len(intervals)
    lo, hi = wilson(k, n)
    verdicts.append(("3. calibrated 90% intervals", lo <= 0.9 <= hi, f"{k}/{n} = {100 * k / n:.0f}% covered, 95% CI [{100 * lo:.0f}%, {100 * hi:.0f}%]"))

    lines += ["## D14 verdict", "", "| criterion | result | evidence |", "|---|---|---|"]
    lines += [f"| {name} | {'✅ pass' if ok else '❌ fail'} | {ev} |" for name, ok, ev in verdicts]
    lines += ["", "## Per block: counted steps vs the ensemble's 90% interval", "",
              "| session | block | kind | counted | estimate ± 90% | inside? |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['session']} | #{r['i']} {r['label']} | {'no-step' if r['kind'] == 'hold' else r['kind']} | {r['steps']} | "
              f"{r['n_hat']} ± {r['half']:.1f} | {'✓' if r['covered'] else '✗'} |" for r in intervals]
    lines += ["", "## Per session totals (usable blocks)", "", "| session | counted | ensemble | B2 |", "|---|---|---|---|"]
    for sid in test:
        cnt = sum(r["steps"] for r in ens_rows if r["session"] == sid)
        e = cnt + sum(r["count_err"] for r in ens_rows if r["session"] == sid)
        b2 = cnt + sum(r["count_err"] for r in b2_rows if r["session"] == sid)
        lines.append(f"| {sid} | {cnt} | {e} ({100 * (e - cnt) / cnt:+.1f}%) | {b2} ({100 * (b2 - cnt) / cnt:+.1f}%) |")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
