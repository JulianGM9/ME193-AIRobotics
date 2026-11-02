"""Train the CNN (milestone M5, PLAN.md D10, D12, D14, D18).

Run from neuralnetstuff/step_counter/:
  ../../.venv/bin/python scripts/train.py --loso                 # CNN vs B2, leave-one-session-out
  ../../.venv/bin/python scripts/train.py                        # one model on every training session
  ../../.venv/bin/python scripts/train.py --loso train.epochs=40 model.dropout=0.2   # config overrides

Only training sessions are used: the locked test sessions (S009, S010) are never touched here.
--loso trains one model per held-out session on the others, and tunes B2 on the same others,
so both are scored on exactly the same unseen blocks -> reports/m5_cnn_loso.md.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.baselines import B2_GRID, Detector, loso
from stepnet.config import load_config
from stepnet.data.blocks import load_session, real_sessions
from stepnet.eval.metrics import block_rows, bootstrap, is_hard, paired_bootstrap, summarize
from stepnet.tracking import Run, dataset_manifest
from stepnet.train import detect_peaks, predict_session, train_model

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SUMMARY_KEYS = [("f1", "stride F1", "{:.3f}"), ("bout_abs_err", "mean abs count error per block with steps", "{:.2f}"),
                ("net_err_pct", "net count error (%)", "{:+.1f}"), ("false_per_min", "false strides / min (no-step blocks)", "{:.1f}")]


def fmt(summary, ci, key, f):
    return f"{f.format(summary[key])} [{f.format(ci[key][0])}, {f.format(ci[key][1])}]"


def run_loso(cfg, data, run):
    sessions = list(data)
    cnn_rows, curves = [], {}
    for held in sessions:
        t0 = time.time()
        train = [data[s] for s in sessions if s != held]
        losses = []
        model, stats = train_model(train, cfg, log=lambda **v: losses.append(v["loss"]))
        prob = predict_session(model, stats, data[held], cfg.data.window, cfg.detect.hop)
        found, _ = detect_peaks(prob, data[held].seq, cfg.detect.threshold, cfg.detect.min_gap_s)
        rows = block_rows(data[held].blocks, {held: found})
        cnn_rows += rows
        curves[held] = losses
        s = summarize(rows)
        run.log(fold=held, final_loss=losses[-1], f1=s["f1"], bout_abs_err=s["bout_abs_err"], seconds=time.time() - t0)
        print(f"  {held}: loss {losses[0]:.3f} -> {losses[-1]:.3f}  F1 {s['f1']:.3f}  |err|/block {s['bout_abs_err']:.2f}"
              f"  false/min {s['false_per_min']:.1f}   ({time.time() - t0:.0f} s)", flush=True)

    b2_rows, _, _ = loso("B2", B2_GRID, data, {s: Detector(d) for s, d in data.items()})
    # same blocks, same order, for the paired comparison
    key = lambda r: (r["session"], r["i"])
    b2_by = {key(r): r for r in b2_rows}
    b2_rows = [b2_by[key(r)] for r in cnn_rows]
    return cnn_rows, b2_rows, curves


def report(cfg, cnn_rows, b2_rows, curves, run, dataset_hash):
    hard_c = [r for r in cnn_rows if is_hard(r)]
    hard_b = [r for r in b2_rows if is_hard(r)]
    easy_c = [r for r in cnn_rows if not is_hard(r)]
    easy_b = [r for r in b2_rows if not is_hard(r)]
    lines = ["# M5: CNN vs B2 (leave-one-session-out on the training sessions)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/train.py --loso`, run `{run.dir.name}`, "
             f"dataset `{dataset_hash}`. Each held-out session is scored with a CNN trained, and a B2 tuned, on the "
             "other sessions. Values: point estimate [95% bootstrap CI over blocks]. "
             "The **hard subset** is every protocol C counted block plus shuffles and foot taps (D14).", ""]
    for title, a, b in (("All blocks", cnn_rows, b2_rows), ("Scripted (not hard)", easy_c, easy_b), ("Hard subset", hard_c, hard_b)):
        sa, sb, ca, cb = summarize(a), summarize(b), bootstrap(a), bootstrap(b)
        lines += [f"## {title} ({len(a)} blocks)", "", "| metric | CNN | B2 | CNN − B2 (paired 95% CI) |", "|---|---|---|---|"]
        for k, label, f in SUMMARY_KEYS:
            d, lo, hi = paired_bootstrap(a, b, k)
            lines.append(f"| {label} | {fmt(sa, ca, k, f)} | {fmt(sb, cb, k, f)} | {f.format(d)} [{f.format(lo)}, {f.format(hi)}] |")
        lines.append("")
    lines += ["## Per activity", "", "| activity | kind | blocks | CNN abs err | B2 abs err | CNN false | B2 false |", "|---|---|---|---|---|---|---|"]
    for act in dict.fromkeys(r["activity"] for r in cnn_rows):
        ra = [r for r in cnn_rows if r["activity"] == act]
        rb = [r for r in b2_rows if r["activity"] == act]
        lines.append(f"| {act} | {'no-step' if ra[0]['kind'] == 'hold' else ra[0]['kind']} | {len(ra)} | "
                     f"{np.mean([abs(r['count_err']) for r in ra]):.2f} | {np.mean([abs(r['count_err']) for r in rb]):.2f} | "
                     f"{sum(r['fp'] for r in ra)} | {sum(r['fp'] for r in rb)} |")
    lines += ["", "## Training loss per fold", "", "| held out | first epoch | last epoch |", "|---|---|---|"]
    lines += [f"| {s} | {c[0]:.3f} | {c[-1]:.3f} |" for s, c in curves.items()]
    lines += ["", "## Config", "", "```", (run.dir / "config.yaml").read_text().strip(), "```", ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "m5_cnn_loso.md").write_text("\n".join(lines))
    (run.dir / "rows.json").write_text(json.dumps({"cnn": cnn_rows, "b2": b2_rows}))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "cnn.yaml"))
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("overrides", nargs="*", help="config overrides like train.epochs=40")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    cfg = load_config(args.config, args.overrides)
    sessions = real_sessions()  # never the locked test sessions
    data = {s: load_session(s) for s in sessions}
    dataset_hash, _ = dataset_manifest(sessions)
    print(f"{len(sessions)} training sessions, {sum(len(b.strides) for d in data.values() for b in d.blocks)} strides, "
          f"dataset {dataset_hash}")

    if args.loso:
        run = Run(cfg, dataset_hash, tag="loso")
        cnn_rows, b2_rows, curves = run_loso(cfg, data, run)
        lines = report(cfg, cnn_rows, b2_rows, curves, run, dataset_hash)
        run.finish({"cnn": summarize(cnn_rows), "b2": summarize(b2_rows)})
        print("\n".join(lines[4:4 + 3 * (len(SUMMARY_KEYS) + 5)]))
        print(f"\nWrote {REPORTS / 'm5_cnn_loso.md'}  (run {run.dir.name})")
    else:
        run = Run(cfg, dataset_hash)
        model, stats = train_model(list(data.values()), cfg, log=run.log)
        run.save_model(model, stats, config=cfg.name)
        run.finish({"sessions": sessions})
        print(f"Saved {run.dir / 'model.pt'}")


if __name__ == "__main__":
    main()
