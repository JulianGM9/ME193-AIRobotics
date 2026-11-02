"""Tune and evaluate the classical baselines with leave-one-session-out (milestone M4, PLAN.md D11, D14).

  ../../.venv/bin/python scripts/evaluate_baselines.py      # -> reports/m4_baselines.md + figures

For each session: grid-search B1/B2 on the OTHER sessions (minimizing total |count error|),
then run the winner on the held-out one. The held-out results of all folds are pooled and
summarized with bootstrap 95% confidence intervals over blocks.

"oracle" uses the labeled strides as detections: its count error is only the doubling error
(steps = 2 x strides is off by one whenever a bout starts and ends on the same foot). No
stride detector can beat it, so it is the floor for every model, the network included.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.baselines import B1_GRID, B2_GRID, Detector, grid
from stepnet.data.blocks import load_session, real_sessions
from stepnet.eval.metrics import block_rows, bootstrap, summarize, tuning_loss

REPORTS = Path(__file__).resolve().parents[1] / "reports"
METRICS = [("f1", "stride F1", "{:.3f}"), ("recall", "recall", "{:.3f}"), ("precision", "precision", "{:.3f}"),
           ("bout_abs_err", "mean abs count error per bout (steps)", "{:.2f}"),
           ("net_err_pct", "net count error, all bouts (%)", "{:+.1f}"),
           ("false_per_min", "false strides / min in no-step blocks", "{:.1f}")]


def loso(name, spec, data, detectors):
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


def fmt_ci(summary, ci, key, f):
    return f"{f.format(summary[key])} [{f.format(ci[key][0])}, {f.format(ci[key][1])}]"


def per_activity(rows):
    out = {}
    for act in dict.fromkeys(r["activity"] for r in rows):
        rs = [r for r in rows if r["activity"] == act]
        kind = rs[0]["kind"]
        minutes = sum(r["seconds"] for r in rs) / 60
        out[act] = {"kind": kind, "blocks": len(rs),
                    "abs_err": float(np.mean([abs(r["count_err"]) for r in rs])),
                    "fp_per_min": sum(r["fp"] for r in rs) / minutes,
                    "recall": sum(r["tp"] for r in rs) / max(1, sum(r["tp"] + r["fn"] for r in rs))}
    return out


def plot(results, data, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"B1": "#2a78d6", "B2": "#eb6834"}  # categorical slots 1-2
    text, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"

    # 1. Per activity: bouts -> mean |count error| per bout; no-step blocks -> false strides per minute
    acts = {name: per_activity(rows) for name, rows in results.items() if name != "oracle"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, kind, key, ylabel in ((ax1, "bout", "abs_err", "mean |count error| per bout (steps)"),
                                  (ax2, "hold", "fp_per_min", "false strides per minute")):
        names = [a for a, v in acts["B1"].items() if v["kind"] == kind]
        x = np.arange(len(names))
        for j, model in enumerate(("B1", "B2")):
            vals = [acts[model][a][key] for a in names]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=colors[model], label=model, edgecolor="white", lw=1)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, color=muted, padding=1)
        if kind == "bout":
            oracle = per_activity(results["oracle"])
            ax.scatter(x, [oracle[a]["abs_err"] for a in names], marker="_", s=300, color=text, zorder=3,
                       label="oracle (doubling error only)")
        ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=25, ha="right", fontsize=8, color=text)
        ax.set_ylabel(ylabel, color=muted, fontsize=9)
        ax.grid(axis="y", color=grid_c, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=muted)
    ax1.set_title("Walking-type bouts (lower is better)", loc="left", fontsize=10, color=text)
    ax2.set_title("No-step blocks (lower is better)", loc="left", fontsize=10, color=text)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_per_activity.png", dpi=130)
    plt.close(fig)

    # 2. What B2's heel-strike gate does: a seated leg swing next to a walking bout
    sess = next(d for d in data.values() if any(b.activity == "leg_swing_seated" for b in d.blocks))
    det = Detector(sess)
    # the parameters this session was actually evaluated with: tuned on the other sessions
    b1_params, b2_params = params["B1"]["per_fold"][sess.session], params["B2"]["per_fold"][sess.session]
    swing = next(b for b in sess.blocks if b.activity == "leg_swing_seated")
    walk = next(b for b in sess.blocks if b.activity.startswith("walk"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharey="row", gridspec_kw={"height_ratios": [1.4, 1]})
    for col, b in enumerate((walk, swing)):
        sel = (sess.seq >= b.first) & (sess.seq < b.last)
        t = (sess.seq[sel] - b.first) / 100
        axes[0, col].plot(t, sess.gyr[sel, 2], color="#1baf7a", lw=0.9)
        axes[1, col].plot(t, det.acc_mag[sel], color="#6250d6", lw=0.9)
        axes[1, col].axhline(b2_params["impact_g"], color=muted, lw=0.8, ls="--")
        for model, p, y, marker in (("B1", b1_params, 0.95, "v"), ("B2", b2_params, 0.86, "v")):
            hits = det.detect(p)
            hits = hits[(hits >= b.first) & (hits < b.last)]
            axes[0, col].scatter((hits - b.first) / 100, np.full(len(hits), y), transform=axes[0, col].get_xaxis_transform(),
                                 marker=marker, color=colors[model], s=28, label=f"{model}: {len(hits)} strides")
        axes[0, col].set_title(f"{b.session} #{b.i} {b.label}: {b.steps} steps counted ({len(b.strides)} labeled strides), "
                               "held-out parameters",
                               loc="left", fontsize=10, color=text)
        axes[0, col].legend(frameon=False, fontsize=8, loc="lower right")
        axes[1, col].set_xlabel("time in block (s)", color=muted, fontsize=9)
    axes[0, 0].set_ylabel("L gyro (deg/s)", color=muted, fontsize=9)
    axes[1, 0].set_ylabel("|accel| (g)", color=muted, fontsize=9)
    axes[1, 1].text(0.99, b2_params["impact_g"], f" B2 impact threshold {b2_params['impact_g']} g", color=muted, fontsize=8,
                    ha="right", va="bottom", transform=axes[1, 1].get_yaxis_transform())
    for ax in axes.flat:
        ax.grid(color=grid_c, lw=0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_heel_strike_gate.png", dpi=130)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    sessions = real_sessions()
    data = {s: load_session(s) for s in sessions}
    detectors = {s: Detector(d) for s, d in data.items()}
    blocks = [b for d in data.values() for b in d.blocks]
    print(f"{len(sessions)} sessions, {len(blocks)} usable blocks, "
          f"{sum(len(b.strides) for b in blocks)} labeled strides")

    results, params = {}, {}
    for name, spec in (("B1", B1_GRID), ("B2", B2_GRID)):
        rows, folds, all_data = loso(name, spec, data, detectors)
        results[name], params[name] = rows, {"per_fold": folds, "all_data": all_data}
    results["oracle"] = block_rows(blocks, {s: np.concatenate([b.strides for b in d.blocks] or [np.array([], int)])
                                            for s, d in data.items()})

    summaries = {k: summarize(v) for k, v in results.items()}
    cis = {k: bootstrap(v) for k, v in results.items()}

    lines = ["# M4: Classical baselines (leave-one-session-out)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/evaluate_baselines.py`. "
             f"{len(sessions)} sessions ({', '.join(sessions)}), {len(blocks)} usable blocks, "
             f"{sum(len(b.strides) for b in blocks)} labeled strides. Each session's numbers come from parameters "
             "tuned on the other four. Values: point estimate [95% bootstrap CI over blocks].", "",
             "| metric | B1 peak detection | B2 + heel-strike gate | oracle (labels) |", "|---|---|---|---|"]
    for key, label, f in METRICS:
        lines.append(f"| {label} | " + " | ".join(fmt_ci(summaries[m], cis[m], key, f) for m in ("B1", "B2", "oracle")) + " |")
    lines += ["", "## Per activity (held-out)", "",
              "| activity | kind | blocks | B1 abs err / bout | B2 abs err / bout | B1 false/min | B2 false/min | B1 recall | B2 recall |",
              "|---|---|---|---|---|---|---|---|---|"]
    a1, a2 = per_activity(results["B1"]), per_activity(results["B2"])
    for act, v in a1.items():
        w = a2[act]
        lines.append(f"| {act} | {'bout' if v['kind'] == 'bout' else 'no-step'} | {v['blocks']} | {v['abs_err']:.2f} | {w['abs_err']:.2f} | "
                     f"{v['fp_per_min']:.1f} | {w['fp_per_min']:.1f} | {v['recall']:.2f} | {w['recall']:.2f} |")
    lines += ["", "## Per session: net count error over all bouts (held-out)", "",
              "| session | counted steps | B1 | B2 | oracle |", "|---|---|---|---|---|"]
    for s in sessions:
        steps = sum(r["steps"] for r in results["B1"] if r["session"] == s and r["kind"] == "bout")
        errs = [sum(r["count_err"] for r in results[m] if r["session"] == s) for m in ("B1", "B2", "oracle")]
        lines.append(f"| {s} | {steps} | " + " | ".join(f"{e:+d} ({100 * e / steps:+.1f}%)" for e in errs) + " |")
    lines += ["", "## Parameters", "", "Tuned on all five sessions (for later live use):", "",
              f"- B1: `{json.dumps(params['B1']['all_data'])}`", f"- B2: `{json.dumps(params['B2']['all_data'])}`", "",
              "Per fold (tuned without that session):", ""]
    for s in sessions:
        lines.append(f"- {s}: B1 `{json.dumps(params['B1']['per_fold'][s])}`, B2 `{json.dumps(params['B2']['per_fold'][s])}`")
    lines += ["", "![per activity](m4_per_activity.png)", "", "![heel-strike gate](m4_heel_strike_gate.png)", ""]
    (REPORTS / "m4_baselines.md").write_text("\n".join(lines))
    (REPORTS / "m4_baselines.json").write_text(json.dumps({"summary": summaries, "ci": cis, "params": params}, indent=2))
    plot(results, data, params)
    print("\n".join(lines[:4 + 2 + len(METRICS)]))
    print(f"\nWrote {REPORTS / 'm4_baselines.md'}")


if __name__ == "__main__":
    main()
