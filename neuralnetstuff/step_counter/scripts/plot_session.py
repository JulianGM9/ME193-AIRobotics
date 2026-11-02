"""Plot a recorded session: body-frame accel and gyro with the labeled blocks shaded.

  ../../.venv/bin/python scripts/plot_session.py S001          # opens a window, saves plot.png
  ../../.venv/bin/python scripts/plot_session.py T001 --no-show
"""
import argparse
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.io.protocol import SAMPLE_RATE_HZ
from stepnet.io.session import RAW_DIR, missing_seqs, read_session
from stepnet.preprocess import BODY_AXES, to_body, to_physical

AXIS_COLORS = {"S": "#2a78d6", "F": "#eb6834", "L": "#1baf7a"}  # categorical slots 1-3 (dataviz palette)
BAND = {"bout": "#d9d8d3", "hold": "#ecebe7", "transition": "#f6f5f3"}  # neutral: identity is in the label
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session_dir = RAW_DIR / args.session
    samples, meta = read_session(session_dir)
    acc, gyr = to_physical(samples, meta.get("accel_calibration"))
    gyr = gyr - meta["gyro_bias_dps"]
    acc_b, gyr_b = to_body(acc), to_body(gyr)
    seq0 = samples[0, 0]
    t = (samples[:, 0] - seq0) / SAMPLE_RATE_HZ

    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for ax, data, unit in ((ax_a, acc_b, "accel (g)"), (ax_g, gyr_b, "gyro (deg/s)")):
        for seg in meta["segments"]:
            x0, x1 = (seg["start_seq"] - seq0) / SAMPLE_RATE_HZ, (seg["end_seq"] - seq0) / SAMPLE_RATE_HZ
            ax.axvspan(x0, x1, color=BAND.get(seg["kind"], BAND["transition"]), lw=0)
            if seg.get("ok") is False:
                ax.axvspan(x0, x1, facecolor="none", edgecolor=MUTED, hatch="//", lw=0)
        for i, axis in enumerate(BODY_AXES):
            ax.plot(t, data[:, i], color=AXIS_COLORS[axis], lw=0.8, label=axis)
        ax.set_ylabel(unit, color=MUTED)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Block labels along the top of the accel panel: number, activity and truth
    for seg in meta["segments"]:
        if seg["kind"] == "transition":
            continue
        x0 = (seg["start_seq"] - seq0) / SAMPLE_RATE_HZ
        truth = f"{seg['steps']} steps" if seg.get("steps") is not None else "? steps"
        ax_a.annotate(f"#{seg['i']} {seg['label']}\n{truth}", (x0, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(2, -2), textcoords="offset points", va="top", fontsize=7, color=TEXT)

    # Legend in the title row, so it never covers the block labels
    fig.legend(*ax_a.get_legend_handles_labels(), loc="upper right", frameon=False, ncol=3, labelcolor=TEXT)
    ax_g.set_xlabel("time (s)", color=MUTED)
    n_missing = len(missing_seqs(samples))
    fig.suptitle(f"{meta['session']}  protocol {meta['protocol']}  {meta['recorded_at']}   "
                 f"{len(samples)} samples, {n_missing} missing   tilt {meta['orientation_error_deg']:.1f} deg",
                 color=TEXT, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = session_dir / "plot.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
