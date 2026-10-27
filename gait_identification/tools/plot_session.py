"""Plot one recorded session: S/F/L acceleration and gyro over time, with the
labeled segments shaded, so you can eyeball whether each activity looks right.

Run from the gait_identification folder:
    python3 tools/plot_session.py data/session_01.json
Saves a PNG next to the JSON and opens a window.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"calibrate": "#cccccc", "standing": "#4c72b0", "walking": "#55a868", "running": "#c44e52",
          "biking": "#8172b2", "lying_still": "#ccb974", "transition": "#ffffff"}


def load(json_path):
    meta = json.loads(Path(json_path).read_text())
    raw = np.loadtxt(Path(json_path).with_suffix(".csv"), delimiter=",", skiprows=1)
    t = (raw[:, 1] - raw[0, 1]) / 1000.0
    acc = raw[:, 2:5] / meta["units"]["accel_lsb_per_g"]
    bias = meta["gyro_bias_dps"] or {"x": 0, "y": 0, "z": 0}
    gyr = raw[:, 5:8] / meta["units"]["gyro_lsb_per_dps"] - np.array([bias["x"], bias["y"], bias["z"]])
    return meta, raw[0, 1], t, acc, gyr


def to_body(data, axis_map):
    """Reorder/flip sensor x/y/z columns into S, F, L using e.g. {"S": "-x"}."""
    cols = {"x": 0, "y": 1, "z": 2}
    return np.column_stack([(-1 if axis_map[k][0] == "-" else 1) * data[:, cols[axis_map[k][1]]] for k in "SFL"])


def main():
    meta, t0_ms, t, acc, gyr = load(sys.argv[1])
    acc, gyr = to_body(acc, meta["axis_map"]), to_body(gyr, meta["axis_map"])
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7))
    for ax, data, unit in ((axes[0], acc, "accel (g)"), (axes[1], gyr, "gyro (deg/s)")):
        for label, start, end, ok in meta["segments"]:
            ax.axvspan((start - t0_ms) / 1000, (end - t0_ms) / 1000, color=COLORS.get(label, "#eeeeee"),
                       alpha=0.25 if ok else 0.08, lw=0)
        for i, name in enumerate("SFL"):
            ax.plot(t, data[:, i], lw=0.6, label=name)
        ax.set_ylabel(unit)
        ax.legend(loc="upper right")
    for label, start, end, ok in meta["segments"]:
        if label != "transition":
            axes[0].text(((start + end) / 2 - t0_ms) / 1000, axes[0].get_ylim()[1] * 0.9,
                         label + ("" if ok else " (bad)"), ha="center", fontsize=9)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Session {meta['session']:02d} ({meta['protocol']}) - {meta['samples']} samples, {meta['gaps']} gaps")
    fig.tight_layout()
    out = Path(sys.argv[1]).with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
