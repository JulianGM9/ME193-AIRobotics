"""One-time six-pose accelerometer calibration (PLAN.md D5, M2).

Run from neuralnetstuff/step_counter/ with the board app running:
  ../../.venv/bin/python scripts/calibrate_accel.py

Hold the board (strapped on or in your hand) so that each side points straight up in turn,
resting it against a table or a book so it stays still. Each pose takes 3 s. The result
(per-axis offset and scale) is saved to data/calibration/accel.json, and every session
recorded afterwards keeps a copy in its meta.json. Re-run it if you ever swap the sensor.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet import calibration as cal
from stepnet.io.live import LiveStream
from stepnet.motion import is_still
from stepnet.preprocess import to_physical

POSE_SECONDS = 3.0
POSES = [
    "ARROW UP      - the end that points toward your knee faces the ceiling (like standing)",
    "ARROW DOWN    - the knee end faces the floor (board upside down)",
    "FORWARD UP    - the side that faces forward when you walk faces the ceiling",
    "FORWARD DOWN  - the forward side faces the floor",
    "OUTER UP      - the outer side (away from your other leg) faces the ceiling",
    "OUTER DOWN    - the outer side faces the floor",
]


def main():
    path = cal.ACCEL_CAL_PATH
    if path.exists() and input(f"{path} exists. Overwrite it? [y/N] ").strip().lower() != "y":
        return
    stream = LiveStream().start()
    print("Waiting for IMU data...")
    if not stream.wait_for_data():
        print("No IMU data. Is the board app running?")
        stream.stop()
        return

    means = []
    try:
        for n, pose in enumerate(POSES, start=1):
            while True:
                input(f"\nPose {n}/6: {pose}\n  Hold it still and press Enter...")
                raw = stream.collect(POSE_SECONDS)
                acc, gyr = to_physical(raw)  # uncalibrated on purpose
                if not is_still(raw, gyr.mean(axis=0)):
                    print("  ! the board moved - try again")
                    continue
                mean = acc.mean(axis=0)
                print(f"  x {mean[0]:+.3f}  y {mean[1]:+.3f}  z {mean[2]:+.3f} g")
                means.append(mean)
                break
    finally:
        stream.stop()

    try:
        fit = cal.fit_accel_calibration(means)
    except ValueError as exc:
        print(f"\n! {exc}\nNothing saved - run it again, holding each pose as described.")
        return

    corrected = cal.apply_accel_calibration(np.array(means), fit)
    norms = np.linalg.norm(corrected, axis=1)
    print("\n        offset (g)   scale")
    for a, off, sc in zip("xyz", fit["offset_g"], fit["scale"]):
        print(f"  {a}    {off:+.3f}      {sc:.3f}")
    print(f"|g| per pose before: {np.round(np.linalg.norm(means, axis=1), 3).tolist()}")
    print(f"|g| per pose after:  {np.round(norms, 3).tolist()}   (should all be ~1.000)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **fit,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_means_g": np.round(means, 4).tolist(),
        "norm_after_g": np.round(norms, 4).tolist(),
    }, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
