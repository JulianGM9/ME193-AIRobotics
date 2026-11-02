"""Preprocessing (PLAN.md D5). For now: raw counts -> physical units -> body frame.

Body frame (D3), fixed by how the board is strapped on (outside of the RIGHT ankle):
  S = up the shin, toward the knee
  F = forward (walking direction)
  L = toward your left, i.e. the INNER side of the right ankle
S, F, L is right-handed (L = S x F), like the sensor's x, y, z. Only two axes can be
chosen freely; flipping L alone would make the frame left-handed and break rotation and
gyro maths later. Verified by the M1 tilt test on 2026-11-02.
At rest an accelerometer reads +1 g pointing *up*, so standing still gives S = +1 g.
"""
import numpy as np

from stepnet.calibration import apply_accel_calibration
from stepnet.io.protocol import ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS

AXIS_MAP = {"S": "-x", "F": "-y", "L": "+z"}  # measured in the gait project, same mount
BODY_AXES = ("S", "F", "L")


def to_physical(raw: np.ndarray, accel_cal: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw (n, 8) samples (seq, t_ms, ax..gz) -> (accel in g, gyro in deg/s), each (n, 3) in sensor xyz.

    accel_cal (from stepnet.calibration) corrects each accel axis's offset and scale.
    """
    raw = np.asarray(raw, dtype=float)
    acc = apply_accel_calibration(raw[:, 2:5] / ACCEL_LSB_PER_G, accel_cal)
    return acc, raw[:, 5:8] / GYRO_LSB_PER_DPS


def to_body(xyz: np.ndarray, axis_map: dict = AXIS_MAP) -> np.ndarray:
    """Sensor-frame (n, 3) vectors -> body frame (n, 3) in S, F, L order."""
    xyz = np.asarray(xyz, dtype=float)
    cols = []
    for axis in BODY_AXES:
        spec = axis_map[axis]  # e.g. "-x"
        sign = -1.0 if spec[0] == "-" else 1.0
        cols.append(sign * xyz[:, "xyz".index(spec[1])])
    return np.stack(cols, axis=1)
