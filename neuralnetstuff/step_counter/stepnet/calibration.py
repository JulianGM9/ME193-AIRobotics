"""Sensor calibration (PLAN.md D3, D5).

Accelerometer (one-time, scripts/calibrate_accel.py): six still poses, each with one sensor
axis pointing straight up (+1 g) or straight down (-1 g). For every axis:
    offset = (up + down) / 2        scale = (up - down) / 2
so the corrected reading is  (raw_g - offset) / scale.

Gyro bias and orientation (every session, from the standing calibration): the gyro's mean
while still is its bias, and the mean acceleration is the gravity direction, which should
point up the shin (+S).
"""
import json
from pathlib import Path

import numpy as np

ACCEL_CAL_PATH = Path(__file__).resolve().parents[1] / "data" / "calibration" / "accel.json"
MIN_POSE_G = 0.7  # the dominant axis of a pose must read at least this much


def fit_accel_calibration(pose_means) -> dict:
    """Six (3,) mean accelerations in g (sensor xyz) -> {"offset_g": [3], "scale": [3]}.

    Each pose is assigned to the axis that dominates it, so the poses may come in any order.
    Raises ValueError unless every axis has exactly one 'up' and one 'down' pose.
    """
    poses = np.asarray(pose_means, dtype=float)
    if poses.shape != (6, 3):
        raise ValueError(f"need 6 poses of 3 values, got shape {poses.shape}")
    up, down = {}, {}
    for i, pose in enumerate(poses):
        axis = int(np.argmax(np.abs(pose)))
        if abs(pose[axis]) < MIN_POSE_G:
            raise ValueError(f"pose {i + 1} has no axis near +/-1 g ({pose.round(2).tolist()}) - not aligned")
        side = up if pose[axis] > 0 else down
        if axis in side:
            raise ValueError(f"poses {side[axis] + 1} and {i + 1} both point axis {'xyz'[axis]} "
                             f"{'up' if side is up else 'down'}")
        side[axis] = i
    up_g = np.array([poses[up[a], a] for a in range(3)])
    down_g = np.array([poses[down[a], a] for a in range(3)])
    return {"offset_g": ((up_g + down_g) / 2).tolist(), "scale": ((up_g - down_g) / 2).tolist()}


def apply_accel_calibration(acc_g, cal):
    """(n, 3) accel in g (sensor xyz) -> corrected. cal=None leaves it unchanged."""
    if cal is None:
        return np.asarray(acc_g, dtype=float)
    return (np.asarray(acc_g, dtype=float) - np.asarray(cal["offset_g"])) / np.asarray(cal["scale"])


def load_accel_calibration(path=ACCEL_CAL_PATH):
    """The saved calibration dict, or None if the sensor hasn't been calibrated yet."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def orientation_error_deg(gravity_body) -> float:
    """Angle between the measured gravity direction (body frame) and straight up the shin (+S)."""
    g = np.asarray(gravity_body, dtype=float)
    return float(np.degrees(np.arccos(np.clip(g[0] / np.linalg.norm(g), -1.0, 1.0))))


def angle_between_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
