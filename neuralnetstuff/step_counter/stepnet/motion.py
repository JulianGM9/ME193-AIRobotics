"""Coarse moving/still detection from the gyro, used by the recorder to end bouts.

The ankle's gyro is the clearest motion signal: walking swings the shin at several hundred
deg/s, while standing still reads only the sensor's bias plus ~1 deg/s of noise.
"""
import numpy as np

from stepnet.io.protocol import GYRO_LSB_PER_DPS

MOVING_DPS = 50.0  # any sample above this -> the leg is moving
STILL_DPS = 15.0  # every sample below this -> the leg is still


def gyro_magnitude_dps(raw, gyro_bias_dps) -> np.ndarray:
    """Bias-corrected angular speed |omega| in deg/s for raw (n, 8) samples."""
    gyro = np.asarray(raw, dtype=float)[:, 5:8] / GYRO_LSB_PER_DPS - np.asarray(gyro_bias_dps)
    return np.linalg.norm(gyro, axis=1)


def is_moving(raw, gyro_bias_dps, threshold_dps=MOVING_DPS) -> bool:
    return len(raw) > 0 and bool(gyro_magnitude_dps(raw, gyro_bias_dps).max() > threshold_dps)


def is_still(raw, gyro_bias_dps, threshold_dps=STILL_DPS) -> bool:
    return len(raw) > 0 and bool(gyro_magnitude_dps(raw, gyro_bias_dps).max() < threshold_dps)
