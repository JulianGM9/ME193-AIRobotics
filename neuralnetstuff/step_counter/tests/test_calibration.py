"""Six-pose accelerometer calibration and the orientation check."""

import numpy as np
import pytest

from stepnet import calibration as cal
from stepnet.preprocess import to_physical

TRUE_OFFSET = np.array([0.01, -0.03, 0.18])  # like this MPU-6050's z axis
TRUE_SCALE = np.array([1.00, 0.98, 1.02])


def six_poses(order=range(6)):
    """What a sensor with TRUE_OFFSET/TRUE_SCALE reads in the six +/-1 g poses."""
    ideal = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return (ideal * TRUE_SCALE + TRUE_OFFSET)[list(order)]


def test_fit_recovers_offset_and_scale():
    fit = cal.fit_accel_calibration(six_poses())
    assert np.allclose(fit["offset_g"], TRUE_OFFSET)
    assert np.allclose(fit["scale"], TRUE_SCALE)


def test_pose_order_does_not_matter():
    fit = cal.fit_accel_calibration(six_poses(order=[4, 0, 3, 5, 1, 2]))
    assert np.allclose(fit["offset_g"], TRUE_OFFSET)


def test_corrected_poses_read_exactly_one_g():
    poses = six_poses()
    corrected = cal.apply_accel_calibration(poses, cal.fit_accel_calibration(poses))
    assert np.allclose(np.linalg.norm(corrected, axis=1), 1.0)


def test_rejects_two_poses_on_the_same_side():
    poses = six_poses()
    poses[1] = poses[0]  # "arrow up" twice, no "arrow down"
    with pytest.raises(ValueError, match="both point"):
        cal.fit_accel_calibration(poses)


def test_rejects_a_tilted_pose():
    poses = six_poses()
    poses[2] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="not aligned"):
        cal.fit_accel_calibration(poses)


def test_to_physical_applies_calibration():
    raw = np.array([[0, 0, 0, 0, int(round(TRUE_OFFSET[2] * 2048)), 0, 0, 0]])
    fit = cal.fit_accel_calibration(six_poses())
    acc, _ = to_physical(raw, fit)
    # x and y read 0 raw -> (0 - offset) / scale; z reads exactly its offset -> ~0
    expected = [-TRUE_OFFSET[0] / TRUE_SCALE[0], -TRUE_OFFSET[1] / TRUE_SCALE[1], 0.0]
    assert np.allclose(acc[0], expected, atol=1e-3)


def test_orientation_error():
    assert cal.orientation_error_deg([1, 0, 0]) == pytest.approx(0.0)
    assert cal.orientation_error_deg([1, 1, 0]) == pytest.approx(45.0)
    assert cal.orientation_error_deg([-1, 0, 0]) == pytest.approx(180.0)
    assert cal.angle_between_deg([1, 0, 0], [0, 0, 2]) == pytest.approx(90.0)
