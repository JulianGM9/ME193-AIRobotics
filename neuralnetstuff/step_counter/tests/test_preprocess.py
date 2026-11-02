"""Unit conversion and the sensor -> body frame mapping."""

import numpy as np

from stepnet.preprocess import to_body, to_physical


def test_standing_reads_plus_one_g_up_the_shin():
    # A real standing sample from the gait project (same mount): accel x is about -1 g.
    raw = np.array([[350, 3947, -2069, -106, 305, -245, -204, 364]])
    acc, gyr = to_physical(raw)
    s, f, l = to_body(acc)[0]
    assert abs(s - 1.0) < 0.05
    assert abs(f) < 0.1 and abs(l) < 0.2
    assert np.allclose(gyr[0], np.array([-245, -204, 364]) / 16.4)


def test_axis_map_signs_and_order():
    xyz = np.array([[1.0, 2.0, 3.0]])
    assert to_body(xyz).tolist() == [[-1.0, -2.0, 3.0]]  # S=-x, F=-y, L=+z
    assert to_body(xyz, {"S": "+z", "F": "-x", "L": "+y"}).tolist() == [[3.0, -1.0, 2.0]]
