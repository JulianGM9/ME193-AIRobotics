import math

from vision.coords import shoulder_width, to_game_space


def make_pose(shoulder, other_shoulder, wrist):
    return {"shoulder": shoulder, "other_shoulder": other_shoulder, "wrist": wrist}


def test_shoulder_width_is_euclidean_distance():
    pose = make_pose(shoulder=(0.6, 0.4), other_shoulder=(0.4, 0.4), wrist=(0.6, 0.6))
    assert math.isclose(shoulder_width(pose), 0.2)


def test_wrist_at_shoulder_maps_to_origin():
    pose = make_pose(shoulder=(0.5, 0.5), other_shoulder=(0.3, 0.5), wrist=(0.5, 0.5))
    x, y = to_game_space(pose)
    assert x == 0.0
    assert y == 0.0


def test_wrist_right_and_above_shoulder_is_positive_x_positive_y():
    # Image coords are y-down; a wrist above the shoulder has a smaller y.
    pose = make_pose(shoulder=(0.5, 0.5), other_shoulder=(0.3, 0.5), wrist=(0.6, 0.4))
    x, y = to_game_space(pose)
    assert x > 0
    assert y > 0


def test_scales_by_shoulder_width_so_distance_from_camera_does_not_matter():
    close_pose = make_pose(shoulder=(0.5, 0.5), other_shoulder=(0.3, 0.5), wrist=(0.6, 0.5))
    far_pose = make_pose(shoulder=(0.5, 0.5), other_shoulder=(0.4, 0.5), wrist=(0.55, 0.5))
    # In both cases the wrist is offset by exactly half a shoulder-width.
    close_x, close_y = to_game_space(close_pose)
    far_x, far_y = to_game_space(far_pose)
    assert math.isclose(close_x, far_x)
    assert math.isclose(close_y, far_y)


def test_degenerate_shoulder_width_returns_none():
    pose = make_pose(shoulder=(0.5, 0.5), other_shoulder=(0.5, 0.5), wrist=(0.6, 0.5))
    assert to_game_space(pose) is None
