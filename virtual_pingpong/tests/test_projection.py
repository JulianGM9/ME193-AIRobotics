import math

from render.projection import make_camera, project_point

PHYSICS_CFG = {"table_length": 2.74, "table_width": 1.525}
RENDER_CFG = {
    "camera_back_offset": 1.8,
    "camera_height": 1.0,
    "focal_length": 700,
    "near_clip": 0.05,
    "screen_width": 960,
    "screen_height": 600,
}
CAMERA = make_camera(PHYSICS_CFG, RENDER_CFG)


def test_point_at_camera_height_and_centered_x_projects_to_screen_center():
    screen_x, screen_y, scale = project_point(0, RENDER_CFG["camera_height"], 0, CAMERA)
    assert math.isclose(screen_x, RENDER_CFG["screen_width"] / 2)
    assert math.isclose(screen_y, RENDER_CFG["screen_height"] / 2)


def test_positive_x_projects_right_of_center():
    screen_x, _, _ = project_point(0.5, RENDER_CFG["camera_height"], 0, CAMERA)
    assert screen_x > RENDER_CFG["screen_width"] / 2


def test_negative_x_projects_left_of_center():
    screen_x, _, _ = project_point(-0.5, RENDER_CFG["camera_height"], 0, CAMERA)
    assert screen_x < RENDER_CFG["screen_width"] / 2


def test_point_above_camera_height_projects_above_screen_center():
    _, screen_y, _ = project_point(0, RENDER_CFG["camera_height"] + 0.5, 0, CAMERA)
    assert screen_y < RENDER_CFG["screen_height"] / 2  # smaller y = higher on screen


def test_point_below_camera_height_projects_below_screen_center():
    _, screen_y, _ = project_point(0, RENDER_CFG["camera_height"] - 0.5, 0, CAMERA)
    assert screen_y > RENDER_CFG["screen_height"] / 2


def test_farther_points_have_smaller_scale():
    _, _, near_scale = project_point(0.5, 1.0, 0.0, CAMERA)
    _, _, far_scale = project_point(0.5, 1.0, -1.0, CAMERA)  # further from camera (more negative z)
    assert far_scale < near_scale


def test_farther_point_with_same_world_x_appears_closer_to_center():
    # Perspective: the same lateral offset should shrink toward the
    # vanishing point as depth increases.
    near_x, _, _ = project_point(0.5, 1.0, 0.0, CAMERA)
    far_x, _, _ = project_point(0.5, 1.0, -1.0, CAMERA)
    center = RENDER_CFG["screen_width"] / 2
    assert abs(far_x - center) < abs(near_x - center)


def test_point_behind_the_camera_returns_none():
    behind_z = CAMERA["z"] + 1.0  # beyond the camera's own position
    assert project_point(0, 1.0, behind_z, CAMERA) is None


def test_point_within_near_clip_returns_none():
    just_in_front_z = CAMERA["z"] - CAMERA["near_clip"] / 2  # depth < near_clip
    assert project_point(0, 1.0, just_in_front_z, CAMERA) is None
