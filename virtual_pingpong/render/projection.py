"""Pure 3D-to-2D projection math, with no pygame dependency, so it can be
unit-tested on its own. The camera sits behind and above the player's end
of the table, looking straight down the table's length (the -z direction,
toward the opponent) with no tilt - a simple pinhole-camera model.
"""


def make_camera(physics_cfg, render_cfg):
    """Build the camera parameters dict used by project_point()."""
    camera_z = physics_cfg["table_length"] / 2 + render_cfg["camera_back_offset"]
    return {
        "z": camera_z,
        "height": render_cfg["camera_height"],
        "focal_length": render_cfg["focal_length"],
        "near_clip": render_cfg["near_clip"],
        "screen_center_x": render_cfg["screen_width"] / 2,
        "screen_center_y": render_cfg["screen_height"] / 2,
    }


def project_point(x, y, z, camera):
    """Project a 3D table-space point (meters) to 2D screen pixels.

    Returns (screen_x, screen_y, scale), or None if the point is at or
    behind the near clip plane (too close to / behind the camera to draw).
    scale = focal_length / depth, shrinking for farther points - multiply
    a base pixel size by it to make objects correctly smaller with
    distance (e.g. ball radius, shadow opacity).
    """
    depth = camera["z"] - z
    if depth <= camera["near_clip"]:
        return None
    scale = camera["focal_length"] / depth
    screen_x = camera["screen_center_x"] + x * scale
    screen_y = camera["screen_center_y"] - (y - camera["height"]) * scale
    return screen_x, screen_y, scale
