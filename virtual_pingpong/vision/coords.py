import math


def shoulder_width(pose):
    sx, sy = pose["shoulder"]
    ox, oy = pose["other_shoulder"]
    return math.hypot(sx - ox, sy - oy)


def to_game_space(pose, point_key="wrist"):
    """Map a pose landmark to game-space coordinates centered on the
    dominant shoulder, scaled by shoulder width - so the game behaves
    consistently regardless of how far the player stands from the camera.

    Returns (x, y) in roughly shoulder-width units, with x increasing to
    the player's right and y increasing upward (flipped from MediaPipe's
    y-down image coordinates). Returns None if the shoulders are too close
    together in-frame to give a meaningful scale (e.g. pose not detected
    cleanly, or player side-on to the camera).
    """
    width = shoulder_width(pose)
    if width < 1e-6:
        return None

    sx, sy = pose["shoulder"]
    px, py = pose[point_key]
    x = (px - sx) / width
    y = -(py - sy) / width
    return x, y
