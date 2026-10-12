"""Ball physics for the table.

Coordinate system: x = across the table width, y = height above the table
surface (0 = table, positive = up), z = along the table length (positive =
player's end, negative = opponent's end, 0 = net). All units are meters;
physics.py knows nothing about the camera/game-space units vision/coords.py
produces - that mapping happens at the integration layer.

Gravity only acts on y, so vx and vz are constant between bounces. That
makes predict_landing_x() exact (not an approximation) for this simplified
point-mass model (no spin, no drag).
"""
from dataclasses import dataclass


@dataclass
class Ball:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    has_bounced: bool = False  # True once it's bounced on either side during this flight


@dataclass
class StepResult:
    bounced_side: str = None  # "player", "opponent", or None
    net_hit: bool = False
    out_of_bounds: bool = False


def step(ball, dt, cfg):
    """Advance the ball by dt seconds. Returns (new_ball, StepResult).
    Semi-implicit Euler: velocity is updated first, then position uses the
    updated velocity - standard for simple game physics.
    """
    vx = ball.vx
    vy = ball.vy + cfg["gravity"] * dt
    vz = ball.vz

    x = ball.x + vx * dt
    y = ball.y + vy * dt
    z = ball.z + vz * dt

    result = StepResult()
    has_bounced = ball.has_bounced

    # Only a fault if this is still the ball's first, un-bounced flight -
    # otherwise this would also catch perfectly normal secondary bounces
    # (height naturally decays each bounce under restitution < 1, and by
    # the time the ball's z position nears the net it may well already be
    # bouncing low) and wrongly call them a net hit.
    crossed_net = (ball.z > 0 > z) or (ball.z < 0 < z)
    if crossed_net and not has_bounced and min(ball.y, y) < cfg["net_height"]:
        result.net_hit = True
        vz = 0.0  # caught by the net

    if y <= 0 and vy < 0:
        y = 0.0
        vy = -vy * cfg["restitution"]
        result.bounced_side = "player" if z > 0 else "opponent"
        has_bounced = True

    half_width = cfg["table_width"] / 2
    half_length = cfg["table_length"] / 2
    out_of_bounds_margin = 0.5  # meters past the table edge before calling it fully gone
    if abs(x) > half_width or abs(z) > half_length + out_of_bounds_margin:
        result.out_of_bounds = True

    return Ball(x, y, z, vx, vy, vz, has_bounced), result


def predict_landing_x(ball, target_z):
    """Predict the ball's x position when it reaches z == target_z. Exact
    under this model, since vx and vz are both constant between bounces.
    Returns None if the ball isn't moving toward target_z at all.
    """
    if ball.vz == 0:
        return ball.x if ball.z == target_z else None
    dt = (target_z - ball.z) / ball.vz
    if dt < 0:
        return None
    return ball.x + ball.vx * dt
