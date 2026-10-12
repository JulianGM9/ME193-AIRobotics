import math

from game.physics import Ball, predict_landing_x, step

CFG = {
    "table_length": 2.74,
    "table_width": 1.525,
    "net_height": 0.1525,
    "gravity": -9.8,
    "restitution": 0.75,
}


def test_ball_falls_under_gravity():
    ball = Ball(x=0, y=1.0, z=0, vx=0, vy=0, vz=0)
    new_ball, result = step(ball, dt=0.1, cfg=CFG)
    assert new_ball.vy < 0  # gravity pulled it downward
    assert new_ball.y < 1.0
    assert result.bounced_side is None


def test_bounce_on_player_side_applies_restitution():
    # Just above the table, falling, on the player's half (z > 0).
    ball = Ball(x=0, y=0.001, z=1.0, vx=0, vy=-2.0, vz=0)
    new_ball, result = step(ball, 0.01, CFG)
    assert result.bounced_side == "player"
    assert new_ball.y == 0.0
    # Restitution should scale the rebound speed down from the impact speed.
    assert new_ball.vy > 0
    assert new_ball.vy < 2.0


def test_bounce_on_opponent_side_is_labeled_correctly():
    ball = Ball(x=0, y=0.001, z=-1.0, vx=0, vy=-2.0, vz=0)
    _, result = step(ball, 0.01, CFG)
    assert result.bounced_side == "opponent"


def test_no_bounce_while_ball_is_above_the_table():
    ball = Ball(x=0, y=1.0, z=0, vx=0, vy=-1.0, vz=0)
    _, result = step(ball, 0.01, CFG)
    assert result.bounced_side is None


def test_low_crossing_of_the_net_is_a_net_hit():
    # Moving from the opponent's side to the player's side, well below net height.
    ball = Ball(x=0, y=0.05, z=-0.05, vx=0, vy=0, vz=2.0)
    _, result = step(ball, 0.1, CFG)
    assert result.net_hit is True


def test_high_crossing_over_the_net_is_not_a_net_hit():
    ball = Ball(x=0, y=0.5, z=-0.05, vx=0, vy=0, vz=2.0)
    _, result = step(ball, 0.1, CFG)
    assert result.net_hit is False


def test_low_crossing_after_already_bouncing_is_not_a_net_hit():
    # Same low crossing as test_low_crossing_of_the_net_is_a_net_hit, but
    # the ball has already legitimately bounced once this flight. This is
    # ordinary bounce-height decay as the ball crosses the table (bounce
    # height shrinks fast under restitution < 1), not a real net fault -
    # regression test for a bug where this froze the ball at the net on
    # nearly every serve.
    ball = Ball(x=0, y=0.05, z=-0.05, vx=0, vy=0, vz=2.0, has_bounced=True)
    new_ball, result = step(ball, 0.1, CFG)
    assert result.net_hit is False
    assert new_ball.vz == 2.0  # velocity preserved, not zeroed


def test_bouncing_sets_has_bounced_true():
    ball = Ball(x=0, y=0.001, z=1.0, vx=0, vy=-2.0, vz=0)
    new_ball, _ = step(ball, 0.01, CFG)
    assert new_ball.has_bounced is True


def test_fresh_ball_defaults_to_not_bounced():
    ball = Ball(x=0, y=0.5, z=0, vx=0, vy=0, vz=0)
    assert ball.has_bounced is False


def test_ball_far_outside_table_width_is_out_of_bounds():
    ball = Ball(x=2.0, y=0.5, z=0, vx=0, vy=0, vz=0)
    _, result = step(ball, 0.01, CFG)
    assert result.out_of_bounds is True


def test_ball_on_table_is_not_out_of_bounds():
    ball = Ball(x=0.3, y=0.1, z=1.0, vx=0, vy=0, vz=0)
    _, result = step(ball, 0.01, CFG)
    assert result.out_of_bounds is False


# --- predict_landing_x ---

def test_predict_landing_x_is_exact_for_constant_velocity():
    # Starting at z=1.0 moving toward z=-1.0 at vz=-2 m/s, with x drifting
    # sideways at vx=0.5 m/s. After 1 second, z = -1.0 (arrives at target).
    ball = Ball(x=0.0, y=0.5, z=1.0, vx=0.5, vy=0, vz=-2.0)
    landing_x = predict_landing_x(ball, target_z=-1.0)
    assert math.isclose(landing_x, 0.5)  # 1 second of travel at vx=0.5


def test_predict_landing_x_returns_none_when_moving_away():
    ball = Ball(x=0, y=0.5, z=1.0, vx=0, vy=0, vz=2.0)  # moving further from target_z=-1.0
    assert predict_landing_x(ball, target_z=-1.0) is None


def test_predict_landing_x_handles_stationary_ball_already_at_target():
    ball = Ball(x=0.25, y=0.5, z=-1.0, vx=0, vy=0, vz=0)
    assert predict_landing_x(ball, target_z=-1.0) == 0.25
