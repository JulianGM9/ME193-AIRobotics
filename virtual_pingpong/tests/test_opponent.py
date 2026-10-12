import random

from game.opponent import Opponent
from game.physics import Ball

CFG = {
    "max_speed": 1.5,
    "reaction_delay": 0.25,
    "miss_probability": 0.15,
    "return_speed_range": (2.5, 3.5),
    "return_target_spread": 0.6,
}
OPPONENT_END_Z = -1.37


def test_does_not_move_before_reaction_delay_elapses():
    opponent = Opponent(CFG)
    opponent.notice_ball_incoming()
    ball = Ball(x=1.0, y=0.5, z=1.0, vx=0.0, vy=0, vz=-2.0)
    opponent.update(ball, dt=0.1, opponent_end_z=OPPONENT_END_Z)  # only 0.1s < 0.25s delay
    assert opponent.x == 0.0


def test_starts_moving_toward_target_after_reaction_delay():
    opponent = Opponent(CFG)
    opponent.notice_ball_incoming()
    ball = Ball(x=1.0, y=0.5, z=1.0, vx=0.0, vy=0, vz=-2.0)  # moving straight, lands at x=1.0
    opponent.update(ball, dt=0.3, opponent_end_z=OPPONENT_END_Z)  # past the 0.25s delay
    assert opponent.x > 0.0  # has started moving toward x=1.0


def test_respects_max_speed_cap():
    opponent = Opponent(CFG)
    opponent.notice_ball_incoming()
    ball = Ball(x=10.0, y=0.5, z=1.0, vx=0.0, vy=0, vz=-2.0)  # far-off target
    # The reaction-delay threshold is crossed partway through the first
    # call, and that same call's dt also drives movement once it fires -
    # so total movement is bounded by max_speed times the TOTAL dt elapsed
    # across both calls (1.25s), not just the second call's dt (1.0s).
    opponent.update(ball, dt=0.25, opponent_end_z=OPPONENT_END_Z)  # triggers reaction
    opponent.update(ball, dt=1.0, opponent_end_z=OPPONENT_END_Z)
    assert opponent.x <= CFG["max_speed"] * 1.25 + 1e-9


def test_eventually_reaches_the_target_without_overshooting():
    opponent = Opponent(CFG)
    opponent.notice_ball_incoming()
    ball = Ball(x=0.5, y=0.5, z=1.0, vx=0.0, vy=0, vz=-2.0)
    for _ in range(50):
        opponent.update(ball, dt=0.05, opponent_end_z=OPPONENT_END_Z)
    assert abs(opponent.x - 0.5) < 1e-9


def test_decide_outcome_is_deterministic_with_a_seeded_rng():
    opponent = Opponent(CFG)
    # miss_probability=0.15: a draw below 0.15 is a miss, at/above is a return.
    assert opponent.decide_outcome(random.Random(1)).__class__ is str
    low_roll_rng = type("FixedRng", (), {"random": lambda self: 0.05})()
    high_roll_rng = type("FixedRng", (), {"random": lambda self: 0.95})()
    assert opponent.decide_outcome(low_roll_rng) == "miss"
    assert opponent.decide_outcome(high_roll_rng) == "return"


def test_choose_return_stays_within_configured_ranges():
    opponent = Opponent(CFG)
    rng = random.Random(42)
    for _ in range(100):
        choice = opponent.choose_return(rng)
        assert CFG["return_speed_range"][0] <= choice["speed"] <= CFG["return_speed_range"][1]
        assert -CFG["return_target_spread"] <= choice["target_x"] <= CFG["return_target_spread"]
