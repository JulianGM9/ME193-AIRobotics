import math

from game.opponent_animator import HIT, IDLE, LOSE, SCORE, OpponentAnimator


def test_starts_idle():
    animator = OpponentAnimator()
    state, progress = animator.current_state(now=0.0)
    assert state == IDLE
    assert progress == 0.0


def test_trigger_hit_is_reported_immediately():
    animator = OpponentAnimator(hit_duration_seconds=0.25)
    animator.trigger_hit(now=10.0)
    state, progress = animator.current_state(now=10.0)
    assert state == HIT
    assert progress == 0.0


def test_hit_progress_increases_partway_through():
    animator = OpponentAnimator(hit_duration_seconds=0.2)
    animator.trigger_hit(now=10.0)
    state, progress = animator.current_state(now=10.1)
    assert state == HIT
    assert math.isclose(progress, 0.5)


def test_hit_falls_back_to_idle_once_duration_elapses():
    animator = OpponentAnimator(hit_duration_seconds=0.2)
    animator.trigger_hit(now=10.0)
    state, progress = animator.current_state(now=10.25)  # safely past, avoids float boundary flakiness
    assert state == IDLE
    assert progress == 0.0


def test_trigger_score_and_lose_use_their_own_durations():
    animator = OpponentAnimator(score_duration_seconds=1.0, lose_duration_seconds=0.5)
    animator.trigger_score(now=0.0)
    assert animator.current_state(now=0.9)[0] == SCORE
    assert animator.current_state(now=1.1)[0] == IDLE  # safely past, avoids float boundary flakiness

    animator.trigger_lose(now=5.0)
    assert animator.current_state(now=5.4)[0] == LOSE
    assert animator.current_state(now=5.6)[0] == IDLE


def test_a_new_trigger_replaces_whatever_event_was_playing():
    animator = OpponentAnimator(hit_duration_seconds=1.0, score_duration_seconds=1.0)
    animator.trigger_hit(now=0.0)
    animator.trigger_score(now=0.1)  # interrupts the hit animation
    state, progress = animator.current_state(now=0.1)
    assert state == SCORE
    assert progress == 0.0


def test_sway_offset_is_zero_at_time_zero():
    animator = OpponentAnimator(sway_period_seconds=2.0, sway_amplitude_deg=5.0)
    assert animator.sway_offset(now=0.0) == 0.0


def test_sway_offset_reaches_amplitude_at_a_quarter_period():
    animator = OpponentAnimator(sway_period_seconds=4.0, sway_amplitude_deg=5.0)
    assert math.isclose(animator.sway_offset(now=1.0), 5.0, abs_tol=1e-9)


def test_sway_offset_is_periodic():
    animator = OpponentAnimator(sway_period_seconds=2.0, sway_amplitude_deg=5.0)
    assert math.isclose(animator.sway_offset(now=0.3), animator.sway_offset(now=2.3))


def test_sway_continues_independently_of_an_active_event():
    # Idle sway should still be computable (e.g. to blend with a pose)
    # while a hit/score/lose event is also playing - it's not gated on
    # current_state().
    animator = OpponentAnimator(sway_period_seconds=4.0, sway_amplitude_deg=5.0)
    animator.trigger_hit(now=0.0)
    assert math.isclose(animator.sway_offset(now=1.0), 5.0, abs_tol=1e-9)
