"""Fuses the vision-tracked paddle position against the IMU swing to
decide whether the player actually hit the ball.
"""
import math
from dataclasses import dataclass

FEEDBACK_MESSAGES = {
    "no_swing": "Swing!",
    "too_early": "Too early!",
    "too_late": "Too late!",
    "wrong_spot": "Wrong spot!",
}


MAX_ANGLE_OFFSET = 2.5  # vx from a dead-on-the-edge-of-hit_radius contact


@dataclass
class HitResult:
    outcome: str  # "hit" or "miss"
    reason: str = None  # set when outcome == "miss": one of FEEDBACK_MESSAGES' keys
    quality: dict = None  # set when outcome == "hit": {"speed_factor", "angle_offset"}


def compute_quality(swing_event, offset_x, hit_radius):
    """Stronger-than-typical swings (peak_intensity above the calibrated
    "1.0 = average practice swing" baseline) speed up the return.

    Direction comes from where on the paddle the ball made contact
    (offset_x: the ball's arrival x minus the paddle's x, same axis as the
    table) rather than how the paddle was moving - contact toward one edge
    sends the ball toward that same side, like deflecting off a real
    paddle, and is much easier to aim deliberately than trying to swing at
    a precise velocity without throwing off swing detection. Normalized by
    hit_radius so a dead-on-the-edge contact gives the same max angle at
    every difficulty level, regardless of how generous hit_radius is
    configured to be.
    """
    speed_factor = 1.0 + 0.3 * max(0.0, swing_event.peak_intensity - 1.0)
    offset_fraction = max(-1.0, min(1.0, offset_x / hit_radius)) if hit_radius > 0 else 0.0
    angle_offset = offset_fraction * MAX_ANGLE_OFFSET
    return {"speed_factor": speed_factor, "angle_offset": angle_offset}


def evaluate_hit(paddle_position, ball_arrival_position, ball_arrival_time,
                  swing_event, hit_radius, timing_window_ms, swing_search_window_ms=500):
    """Returns a HitResult.

    paddle_position / ball_arrival_position: (x, y) tuples, same units.
    ball_arrival_time / swing_event.timestamp: seconds (time.monotonic()-style).
    swing_event: a SwingEvent (see imu/swing_detector.py), or None if no
        swing was detected anywhere near this ball arrival.
    """
    distance = math.hypot(
        paddle_position[0] - ball_arrival_position[0],
        paddle_position[1] - ball_arrival_position[1],
    )
    position_ok = distance <= hit_radius

    if swing_event is None:
        return HitResult(outcome="miss", reason="no_swing")

    timing_error_ms = (swing_event.timestamp - ball_arrival_time) * 1000

    if abs(timing_error_ms) > swing_search_window_ms:
        # The nearest swing was so far from this ball's arrival that it's
        # not meaningfully related to it - treat it as if there were none.
        return HitResult(outcome="miss", reason="no_swing")

    if abs(timing_error_ms) > timing_window_ms:
        reason = "too_early" if timing_error_ms < 0 else "too_late"
        return HitResult(outcome="miss", reason=reason)

    if not position_ok:
        return HitResult(outcome="miss", reason="wrong_spot")

    offset_x = ball_arrival_position[0] - paddle_position[0]
    return HitResult(outcome="hit", quality=compute_quality(swing_event, offset_x, hit_radius))
