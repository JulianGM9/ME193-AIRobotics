import math

from game.hit_detection import MAX_ANGLE_OFFSET, evaluate_hit
from imu.swing_detector import SwingEvent

HIT_RADIUS = 0.2
TIMING_WINDOW_MS = 150

PADDLE_POS = (0.5, 0.5)
BALL_POS = (0.52, 0.48)  # close to paddle, within hit_radius
FAR_BALL_POS = (5.0, 5.0)  # nowhere near the paddle
ARRIVAL_TIME = 10.0


def swing_at(timestamp, intensity=1.0, direction="forehand", axis="x"):
    return SwingEvent(timestamp=timestamp, peak_intensity=intensity, direction=direction, dominant_axis=axis)


def test_no_swing_at_all_is_a_miss_with_swing_feedback():
    result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME,
                           swing_event=None, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "miss"
    assert result.reason == "no_swing"


def test_good_position_and_good_timing_is_a_hit():
    swing = swing_at(ARRIVAL_TIME + 0.05)  # 50ms late, within the 150ms window
    result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME,
                           swing, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "hit"
    assert result.quality is not None


def test_swing_too_early_outside_window_is_a_miss_with_too_early_feedback():
    swing = swing_at(ARRIVAL_TIME - 0.3)  # 300ms early, outside 150ms window, inside 500ms search window
    result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME,
                           swing, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "miss"
    assert result.reason == "too_early"


def test_swing_too_late_outside_window_is_a_miss_with_too_late_feedback():
    swing = swing_at(ARRIVAL_TIME + 0.3)
    result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME,
                           swing, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "miss"
    assert result.reason == "too_late"


def test_good_timing_but_wrong_position_is_a_miss_with_wrong_spot_feedback():
    swing = swing_at(ARRIVAL_TIME)  # perfect timing
    result = evaluate_hit(PADDLE_POS, FAR_BALL_POS, ARRIVAL_TIME,
                           swing, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "miss"
    assert result.reason == "wrong_spot"


def test_swing_way_outside_search_window_counts_as_no_swing():
    swing = swing_at(ARRIVAL_TIME - 10.0)  # 10 seconds early - unrelated to this ball
    result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME,
                           swing, hit_radius=HIT_RADIUS, timing_window_ms=TIMING_WINDOW_MS)
    assert result.outcome == "miss"
    assert result.reason == "no_swing"


def test_stronger_swing_increases_speed_factor():
    weak = swing_at(ARRIVAL_TIME, intensity=1.0)
    strong = swing_at(ARRIVAL_TIME, intensity=2.0)
    weak_result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME, weak, HIT_RADIUS, TIMING_WINDOW_MS)
    strong_result = evaluate_hit(PADDLE_POS, BALL_POS, ARRIVAL_TIME, strong, HIT_RADIUS, TIMING_WINDOW_MS)
    assert strong_result.quality["speed_factor"] > weak_result.quality["speed_factor"]


def test_position_exactly_at_hit_radius_boundary_counts_as_ok():
    # Paddle and ball at the same point - distance 0, trivially within radius.
    swing = swing_at(ARRIVAL_TIME)
    result = evaluate_hit((0.5, 0.5), (0.5, 0.5), ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert result.outcome == "hit"


# --- Direction from where on the paddle the ball made contact (not
# paddle velocity - deliberately removed, since it was hard to control
# without accidentally throwing off swing detection too). ---

def test_contact_right_of_paddle_center_angles_the_return_right():
    swing = swing_at(ARRIVAL_TIME)
    ball_right_of_paddle = (0.6, 0.5)  # within HIT_RADIUS=0.2 of (0.5, 0.5)
    result = evaluate_hit((0.5, 0.5), ball_right_of_paddle, ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert result.quality["angle_offset"] > 0


def test_contact_left_of_paddle_center_angles_the_return_left():
    swing = swing_at(ARRIVAL_TIME)
    ball_left_of_paddle = (0.4, 0.5)
    result = evaluate_hit((0.5, 0.5), ball_left_of_paddle, ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert result.quality["angle_offset"] < 0


def test_contact_dead_center_gives_no_angle():
    swing = swing_at(ARRIVAL_TIME)
    result = evaluate_hit((0.5, 0.5), (0.5, 0.5), ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert result.quality["angle_offset"] == 0


def test_contact_at_the_very_edge_of_hit_radius_gives_the_max_angle():
    swing = swing_at(ARRIVAL_TIME)
    ball_at_edge = (0.5 + HIT_RADIUS, 0.5)  # exactly hit_radius away, to the right
    result = evaluate_hit((0.5, 0.5), ball_at_edge, ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert math.isclose(result.quality["angle_offset"], MAX_ANGLE_OFFSET)


def test_max_angle_is_the_same_regardless_of_hit_radius_size():
    # A generous (beginner-sized) hit_radius and a tight (expert-sized) one
    # should both give the same max angle for a dead-on-the-edge contact -
    # the angle is normalized to a fraction of hit_radius, not an absolute
    # distance, so easier levels aren't also capable of sharper angles.
    swing = swing_at(ARRIVAL_TIME)
    generous_radius = 0.7
    tight_radius = 0.2
    generous_result = evaluate_hit((0.0, 0.0), (generous_radius, 0.0), ARRIVAL_TIME,
                                    swing, generous_radius, TIMING_WINDOW_MS)
    tight_result = evaluate_hit((0.0, 0.0), (tight_radius, 0.0), ARRIVAL_TIME,
                                 swing, tight_radius, TIMING_WINDOW_MS)
    assert math.isclose(generous_result.quality["angle_offset"], MAX_ANGLE_OFFSET)
    assert math.isclose(tight_result.quality["angle_offset"], MAX_ANGLE_OFFSET)


def test_vertical_offset_alone_does_not_angle_the_return():
    # Only the x-axis (across the table) should steer the ball - a ball
    # that arrives above/below paddle center but at the same x shouldn't
    # angle it sideways.
    swing = swing_at(ARRIVAL_TIME)
    result = evaluate_hit((0.5, 0.5), (0.5, 0.35), ARRIVAL_TIME, swing, HIT_RADIUS, TIMING_WINDOW_MS)
    assert result.quality["angle_offset"] == 0
