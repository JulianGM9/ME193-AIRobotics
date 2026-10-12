import cv2
import cv2.aruco as aruco
import numpy as np

from vision.apriltag import DICTIONARY, LevelSelector, create_detector, detect_tags

TAG_TO_LEVEL = {0: "beginner", 1: "intermediate", 2: "advanced", 3: "expert"}


def make_tag_frame(tag_id, size=200, margin=40):
    tag_img = aruco.generateImageMarker(DICTIONARY, tag_id, size)
    padded = cv2.copyMakeBorder(tag_img, margin, margin, margin, margin,
                                 cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# --- detect_tags on synthetic images ---

def test_detects_a_generated_tag():
    detector = create_detector()
    frame = make_tag_frame(tag_id=2)
    tags = detect_tags(frame, detector)
    assert len(tags) == 1
    tag_id, corners = tags[0]
    assert tag_id == 2
    assert corners.shape == (4, 2)


def test_blank_frame_has_no_tags():
    detector = create_detector()
    frame = np.full((280, 280, 3), 255, dtype=np.uint8)
    assert detect_tags(frame, detector) == []


# --- LevelSelector, using a fake clock (plain floats, no real sleeping) ---

def make_selector(**overrides):
    kwargs = dict(
        tag_to_level=TAG_TO_LEVEL,
        confirm_hold_seconds=1.0,
    )
    kwargs.update(overrides)
    return LevelSelector(**kwargs)


def test_brief_tag_sighting_does_not_confirm():
    selector = make_selector()
    selector.start(now=0.0)
    assert selector.update([2], now=0.0) is None
    assert selector.update([2], now=0.3) is None  # under the 1.0s hold requirement


def test_sustained_tag_sighting_confirms_after_hold_time():
    selector = make_selector()
    selector.start(now=0.0)
    selector.update([2], now=0.0)
    selector.update([2], now=0.5)
    result = selector.update([2], now=1.0)
    assert result == "advanced"
    assert selector.confirmed_level == "advanced"


def test_switching_tags_restarts_the_hold_timer():
    selector = make_selector()
    selector.start(now=0.0)
    selector.update([0], now=0.0)
    selector.update([0], now=0.9)  # almost confirmed as "beginner"
    selector.update([1], now=0.95)  # switched to a different tag just before confirming
    result = selector.update([1], now=1.4)  # only 0.45s on tag 1 so far
    assert result is None
    result = selector.update([1], now=1.95)  # now 1.0s on tag 1
    assert result == "intermediate"


def test_first_recognized_tag_wins_when_multiple_are_visible():
    selector = make_selector()
    selector.start(now=0.0)
    selector.update([3, 1], now=0.0)
    result = selector.update([3, 1], now=1.0)
    assert result == "expert"


def test_waits_indefinitely_with_no_tag_shown_rather_than_timing_out():
    selector = make_selector()
    selector.start(now=0.0)
    selector.update([], now=1.0)
    result = selector.update([], now=1_000_000.0)  # arbitrarily far in the future
    assert result is None
    assert selector.confirmed_level is None


def test_select_by_key_confirms_immediately():
    selector = make_selector()
    selector.start(now=0.0)
    result = selector.select_by_key("expert")
    assert result == "expert"
    assert selector.update([0], now=100.0) == "expert"  # stays confirmed regardless


def test_progress_reports_fraction_of_hold_time_elapsed():
    selector = make_selector(confirm_hold_seconds=2.0)
    selector.start(now=0.0)
    selector.update([0], now=0.0)
    selector.update([0], now=1.0)
    assert selector.progress(now=1.0) == 0.5


def test_progress_is_zero_with_nothing_held_up():
    selector = make_selector()
    selector.start(now=0.0)
    assert selector.progress(now=0.0) == 0.0


def test_current_candidate_reflects_what_is_being_held():
    selector = make_selector()
    selector.start(now=0.0)
    assert selector.current_candidate() is None
    selector.update([2], now=0.0)
    assert selector.current_candidate() == "advanced"


def test_current_candidate_is_none_after_hand_drops_the_tag():
    selector = make_selector()
    selector.start(now=0.0)
    selector.update([2], now=0.0)
    selector.update([], now=0.1)
    assert selector.current_candidate() is None
