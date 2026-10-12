from vision.paddle_detector import PaddleDetector

CFG = {"smoothing_alpha": 0.5}


def make_pose(wrist=(0.5, 0.7)):
    return {"wrist": wrist}


# --- fallback selection ---

def test_uses_hand_position_when_available():
    detector = PaddleDetector(CFG)
    pose = make_pose(wrist=(0.5, 0.7))
    detector.update(pose, hand_position=(0.6, 0.6), dt=0.0)
    assert detector.last_source == "hand"


def test_falls_back_to_wrist_when_hand_not_detected():
    detector = PaddleDetector(CFG)
    pose = make_pose(wrist=(0.5, 0.7))
    detector.update(pose, hand_position=None, dt=0.0)
    assert detector.last_source == "wrist"


def test_first_update_snaps_directly_to_the_detected_point():
    detector = PaddleDetector(CFG)
    pose = make_pose()
    smoothed, velocity = detector.update(pose, hand_position=(0.6, 0.6), dt=0.0)
    assert smoothed == (0.6, 0.6)
    assert velocity == (0.0, 0.0)


# --- smoothing / velocity ---

def test_smoothing_moves_partway_toward_target():
    detector = PaddleDetector(CFG)
    pose = make_pose()
    detector.update(pose, hand_position=(0.0, 0.0), dt=0.0)  # establishes smoothed_pos at (0, 0)
    smoothed, _ = detector.update(pose, hand_position=(1.0, 0.0), dt=1.0)
    assert smoothed == (0.5, 0.0)  # halfway, since smoothing_alpha=0.5


def test_velocity_reflects_the_smoothed_movement_over_dt():
    detector = PaddleDetector(CFG)
    pose = make_pose()
    detector.update(pose, hand_position=(0.0, 0.0), dt=0.0)
    _, velocity = detector.update(pose, hand_position=(1.0, 0.0), dt=1.0)
    assert velocity[0] == 0.5  # moved 0.5 units in 1 second
    assert velocity[1] == 0.0


def test_zero_dt_does_not_divide_by_zero_and_keeps_prior_velocity():
    detector = PaddleDetector(CFG)
    pose = make_pose()
    detector.update(pose, hand_position=(0.0, 0.0), dt=0.0)
    detector.update(pose, hand_position=(1.0, 0.0), dt=1.0)
    prior_velocity = detector.velocity
    detector.update(pose, hand_position=(2.0, 0.0), dt=0.0)
    assert detector.velocity == prior_velocity


def test_switching_from_hand_to_wrist_still_smooths_continuously():
    detector = PaddleDetector(CFG)
    pose = make_pose(wrist=(1.0, 0.0))
    detector.update(pose, hand_position=(0.0, 0.0), dt=0.0)  # hand detected, snaps to (0, 0)
    smoothed, _ = detector.update(pose, hand_position=None, dt=1.0)  # hand lost, falls back to wrist
    assert detector.last_source == "wrist"
    assert smoothed == (0.5, 0.0)  # smooths halfway toward the wrist position, not a hard jump
