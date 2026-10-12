import time

from imu.swing_detector import Calibration, IMUSample, SwingDetector

CALIBRATION = Calibration(
    gyro_noise_floor=1.0,
    gyro_avg_peak=5.0,
    accel_noise_floor=1.0,
    accel_avg_peak=5.0,
)


def make_detector(**overrides):
    kwargs = dict(
        smoothing_alpha=1.0,  # no smoothing lag, so tests are deterministic
        gravity_ema_alpha=1.0,  # gravity estimate tracks accel instantly
        refractory_ms=300,
        threshold_fraction=1.0,
        calibration=CALIBRATION,
    )
    kwargs.update(overrides)
    return SwingDetector(**kwargs)


def feed(detector, samples):
    events = []
    for sample in samples:
        event = detector.update(sample)
        if event is not None:
            events.append(event)
    return events


def test_resting_samples_produce_no_swing():
    detector = make_detector()
    samples = [
        IMUSample(timestamp=i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(1.0, 0.0, 0.0))
        for i in range(20)
    ]
    assert feed(detector, samples) == []


def test_a_clear_swing_is_detected_exactly_once():
    detector = make_detector()
    # Bell-curve-shaped gyro magnitude: rest -> ramp up past threshold -> back down to rest.
    gyro_profile = [1.0, 1.0, 2.0, 4.0, 6.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.0, 1.0]
    samples = [
        IMUSample(timestamp=i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    events = feed(detector, samples)
    assert len(events) == 1
    assert events[0].dominant_axis == "x"
    assert events[0].direction == "forehand"  # positive gyro x


def test_negative_axis_is_classified_as_backhand():
    detector = make_detector()
    gyro_profile = [-1.0, -2.0, -6.0, -8.0, -6.0, -2.0, -1.0, -1.0]
    samples = [
        IMUSample(timestamp=i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    events = feed(detector, samples)
    assert len(events) == 1
    assert events[0].direction == "backhand"


def test_refractory_period_suppresses_a_second_swing_too_soon():
    detector = make_detector(refractory_ms=300)
    gyro_profile = [1.0, 6.0, 8.0, 6.0, 1.0] + [1.0, 6.0, 8.0, 6.0, 1.0]
    # 10 samples at 15ms apart = 135ms total, well under the 300ms refractory window.
    samples = [
        IMUSample(timestamp=i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    events = feed(detector, samples)
    assert len(events) == 1


def test_second_swing_after_refractory_window_is_detected():
    detector = make_detector(refractory_ms=100)
    first = [1.0, 6.0, 8.0, 6.0, 1.0]
    gap = [1.0] * 10  # 150ms of rest, longer than the 100ms refractory period
    second = [1.0, 6.0, 8.0, 6.0, 1.0]
    gyro_profile = first + gap + second
    samples = [
        IMUSample(timestamp=i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    events = feed(detector, samples)
    assert len(events) == 2


def test_recent_swing_within_window_returns_event():
    # recent_swing() compares against real wall-clock time.monotonic(), so
    # synthetic timestamps must be anchored to "now", not to an arbitrary
    # zero - otherwise "now minus timestamp" is just the process uptime.
    detector = make_detector()
    now = time.monotonic()
    gyro_profile = [1.0, 6.0, 8.0, 6.0, 1.0]
    samples = [
        IMUSample(timestamp=now + i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    feed(detector, samples)
    assert detector.recent_swing(window_ms=5000) is not None


def test_recent_swing_outside_window_returns_none():
    detector = make_detector()
    now = time.monotonic()
    gyro_profile = [1.0, 6.0, 8.0, 6.0, 1.0]
    samples = [
        IMUSample(timestamp=now + i * 0.015, accel=(1.0, 0.0, 0.0), gyro=(g, 0.0, 0.0))
        for i, g in enumerate(gyro_profile)
    ]
    feed(detector, samples)
    time.sleep(0.05)
    assert detector.recent_swing(window_ms=10) is None
