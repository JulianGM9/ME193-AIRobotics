import threading
import time

import legoeducation as le

import imu.motor_connection as motor_connection_module
from imu.motor_connection import MotorConnection


class FakeMotor:
    """Stands in for le.DoubleMotor() so these tests don't need real BLE
    hardware - just records each motor_run_for_time() call."""

    def __init__(self):
        self.connected = True
        self.calls = []

    def motor_run_for_time(self, time_ms, *, direction, motor, speed, blocking):
        self.calls.append({
            "time_ms": time_ms, "direction": direction, "motor": motor,
            "speed": speed, "blocking": blocking,
        })


class SlowFakeMotor(FakeMotor):
    """Like FakeMotor, but each call blocks until released - for
    simulating a pulse that's still in flight when another is requested."""

    def __init__(self):
        super().__init__()
        self.call_started = threading.Event()
        self.release = threading.Event()

    def motor_run_for_time(self, *a, **k):
        super().motor_run_for_time(*a, **k)
        self.call_started.set()
        self.release.wait(timeout=2.0)


def make_connection():
    connection = MotorConnection()
    connection.motor = FakeMotor()
    return connection


def test_run_haptic_pulse_issues_the_requested_number_of_segments():
    connection = make_connection()
    connection._run_haptic_pulse(pulse_ms=50, pulses=4, speed=70)
    assert len(connection.motor.calls) == 4


def test_run_haptic_pulse_alternates_direction_each_segment():
    connection = make_connection()
    connection._run_haptic_pulse(pulse_ms=50, pulses=4, speed=70)
    directions = [call["direction"] for call in connection.motor.calls]
    assert directions == [
        le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
        le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
        le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
        le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
    ]


def test_run_haptic_pulse_drives_both_motors_together():
    connection = make_connection()
    connection._run_haptic_pulse(pulse_ms=50, pulses=2, speed=70)
    assert all(call["motor"] == le.MOTOR_BOTH for call in connection.motor.calls)


def test_run_haptic_pulse_passes_through_duration_and_speed():
    connection = make_connection()
    connection._run_haptic_pulse(pulse_ms=35, pulses=1, speed=55)
    call = connection.motor.calls[0]
    assert call["time_ms"] == 35
    assert call["speed"] == 55


def test_run_haptic_pulse_survives_a_mid_pulse_ble_error():
    connection = make_connection()

    def raise_on_second_call(*a, **k):
        raise_on_second_call.n += 1
        if raise_on_second_call.n == 2:
            raise RuntimeError("BLE hiccup")
    raise_on_second_call.n = 0
    connection.motor.motor_run_for_time = raise_on_second_call

    connection._run_haptic_pulse(pulse_ms=50, pulses=4, speed=70)  # must not raise


def test_haptic_pulse_does_nothing_when_not_connected():
    connection = make_connection()
    connection.motor.connected = False
    connection.haptic_pulse(pulse_ms=50, pulses=4, speed=70)
    assert connection.motor.calls == []


def test_haptic_pulse_runs_asynchronously_without_blocking_the_caller():
    connection = make_connection()
    connection.haptic_pulse(pulse_ms=50, pulses=4, speed=70)
    # The call above must return immediately (it's spawned a background
    # thread) rather than blocking for the full ~200ms pulse sequence -
    # that's the whole point of not stalling the game loop.
    deadline = time.monotonic() + 1.0
    while len(connection.motor.calls) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(connection.motor.calls) == 4


def test_haptic_pulse_skips_overlap_while_a_previous_one_is_still_running():
    # Two pulses overlapping would mean two threads issuing
    # motor_run_for_time() on the same BLE connection concurrently, which
    # isn't safe (this was the cause of a real crash after enough rapid
    # successive hits) - a second call made while the first is still in
    # flight must be skipped outright, not queued or run concurrently.
    connection = MotorConnection()
    connection.motor = SlowFakeMotor()

    connection.haptic_pulse(pulse_ms=50, pulses=4, speed=70)
    assert connection.motor.call_started.wait(timeout=1.0)  # first pulse is now mid-flight

    connection.haptic_pulse(pulse_ms=50, pulses=4, speed=70)  # should be skipped, not queued
    time.sleep(0.1)
    assert len(connection.motor.calls) == 1  # still just the first pulse's first segment

    connection.motor.release.set()  # let the first pulse finish


def test_haptic_pulse_works_again_once_the_previous_one_has_finished():
    connection = make_connection()
    connection.haptic_pulse(pulse_ms=10, pulses=1, speed=70)
    deadline = time.monotonic() + 1.0
    while len(connection.motor.calls) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    # Give the background thread a moment to clear the busy flag after its
    # single segment completes, then confirm a fresh pulse isn't skipped.
    deadline = time.monotonic() + 1.0
    while connection._haptic_busy_since is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    connection.haptic_pulse(pulse_ms=10, pulses=1, speed=70)
    deadline = time.monotonic() + 1.0
    while len(connection.motor.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(connection.motor.calls) == 2


def test_haptic_pulse_skips_when_a_pulse_recently_started():
    connection = make_connection()
    connection._haptic_busy_since = time.monotonic()  # "just started, still well within timeout"
    connection.haptic_pulse(pulse_ms=10, pulses=1, speed=70)
    time.sleep(0.05)
    assert connection.motor.calls == []


def test_haptic_pulse_self_heals_after_a_stuck_busy_flag_times_out():
    # Simulates a thread that hung forever (e.g. a BLE confirmation that
    # never arrived) and so never cleared _haptic_busy_since - this used to
    # mean every future haptic_pulse() call silently no-opped for the rest
    # of the session (confirmed: "buzzed once, then never again"). A call
    # long after HAPTIC_STUCK_TIMEOUT_SECONDS must now proceed anyway.
    connection = make_connection()
    connection._haptic_busy_since = (
        time.monotonic() - motor_connection_module.HAPTIC_STUCK_TIMEOUT_SECONDS - 1.0
    )
    connection.haptic_pulse(pulse_ms=10, pulses=1, speed=70)
    deadline = time.monotonic() + 1.0
    while len(connection.motor.calls) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(connection.motor.calls) == 1


def test_run_haptic_pulse_uses_non_blocking_motor_calls():
    # blocking=True previously meant a dropped BLE confirmation could hang
    # the background thread forever (see the self-heal test above) - the
    # sequence is now paced by our own sleep instead, so each individual
    # motor_run_for_time() call must be fire-and-forget.
    connection = make_connection()
    connection._run_haptic_pulse(pulse_ms=10, pulses=2, speed=70)
    assert all(call["blocking"] is False for call in connection.motor.calls)
