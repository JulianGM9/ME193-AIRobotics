import threading
import time

import legoeducation as le

RECONNECT_POLL_SECONDS = 1.0
HAPTIC_PULSE_MS = 60  # duration of each back-and-forth jiggle segment
HAPTIC_PULSE_COUNT = 5  # alternations, for a short buzz (~300ms total)
HAPTIC_SPEED_PERCENT = 100  # max - stronger torque per jolt reads as more "violent" than more, weaker ones
# Generous upper bound on how long a pulse sequence should ever take
# (actual expected duration is HAPTIC_PULSE_COUNT * HAPTIC_PULSE_MS, well
# under a second) - see haptic_pulse()'s docstring for why this exists.
HAPTIC_STUCK_TIMEOUT_SECONDS = 2.0


class MotorConnection:
    """Owns the Double Motor BLE connection, with status tracking and a
    background auto-reconnect loop so a dropped link doesn't require
    restarting the whole game.
    """

    def __init__(self, card_color=None, card_serial=None, device_notification_delay=15):
        self.card_color = card_color
        self.card_serial = card_serial
        # The library's default notification delay is 100ms, which is far
        # too coarse for swing timing (a swing may only last ~200-400ms
        # total). 15ms is the library's documented minimum.
        self.device_notification_delay = device_notification_delay
        self.motor = le.DoubleMotor()
        self.status = "disconnected"  # disconnected, connecting, connected
        self._watchdog_thread = None
        self._stop = threading.Event()
        self._haptic_busy_since = None  # time.monotonic() timestamp, or None if idle

    @property
    def connected(self):
        return bool(self.motor.connected)

    def connect(self):
        self.status = "connecting"
        kwargs = {"device_notification_delay": self.device_notification_delay}
        if self.card_color is not None:
            kwargs["card_color"] = self.card_color
        if self.card_serial is not None:
            kwargs["card_serial"] = self.card_serial
        self.motor.connect(**kwargs)
        self.status = "connected"

    def disconnect(self):
        self._stop.set()
        if self.motor.connected:
            self.motor.disconnect()
        self.status = "disconnected"

    def start_watchdog(self):
        """Run a background thread that reconnects automatically if the
        BLE link drops. Call this once, after the first connect()."""
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        while not self._stop.is_set():
            if not self.motor.connected:
                self.status = "connecting"
                try:
                    self.connect()
                except Exception as err:
                    print(f"Reconnect attempt failed: {err}")
                    self.status = "disconnected"
            time.sleep(RECONNECT_POLL_SECONDS)

    def haptic_pulse(self, pulse_ms=HAPTIC_PULSE_MS, pulses=HAPTIC_PULSE_COUNT, speed=HAPTIC_SPEED_PERCENT):
        """Briefly jiggles both motors back and forth to simulate the
        physical "thwack" of hitting the ball. Fire-and-forget: runs on its
        own thread, paced by our own time.sleep() between segments (not by
        waiting on the BLE device's confirmation - see _run_haptic_pulse)
        so it never stalls the game loop.

        No-ops if not connected, or if a previous pulse is still (recently)
        running - two pulses overlapping would mean two threads issuing
        motor_run_for_time() on the same BLE connection concurrently, which
        isn't safe (confirmed to crash the game after enough rapid
        successive hits). "Recently" is the key word: this used to be a
        plain Lock, released in a finally block - but a single
        motor_run_for_time(blocking=True) call that never got its BLE
        confirmation (a dropped packet, not even an error, just silence)
        hung that thread forever, which meant the lock was never released
        and every single haptic pulse for the rest of the session silently
        no-opped (confirmed: "buzzed once, then never again"). Timing out
        the busy-check after HAPTIC_STUCK_TIMEOUT_SECONDS instead means a
        stuck thread (should it ever happen again, despite the
        blocking=False fix below) can only break one haptic pulse, not the
        rest of the session.
        """
        if not self.motor.connected:
            return
        now = time.monotonic()
        if self._haptic_busy_since is not None and (now - self._haptic_busy_since) < HAPTIC_STUCK_TIMEOUT_SECONDS:
            return
        self._haptic_busy_since = now
        threading.Thread(target=self._run_haptic_pulse_then_clear, args=(pulse_ms, pulses, speed),
                          daemon=True).start()

    def _run_haptic_pulse_then_clear(self, pulse_ms, pulses, speed):
        try:
            self._run_haptic_pulse(pulse_ms, pulses, speed)
        finally:
            self._haptic_busy_since = None

    def _run_haptic_pulse(self, pulse_ms, pulses, speed):
        # blocking=False: fire the direction command and move on, pacing
        # the sequence with our own sleep rather than waiting for the
        # Double Motor's BLE confirmation of each segment. The previous
        # blocking=True version hung this thread forever if a confirmation
        # was ever dropped (see haptic_pulse()'s docstring) - sleeping a
        # fixed duration can't hang no matter what the BLE link does.
        direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE
        try:
            for _ in range(pulses):
                self.motor.motor_run_for_time(
                    pulse_ms, direction=direction, motor=le.MOTOR_BOTH, speed=speed, blocking=False,
                )
                time.sleep(pulse_ms / 1000.0)
                direction = (
                    le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                    if direction == le.MOTOR_MOVE_DIRECTION_CLOCKWISE
                    else le.MOTOR_MOVE_DIRECTION_CLOCKWISE
                )
        except Exception as err:
            # Best-effort feedback - a BLE hiccup mid-pulse shouldn't crash
            # or otherwise affect the game.
            print(f"Haptic pulse failed: {err}")
