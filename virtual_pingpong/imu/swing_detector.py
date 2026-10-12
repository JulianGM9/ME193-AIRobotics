import math
import threading
import time
from collections import deque
from dataclasses import dataclass

# NOTE ON UNITS: the legoeducation library's docs (LEGO/LEGOEducation on
# GitHub) document accelerometerX/Y/Z and gyroscopeX/Y/Z as fields on
# imu_device, but never state their units or scale, and no example uses
# them. Rather than guess, this detector is unit-agnostic by design: every
# threshold is derived from calibrate() measuring this specific paddle at
# rest and during practice swings, and all comparisons are ratios against
# those measured values. Whatever units the hardware actually reports in,
# the math below still works.


@dataclass
class IMUSample:
    timestamp: float  # seconds, from time.monotonic()
    accel: tuple  # (x, y, z), raw units from imu_device
    gyro: tuple  # (x, y, z), raw units from imu_device


@dataclass
class SwingEvent:
    timestamp: float
    peak_intensity: float
    direction: str  # "forehand", "backhand", or "unknown"
    dominant_axis: str  # "x", "y", or "z" - which gyro axis drove the swing


@dataclass
class Calibration:
    gyro_noise_floor: float
    gyro_avg_peak: float
    accel_noise_floor: float
    accel_avg_peak: float


def _magnitude(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


class SwingDetector:
    """Pure logic: feed it IMUSamples, get back SwingEvents. No threading
    or hardware access here, so this can be unit-tested with synthetic
    data (see tests/test_swing_detector.py).
    """

    def __init__(self, *, smoothing_alpha=0.3, gravity_ema_alpha=0.02,
                 refractory_ms=300, threshold_fraction=1.0, calibration=None):
        self.smoothing_alpha = smoothing_alpha
        self.gravity_ema_alpha = gravity_ema_alpha
        self.refractory_ms = refractory_ms
        self.threshold_fraction = threshold_fraction
        self.calibration = calibration

        self._gravity_estimate = None  # slow EMA of accel, approximates gravity/orientation DC offset
        self._smoothed_intensity = 0.0
        self._last_swing_time = None
        self._recent_events = deque(maxlen=20)

        # Track the sample with the peak intensity since the last time we
        # were below threshold, so a detected swing can report which gyro
        # axis/direction actually drove it.
        self._armed = False
        self._peak_intensity = 0.0
        self._peak_sample = None

    def _normalized(self, value, noise_floor, avg_peak):
        span = max(avg_peak - noise_floor, 1e-6)
        return max(0.0, (value - noise_floor) / span)

    def update(self, sample: IMUSample):
        """Feed one new sample. Returns a SwingEvent if this sample marks
        the peak of a completed swing, else None."""
        gyro_mag = _magnitude(sample.gyro)
        accel_mag = _magnitude(sample.accel)

        if self._gravity_estimate is None:
            self._gravity_estimate = accel_mag
        else:
            self._gravity_estimate += self.gravity_ema_alpha * (accel_mag - self._gravity_estimate)
        accel_hp = abs(accel_mag - self._gravity_estimate)

        if self.calibration is not None:
            # Use whichever channel is more excited, not the sum - a real
            # swing may be gyro-dominant or accel-dominant, and summing
            # would under-count a swing that only strongly excites one.
            intensity = max(
                self._normalized(gyro_mag, self.calibration.gyro_noise_floor, self.calibration.gyro_avg_peak),
                self._normalized(accel_hp, self.calibration.accel_noise_floor, self.calibration.accel_avg_peak),
            )
        else:
            # Uncalibrated fallback: just the raw combined magnitude. Not
            # meaningful across different units/hardware - calibrate()
            # should be run before relying on this for real detection.
            intensity = max(gyro_mag, accel_hp)

        self._smoothed_intensity += self.smoothing_alpha * (intensity - self._smoothed_intensity)
        threshold = self.threshold_fraction  # 1.0 = about as strong as the calibrated average peak

        in_refractory = (
            self._last_swing_time is not None
            and (sample.timestamp - self._last_swing_time) * 1000 < self.refractory_ms
        )

        event = None
        if self._smoothed_intensity > threshold and not in_refractory:
            if not self._armed or self._smoothed_intensity > self._peak_intensity:
                self._armed = True
                self._peak_intensity = self._smoothed_intensity
                self._peak_sample = sample
        elif self._armed:
            # Intensity dropped back below threshold - the swing peaked
            # and finished. Report it using the peak sample we tracked.
            event = self._finish_swing()

        return event

    def _finish_swing(self):
        sample = self._peak_sample
        axes = ["x", "y", "z"]
        dominant_index = max(range(3), key=lambda i: abs(sample.gyro[i]))
        dominant_axis = axes[dominant_index]
        direction = "forehand" if sample.gyro[dominant_index] >= 0 else "backhand"

        event = SwingEvent(
            timestamp=sample.timestamp,
            peak_intensity=self._peak_intensity,
            direction=direction,
            dominant_axis=dominant_axis,
        )
        self._recent_events.append(event)
        self._last_swing_time = sample.timestamp
        self._armed = False
        self._peak_intensity = 0.0
        self._peak_sample = None
        return event

    def recent_swing(self, window_ms):
        """Return the most recent SwingEvent if it happened within the
        last window_ms (measured from now), else None."""
        if not self._recent_events:
            return None
        latest = self._recent_events[-1]
        if (time.monotonic() - latest.timestamp) * 1000 <= window_ms:
            return latest
        return None


def calibrate(read_sample, *, hold_seconds=3, num_swings=5, prompt=print, poll_interval_ms=15):
    """Measure this paddle's noise floor (held still) and typical swing
    peak (practice swings), for unit-agnostic thresholding.

    read_sample: a zero-argument callable returning a fresh IMUSample.
    prompt: callable(str) used to tell the player what to do next -
        defaults to print(), pass a UI callback to show it on screen
        instead.
    """
    poll_interval = poll_interval_ms / 1000.0

    prompt(f"Hold the paddle still for {hold_seconds} seconds...")
    still_gyro_mags = []
    still_accel_mags = []
    end_time = time.monotonic() + hold_seconds
    while time.monotonic() < end_time:
        sample = read_sample()
        still_gyro_mags.append(_magnitude(sample.gyro))
        still_accel_mags.append(_magnitude(sample.accel))
        time.sleep(poll_interval)

    gyro_noise_floor = sum(still_gyro_mags) / len(still_gyro_mags)
    accel_noise_floor = sum(still_accel_mags) / len(still_accel_mags)

    gyro_peaks = []
    accel_peaks = []
    for i in range(num_swings):
        prompt(f"Practice swing {i + 1} of {num_swings} - swing now!")
        peak_gyro = 0.0
        peak_accel = 0.0
        window_end = time.monotonic() + 1.5
        while time.monotonic() < window_end:
            sample = read_sample()
            peak_gyro = max(peak_gyro, _magnitude(sample.gyro))
            peak_accel = max(peak_accel, _magnitude(sample.accel))
            time.sleep(poll_interval)
        gyro_peaks.append(peak_gyro)
        accel_peaks.append(peak_accel)

    calibration = Calibration(
        gyro_noise_floor=gyro_noise_floor,
        gyro_avg_peak=sum(gyro_peaks) / len(gyro_peaks),
        accel_noise_floor=accel_noise_floor,
        accel_avg_peak=sum(accel_peaks) / len(accel_peaks),
    )
    prompt(f"Calibration complete: {calibration}")
    return calibration


def run_in_thread(motor_connection, detector, *, poll_interval_ms=15, on_swing=None):
    """Background thread: continuously read imu_device from the (already
    connected) motor and feed detector.update(). Returns the Thread,
    already started; call the returned stop_event.set() to end it.
    """
    stop_event = threading.Event()
    poll_interval = poll_interval_ms / 1000.0

    def loop():
        while not stop_event.is_set():
            imu = motor_connection.motor.imu_device
            sample = IMUSample(
                timestamp=time.monotonic(),
                accel=(imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ),
                gyro=(imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ),
            )
            event = detector.update(sample)
            if event is not None and on_swing is not None:
                on_swing(event)
            time.sleep(poll_interval)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread, stop_event
