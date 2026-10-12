"""Standalone tool: run calibration, then live swing detection against the
real Double Motor, printing each detected SwingEvent as it happens. For
validating the imu/swing_detector subsystem end-to-end, on real hardware,
before building the rest of the game on top of it.
Run with: python3 tools/swing_test.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from imu.motor_connection import MotorConnection
from imu.swing_detector import IMUSample, SwingDetector, calibrate, run_in_thread


def make_read_sample(connection):
    def read_sample():
        imu = connection.motor.imu_device
        return IMUSample(
            timestamp=time.monotonic(),
            accel=(imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ),
            gyro=(imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ),
        )
    return read_sample


def main():
    cfg = config.load()
    motor_cfg = cfg.get("motor", {})
    sd_cfg = cfg["swing_detector"]

    connection = MotorConnection(
        card_color=motor_cfg.get("card_color"),
        card_serial=motor_cfg.get("card_serial"),
        device_notification_delay=sd_cfg["poll_interval_ms"],
    )
    print("Connecting to Double Motor...")
    connection.connect()
    print("Connected.")

    calibration = calibrate(
        make_read_sample(connection),
        hold_seconds=sd_cfg["calibration_hold_seconds"],
        num_swings=sd_cfg["calibration_num_swings"],
        poll_interval_ms=sd_cfg["poll_interval_ms"],
    )

    sd_cfg["calibration"] = {
        "gyro_noise_floor": calibration.gyro_noise_floor,
        "gyro_avg_peak": calibration.gyro_avg_peak,
        "accel_noise_floor": calibration.accel_noise_floor,
        "accel_avg_peak": calibration.accel_avg_peak,
    }
    config.save(cfg)
    print("Calibration saved to config.yaml.")

    detector = SwingDetector(
        smoothing_alpha=sd_cfg["smoothing_alpha"],
        gravity_ema_alpha=sd_cfg["gravity_ema_alpha"],
        refractory_ms=sd_cfg["refractory_ms"],
        threshold_fraction=sd_cfg["threshold_fraction"],
        calibration=calibration,
    )

    def on_swing(event):
        print(f"SWING  intensity={event.peak_intensity:.2f}  "
              f"direction={event.direction}  axis={event.dominant_axis}")

    print("Live detection running - swing the paddle a few times. Ctrl+C to stop.")
    thread, stop_event = run_in_thread(
        connection, detector,
        poll_interval_ms=sd_cfg["poll_interval_ms"],
        on_swing=on_swing,
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        connection.disconnect()


if __name__ == "__main__":
    main()
