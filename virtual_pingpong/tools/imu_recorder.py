"""Standalone tool: record raw IMU samples from the paddle's Double Motor
to a CSV file, so swing thresholds can be tuned offline without needing
the whole game running. Run with: python3 tools/imu_recorder.py
"""
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from imu.motor_connection import MotorConnection

OUTPUT_PATH = PROJECT_ROOT / "imu_recording.csv"


def main():
    cfg = config.load()
    motor_cfg = cfg.get("motor", {})
    poll_interval = cfg["swing_detector"]["poll_interval_ms"] / 1000.0

    connection = MotorConnection(
        card_color=motor_cfg.get("card_color"),
        card_serial=motor_cfg.get("card_serial"),
        device_notification_delay=cfg["swing_detector"]["poll_interval_ms"],
    )
    print("Connecting to Double Motor...")
    connection.connect()
    print(f"Connected. Recording to {OUTPUT_PATH} - press Ctrl+C to stop.")

    start = time.monotonic()
    try:
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "yaw", "pitch", "roll",
                "accelerometerX", "accelerometerY", "accelerometerZ",
                "gyroscopeX", "gyroscopeY", "gyroscopeZ",
            ])
            while True:
                imu = connection.motor.imu_device
                writer.writerow([
                    f"{time.monotonic() - start:.4f}",
                    imu.yaw, imu.pitch, imu.roll,
                    imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
                    imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ,
                ])
                f.flush()
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        connection.disconnect()
        print(f"\nSaved recording to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
