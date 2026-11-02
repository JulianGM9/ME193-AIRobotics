"""Pretend to be the board: publish a synthetic 100 Hz IMU stream over MQTT (PLAN.md testing strategy).

Uses the same topics and formats as board/python/main.py, so scripts/listen.py (and later
the recorder and UI) can be tested without hardware. Don't run it while the real board app
is running, unless both sides use a private topic:  STEPNET_BASE_TOPIC=test/xyz ...

  ../../.venv/bin/python scripts/fake_board.py                 # board standing still, S up
  ../../.venv/bin/python scripts/fake_board.py --drop 0.01     # also lose 1% of batches
  ../../.venv/bin/python scripts/fake_board.py --walk-cycle    # 12 s still / 8 s "walking", repeating
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.io import protocol as p

BATCH_SIZE = 25  # like the real relay: one MQTT message every 0.25 s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", type=float, default=0.0, help="probability of dropping each batch")
    ap.add_argument("--walk-cycle", action="store_true", help="alternate 12 s still and 8 s of leg swing")
    args = ap.parse_args()

    run = "fake_" + time.strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_message(c, userdata, msg):
        data = json.loads(msg.payload)
        if data.get("type") == "ping":
            c.publish(p.TOPIC_STATUS, json.dumps({"type": "pong", "run": run, "id": data.get("id"), "t": data.get("t")}))

    client.on_connect = lambda c, *a: c.subscribe(p.TOPIC_CMD)
    client.on_message = on_message
    client.connect(p.BROKER, p.PORT)
    client.loop_start()
    print(f"Fake board {run} publishing to {p.BASE_TOPIC}/... (Ctrl+C to stop)", flush=True)

    # Standing still with the gait mount (S = -x): accel x = -1 g, plus noise and a gyro bias.
    gravity = np.array([-p.ACCEL_LSB_PER_G, 0, 0])
    gyro_bias = np.array([-7.9, 1.3, 0.0]) * p.GYRO_LSB_PER_DPS
    seq, sent, dropped = 0, 0, 0
    t0 = time.monotonic()
    next_status = t0 + 1.0
    try:
        while True:
            # Publish on a fixed 4 batches/s grid, like the relay on the MCU's 100 Hz timer.
            time.sleep(max(0.0, t0 + (seq + BATCH_SIZE) / p.SAMPLE_RATE_HZ - time.monotonic()))
            n = np.arange(seq, seq + BATCH_SIZE)
            t_ms = (n * 1000 / p.SAMPLE_RATE_HZ).astype(int)
            acc = gravity + rng.normal(0, 8, (BATCH_SIZE, 3))
            gyr = gyro_bias + rng.normal(0, 3, (BATCH_SIZE, 3))
            if args.walk_cycle:
                # ~1 stride/s: the shin swings +/-250 deg/s about the lateral axis (z)
                t = n / p.SAMPLE_RATE_HZ
                walking = (t % 20.0) >= 12.0
                gyr[:, 2] += walking * 250 * np.sin(2 * np.pi * t) * p.GYRO_LSB_PER_DPS
            samples = np.column_stack([n, t_ms, acc, gyr]).round().astype(int)
            seq += BATCH_SIZE
            if rng.random() < args.drop:
                dropped += 1
            else:
                client.publish(p.TOPIC_IMU, json.dumps({"run": run, "t": time.time(), "b": p.format_batch(samples)}))
                sent += BATCH_SIZE
            if time.monotonic() >= next_status:
                client.publish(p.TOPIC_STATUS, json.dumps({
                    "type": "status", "run": run, "online": True, "mcu": "fake MCU", "samples": sent,
                    "gaps": 0, "rate_hz": 100.0, "uptime_s": round(time.monotonic() - t0, 1), "log": "none"}))
                next_status += 1.0
    except KeyboardInterrupt:
        print(f"\nsent {sent} samples, dropped {dropped} batches")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
