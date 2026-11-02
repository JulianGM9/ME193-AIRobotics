"""Live check of the board stream (milestone M1).

Run from neuralnetstuff/step_counter/:
  ../../.venv/bin/python scripts/listen.py             # stats every second, Ctrl+C to stop
  ../../.venv/bin/python scripts/listen.py --test 300  # 5 minutes, then PASS/FAIL (PLAN.md M1)

Every second it prints the sample rate, gaps, round-trip latency, and the mean
acceleration and gyro in the body frame (S = up the shin, F = forward, L = toward your
left, which is the inner side of the right ankle).
Tilt test: point each body axis at the ceiling in turn. That axis should read +1.00 g.
"""
import argparse
import queue
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.io.mqtt_link import BoardLink
from stepnet.io.protocol import SAMPLE_RATE_HZ
from stepnet.io.stream import SeqTracker
from stepnet.preprocess import BODY_AXES, to_body, to_physical

PRINT_EVERY_S = 1.0
PING_EVERY_S = 2.0
RATE_TOLERANCE_HZ = 0.5  # PLAN.md M1 pass criterion
AXIS_HINTS = {"S": "shin axis (arrow toward knee)", "F": "forward side", "L": "left = inner side of right ankle"}
NO_DATA_HINT_S = 15
NO_DATA_HINT = """
  No IMU data for a while, but MQTT works. If the board's console shows no [MCU] lines,
  the MCU is probably stuck at startup: App Lab re-flashed it while it was mid-way
  through an I2C read, and the MPU-6050 is still holding the bus (PLAN.md change log).
  Fix: unplug the MPU's VCC jumper for 2 s (or power-cycle the board), then press Run again.
"""


def body_text(vec, fmt):
    return " ".join(f"{a} {v:{fmt}}" for a, v in zip(BODY_AXES, vec))


def gravity_text(acc):
    """Which body axis gravity is on, e.g. 'S+ up: shin axis (arrow toward knee)'."""
    i = int(np.argmax(np.abs(acc)))
    up = acc[i] > 0
    return f"{BODY_AXES[i]}{'+' if up else '-'} {'up' if up else 'down'}: {AXIS_HINTS[BODY_AXES[i]]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=float, metavar="SECONDS",
                    help="run for this long after the first sample, then print PASS/FAIL")
    args = ap.parse_args()

    link = BoardLink().start()
    print(f"Listening on {link.broker} ... (Ctrl+C to stop)", flush=True)
    tracker = SeqTracker()
    interval = []  # samples since the last printed line
    rtts = deque(maxlen=10)
    first = {"rx": None, "n": 0}  # first batch: its arrival time and size (excluded from the rate)
    last_rx = None
    board = {"mcu": None, "gaps_start": None, "gaps_end": None}
    next_print = next_ping = started = time.monotonic()
    deadline = None
    hinted = False

    try:
        while True:
            try:
                event = link.events.get(timeout=0.05)
            except queue.Empty:
                event = None

            if event and event[0] == "imu":
                _, run, _, samples = event
                new_missing = tracker.update(run, samples[:, 0])
                if new_missing:
                    print(f"  ! {new_missing} samples missing before seq {samples[0, 0]}", flush=True)
                last_rx = time.monotonic()
                if first["rx"] is None:
                    first.update(rx=last_rx, n=len(samples))
                    if args.test:
                        deadline = last_rx + args.test
                        print(f"First sample received - testing for {args.test:.0f} s", flush=True)
                interval.append(samples)
            elif event and event[0] == "status":
                msg = event[1]
                if not msg.get("online", True):
                    print(f"[BOARD] run {msg.get('run')} went offline", flush=True)
                    continue
                if msg.get("mcu") != board["mcu"]:
                    board["mcu"] = msg.get("mcu")
                    print(f"[BOARD] run {msg.get('run')}: MCU says {board['mcu']!r}, "
                          f"backup log {msg.get('log')}", flush=True)
                if first["rx"] is not None:
                    if board["gaps_start"] is None:
                        board["gaps_start"] = msg.get("gaps", 0)
                    board["gaps_end"] = msg.get("gaps", 0)
            elif event and event[0] == "pong":
                rtts.append(event[1])
            elif event and event[0] == "link":
                print(f"[MQTT] {event[2]}", flush=True)

            now = time.monotonic()
            if now >= next_ping:
                link.ping()
                next_ping = now + PING_EVERY_S
            if now >= next_print:
                print_line(interval, tracker, rtts, now - next_print + PRINT_EVERY_S)
                no_data_s = now - (last_rx or started)
                if no_data_s > NO_DATA_HINT_S and rtts and not hinted:
                    print(NO_DATA_HINT, flush=True)
                    hinted = True
                elif no_data_s < 1:
                    hinted = False
                interval = []
                next_print = now + PRINT_EVERY_S
            if deadline is not None and now >= deadline:
                break
    except KeyboardInterrupt:
        print()
    finally:
        link.stop()

    summarize(args, tracker, first, last_rx, rtts, board)


def print_line(interval, tracker, rtts, dt):
    stamp = time.strftime("%H:%M:%S")
    rtt = f"{statistics.median(rtts) * 1000:4.0f} ms" if rtts else "   - ms"
    if not interval:
        print(f"{stamp}  no IMU data   rtt {rtt}", flush=True)
        return
    samples = np.concatenate(interval)
    acc, gyr = to_physical(samples)
    acc_b, gyr_b = to_body(acc).mean(axis=0), to_body(gyr).mean(axis=0)
    print(f"{stamp}  rx {len(samples) / dt:5.1f} Hz  missing {tracker.missing}  rtt {rtt} | "
          f"acc {body_text(acc_b, '+.2f')} g  |a| {np.linalg.norm(acc_b):.2f} -> {gravity_text(acc_b):38s} | "
          f"gyro {body_text(gyr_b, '+6.1f')} dps", flush=True)


def summarize(args, tracker, first, last_rx, rtts, board):
    print("\n==== summary ====")
    if first["rx"] is None:
        print("No IMU data received. Is the board app running, and does its LED show 'C'?")
        return
    elapsed = last_rx - first["rx"]
    rate = (tracker.received - first["n"]) / elapsed if elapsed > 0 else float("nan")
    board_gaps = (board["gaps_end"] or 0) - (board["gaps_start"] or 0)
    print(f"duration           {elapsed:8.1f} s")
    print(f"samples received   {tracker.received:8d}")
    print(f"rate (laptop clock){rate:8.2f} Hz")
    print(f"missing samples    {tracker.missing:8d}   (of which lost on the board, before MQTT: {board_gaps})")
    print(f"board restarts     {tracker.restarts:8d}")
    if rtts:
        print(f"round trip (median){statistics.median(rtts) * 1000:8.0f} ms")
    if not args.test:
        return

    checks = [
        ("ran for the full test", elapsed >= args.test - 1.0),
        (f"rate {SAMPLE_RATE_HZ:.0f} +/- {RATE_TOLERANCE_HZ} Hz", abs(rate - SAMPLE_RATE_HZ) <= RATE_TOLERANCE_HZ),
        ("0 missing samples", tracker.missing == 0),
        ("0 board restarts", tracker.restarts == 0),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("M1 stream test:", "PASS" if all(ok for _, ok in checks) else "FAIL")


if __name__ == "__main__":
    main()
