"""Guided data collection (MODE = "collect" in main.py).

Each run of the app records ONE labeled session (PLAN.md D13):
  1. waits for IMU data to arrive from the MCU
  2. walks you through a fixed schedule of activities, printing big cues and
     countdowns in the App Lab console
  3. saves data/session_NN.csv (raw samples) and data/session_NN.json
     (who/when/axis map/gyro bias + the labeled segments)

Set PROTOCOL below: "test" is a ~1.5 minute dry run, "full" is the real
~8 minute session. If something goes wrong during a segment (trip, stop,
cable pulled), note it and set that segment's "ok" to false in the JSON
afterwards - it will then be excluded from training.
"""
import json
import threading
import time
from pathlib import Path

from arduino.app_utils import *

# ---- edit these per session ----
PROTOCOL = "full"  # "test" (short dry run) or "full" (real session)
PERSON = "julian"
MOUNT_NOTES = "UNO Q + MPU-6050 on outside of ankle, jumper wires"
# ---------------------------------

ACCEL_LSB_PER_G = 2048.0  # +/-16 g range
GYRO_LSB_PER_DPS = 16.4  # +/-2000 deg/s range
AXIS_MAP = {"S": "-x", "F": "-y", "L": "+z"}  # measured in stage 1 (PLAN.md D6)

# (label, seconds). "calibrate" = stand still for the gyro bias (D7),
# "transition" = time to change activity, never used for training.
PROTOCOLS = {
    "test": [
        ("transition", 10), ("calibrate", 5), ("standing", 15), ("transition", 5), ("walking", 15),
        ("transition", 5), ("running", 15), ("transition", 10), ("biking", 15),
        ("transition", 5), ("lying_still", 10),
    ],
    "full": [
        ("transition", 10), ("calibrate", 5), ("standing", 90), ("transition", 10), ("walking", 90),
        ("transition", 10), ("running", 90), ("transition", 20), ("biking", 90),
        ("transition", 10), ("lying_still", 30),
    ],
}
INSTRUCTIONS = {
    "calibrate": "STAND COMPLETELY STILL (gyro calibration)",
    "standing": "STAND STILL",
    "walking": "WALK (vary your pace a little)",
    "running": "RUN (jog to fast)",
    "biking": "LIE ON YOUR BACK AND AIR-BIKE (vary cadence)",
    "lying_still": "LIE STILL ON YOUR BACK (posture check - not used for training)",
    "transition": "GET READY FOR THE NEXT ACTIVITY",
}
COUNTDOWN_EVERY_S = 10

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def next_session_number():
    nums = [int(p.stem.split("_")[1]) for p in DATA_DIR.glob("session_*.json")]
    return max(nums, default=0) + 1


SESSION = next_session_number()
CSV_PATH = DATA_DIR / f"session_{SESSION:02d}.csv"
JSON_PATH = DATA_DIR / f"session_{SESSION:02d}.json"

lock = threading.Lock()
state = {"last_seq": None, "last_t_ms": None, "samples": 0, "gaps": 0}
csv_file = open(CSV_PATH, "w")
csv_file.write("seq,t_ms,ax,ay,az,gx,gy,gz\n")


def imu_batch(batch: str):
    """Called by the Bridge (in its own thread) with one batch string from the MCU."""
    try:
        _store_batch(batch)
    except Exception as exc:  # the Bridge may swallow exceptions silently, so report them here
        print(f"[ERROR] bad IMU batch ({exc!r}): {batch[:80]!r}", flush=True)


def _store_batch(batch: str):
    with lock:
        for sample in batch.split(";"):
            if not sample:
                continue
            seq, t_ms = (int(x) for x in sample.split(",")[:2])
            if state["last_seq"] is not None and seq != state["last_seq"] + 1:
                state["gaps"] += seq - state["last_seq"] - 1
            state["last_seq"], state["last_t_ms"] = seq, t_ms
            state["samples"] += 1
            csv_file.write(sample + "\n")


def imu_status(message: str):
    print(f"[MCU] {message}")


Bridge.provide("imu_batch", imu_batch)
Bridge.provide("imu_status", imu_status)


def latest_t_ms():
    with lock:
        return state["last_t_ms"]


def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60, flush=True)


schedule = PROTOCOLS[PROTOCOL]
segments = []  # [label, start_ms, end_ms, ok]
run = {"index": -1, "seg_start": None, "seg_t0": None, "last_countdown": None, "done": False,
       "wait_started": time.monotonic(), "last_wait_msg": time.monotonic()}

print(f"Session {SESSION:02d} ({PROTOCOL} protocol) -> {CSV_PATH.name}, {JSON_PATH.name}")
print("Waiting for IMU data...", flush=True)


def gyro_bias():
    """Mean raw gyro over the calibration segment, in deg/s per sensor axis."""
    cal = next((s for s in segments if s[0] == "calibrate"), None)
    if cal is None:
        return None
    sums, n = [0.0, 0.0, 0.0], 0
    with open(CSV_PATH) as f:
        next(f)
        for line in f:
            vals = [int(x) for x in line.split(",")]
            if cal[1] <= vals[1] <= cal[2]:
                for i in range(3):
                    sums[i] += vals[5 + i]
                n += 1
    return {ax: round(sums[i] / n / GYRO_LSB_PER_DPS, 3) for i, ax in enumerate("xyz")} if n else None


def finish():
    with lock:
        csv_file.flush()
        samples, gaps = state["samples"], state["gaps"]
    meta = {
        "session": SESSION,
        "protocol": PROTOCOL,
        "person": PERSON,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mount_notes": MOUNT_NOTES,
        "axis_map": AXIS_MAP,
        "units": {"accel_lsb_per_g": ACCEL_LSB_PER_G, "gyro_lsb_per_dps": GYRO_LSB_PER_DPS},
        "gyro_bias_dps": gyro_bias(),
        "samples": samples,
        "gaps": gaps,
        "segments": segments,
    }
    JSON_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    banner(f"SESSION {SESSION:02d} DONE - you can stop moving")
    print(f"Samples: {samples}, gaps: {gaps} ({100.0 * gaps / max(1, samples + gaps):.2f}%)")
    print(f"Gyro bias (deg/s): {meta['gyro_bias_dps']}")
    for label, start, end, _ in segments:
        print(f"  {label:12s} {(end - start) / 1000:6.1f} s")
    print(f"Saved {CSV_PATH} and {JSON_PATH}")
    print("If any segment went wrong, set its 'ok' to false in the JSON. Stop the app now.", flush=True)


def loop():
    time.sleep(0.05)
    if run["done"]:
        return
    t_ms = latest_t_ms()
    if t_ms is None:
        now = time.monotonic()
        if now - run["last_wait_msg"] >= 3:
            run["last_wait_msg"] = now
            print(f"Still no IMU data after {now - run['wait_started']:.0f} s - check the MPU-6050 jumper wires "
                  "and that the sketch uploaded.", flush=True)
        return

    now = time.monotonic()
    if run["index"] >= 0:
        label, seconds = schedule[run["index"]]
        elapsed = now - run["seg_start"]
        remaining = seconds - elapsed
        if remaining > 0:
            # Countdown every 10 s, then every second for the last 5.
            tick = int(remaining) + 1
            if tick != run["last_countdown"] and (tick <= 5 or tick % COUNTDOWN_EVERY_S == 0):
                run["last_countdown"] = tick
                print(f"    {label.upper():12s} {tick:3d} s left", flush=True)
            return
        segments.append([label, run["seg_t0"], t_ms, True])

    run["index"] += 1
    if run["index"] >= len(schedule):
        run["done"] = True
        finish()
        return

    label, seconds = schedule[run["index"]]
    upcoming = next((l for l, _ in schedule[run["index"] + 1:] if l != "transition"), None)
    extra = f"   (next: {upcoming})" if label == "transition" and upcoming else ""
    banner(f"{INSTRUCTIONS[label]} - {seconds} s{extra}")
    run.update(seg_start=now, seg_t0=t_ms, last_countdown=None)

