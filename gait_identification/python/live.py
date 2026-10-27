"""Live classification (MODE = "live" in main.py) - PLAN.md D16.

  1. asks you to stand still for 3 s to measure the gyro offset (D7)
  2. every second, takes the last 2 s of IMU data and classifies it with BOTH
     models, smoothing each with a majority vote over the last 3 predictions
  3. shows the chosen model's smoothed activity on the LED matrix (letter +
     model number), prints both models to the console, and logs every
     prediction to data/live_<time>.csv
  4. publishes every prediction over MQTT and accepts model-switch commands,
     so a laptop UI can show the activity and pick the model (see MQTT below)

Needs models/model1.npz and models/model2.npz (from tools/train.py) next to
this file.
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import *

import gaitlib as g

MODEL = 2  # which model's answer is shown at startup: 1 or 2 (the laptop UI can switch it)

# MQTT: one topic, JSON messages tagged with "type".
#   board -> UI:  {"type": "status", "activity": ..., "model": 1|2, "m1": {...}, "m2": {...}, ...}
#   UI -> board:  {"type": "set_model", "model": 1|2}
# Everyone on the topic sees every message, so each side ignores the types it didn't ask for.
BROKER = "broker.emqx.io"  # test.mosquitto.org was unreachable on 2026-10-26 (tools/mqtt_probe.py)
TOPIC = "ME193/Julian/GaitDetection"

ACCEL_LSB_PER_G = 2048.0  # +/-16 g range
GYRO_LSB_PER_DPS = 16.4  # +/-2000 deg/s range
AXIS_MAP = {"S": "-x", "F": "-y", "L": "+z"}  # measured in stage 1 (PLAN.md D6)
CALIBRATE_S = 3.0

HERE = Path(__file__).resolve().parent
MODELS = {m: g.KNN.load(HERE / "models" / f"model{m}.npz") for m in (1, 2)}
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.csv"

lock = threading.Lock()
incoming = deque()  # (seq, t_ms, ax, ay, az, gx, gy, gz) raw ints, filled by the Bridge thread


def imu_batch(batch: str):
    """Bridge thread: only queue the samples - all processing happens in loop()."""
    try:
        rows = [tuple(int(x) for x in s.split(",")) for s in batch.split(";") if s]
        with lock:
            incoming.extend(rows)
    except Exception as exc:  # the Bridge may swallow exceptions silently, so report them here
        print(f"[ERROR] bad IMU batch ({exc!r}): {batch[:80]!r}", flush=True)


def imu_status(message: str):
    print(f"[MCU] {message}")


Bridge.provide("imu_batch", imu_batch)
Bridge.provide("imu_status", imu_status)

meta = {"units": {"accel_lsb_per_g": ACCEL_LSB_PER_G, "gyro_lsb_per_dps": GYRO_LSB_PER_DPS},
        "axis_map": AXIS_MAP, "gyro_bias_dps": None}
buf = deque(maxlen=g.WINDOW)  # last 2 s of raw samples
state = {"calib": [], "calib_start": None, "new_since_pred": 0, "last_seq": None, "gap_in_buf": 0}
smoothers = {m: g.Smoother() for m in (1, 2)}
shown_model = {"value": MODEL}  # changed at runtime by set_model messages
log = open(LOG_PATH, "w")
log.write("time,model1_raw,model1_smoothed,model2_raw,model2_smoothed,shown_model\n")


def show(activity_name):
    idx = g.CLASSES.index(activity_name) if activity_name in g.CLASSES else -1
    try:
        Bridge.call("show", idx, shown_model["value"])
    except Exception as exc:
        print(f"[WARN] LED update failed: {exc}", flush=True)


def calibrate(samples):
    """Collect CALIBRATE_S of still data, then set the gyro offset. Returns True when done."""
    if state["calib_start"] is None:
        state["calib_start"] = time.monotonic()
        print("\n=== STAND COMPLETELY STILL for 3 s (gyro calibration) ===", flush=True)
    state["calib"].extend(samples)
    if time.monotonic() - state["calib_start"] < CALIBRATE_S:
        return False
    gyro = np.array([s[5:8] for s in state["calib"]], float).mean(axis=0) / GYRO_LSB_PER_DPS
    meta["gyro_bias_dps"] = dict(zip("xyz", gyro.tolist()))
    print(f"Gyro offset (deg/s): x {gyro[0]:+.2f}  y {gyro[1]:+.2f}  z {gyro[2]:+.2f}")
    print(f"Classifying. LED shows Model {shown_model['value']}. Logging to {LOG_PATH}\n", flush=True)
    return True


def classify():
    """Returns ({model: (raw, smoothed)}, cadence_hz)."""
    raw = np.array(buf, float)
    acc, gyr = g.raw_to_body(raw[:, 2:5], raw[:, 5:8], meta)
    out, cadence = {}, 0.0
    for m, knn in MODELS.items():
        f = g.window_features(acc, gyr, m, knn.meta.get("axis", "S"), knn.meta.get("stat", "max"))
        pred = knn.predict(f[None, :])[0]
        out[m] = (pred, smoothers[m].update(pred))
        if m == 2:
            cadence = float(f[g.MODEL2_FEATURE_NAMES.index("cadence_hz")])
    return out, cadence


# ---------------------------------------------------------------- MQTT

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] connected to {BROKER} ({reason_code}), topic {TOPIC}", flush=True)
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    if data.get("type") == "set_model" and data.get("model") in (1, 2):
        shown_model["value"] = data["model"]
        print(f"[MQTT] switched to Model {data['model']}", flush=True)


def publish(out, cadence):
    m = shown_model["value"]
    client.publish(TOPIC, json.dumps({
        "type": "status",
        "time": time.time(),
        "activity": out[m][1],
        "model": m,
        "cadence_hz": round(cadence, 2),
        "m1": {"raw": out[1][0], "smoothed": out[1][1]},
        "m2": {"raw": out[2][0], "smoothed": out[2][1]},
    }))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
try:
    client.connect(BROKER)  # same as the other ME193 MQTT scripts
except OSError as exc:
    # Don't stop the classifier: report it and let paho keep retrying in the background.
    print(f"[MQTT] could not connect to {BROKER}: {exc!r} - classifying without MQTT, retrying", flush=True)
    client.connect_async(BROKER)
client.loop_start()


def loop():
    time.sleep(0.02)
    with lock:
        samples = list(incoming)
        incoming.clear()
    if not samples:
        return

    if meta["gyro_bias_dps"] is None:
        calibrate(samples)
        return

    for s in samples:
        if state["last_seq"] is not None and s[0] != state["last_seq"] + 1:
            state["gap_in_buf"] = g.WINDOW  # a gap: wait until it has left the window
        state["last_seq"] = s[0]
        buf.append(s)
        state["gap_in_buf"] = max(0, state["gap_in_buf"] - 1)
        state["new_since_pred"] += 1

        if len(buf) == g.WINDOW and state["new_since_pred"] >= g.HOP:
            state["new_since_pred"] = 0
            if state["gap_in_buf"]:
                print("  (dropped samples in this window - skipping)", flush=True)
                continue
            out, cadence = classify()
            shown = out[shown_model["value"]][1]
            show(shown)
            publish(out, cadence)
            stamp = time.strftime("%H:%M:%S")
            print(f"{stamp}  >> {shown.upper():8s} <<   "
                  f"M1 {out[1][0]:8s} (smoothed {out[1][1]:8s})   M2 {out[2][0]:8s} (smoothed {out[2][1]:8s})",
                  flush=True)
            log.write(f"{stamp},{out[1][0]},{out[1][1]},{out[2][0]},{out[2][1]},{shown_model['value']}\n")
            log.flush()
