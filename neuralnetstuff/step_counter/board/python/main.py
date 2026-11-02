"""Step counter - Linux side: a thin relay (PLAN.md D1, D2).

  1. receives IMU batches from the MCU over the Bridge
  2. appends every sample to a local backup log, data/stream_<run>.csv (gap-free
     copy for when MQTT drops messages; pull it with adb)
  3. publishes the batches to MQTT  .../imu, grouped every PUBLISH_EVERY_S
  4. publishes a status message every second to  .../status
  5. answers {"type": "ping"} commands on  .../cmd  with a pong on .../status,
     so the laptop can measure round-trip latency without synced clocks

The MQTT topics and message formats must match stepnet/io/protocol.py on the laptop.
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt
from arduino.app_utils import *

BROKER = "broker.emqx.io"  # same broker as the other ME193 projects
BASE_TOPIC = "ME193/Julian/StepCounter"
TOPIC_IMU = BASE_TOPIC + "/imu"
TOPIC_STATUS = BASE_TOPIC + "/status"
TOPIC_CMD = BASE_TOPIC + "/cmd"
STATUS_EVERY_S = 1.0
# broker.emqx.io starts dropping messages above ~8/s per client (measured 2026-11-02), so
# the MCU's 10 batches/s are grouped into 4 messages/s (~25 samples each).
PUBLISH_EVERY_S = 0.25

LED_NO_MQTT, LED_STREAMING = 0, 1  # show() codes in the sketch
LED_CALL_TIMEOUT_S = 2.0

RUN_ID = time.strftime("%Y%m%d_%H%M%S")  # new every app start, so the laptop can spot restarts
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / f"stream_{RUN_ID}.csv"

lock = threading.Lock()
incoming = deque()  # batch strings, filled by the Bridge thread
state = {"mcu": "waiting for MCU", "samples": 0, "gaps": 0, "last_seq": None,
         "interval_samples": 0, "last_status": time.monotonic(), "mqtt": False,
         "led": None, "start": time.monotonic(), "pending": [], "last_publish": time.monotonic()}


def imu_batch(batch: str):
    """Bridge thread: only queue the batch - all work happens in loop()."""
    with lock:
        incoming.append(batch)


def imu_status(message: str):
    print(f"[MCU] {message}", flush=True)
    state["mcu"] = message


Bridge.provide("imu_batch", imu_batch)
Bridge.provide("imu_status", imu_status)

log = open(LOG_PATH, "w")
log.write("seq,t_ms,ax,ay,az,gx,gy,gz\n")
print(f"Run {RUN_ID}: logging to {LOG_PATH}", flush=True)


# ---------------------------------------------------------------- MQTT

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] connected to {BROKER} ({reason_code}), base topic {BASE_TOPIC}", flush=True)
    client.subscribe(TOPIC_CMD)
    state["mqtt"] = True


def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] disconnected ({reason_code}) - still logging locally, retrying", flush=True)
    state["mqtt"] = False


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    if data.get("type") == "ping":
        client.publish(TOPIC_STATUS, json.dumps({"type": "pong", "run": RUN_ID,
                                                 "id": data.get("id"), "t": data.get("t")}))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
# If the board vanishes, the broker tells the laptop on our behalf.
client.will_set(TOPIC_STATUS, json.dumps({"type": "status", "run": RUN_ID, "online": False}))
try:
    client.connect(BROKER)
except OSError as exc:
    # Don't stop relaying to the local log: let paho keep retrying in the background.
    print(f"[MQTT] could not connect to {BROKER}: {exc!r} - logging locally, retrying", flush=True)
    client.connect_async(BROKER)
client.loop_start()


def count_samples(batch):
    """Update sample and gap counters from one batch (seq is the first field of each sample)."""
    for sample in batch.split(";"):
        seq = int(sample.split(",", 1)[0])
        if state["last_seq"] is not None and seq > state["last_seq"] + 1:
            state["gaps"] += seq - state["last_seq"] - 1
        state["last_seq"] = seq
        state["samples"] += 1
        state["interval_samples"] += 1


def publish_status():
    now = time.monotonic()
    dt = now - state["last_status"]
    client.publish(TOPIC_STATUS, json.dumps({
        "type": "status",
        "run": RUN_ID,
        "online": True,
        "mcu": state["mcu"],
        "samples": state["samples"],
        "gaps": state["gaps"],
        "rate_hz": round(state["interval_samples"] / dt, 2),
        "uptime_s": round(now - state["start"], 1),
        "log": LOG_PATH.name,
    }))
    state["last_status"] = now
    state["interval_samples"] = 0


def led_worker():
    """Keep the LED matrix in sync with the MQTT state, in its own thread.

    Bridge.call blocks until the MCU answers (or times out), so it must never run in the
    relay loop. It also waits for the first IMU sample: until then the sketch may still be
    flashing or booting, and a call to show() would just hang until the timeout.
    """
    while True:
        time.sleep(0.5)
        if state["samples"] == 0:
            continue
        want = LED_STREAMING if state["mqtt"] else LED_NO_MQTT
        if want == state["led"]:
            continue
        try:
            Bridge.call("show", want, timeout=LED_CALL_TIMEOUT_S)
            state["led"] = want
        except Exception as exc:  # retried on the next pass
            print(f"[WARN] LED update failed: {exc}", flush=True)


threading.Thread(target=led_worker, daemon=True).start()


def loop():
    time.sleep(0.02)
    with lock:
        batches = list(incoming)
        incoming.clear()

    for batch in batches:
        try:
            count_samples(batch)
        except ValueError:
            print(f"[ERROR] bad IMU batch: {batch[:80]!r}", flush=True)
            continue
        log.write(batch.replace(";", "\n") + "\n")
        state["pending"].append(batch)

    if state["pending"] and time.monotonic() - state["last_publish"] >= PUBLISH_EVERY_S:
        client.publish(TOPIC_IMU, json.dumps({"run": RUN_ID, "t": time.time(), "b": ";".join(state["pending"])}))
        state["pending"] = []
        state["last_publish"] = time.monotonic()

    if time.monotonic() - state["last_status"] >= STATUS_EVERY_S:
        log.flush()
        publish_status()


App.run(user_loop=loop)
