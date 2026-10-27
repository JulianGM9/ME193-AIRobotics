"""Part 1 MQTT check: watch the gait detector's messages and switch its model.

    python3 tools/mqtt_monitor.py

Prints every status message the board publishes on ME193/Julian/GaitDetection.
Type 1 or 2 and press Enter to tell the board which model to show; q quits.
"""
import json
import sys
import time

import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"  # test.mosquitto.org was unreachable on 2026-10-26 (tools/mqtt_probe.py)
TOPIC = "ME193/Julian/GaitDetection"  # must match python/live.py


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to {BROKER} ({reason_code}). Listening on {TOPIC}")
    print("Type 1 or 2 + Enter to switch model, q to quit.\n")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        print(f"(non-JSON message: {msg.payload[:60]!r})")
        return
    if data.get("type") == "status":
        lag = time.time() - data.get("time", time.time())
        print(f"{time.strftime('%H:%M:%S')}  >> {data['activity'].upper():8s} <<  model {data['model']}  "
              f"cadence {data['cadence_hz']:.2f} Hz   "
              f"M1 {data['m1']['raw']}/{data['m1']['smoothed']}   M2 {data['m2']['raw']}/{data['m2']['smoothed']}   "
              f"(lag {lag:.2f} s)")
    elif data.get("type") == "set_model":
        print(f"-- set_model {data.get('model')} sent")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to {BROKER}...")
    try:
        client.connect(BROKER)  # same as the other ME193 MQTT scripts: blocks until connected
    except OSError as exc:
        sys.exit(f"Could not connect to {BROKER}: {exc!r}\n"
                 "Check the laptop's internet connection. Some networks block MQTT's port 1883.")
    client.loop_start()
    try:
        while True:
            cmd = input().strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("1", "2"):
                client.publish(TOPIC, json.dumps({"type": "set_model", "model": int(cmd)}))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
