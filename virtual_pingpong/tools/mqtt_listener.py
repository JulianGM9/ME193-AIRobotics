"""Standalone tool: subscribe to the score topic and print every value
received, to verify net/mqtt_publisher.py is actually publishing.
Run with: python3 tools/mqtt_listener.py
"""
import sys
from pathlib import Path

import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def main():
    cfg = config.load()["mqtt"]

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(cfg["topic"])
        print(f"Connected to {cfg['broker']}, listening on '{cfg['topic']}'")

    def on_message(client, userdata, msg):
        print(f"{msg.topic}: {msg.payload.decode()}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg["broker"], cfg["port"])

    print("Listening... Ctrl+C to quit.")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
