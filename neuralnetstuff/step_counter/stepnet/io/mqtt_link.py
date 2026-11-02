"""Laptop-side MQTT connection to the board (PLAN.md D2).

paho runs its network loop in a background thread. BoardLink turns every message into
an event on a thread-safe queue, so the caller drains events in its own loop:

  ("imu",    run, board_time, samples (n, 8) int array)
  ("status", message dict)        board status, once a second
  ("pong",   rtt_s)               answer to ping(), round trip on the laptop clock
  ("link",   connected: bool, text)
"""
import itertools
import json
import queue
import time

import paho.mqtt.client as mqtt

from stepnet.io import protocol as p


class BoardLink:
    def __init__(self, broker=p.BROKER, port=p.PORT):
        self.broker, self.port = broker, port
        self.events = queue.Queue()
        self._ping_ids = itertools.count()
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self):
        try:
            self._client.connect(self.broker, self.port)
        except OSError as exc:
            self.events.put(("link", False, f"can't reach {self.broker}: {exc}"))
            self._client.connect_async(self.broker, self.port)  # keep retrying in the background
        self._client.loop_start()
        return self

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()

    def ping(self):
        self._client.publish(p.TOPIC_CMD, json.dumps(
            {"type": "ping", "id": next(self._ping_ids), "t": time.time()}))

    # ---- paho callbacks (background thread) ----

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe([(p.TOPIC_IMU, 0), (p.TOPIC_STATUS, 0)])
        self.events.put(("link", True, f"connected to {self.broker}"))

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.events.put(("link", False, f"disconnected ({reason_code})"))

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            if msg.topic == p.TOPIC_IMU:
                self.events.put(("imu", data["run"], data["t"], p.parse_batch(data["b"])))
            elif data.get("type") == "pong":
                self.events.put(("pong", time.time() - data["t"]))
            elif data.get("type") == "status":
                self.events.put(("status", data))
        except (ValueError, KeyError, TypeError) as exc:
            self.events.put(("link", True, f"ignored bad message on {msg.topic}: {exc!r}"))
