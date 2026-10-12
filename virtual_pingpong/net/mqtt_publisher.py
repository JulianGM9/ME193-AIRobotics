"""Publishes the session's record streak to MQTT. Must never block or
crash the game: the network loop runs in paho's own background thread
(loop_start()), and every publish is wrapped so a broker outage just
leaves `connected` False instead of raising.
"""
import paho.mqtt.client as mqtt


def format_payload(record_streak):
    """The spec requires a plain floating-point string, e.g. "7.0" - no
    JSON, no extra text."""
    return f"{float(record_streak):.1f}"


class MqttPublisher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.connected = False
        self._last_sent = None
        self._last_heartbeat = 0.0

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = (reason_code == 0)
        if self.connected:
            # Re-send the current value (or 0.0 if nothing's been sent yet)
            # so a reconnect doesn't leave subscribers with a stale value.
            self.publish(self._last_sent if self._last_sent is not None else 0.0, force=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False

    def start(self):
        try:
            self._client.connect(self.cfg["broker"], self.cfg["port"])
        except Exception as exc:
            print(f"MQTT: initial connect failed ({exc}); will keep retrying in the background.")
        self._client.loop_start()

    def publish(self, record_streak, force=False):
        if not force and record_streak == self._last_sent:
            return
        self._last_sent = record_streak
        try:
            self._client.publish(self.cfg["topic"], format_payload(record_streak), qos=self.cfg["qos"])
        except Exception as exc:
            print(f"MQTT: publish failed ({exc})")

    def maybe_heartbeat(self, record_streak, now):
        """Call periodically (e.g. once per game loop tick); resends the
        current value once heartbeat_seconds has elapsed, so subscribers
        always have a current value even with no recent change."""
        if now - self._last_heartbeat >= self.cfg["heartbeat_seconds"]:
            self._last_heartbeat = now
            self.publish(record_streak, force=True)

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
