"""Publishes a plain-English status line (current streak, best streak,
score) to MQTT. Must never block or crash the game: the network loop runs
in paho's own background thread (loop_start()), and every publish is
wrapped so a broker outage just leaves `connected` False instead of
raising.
"""
import paho.mqtt.client as mqtt


def format_payload(current_streak, record_streak, player_score, opponent_score):
    """Human-readable, not JSON/structured - meant to be read directly off
    an MQTT monitor, not parsed."""
    return (f"Current streak: {current_streak} | Best streak: {record_streak} | "
            f"Score: {player_score}-{opponent_score}")


class MqttPublisher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.connected = False
        self._last_sent = None  # last (current_streak, record_streak, player_score, opponent_score) tuple
        self._last_heartbeat = 0.0

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = (reason_code == 0)
        if self.connected:
            # Re-send the current values (or all zeros if nothing's been
            # sent yet) so a reconnect doesn't leave subscribers stale.
            current_streak, record_streak, player_score, opponent_score = self._last_sent or (0, 0, 0, 0)
            self.publish(current_streak, record_streak, player_score, opponent_score, force=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False

    def start(self):
        try:
            self._client.connect(self.cfg["broker"], self.cfg["port"])
        except Exception as exc:
            print(f"MQTT: initial connect failed ({exc}); will keep retrying in the background.")
        self._client.loop_start()

    def publish(self, current_streak, record_streak, player_score, opponent_score, force=False):
        values = (current_streak, record_streak, player_score, opponent_score)
        if not force and values == self._last_sent:
            return
        self._last_sent = values
        try:
            self._client.publish(self.cfg["topic"], format_payload(*values), qos=self.cfg["qos"])
        except Exception as exc:
            print(f"MQTT: publish failed ({exc})")

    def maybe_heartbeat(self, current_streak, record_streak, player_score, opponent_score, now):
        """Call periodically (e.g. once per game loop tick); resends the
        current values once heartbeat_seconds has elapsed, so subscribers
        always have current values even with no recent change."""
        if now - self._last_heartbeat >= self.cfg["heartbeat_seconds"]:
            self._last_heartbeat = now
            self.publish(current_streak, record_streak, player_score, opponent_score, force=True)

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
