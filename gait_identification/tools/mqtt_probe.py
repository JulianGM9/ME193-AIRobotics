"""Find an MQTT broker/port combination this network allows.

    python3 tools/mqtt_probe.py

Tries public test brokers on plain MQTT, TLS, WebSockets and secure WebSockets,
and for each one reports whether a connect + publish + receive round trip works.
"""
import json
import ssl
import threading
import time
import uuid

import paho.mqtt.client as mqtt

TIMEOUT_S = 6
TOPIC = f"ME193/Julian/GaitDetection/probe/{uuid.uuid4().hex[:8]}"

# (broker, port, transport, tls)
CANDIDATES = [
    ("test.mosquitto.org", 1883, "tcp", False),
    ("test.mosquitto.org", 8883, "tcp", True),
    ("test.mosquitto.org", 8080, "websockets", False),
    ("test.mosquitto.org", 8081, "websockets", True),
    ("broker.hivemq.com", 1883, "tcp", False),
    ("broker.hivemq.com", 8000, "websockets", False),
    ("broker.hivemq.com", 8884, "websockets", True),
    ("broker.emqx.io", 1883, "tcp", False),
    ("broker.emqx.io", 8083, "websockets", False),
    ("broker.emqx.io", 8084, "websockets", True),
]


def probe(broker, port, transport, tls):
    got = threading.Event()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport=transport)
    if transport == "websockets":
        client.ws_set_options(path="/mqtt")
    if tls:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.on_connect = lambda c, u, f, rc, p: c.subscribe(TOPIC)
    client.on_subscribe = lambda c, u, mid, rc, p: c.publish(TOPIC, json.dumps({"type": "probe"}))
    client.on_message = lambda c, u, m: got.set()
    start = time.time()
    try:
        client.connect(broker, port, keepalive=10)
    except Exception as exc:
        return f"FAIL  ({type(exc).__name__}: {exc})"
    client.loop_start()
    ok = got.wait(TIMEOUT_S)
    client.loop_stop()
    client.disconnect()
    return f"OK    round trip {time.time() - start:.2f} s" if ok else "FAIL  (connected but no message came back)"


def main():
    print(f"Probing (up to {TIMEOUT_S} s each)...\n")
    for broker, port, transport, tls in CANDIDATES:
        kind = ("wss" if tls else "ws") if transport == "websockets" else ("mqtts" if tls else "mqtt")
        print(f"{broker:20s} {port:5d} {kind:6s} ", end="", flush=True)
        print(probe(broker, port, transport, tls))


if __name__ == "__main__":
    main()
