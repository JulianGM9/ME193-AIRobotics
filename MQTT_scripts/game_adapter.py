import queue

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
GAME_TOPIC = "ME193/Julian"
ANNOUNCE_TOPIC = "ME193/Rogers"

# What each role is called in an announcement, and who its opponent is.
ROLE_WORD = {"goalie": "goalie", "shooter": "ball"}
OPPONENT = {"goalie": "shooter", "shooter": "goalie"}

incoming_messages = queue.Queue()


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(GAME_TOPIC)
    client.subscribe(ANNOUNCE_TOPIC)


def on_message(client, userdata, message):
    incoming_messages.put((message.topic, message.payload.decode().strip()))


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER)
    client.loop_start()

    role = None
    print(f"Waiting for a role on '{GAME_TOPIC}' ('goalie' or 'shooter')...")

    try:
        while True:
            topic, payload = incoming_messages.get()
            lower = payload.lower()

            if topic == ANNOUNCE_TOPIC:
                print(f"[{ANNOUNCE_TOPIC}] {payload}")
                continue

            if role is None:
                if lower in ("goalie", "shooter"):
                    role = lower
                    print(f"We are the {role}. Opponent is the {OPPONENT[role]}.")
                continue

            if lower == "win":
                announcement = f"{ROLE_WORD[role]} wins"
                client.publish(ANNOUNCE_TOPIC, announcement)
                print(f"We won as {role}. Broadcast '{announcement}' to '{ANNOUNCE_TOPIC}'.")
                role = None
                print(f"Waiting for a role on '{GAME_TOPIC}'...")
            elif lower == "lose":
                opponent_role = OPPONENT[role]
                announcement = f"{ROLE_WORD[opponent_role]} wins"
                client.publish(ANNOUNCE_TOPIC, announcement)
                print(f"We lost as {role}. Broadcast '{announcement}' to '{ANNOUNCE_TOPIC}'.")
                role = None
                print(f"Waiting for a role on '{GAME_TOPIC}'...")
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
