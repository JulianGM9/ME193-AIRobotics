import time

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
TOPIC = "ME193/Rogers"

def on_message(client, userdata, message):
    print(f"Got it back: [{message.topic}] {message.payload.decode()}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect(BROKER)
client.loop_start()

client.subscribe(TOPIC)
time.sleep(1)  # give the subscription time to reach the broker

client.publish(TOPIC, "start")
print(f"Published 'start' to '{TOPIC}' on {BROKER}")

time.sleep(1)  # give the message time to come back before disconnecting

client.loop_stop()
client.disconnect()
