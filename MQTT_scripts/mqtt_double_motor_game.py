import ast
import queue
import time

import legoeducation as le
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
TOPIC = "ME193/Julian"
ANNOUNCE_TOPIC = "ME193"

STATE_WAITING = "waiting"
STATE_PLAYING = "playing"

# Shooter/ball mode: reflected light level (0-100) above which the shooter
# (acting as the ball) considers itself "found" and loses. Edit this value
# to tune sensitivity.
LIGHT_LOSE_THRESHOLD = 20

LIGHT_POLL_INTERVAL = 0.05

incoming_messages = queue.Queue()


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    incoming_messages.put(message.payload.decode().strip())


def parse_drive_command(payload):
    """Return (speed, turn) floats if payload is a driving duple, else None."""
    try:
        value = ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        parts = payload.strip("()[]").split(",")
        if len(parts) != 2:
            return None
        try:
            value = (float(parts[0]), float(parts[1]))
        except ValueError:
            return None

    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def drive(motor, speed, turn):
    speed = clamp(speed)
    turn = clamp(turn)
    left = clamp(speed + turn) * 100
    right = clamp(speed - turn) * 100
    motor.movement_move_tank(left, right, blocking=False)
    print(f"Driving: speed={speed:+.2f} turn={turn:+.2f} -> left={left:+.0f}% right={right:+.0f}%")


def celebrate(motor):
    print("Won! Celebrating.")
    for frequency in (440, 554, 659, 880):
        motor.beep(frequency=frequency, count=1, blocking=True)
    motor.movement_move_tank(60, -60, blocking=False)
    time.sleep(2)
    motor.movement_stop()


def mourn(motor):
    print("Lost. Playing a sad tune.")
    motor.movement_stop()
    for frequency in (1200, 1000, 800, 600):
        motor.beep(frequency=frequency, count=1, blocking=True)


def main():
    motor = le.DoubleMotor()
    print("Connecting to Double Motor...")
    motor.connect()
    print("Connected.")

    colorsensor = le.ColorSensor()
    print("Connecting to Color Sensor...")
    colorsensor.connect()
    print("Connected.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER)
    client.loop_start()

    state = STATE_WAITING
    role = None
    ball_lose_triggered = False
    print(f"Waiting for a role on '{TOPIC}' ('goalie' or 'shooter')...")

    try:
        while True:
            try:
                payload = incoming_messages.get(timeout=LIGHT_POLL_INTERVAL)
            except queue.Empty:
                payload = None

            if payload is not None:
                if state == STATE_WAITING:
                    if payload.lower() in ("goalie", "shooter"):
                        role = payload.lower()
                        state = STATE_PLAYING
                        ball_lose_triggered = False
                        print(f"Assigned role: {role}. Entering playing state.")
                else:
                    lower = payload.lower()
                    if lower == "win":
                        celebrate(motor)
                        state = STATE_WAITING
                        role = None
                        print(f"Waiting for a role on '{TOPIC}'...")
                    elif lower == "lose":
                        mourn(motor)
                        state = STATE_WAITING
                        role = None
                        print(f"Waiting for a role on '{TOPIC}'...")
                    else:
                        drive_command = parse_drive_command(payload)
                        if drive_command is not None:
                            drive(motor, *drive_command)
                        else:
                            print(f"Ignoring unrecognized message: {payload!r}")

            if state == STATE_PLAYING and role == "shooter" and not ball_lose_triggered:
                if colorsensor.sensor.reflection > LIGHT_LOSE_THRESHOLD:
                    ball_lose_triggered = True
                    print(f"Light level {colorsensor.sensor.reflection} exceeded threshold. Ball caught.")
                    client.publish(ANNOUNCE_TOPIC, "ball loses")
                    client.publish(TOPIC, "lose")
    except KeyboardInterrupt:
        pass
    finally:
        motor.movement_stop()
        client.loop_stop()
        client.disconnect()
        motor.disconnect()
        colorsensor.disconnect()


if __name__ == "__main__":
    main()
