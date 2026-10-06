import time

import paho.mqtt.client as mqtt
from arduino.app_utils import *

# Must match BROKER and the blue topic in track_both_minifigs.py on the computer.
BROKER = "test.mosquitto.org"
TOPIC = "minifig_blue"

# PD gains on x error (x in [-1, 1], 0 = centered). Edit these to tune:
# raise KP if the correction is too weak/slow, raise KD if it overshoots
# and oscillates back and forth around center.
KP = 75.0
KD = 60.0

MAX_SPEED = 255  # PWM clamp, do not exceed 255

# PWM below this may not be enough to overcome motor/gearbox friction, so
# a weak correction just sits still instead of actually moving. Edit this
# if the car still doesn't move at small errors, or is jumpy very close
# to center.
MIN_SPEED = 90

# Hysteresis around center, so the car doesn't flicker on and off right at
# a single threshold. Once stopped (|x| < CENTER_ZONE), it stays stopped
# until the minifig drifts past the wider RESUME_ZONE before correcting
# again. RESUME_ZONE must be >= CENTER_ZONE.
CENTER_ZONE = 0.10
RESUME_ZONE = 0.18

# Set to -1 if testing shows the car drives the wrong way (away from
# center instead of toward it).
DIRECTION_SIGN = 1

TIMEOUT = 0.5  # stop driving if no minifig message arrives for this long

latest = {"x": None, "time": 0.0}
prev_error = 0.0
have_prev = False  # True once prev_error holds a real reading, not a reset placeholder
centered = False  # True while stopped in the hysteresis zone around center
last_tick = time.time()


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(TOPIC)
    print(f"Connected to {BROKER}, listening on '{TOPIC}'")


def parse_x(payload):
    """Pull x out of a "(x, y)" payload string; return None if unparseable."""
    try:
        parts = payload.strip("()").split(",")
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def on_message(client, userdata, msg):
    x = parse_x(msg.payload.decode().strip())
    if x is not None:
        latest["x"] = x
        latest["time"] = time.time()


def apply_min_speed(value):
    """Bump a nonzero speed up to MIN_SPEED (preserving sign) so a weak
    correction isn't too small for the motor to actually turn."""
    if value == 0:
        return 0
    if abs(value) < MIN_SPEED:
        return MIN_SPEED if value > 0 else -MIN_SPEED
    return value


def control(x, dt):
    """Return a single drive speed (-255..255) to apply to both motors so
    the car drives straight forward/backward to bring x to 0."""
    global prev_error, have_prev, centered
    if have_prev and dt > 0:
        derivative = (x - prev_error) / dt
    else:
        # First reading after startup or after re-acquiring a lost minifig -
        # there's no real previous sample to compare against, so treat the
        # derivative as 0 instead of spiking off a stale/reset prev_error.
        derivative = 0.0
    prev_error = x
    have_prev = True

    if centered:
        if abs(x) < RESUME_ZONE:
            return 0
        centered = False  # drifted far enough out to start correcting again
    elif abs(x) < CENTER_ZONE:
        centered = True
        return 0

    speed = DIRECTION_SIGN * (KP * x + KD * derivative)
    speed = max(-MAX_SPEED, min(MAX_SPEED, speed))
    return apply_min_speed(int(speed))


def set_motors(m1, m2):
    """Call the sketch's set_motors, tolerating the bridge not being ready
    yet (e.g. the microcontroller side is still booting)."""
    try:
        Bridge.call("set_motors", m1, m2)
    except ValueError as err:
        print(f"set_motors bridge call failed (will keep retrying): {err}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()


def loop():
    global have_prev, centered, last_tick
    now = time.time()
    dt = now - last_tick
    last_tick = now

    x = latest["x"]
    found = x is not None and now - latest["time"] <= TIMEOUT

    if not found:
        set_motors(0, 0)
        have_prev = False  # next reading is a fresh start, not a continuation
        centered = False
        print("No minifig detected - stopped.")
    else:
        speed = control(x, dt)
        set_motors(speed, speed)
        print(f"x={x:+.2f} -> motors {speed}, {speed}")

    time.sleep(0.05)


# Give the microcontroller side a moment to boot and register its Bridge
# methods before the loop starts calling them - without this, the first
# set_motors call can fail with "method not available" and crash the app.
time.sleep(2)
App.run(user_loop=loop)
