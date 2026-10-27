"""
mediapipetest.py

Drive a LEGO Education Double Motor "car" using one hand tracked by MediaPipe.

Control scheme (mirrored webcam view, so it feels like a mirror):
  - Hand detected  -> motors engaged (only while ARMED)
  - No hand seen   -> motors stop (safety default)
  - Vertical hand position (relative to the center of the frame):
        higher (near top)    -> drive forward, faster the higher it is
        lower (near bottom)  -> drive backward, faster the lower it is
        near vertical center -> stop (a deadzone prevents jitter)
  - Horizontal hand position (relative to the center of the frame):
        left of center  -> steer left
        right of center -> steer right
        near center     -> drive straight

  Throttle and steering are combined with differential ("tank") mixing:
        left_speed  = throttle + steering
        right_speed = throttle - steering
  and then scaled down proportionally if either side exceeds MAX_SPEED, so
  turning never silently changes the shape of the turn.

Keys:
    SPACE  arm / disarm the motors (starts DISARMED for safety)
    q      quit

Requirements (already installed in this project's .venv):
    mediapipe 1.0.1, opencv-contrib-python 5.0.0, legoeducation 1.1.1

  IMPORTANT: run this with the project virtualenv, not the system Python:
      .venv/bin/python mediapipetest.py
  The system Python here is 3.14 and does not have these packages installed.

MacOS camera permission:
  The first run asks for camera access for whichever app hosts the terminal
  (Terminal, iTerm, VS Code...). If you see
      "OpenCV: not authorized to capture video"
  grant it in System Settings -> Privacy & Security -> Camera, then fully
  quit and reopen that app. Permission is granted to the host app, not to
  Python, so a restart is required.

MediaPipe on Apple Silicon -- why the settings below are what they are:
  In mediapipe 1.0.1 on macOS, HandLandmarker hard-crashes the process
  (SIGABRT, not a catchable Python exception) unless you use BOTH:
      delegate = BaseOptions.Delegate.GPU
      image format = mp.ImageFormat.SRGBA   (4-channel!)
  The CPU delegate dies with "Check failed: service_ Service is unavailable"
  and the GPU delegate with 3-channel SRGB dies with "unsupported ImageFrame
  format: 1". Both combinations were verified to abort on this machine.

LEGO Education Python API reference:
    https://github.com/LEGO/LEGOEducation/blob/main/doublemotor.md
    https://github.com/LEGO/LEGOEducation/blob/main/constants.md

Before running:
    1. Update CARD_COLOR / CARD_SERIAL to match your Connection Card.
    2. Power on the Double Motor.
    3. If the car misbehaves, adjust INVERT_LEFT / INVERT_RIGHT / SWAP_SIDES
       (see the notes next to those settings).
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)

import legoeducation as le

# --------------------------------------------------------------------------
# Configuration -- edit these for your setup
# --------------------------------------------------------------------------

CARD_COLOR = le.LEGO_COLOR_YELLOW  # Connection Card color
CARD_SERIAL = "1131"               # Connection Card serial number

# Drive-geometry fixes. Start with all False and change only what's wrong:
#   car drives BACKWARD when your hand is high  -> INVERT_DRIVE = True
#   car SPINS IN PLACE instead of driving straight
#       (one motor's gear train is mirrored)    -> INVERT_LEFT or INVERT_RIGHT
#   car turns the WRONG WAY (left/right mirrored) -> SWAP_SIDES = True
INVERT_DRIVE = False
INVERT_LEFT = False
INVERT_RIGHT = False
SWAP_SIDES = False

MAX_SPEED = 70            # cap motor speed (0-100); keep modest for safety
DEADZONE = 0.08           # ignore hand movement this close to center (0-0.5)
SMOOTHING_TAU = 0.12      # seconds; larger = smoother/laggier. 0 disables.
COMMAND_INTERVAL = 0.08   # min seconds between BLE motor commands
KEEPALIVE_INTERVAL = 0.5  # resend the current command at least this often
SEND_THRESHOLD = 2        # only resend if a side changed by >= this many %
CAMERA_INDEX = 0          # change if you have multiple webcams
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_CAMERA_RETRIES = 30   # tolerate this many consecutive dropped frames

# Model file for the MediaPipe Tasks hand-landmark API, downloaded on first run.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MIN_MODEL_BYTES = 1_000_000  # a truncated download is smaller than this

WINDOW_NAME = "LEGO Hand Control"
PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + base of each finger


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def apply_deadzone(value):
    """Map a -0.5..0.5 offset to -1..1, with a dead band around zero."""
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    scaled = (abs(value) - DEADZONE) / (0.5 - DEADZONE)
    return sign * clamp(scaled, 0.0, 1.0)


def mix_tank(throttle, steer):
    """Differential mixing, scaled proportionally so neither side clips.

    Hard-clamping each side independently would quietly change the turn
    ratio (e.g. full-forward + full-left becomes a much sharper turn than
    asked for), so instead scale both sides by the same factor.
    """
    left = throttle + steer
    right = throttle - steer

    peak = max(abs(left), abs(right))
    if peak > 1.0:
        left /= peak
        right /= peak

    left *= MAX_SPEED
    right *= MAX_SPEED

    if INVERT_DRIVE:
        left, right = -left, -right
    if SWAP_SIDES:
        left, right = right, left
    if INVERT_LEFT:
        left = -left
    if INVERT_RIGHT:
        right = -right

    # movement_move_tank() validates -100..100 and raises ValueError outside it.
    return clamp(left, -100, 100), clamp(right, -100, 100)


def ensure_model_downloaded():
    """Download the model if missing, atomically so a partial file can't stick."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_MODEL_BYTES:
        return

    if os.path.exists(MODEL_PATH):
        print("Existing model file looks truncated; re-downloading.")
        os.remove(MODEL_PATH)

    print("Downloading hand-landmark model (one-time, ~8 MB)...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        if os.path.getsize(tmp_path) < MIN_MODEL_BYTES:
            raise RuntimeError("downloaded model is too small to be valid")
        os.replace(tmp_path, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Error: could not download the hand-landmark model: {exc}")
        sys.exit(1)
    print("Model downloaded.")


def create_landmarker():
    """Build the HandLandmarker with the only config that is stable on macOS."""
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
                # Required on macOS -- see the module docstring. The CPU
                # delegate aborts the whole process here.
                delegate=BaseOptions.Delegate.GPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            "Error: could not open the webcam.\n"
            "  - On macOS this is almost always a permissions problem. Grant\n"
            "    camera access to your terminal app in System Settings ->\n"
            "    Privacy & Security -> Camera, then fully quit and reopen it.\n"
            f"  - Otherwise try a different CAMERA_INDEX (currently {CAMERA_INDEX})."
        )
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # isOpened() can lie on macOS: the real test is whether a frame arrives.
    for _ in range(10):
        if cap.read()[0]:
            return cap
    print("Error: the webcam opened but produced no frames (check permissions).")
    cap.release()
    return None


def connect_double_motor():
    print("Connecting to LEGO Double Motor...")
    motor = le.DoubleMotor()
    try:
        motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    except Exception as exc:
        print(f"Error: connecting to the Double Motor failed: {exc}")
        return None

    if not motor.connected:
        print(
            "Error: could not connect to the Double Motor. Check that it is\n"
            "powered on and that CARD_COLOR/CARD_SERIAL match its Connection Card."
        )
        return None

    print("Connected.")
    return motor


def stop_motors(motor):
    """Stop both sides. Returns True if the command went out."""
    try:
        motor.movement_stop(blocking=False)
        return True
    except Exception as exc:
        print(f"Warning: stop command failed: {exc}")
        return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(cap, landmarker, doublemotor):
    smoothed_throttle = 0.0
    smoothed_steer = 0.0
    last_command_time = 0.0
    last_sent = None            # (left, right) actually transmitted, or "stop"
    last_frame_time = time.monotonic()
    last_timestamp_ms = -1
    dropped_frames = 0
    armed = False

    print("Ready. Press SPACE to arm the motors, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            dropped_frames += 1
            if dropped_frames > MAX_CAMERA_RETRIES:
                print("Error: lost the camera feed.")
                return
            continue
        dropped_frames = 0

        # Bail out if the BLE link dies, rather than leaving the car running
        # on its last command with no way to stop it.
        if not doublemotor.connected:
            print("Error: lost the connection to the Double Motor.")
            return

        frame = cv2.flip(frame, 1)  # mirror for intuitive control
        height, width = frame.shape[:2]

        # The GPU delegate needs 4-channel SRGBA; 3-channel SRGB aborts.
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba)
        )

        # detect_for_video() requires STRICTLY increasing timestamps and raises
        # ValueError otherwise, so force the counter forward on ties.
        timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except ValueError as exc:
            print(f"Warning: skipping frame ({exc})")
            continue

        hand_present = bool(result.hand_landmarks)
        raw_throttle = 0.0
        raw_steer = 0.0

        if hand_present:
            landmarks = result.hand_landmarks[0]
            drawing_utils.draw_landmarks(
                frame, landmarks, HandLandmarksConnections.HAND_CONNECTIONS
            )

            # Average a few stable palm points instead of one landmark.
            cx = sum(landmarks[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
            cy = sum(landmarks[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)

            # x, y are normalized 0..1 with (0, 0) at the top-left.
            raw_steer = apply_deadzone(cx - 0.5)   # right of center -> steer right
            raw_throttle = apply_deadzone(0.5 - cy)  # above center -> forward

        # Frame-rate-independent exponential smoothing, so the feel doesn't
        # change when the detector speeds up or slows down.
        now = time.monotonic()
        dt = now - last_frame_time
        last_frame_time = now
        alpha = 1.0 if SMOOTHING_TAU <= 0 else 1.0 - math.exp(-dt / SMOOTHING_TAU)
        smoothed_throttle += (raw_throttle - smoothed_throttle) * alpha
        smoothed_steer += (raw_steer - smoothed_steer) * alpha

        left_speed, right_speed = mix_tank(smoothed_throttle, smoothed_steer)

        should_stop = not armed or not hand_present or (
            abs(left_speed) < 1 and abs(right_speed) < 1
        )
        target = "stop" if should_stop else (round(left_speed), round(right_speed))

        # Only talk to the motor when something actually changed, plus a slow
        # keepalive. Re-sending 12 identical commands a second saturates the
        # BLE link for no benefit and adds latency to the commands that matter.
        if last_sent == "stop" or target == "stop":
            changed = target != last_sent
        else:
            changed = (
                abs(target[0] - last_sent[0]) >= SEND_THRESHOLD
                or abs(target[1] - last_sent[1]) >= SEND_THRESHOLD
            )
        due = now - last_command_time >= KEEPALIVE_INTERVAL

        if (changed or due) and now - last_command_time >= COMMAND_INTERVAL:
            last_command_time = now
            try:
                if target == "stop":
                    stop_motors(doublemotor)
                else:
                    # One tank command instead of two motor_run() calls: it
                    # takes signed speeds (so no direction juggling), and
                    # blocking=False keeps a dropped BLE ack from stalling
                    # the video loop.
                    doublemotor.movement_move_tank(
                        target[0], target[1], blocking=False
                    )
                last_sent = target
            except Exception as exc:
                print(f"Warning: motor command failed: {exc}")
                last_sent = None

        # --- on-screen debug overlay ---
        if not armed:
            status, color = "DISARMED - press SPACE", (0, 165, 255)
        elif hand_present:
            status, color = "DRIVING", (0, 255, 0)
        else:
            status, color = "NO HAND - STOPPED", (0, 0, 255)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            frame,
            f"throttle: {smoothed_throttle * MAX_SPEED:+.0f}  "
            f"steer: {smoothed_steer * MAX_SPEED:+.0f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"L: {left_speed:+.0f}  R: {right_speed:+.0f}",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.line(frame, (width // 2, 0), (width // 2, height), (80, 80, 80), 1)
        cv2.line(frame, (0, height // 2), (width, height // 2), (80, 80, 80), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return
        if key == ord(" "):
            armed = not armed
            print("ARMED" if armed else "DISARMED")
            if not armed:
                stop_motors(doublemotor)
                last_sent = "stop"
                last_command_time = now

        # Treat closing the window as quitting.
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return


def main():
    # Set up the camera and the detector BEFORE connecting over BLE, so a
    # failure in either one can't leave a connected motor behind.
    ensure_model_downloaded()

    cap = open_camera()
    if cap is None:
        sys.exit(1)

    try:
        landmarker = create_landmarker()
    except Exception as exc:
        print(f"Error: could not create the hand landmarker: {exc}")
        cap.release()
        sys.exit(1)

    doublemotor = connect_double_motor()
    if doublemotor is None:
        landmarker.close()
        cap.release()
        sys.exit(1)

    try:
        run(cap, landmarker, doublemotor)
    finally:
        print("Shutting down...")
        # Stop first, and block this time -- we want the command to land
        # before we tear the connection down.
        try:
            doublemotor.movement_stop()
        except Exception as exc:
            print(f"Warning: final stop failed: {exc}")
        try:
            doublemotor.disconnect()
        except Exception as exc:
            print(f"Warning: disconnect failed: {exc}")
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
