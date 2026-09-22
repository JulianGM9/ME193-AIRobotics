"""
apriltagparking_multicam.py

Same as apriltagparking.py (see that file for the full parking control
scheme), but the camera source can be switched live between your Mac's
built-in webcam and your iPhone via Continuity Camera, without restarting
the script.

Press 'c' at any time to toggle between the two camera sources in
CAMERA_SOURCES below. If the new source fails to open, the script prints
a warning and keeps using whichever camera was already running -- it
never leaves the robot without a working camera.

--------------------------------------------------------------------------
Setting up Continuity Camera (uses your iPhone as a webcam)
--------------------------------------------------------------------------
I can't test this myself (no iPhone/Continuity Camera hardware here), so
verify each step against what you actually see on your machine. As of
recent macOS/iOS versions, Continuity Camera needs:

  1. macOS Ventura (13) or later on the Mac, iOS 16 or later on the iPhone.
  2. Both devices signed into the same Apple ID, with two-factor
     authentication enabled on that Apple ID.
  3. Wi-Fi AND Bluetooth turned on on both devices (they don't need to be
     on the same Wi-Fi network, but both radios need to be on).
  4. Handoff enabled on both: Mac System Settings -> General -> AirDrop &
     Handoff; iPhone Settings -> General -> AirPlay & Handoff.
  5. The iPhone near the Mac (Bluetooth range) and awake (screen doesn't
     need to stay on once connected, but shouldn't be face-down/covered).
  6. Once connected, the iPhone shows a small "Connected as a camera"
     indicator, and any Mac app that lists cameras -- including OpenCV's
     cv2.VideoCapture, since it goes through the system's AVFoundation
     camera list on macOS -- will offer it as a source. No special code
     is needed beyond picking the right index, which is what this file's
     toggle does.
  7. A stand/mount for the iPhone helps a lot here, since it needs to
     stay aimed at the parking area -- any stable mount works, an
     official Belkin/Apple mount is not required.

Camera index caveat: cv2.VideoCapture only takes a numeric index, and
macOS does not guarantee *which* index is the iPhone vs. the built-in
webcam -- it can depend on connection order and isn't guaranteed stable
across reboots/reconnects. PROBE_CAMERAS_ON_START (below) runs a quick
scan at startup that opens each index briefly and prints its resolution,
which is usually enough to tell them apart (the iPhone typically reports
a very different resolution than a laptop's built-in camera). Update
CAMERA_SOURCES with whatever indices you see, then use 'c' to confirm
visually in the preview window which one is active.

Keys:
    c      toggle between camera sources (see CAMERA_SOURCES)
    SPACE  arm / disarm the motors (starts DISARMED for safety)
    q      quit

Run with the project virtualenv:
    .venv/bin/python apriltagparking_multicam.py
"""

import sys
import time

import cv2
import numpy as np

import legoeducation as le
from lelib import doubleMotor

# --------------------------------------------------------------------------
# Configuration -- edit these for your setup
# --------------------------------------------------------------------------

CARD_COLOR = le.LEGO_COLOR_YELLOW  # Connection Card color
CARD_SERIAL = "1131"               # Connection Card serial number

# Drive-geometry fixes. Start with all False and change only what's wrong
# -- see the module docstring for how to tell which one you need.
INVERT_LEFT = False
INVERT_RIGHT = False
REVERSE_PARK_DIRECTION = False

TAG_FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

KP_PARK = 0.15           # forward/backward speed (%) per pixel of
                          # horizontal error between the tag and frame center
MAX_PARK_SPEED = 35       # cap on the parking speed (%)
PARK_TOLERANCE_PX = 15    # horizontal error (px) within which the car is
                           # considered "parked" and stops

# Camera sources to toggle between with the 'c' key: (label, cv2 index).
# The indices are guesses -- 0 is almost always the built-in webcam, but
# the iPhone's index isn't guaranteed. Confirm/adjust using the startup
# probe below or by pressing 'c' and watching the preview window.
CAMERA_SOURCES = [
    ("webcam", 0),
    ("iphone", 1),
]
PROBE_CAMERAS_ON_START = True   # print a quick index/resolution scan at
                                  # startup to help identify each source
CAMERA_TOGGLE_KEY = ord("c")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COMMAND_INTERVAL = 0.08     # min seconds between BLE motor commands
STATUS_INTERVAL = 0.5       # console status print rate
MAX_CAMERA_RETRIES = 30     # tolerate this many consecutive dropped frames

WINDOW_NAME = "AprilTag Parking - multicam (press 'q' to quit)"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def probe_camera_indices(max_index=4):
    """Briefly open indices 0..max_index-1 and report whether each one
    opens and what resolution it reports, to help tell the iPhone and the
    built-in webcam apart. Purely informational -- doesn't change
    CAMERA_SOURCES for you."""
    print(f"Probing camera indices 0-{max_index - 1}...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  index {i}: OK, {w}x{h}")
            else:
                print(f"  index {i}: opened but produced no frame")
        else:
            print(f"  index {i}: not available")
        cap.release()


def try_open_camera(index):
    """Like apriltagparking.py's open_camera(), but for an arbitrary
    index so the source can be switched at runtime. Returns None (and
    prints why) instead of exiting, so a failed switch doesn't kill the
    whole script."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(
            f"Error: could not open camera index {index}.\n"
            "  - On macOS this is almost always a permissions problem. Grant\n"
            "    camera access to your terminal app in System Settings ->\n"
            "    Privacy & Security -> Camera, then fully quit and reopen it.\n"
            "  - If this is meant to be the iPhone, see the Continuity Camera\n"
            "    setup notes in this file's docstring."
        )
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # isOpened() can lie on macOS: the real test is whether a frame arrives.
    for _ in range(10):
        if cap.read()[0]:
            return cap
    print(f"Error: camera index {index} opened but produced no frames.")
    cap.release()
    return None


def build_detectors():
    """One ArucoDetector per AprilTag family, so the wrong family doesn't
    silently look like "no tag visible" (see apriltagtest.py)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return {
        name: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family_id), params)
        for name, family_id in TAG_FAMILIES.items()
    }


def find_largest_tag(gray, detectors):
    """Return (center_x, center_y, size, corners) for the largest AprilTag
    seen across all families this frame, or None if none is visible.
    "Largest" picks the closest/most prominent tag if more than one
    candidate is (spuriously) detected -- only one tag is expected here."""
    best = None
    for detector in detectors.values():
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            continue
        for tag_corners in corners:
            points = tag_corners[0]
            cx, cy = points.mean(axis=0)
            side_lengths = [
                float(np.linalg.norm(points[i] - points[(i + 1) % 4])) for i in range(4)
            ]
            size = sum(side_lengths) / len(side_lengths)
            if best is None or size > best[2]:
                best = (cx, cy, size, tag_corners)
    return best


def park_speed_for_error(error_x):
    """Proportional speed command driven by horizontal pixel error.
    Positive error (tag right of center) drives forward; negative
    (tag left of center) drives backward."""
    return clamp(KP_PARK * error_x, -MAX_PARK_SPEED, MAX_PARK_SPEED)


def straight_wheel_speeds(speed):
    """Both wheels get the same signed speed -- driving straight forward
    or backward along the car's own axis, no turning."""
    if REVERSE_PARK_DIRECTION:
        speed = -speed

    left = right = speed

    if INVERT_LEFT:
        left = -left
    if INVERT_RIGHT:
        right = -right

    # movement_move_tank() validates -100..100 and raises ValueError outside it.
    return clamp(left, -100, 100), clamp(right, -100, 100)


def connect_double_motor():
    print("Connecting to Double Motor...")
    dm = doubleMotor()
    try:
        dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    except Exception as exc:
        print(f"Error: connecting to the Double Motor failed: {exc}")
        return None
    print("Connected.")
    return dm


def stop_motors(dm):
    try:
        dm.movement_stop(blocking=False)
    except Exception as exc:
        print(f"Warning: stop command failed: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(cap, source_index, detectors, dm):
    """source_index is the position in CAMERA_SOURCES of the currently
    open `cap`, tracked here so 'c' knows which source to open next."""
    armed = False
    state = "WAITING"
    dropped_frames = 0
    last_command_time = 0.0
    last_status_time = 0.0

    print("Ready. Press 'c' to switch camera, SPACE to arm the motors, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            dropped_frames += 1
            if dropped_frames > MAX_CAMERA_RETRIES:
                print("Error: lost the camera feed.")
                return
            continue
        dropped_frames = 0

        if not dm.connected:
            print("Error: lost the connection to the Double Motor.")
            return

        height, width = frame.shape[:2]
        frame_center_x = width / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tag = find_largest_tag(gray, detectors)

        if tag is not None:
            cx, cy, size, tag_corners = tag
            error_x = cx - frame_center_x
            if abs(error_x) <= PARK_TOLERANCE_PX:
                state = "PARKED"
                speed = 0.0
            else:
                state = "PARKING"
                speed = park_speed_for_error(error_x)

            cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            error_x = 0.0
            size = 0.0
            speed = 0.0
            state = "WAITING"

        left_speed, right_speed = straight_wheel_speeds(speed)

        now = time.monotonic()
        if now - last_command_time >= COMMAND_INTERVAL:
            last_command_time = now
            try:
                if not armed or state in ("PARKED", "WAITING"):
                    stop_motors(dm)
                else:
                    dm.movement_move_tank(round(left_speed), round(right_speed), blocking=False)
            except Exception as exc:
                print(f"Warning: motor command failed: {exc}")

        if now - last_status_time >= STATUS_INTERVAL:
            last_status_time = now
            source_name = CAMERA_SOURCES[source_index][0]
            if tag is not None:
                print(
                    f"[{state}] ({source_name}) x_err={error_x:+7.1f}px  size={size:6.1f}px  "
                    f"L={left_speed:+.0f}  R={right_speed:+.0f}"
                )
            else:
                print(f"[{state}] ({source_name}) no tag visible  L={left_speed:+.0f}  R={right_speed:+.0f}")

        # --- on-screen debug overlay ---
        colors = {"WAITING": (0, 165, 255), "PARKING": (0, 255, 255), "PARKED": (0, 255, 0)}
        status = state if armed else "DISARMED - press SPACE"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colors.get(state, (255, 255, 255)), 2)
        cv2.putText(frame, f"L: {left_speed:+.0f}  R: {right_speed:+.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {CAMERA_SOURCES[source_index][0]} (press 'c' to switch)",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (80, 80, 80), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return
        if key == ord(" "):
            armed = not armed
            print("ARMED" if armed else "DISARMED")
            if not armed:
                stop_motors(dm)
                last_command_time = now
        if key == CAMERA_TOGGLE_KEY:
            # Stop driving during the switch -- there's no frame from the
            # new source yet, so nothing should be commanded blind.
            stop_motors(dm)

            next_index = (source_index + 1) % len(CAMERA_SOURCES)
            next_name, next_cv_index = CAMERA_SOURCES[next_index]
            print(f"Switching camera to '{next_name}' (index {next_cv_index})...")

            new_cap = try_open_camera(next_cv_index)
            if new_cap is None:
                print(f"Could not switch to '{next_name}'; staying on "
                      f"'{CAMERA_SOURCES[source_index][0]}'.")
            else:
                cap.release()
                cap = new_cap
                source_index = next_index
                dropped_frames = 0
                print(f"Now using '{next_name}'.")

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return


def main():
    if PROBE_CAMERAS_ON_START:
        probe_camera_indices()

    # Camera and detectors set up before connecting over BLE, so a failure
    # in either one can't leave a connected motor behind.
    source_index = 0
    cap = try_open_camera(CAMERA_SOURCES[source_index][1])
    if cap is None:
        sys.exit(1)

    detectors = build_detectors()

    dm = connect_double_motor()
    if dm is None:
        cap.release()
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)

    try:
        run(cap, source_index, detectors, dm)
    finally:
        print("Shutting down...")
        try:
            dm.movement_stop()  # blocking, so it lands before we disconnect
        except Exception as exc:
            print(f"Warning: final stop failed: {exc}")
        try:
            dm.disconnect()
        except Exception as exc:
            print(f"Warning: disconnect failed: {exc}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
