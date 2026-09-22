"""
apriltagparking.py

A fixed webcam is the "parking spot," positioned off to the car's RIGHT
side rather than out in front of it. An AprilTag is mounted on the car
facing the camera. The car stays put until the tag is visible in the
camera's view, then closed-loop drives straight FORWARD or BACKWARD --
no turning -- to slide the tag to the horizontal center of the camera's
frame. Since the camera watches from the side, "sliding along the car's
own driving axis" is exactly what moves the tag left/right in the
camera's view, so the car parks perpendicular to the camera, right in
front of it.

State machine (see main()):
    WAITING -- no tag visible: car stays stopped (it never drives blind).
    PARKING -- tag visible but off-center: proportional control drives
               straight forward (tag right of center) or straight
               backward (tag left of center) -- both wheels get the
               same speed, just forward or reverse. Because it's P-only
               (no damping), it's normal for the car to overshoot and
               correct a few times -- that back-and-forth is exactly
               how it settles into place.
    PARKED  -- horizontal error is within PARK_TOLERANCE_PX: car stops.
    If the tag disappears again after being seen, the car falls back to
    WAITING and stops until it reacquires.

Proportional control:
    error = tag_center_x - frame_center_x   (+ = tag right of center)
    speed = clamp(KP_PARK * error, -MAX_PARK_SPEED, MAX_PARK_SPEED)
    left  = right = speed
Positive error drives both wheels forward; negative error reverses both
-- there is no left/right differential here, unlike griptest.py's tank
mixing, because centering the tag means moving along the car's own axis,
not turning.

Calibration (all camera- and hardware-specific -- there's no way to know
these ahead of time without your actual rig):
    KP_PARK / MAX_PARK_SPEED -- how hard it corrects. Too low = crawls in
        forever; too high = big overshoots each pass.
    PARK_TOLERANCE_PX -- how close to dead-center (in pixels) counts as
        "parked." Too tight and it may hunt back and forth forever
        chasing camera/detection noise; too loose and it stops visibly
        off-center.
    REVERSE_PARK_DIRECTION -- if the car drives away from center instead
        of toward it (tag right of center should drive forward, but
        drives backward instead), set this to True. Depends on which way
        the tag faces and how it's mounted relative to the camera.
    INVERT_LEFT / INVERT_RIGHT -- same drive-geometry fixes as
        griptest.py, for when a motor's gear train runs backward, so
        sending the same signed speed to both sides doesn't actually
        drive straight.

Keys:
    SPACE  arm / disarm the motors (starts DISARMED for safety)
    q      quit

Run with the project virtualenv:
    .venv/bin/python apriltagparking.py
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

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COMMAND_INTERVAL = 0.08     # min seconds between BLE motor commands
STATUS_INTERVAL = 0.5       # console status print rate
MAX_CAMERA_RETRIES = 30     # tolerate this many consecutive dropped frames

WINDOW_NAME = "AprilTag Parking (press 'q' to quit)"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


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

def run(cap, detectors, dm):
    armed = False
    state = "WAITING"
    dropped_frames = 0
    last_command_time = 0.0
    last_status_time = 0.0

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
            if tag is not None:
                print(
                    f"[{state}] x_err={error_x:+7.1f}px  size={size:6.1f}px  "
                    f"L={left_speed:+.0f}  R={right_speed:+.0f}"
                )
            else:
                print(f"[{state}] no tag visible  L={left_speed:+.0f}  R={right_speed:+.0f}")

        # --- on-screen debug overlay ---
        colors = {"WAITING": (0, 165, 255), "PARKING": (0, 255, 255), "PARKED": (0, 255, 0)}
        status = state if armed else "DISARMED - press SPACE"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colors.get(state, (255, 255, 255)), 2)
        cv2.putText(frame, f"L: {left_speed:+.0f}  R: {right_speed:+.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return


def main():
    # Camera and detectors set up before connecting over BLE, so a failure
    # in either one can't leave a connected motor behind.
    cap = open_camera()
    if cap is None:
        sys.exit(1)

    detectors = build_detectors()

    dm = connect_double_motor()
    if dm is None:
        cap.release()
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)

    try:
        run(cap, detectors, dm)
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
