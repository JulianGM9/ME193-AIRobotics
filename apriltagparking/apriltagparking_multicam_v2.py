"""
apriltagparking_multicam.py

Same as apriltagparking.py (see that file for the full parking control
scheme), but with three differences:

  1. The camera source can be switched live between your Mac's built-in
     webcam and your iPhone via Continuity Camera, without restarting the
     script (press 'c' -- see the Continuity Camera section below).

  2. Heading correction: if the tag isn't square-on to the camera --
     meaning the car's heading has drifted so it isn't driving parallel
     to the camera -- the car turns to straighten out at the same time
     as it translates to center the tag horizontally, instead of only
     ever driving dead straight. This is detected from the tag's
     corners: a tag facing the camera square-on projects as a symmetric
     quadrilateral, but as the car's heading skews, one side of the tag
     is nearer the camera than the other, so its left and right edges
     project at different lengths (see tag_skew()).

     Tuning note, in case turning still looks off: a proper 3D pinhole
     projection simulation (cv2.projectPoints, not just an image warp)
     shows this skew value is small even for a large real heading error
     -- it tops out around +-0.13 even at a 89-degree edge-on angle, and
     is NOT affected by the car's lateral position (confirmed both
     things numerically before tuning the constants below). That means
     genuine heading-error signal is inherently small, so single-frame
     corner-detection noise/jitter -- which is very easy to get on a
     small or motion-blurred tag -- can dominate over the real signal
     and read as a much larger, faster-swinging "error" than any real
     misalignment could produce. KP_HEADING is set high enough to
     actually correct real (small) skew values, and CONTROL_SMOOTHING_TAU
     / RAW_SKEW_CLAMP / MAX_HEADING_TURN_CHANGE exist specifically to
     keep that higher gain from overreacting to noisy single frames --
     see their comments below.

     That asymmetry, once smoothed and clamped, drives a second
     proportional controller that differentially speeds up/slows down
     the two wheels to turn the car, layered on top of the existing
     forward/backward translation.

  3. Base state without a tag: instead of stopping, the car does a slow
     random walk (slow forward crawl + slow random turning, re-rolled
     every couple of seconds) so it actively searches instead of sitting
     still. This replaces the "never drives blind" safety behavior from
     apriltagparking.py -- that original file is left untouched with the
     old behavior if you want it back.

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

State machine (see run()):
    SEARCHING -- no tag visible, and either none has ever been seen or
                 TAG_LOST_GRACE_PERIOD has elapsed since it was last
                 seen: slow random-walk crawl (see SEARCH_FORWARD_RANGE /
                 SEARCH_TURN_RANGE below).
    PAUSED    -- the tag was being tracked (PARKING/PARKED) and just
                 stopped being detected: car stops and holds still for
                 up to TAG_LOST_GRACE_PERIOD seconds, in case it reappears.
                 This exists because a single bad/missed detection frame
                 shouldn't be treated the same as "the tag is truly gone"
                 -- without this pause, one flaky frame would immediately
                 kick off a random-walk turn or forward crawl in
                 whatever direction happened to be rolled, right as the
                 car may have been mid-correction toward the tag.
    PARKING   -- tag visible but not yet centered and/or not yet
                 parallel: proportional translation (forward/backward)
                 and proportional heading correction (turning) run at
                 the same time.
    PARKED    -- horizontal error within PARK_TOLERANCE_PX AND heading
                 skew within PARALLEL_TOLERANCE: car stops.

Calibration additions on top of apriltagparking.py's (KP_PARK,
MAX_PARK_SPEED, PARK_TOLERANCE_PX, REVERSE_PARK_DIRECTION, INVERT_LEFT,
INVERT_RIGHT all still apply the same way):
    KP_HEADING / MAX_HEADING_TURN -- how hard the car corrects its
        heading. Too low and it stays crooked; too high and it
        overcorrects/oscillates. Real skew values are small (see the
        tuning note above), so KP_HEADING needs to be much larger than
        KP_PARK to have any real effect -- MAX_HEADING_TURN and
        MAX_HEADING_TURN_CHANGE are what actually keep it safe.
    CONTROL_SMOOTHING_TAU -- exponential smoothing (same technique as
        griptest.py/mediapipetest.py) applied to both the horizontal
        error and the skew before they're used for anything, so a single
        noisy detection frame can't immediately snap the car into a
        large correction. Larger = smoother but slower to react.
    RAW_SKEW_CLAMP -- clamps a single frame's raw skew reading before
        smoothing, so one badly-detected frame (e.g. skew=1.5 from a
        corrupted corner) can't inject a huge outlier into the smoothed
        value. Set comfortably above the real max (~0.13, see above) so
        it only rejects clearly-implausible readings, not real signal.
    MAX_HEADING_TURN_CHANGE -- slew-rate limit: the actual commanded
        turn can change by at most this many percentage points between
        consecutive motor commands, so even a big target correction is
        approached gradually instead of snapping the car around fast
        enough to lose sight of the tag.
    PARALLEL_TOLERANCE -- how small the (smoothed) skew must be to count
        as "parallel enough." Skew is a dimensionless ratio (0 = perfectly
        square-on, ~0.13 is about as large as it physically gets); start
        around 0.04-0.06 and adjust by watching the console's "skew="
        reading.
    REVERSE_HEADING_CORRECTION -- the sign relating "which edge is
        shorter" to "which way to turn" depends on which side of the car
        the tag is mounted on and which side the camera is on, which
        can't be known ahead of time. If the car turns away from
        parallel instead of toward it, set this to True.
    SEARCH_FORWARD_RANGE / SEARCH_TURN_RANGE / SEARCH_STEP_SECONDS --
        the random walk's speed while no tag is visible, and how often
        (in seconds, picked randomly in that range) it re-rolls new
        random values.
    TAG_LOST_GRACE_PERIOD -- how long (seconds) to pause and wait for a
        just-lost tag to reappear before giving up and starting the
        random walk. Too short and a single bad detection frame can
        still trigger an unwanted walk; too long and the car sits idle
        for a while after a tag that's genuinely gone.

Keys:
    c      toggle between camera sources (see CAMERA_SOURCES)
    SPACE  arm / disarm the motors (starts DISARMED for safety)
    q      quit

Run with the project virtualenv:
    .venv/bin/python apriltagparking_multicam.py
"""

import math
import random
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
REVERSE_HEADING_CORRECTION = True   # flipped: heading correction was turning
                                      # the wrong way on the real car

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

KP_HEADING = 150.0        # turn correction (motor speed %) per unit of
                          # tag skew (a dimensionless left/right edge-
                          # length ratio, not pixels -- see tag_skew()).
                          # Much higher than KP_PARK because real skew
                          # values are inherently small (see docstring).
MAX_HEADING_TURN = 18       # cap on the heading-correction turn (%)
MAX_HEADING_TURN_CHANGE = 6  # max change in the commanded turn (%) between
                              # consecutive motor commands (slew limit)
PARALLEL_TOLERANCE = 0.05  # skew within which the car counts as "parallel"

CONTROL_SMOOTHING_TAU = 0.3  # seconds; exponential smoothing on error_x
                              # and skew before they're used for anything
RAW_SKEW_CLAMP = 0.25        # reject implausible single-frame skew spikes
                              # (real skew tops out around 0.13) before
                              # they ever reach the smoothing filter

# Random-walk search behavior while no tag is visible.
SEARCH_FORWARD_RANGE = (10, 20)   # always forward, never backward (%)
SEARCH_TURN_RANGE = (-15, 15)     # random turn bias each step (%)
SEARCH_STEP_SECONDS = (1.5, 4.0)  # how long a random step lasts before
                                    # re-rolling new random values

# How long to pause (car stopped) after losing a tracked tag before
# giving up and starting the random walk -- guards against a single bad
# detection frame kicking off a walk in the wrong direction mid-approach.
TAG_LOST_GRACE_PERIOD = 1.0  # seconds

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


def tag_skew(tag_corners):
    """Signed, size-normalized left/right edge-length asymmetry of a
    detected tag's quadrilateral -- ~0 when the tag faces the camera
    square-on (car driving parallel to the camera), growing as the car's
    heading turns the tag's face away from that (basic keystone/
    perspective distortion). Verified against a proper 3D pinhole
    projection (not just an image warp): 0.0 with zero heading error
    regardless of lateral position, and a consistent-sign value that
    grows monotonically with heading error, saturating around +-0.13
    even at a 89-degree edge-on angle -- see the module docstring's
    tuning note for why that matters for KP_HEADING.

    Corner order is OpenCV's aruco convention: top-left, top-right,
    bottom-right, bottom-left."""
    points = tag_corners[0]
    left_edge = float(np.linalg.norm(points[3] - points[0]))
    right_edge = float(np.linalg.norm(points[2] - points[1]))
    avg_edge = (left_edge + right_edge) / 2.0
    if avg_edge < 1e-6:
        return 0.0
    return (right_edge - left_edge) / avg_edge


def heading_correction_for_skew(skew):
    """Proportional turn command driven by tag_skew()."""
    return clamp(KP_HEADING * skew, -MAX_HEADING_TURN, MAX_HEADING_TURN)


def mix_tank(forward, turn):
    """Differential tank mixing: `forward` moves both wheels the same
    direction (along the car's own axis), `turn` speeds up one side and
    slows the other. Used both for parking (forward = translation toward
    center, turn = heading correction) and for the random-walk search."""
    left = forward + turn
    right = forward - turn

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
    state = "SEARCHING"
    dropped_frames = 0
    last_command_time = 0.0
    last_status_time = 0.0

    # Random-walk search state, persisted across frames so it reads as a
    # walk (steps that last a while) rather than jitter (re-rolled every
    # frame). next_search_reroll = 0.0 forces an immediate fresh roll the
    # first time (and every time) SEARCHING starts.
    search_forward = 0.0
    search_turn = 0.0
    next_search_reroll = 0.0

    # None until a tag has been seen at least once; then holds the
    # monotonic time it was last seen, so a fresh loss can be told apart
    # from "still within the grace period" vs. "grace period expired."
    last_tag_time = None

    # Smoothed error/skew (see CONTROL_SMOOTHING_TAU) and the slew-limited
    # turn command actually sent last time, all persisted across frames.
    smoothed_error_x = 0.0
    smoothed_skew = 0.0
    prev_turn_command = 0.0
    was_tracking = False
    last_frame_time = time.monotonic()

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

        now = time.monotonic()
        height, width = frame.shape[:2]
        frame_center_x = width / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tag = find_largest_tag(gray, detectors)

        dt = now - last_frame_time
        last_frame_time = now

        if tag is not None:
            cx, cy, size, tag_corners = tag
            raw_error_x = cx - frame_center_x
            raw_skew = clamp(tag_skew(tag_corners), -RAW_SKEW_CLAMP, RAW_SKEW_CLAMP)
            last_tag_time = now
            next_search_reroll = 0.0  # fresh random walk next time it's lost

            if not was_tracking:
                # Just reacquired after a gap -- don't blend in a stale
                # smoothed value from a possibly very different moment;
                # start fresh from this frame's reading instead.
                smoothed_error_x = raw_error_x
                smoothed_skew = raw_skew
            else:
                alpha = 1.0 if CONTROL_SMOOTHING_TAU <= 0 else 1.0 - math.exp(-dt / CONTROL_SMOOTHING_TAU)
                smoothed_error_x += (raw_error_x - smoothed_error_x) * alpha
                smoothed_skew += (raw_skew - smoothed_skew) * alpha
            was_tracking = True

            error_x, skew = smoothed_error_x, smoothed_skew

            centered = abs(error_x) <= PARK_TOLERANCE_PX
            parallel = abs(skew) <= PARALLEL_TOLERANCE
            if centered and parallel:
                state = "PARKED"
                translate_speed = 0.0
                turn_target = 0.0
            else:
                state = "PARKING"
                translate_speed = park_speed_for_error(error_x)
                turn_target = heading_correction_for_skew(skew)
                if REVERSE_PARK_DIRECTION:
                    translate_speed = -translate_speed
                if REVERSE_HEADING_CORRECTION:
                    turn_target = -turn_target

            # Slew-limit the turn command so even a large target
            # correction is approached gradually instead of snapping the
            # car around fast enough to lose sight of the tag.
            turn_correction = clamp(
                turn_target,
                prev_turn_command - MAX_HEADING_TURN_CHANGE,
                prev_turn_command + MAX_HEADING_TURN_CHANGE,
            )
            prev_turn_command = turn_correction

            left_speed, right_speed = mix_tank(translate_speed, turn_correction)

            cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            error_x = 0.0
            size = 0.0
            skew = 0.0
            was_tracking = False
            prev_turn_command = 0.0  # matches the 0 actually being sent below

            just_lost = last_tag_time is not None and (now - last_tag_time) < TAG_LOST_GRACE_PERIOD
            if just_lost:
                # Hold still and see if it comes back -- don't touch the
                # random-walk state, so SEARCHING (if the grace period
                # expires) still starts with a fresh reroll.
                state = "PAUSED"
                left_speed, right_speed = 0, 0
            else:
                state = "SEARCHING"

                if now >= next_search_reroll:
                    search_forward = random.uniform(*SEARCH_FORWARD_RANGE)
                    search_turn = random.uniform(*SEARCH_TURN_RANGE)
                    next_search_reroll = now + random.uniform(*SEARCH_STEP_SECONDS)

                left_speed, right_speed = mix_tank(search_forward, search_turn)

        if now - last_command_time >= COMMAND_INTERVAL:
            last_command_time = now
            try:
                if not armed or state in ("PARKED", "PAUSED"):
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
                    f"[{state}] ({source_name}) x_err={error_x:+7.1f}px  skew={skew:+.3f}  "
                    f"size={size:6.1f}px  L={left_speed:+.0f}  R={right_speed:+.0f}"
                )
            else:
                print(f"[{state}] ({source_name}) no tag visible  L={left_speed:+.0f}  R={right_speed:+.0f}")

        # --- on-screen debug overlay ---
        colors = {"SEARCHING": (0, 165, 255), "PAUSED": (0, 0, 255),
                  "PARKING": (0, 255, 255), "PARKED": (0, 255, 0)}
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
