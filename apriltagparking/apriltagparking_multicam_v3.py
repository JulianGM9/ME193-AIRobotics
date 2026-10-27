"""
apriltagparking_multicam_v3.py

Same as apriltagparking_multicam_v2.py (camera-source toggle, heading
correction, random-walk search, pause-before-giving-up-on-a-lost-tag --
see that file's docstring for all of that), plus one new capability:

  4. Closing the distance once parked. A single stationary parking spot
     only gets the car so close to the camera before either side runs
     into the room's other constraints (or the tag simply isn't very
     precise to center on from far away). So once "PARKED", instead of
     just sitting there, the car repeatedly hops itself closer:
         a. Turn CLOSE_IN_TURN_DEGREES (90 by default) toward the tag.
            This points the car's front where the camera roughly is,
            since the tag/camera are to the car's right when parked.
         b. Drive forward CLOSE_IN_DRIVE_DEGREES, blind -- the tag is
            mounted on the car's side, so once the car has turned to
            face the tag/camera, the tag itself now faces sideways-away
            from the camera and can't be seen. There is no way to
            correct mid-drive; this step is pure open-loop dead
            reckoning.
         c. Turn back CLOSE_IN_TURN_DEGREES the other way, restoring the
            original heading so the tag faces the camera again.
     The car then re-runs the normal PARKING/PARKED closed loop to
     recenter itself (the hop rarely lands it perfectly square), and once
     PARKED again, checks the tag's apparent HEIGHT against the actual
     camera frame's height: if it's now at least
     CLOSE_ENOUGH_HEIGHT_FRACTION of the frame, the car is close enough
     and hopping stops (it stays running normally, just no more hops).
     Otherwise it hops again, up to MAX_CLOSE_IN_HOPS times before giving
     up.

     Each hop is a BLOCKING sequence of turn/drive/turn commands (using
     the Double Motor's own movement_turn_for_degrees/
     movement_move_for_degrees, which use the IMU to confirm each turn,
     not the tank-mixing used elsewhere in this file) -- the preview
     window will appear to freeze and keys won't register for the
     several seconds a hop takes. That's expected: there's nothing to
     show or react to while the tag is out of view anyway. Camera frames
     that queued up during the hop are flushed afterward so the vision
     loop doesn't react to stale video once it resumes.

     Hops only happen while ARMED, same as every other motor command in
     this file. CLOSE_IN_TURN_REVERSED / CLOSE_IN_DRIVE_REVERSED are
     calibration flags (see below) for the same reason every other
     directional flag in this file exists: I have no way to verify which
     physical direction is which on your actual car.

See apriltagparking_multicam_v2.py's docstring for the camera-toggle,
heading-correction, and random-walk/pause details, and for the
Continuity Camera setup steps -- all of that carries over unchanged.

State machine (see run()):
    SEARCHING -- no tag visible, and either none has ever been seen or
                 TAG_LOST_GRACE_PERIOD has elapsed since it was last
                 seen: slow random-walk crawl.
    PAUSED    -- the tag was being tracked and just stopped being
                 detected: car stops and holds still for up to
                 TAG_LOST_GRACE_PERIOD seconds, in case it reappears. If
                 it doesn't, before falling back to SEARCHING the car
                 backs up a small blind nudge (see perform_backup_nudge())
                 -- most losses happen because the car drove or turned
                 slightly past the tag, so this alone often re-frames it.
    PARKING   -- tag visible but not yet centered and/or not yet
                 parallel: proportional translation and heading
                 correction run at the same time.
    PARKED    -- horizontal error within PARK_TOLERANCE_PX AND heading
                 skew within PARALLEL_TOLERANCE. If the tag's height is
                 still under CLOSE_ENOUGH_HEIGHT_FRACTION of the frame,
                 performs one blind hop (see above) and returns to
                 PARKING/SEARCHING to reacquire and recenter; once it's
                 tall enough, or MAX_CLOSE_IN_HOPS is reached, the car
                 just stays parked.
    SETTLING  -- right after a hop's final turn-back: motors held at 0
                 for POST_HOP_SETTLE_SECONDS while the camera loop keeps
                 running normally (frames read/shown, keys handled) --
                 this is NOT a time.sleep() pausing the whole script,
                 just a state that ignores detection until the video feed
                 (which lags a bit behind the motors physically stopping)
                 catches up.

Calibration additions on top of apriltagparking_multicam_v2.py's (all of
KP_PARK, MAX_PARK_SPEED, PARK_TOLERANCE_PX, KP_HEADING, MAX_HEADING_TURN,
MAX_HEADING_TURN_CHANGE, PARALLEL_TOLERANCE, CONTROL_SMOOTHING_TAU,
RAW_SKEW_CLAMP, REVERSE_PARK_DIRECTION, REVERSE_HEADING_CORRECTION,
INVERT_LEFT, INVERT_RIGHT, SEARCH_FORWARD_RANGE, SEARCH_TURN_RANGE,
SEARCH_STEP_SECONDS, TAG_LOST_GRACE_PERIOD still apply the same way):
    CLOSE_ENOUGH_HEIGHT_FRACTION -- fraction of the camera frame's actual
        height the tag must reach to count as "close enough," ending the
        hop sequence (0.5 = half the frame height). Measured against the
        real captured frame size each frame, not the FRAME_HEIGHT
        request below, since a camera (especially Continuity Camera) can
        silently ignore that request and deliver a different resolution.
    CLOSE_IN_TURN_DEGREES -- how far to turn for each hop (90 matches
        the camera-to-the-right geometry this whole file assumes).
    CLOSE_IN_DRIVE_DEGREES -- how far (motor shaft degrees, not distance)
        to drive blind toward the tag each hop. Purely a guess since
        there's no physical distance calibration here -- watch how far
        the car actually moves on the first hop and adjust.
    CLOSE_IN_SPEED -- speed (%) used for the hop's turns and drive.
    MAX_CLOSE_IN_HOPS -- safety cap on how many hops to attempt before
        giving up (in case CLOSE_ENOUGH_HEIGHT_FRACTION is miscalibrated
        or the car gets physically stuck).
    CLOSE_IN_TURN_REVERSED -- if the initial turn swings the car's front
        away from the tag/camera instead of toward it, set this to True.
    CLOSE_IN_DRIVE_REVERSED -- if the blind drive moves the car away
        from the tag/camera instead of toward it, set this to True.
    POST_HOP_SETTLE_SECONDS -- pause after a hop's final turn-back before
        the vision loop trusts camera frames again, since the video feed
        lags a bit behind the motors actually stopping.
    BACKUP_NUDGE_DEGREES / BACKUP_NUDGE_SPEED -- how far and how fast the
        car backs up once TAG_LOST_GRACE_PERIOD expires, before starting
        the random walk (see the PAUSED state above).
    BACKUP_NUDGE_REVERSED -- if that backup nudge moves the car further
        away instead of back toward roughly where the tag was, set this
        to True.

Keys:
    c      toggle between camera sources (see CAMERA_SOURCES)
    SPACE  arm / disarm the motors (starts DISARMED for safety)
    q      quit

Run with the project virtualenv:
    .venv/bin/python apriltagparking_multicam_v3.py
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
MAX_HEADING_TURN = 8         # cap on the heading-correction turn (%)
MAX_HEADING_TURN_CHANGE = 3  # max change in the commanded turn (%) between
                              # consecutive motor commands (slew limit)
PARALLEL_TOLERANCE = 0.05  # skew within which the car counts as "parallel"

CONTROL_SMOOTHING_TAU = 0.3  # seconds; exponential smoothing on error_x
                              # and skew before they're used for anything
RAW_SKEW_CLAMP = 0.25        # reject implausible single-frame skew spikes
                              # (real skew tops out around 0.13) before
                              # they ever reach the smoothing filter

# Random-walk search behavior while no tag is visible.
SEARCH_FORWARD_RANGE = (10, 20)   # always forward, never backward (%)
SEARCH_TURN_RANGE = (-6, 6)       # random turn bias each step (%)
SEARCH_STEP_SECONDS = (1.5, 4.0)  # how long a random step lasts before
                                    # re-rolling new random values

# How long to pause (car stopped) after losing a tracked tag before
# giving up and starting the random walk -- guards against a single bad
# detection frame kicking off a walk in the wrong direction mid-approach.
TAG_LOST_GRACE_PERIOD = 1.0  # seconds

# --- "Hop closer" behavior once parked (new in v3) ---
# A fraction of the actual captured frame height, not a fixed pixel count
# -- FRAME_HEIGHT below is only what we ask the camera for, and it can
# silently deliver a different (often much larger, e.g. Continuity
# Camera) resolution instead, which would make a fixed pixel threshold
# meaningless. The real frame height is measured fresh every frame.
CLOSE_ENOUGH_HEIGHT_FRACTION = 0.5   # tag height >= this fraction of the
                                       # frame's height counts as "close enough"
CLOSE_IN_TURN_DEGREES = 90    # how far to turn for each blind hop
CLOSE_IN_DRIVE_DEGREES = 150  # how far (motor shaft degrees) to drive
                                # blind each hop -- a guess; tune by watching
                                # how far the car actually moves on one hop
CLOSE_IN_SPEED = 40           # speed (%) for the hop's turns and drive
MAX_CLOSE_IN_HOPS = 8         # give up hopping after this many attempts
CLOSE_IN_TURN_REVERSED = True     # flipped: hop turn was going the wrong
                                    # way on the real car
CLOSE_IN_DRIVE_REVERSED = False   # flip if the blind drive moves away
                                    # from the tag/camera instead of toward it
POST_HOP_SETTLE_SECONDS = 0.5   # pause after the hop's final turn-back,
                                  # before trusting camera frames again --
                                  # the vision feed lags a bit behind the
                                  # motors actually stopping

# Small blind backup used when giving up on a lost tag (see
# TAG_LOST_GRACE_PERIOD) and falling back to the random walk -- most
# losses happen because the car drove/turned slightly past the tag, so
# backing up a touch first often re-frames it without a full search.
BACKUP_NUDGE_DEGREES = 60   # how far (motor shaft degrees) to back up --
                              # a guess; tune by watching how far it moves
BACKUP_NUDGE_SPEED = 30     # speed (%) for the backup nudge
BACKUP_NUDGE_REVERSED = False   # flip if this nudge moves the car further
                                  # away instead of back toward the tag

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
POST_HOP_FRAME_FLUSH = 5    # frames to discard right after a hop, so the
                              # vision loop doesn't react to stale video
                              # that queued up in the camera buffer

WINDOW_NAME = "AprilTag Parking - multicam v3 (press 'q' to quit)"


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
    candidate is (spuriously) detected -- only one tag is expected here.

    `size` is the tag's apparent HEIGHT in pixels (average of its two
    roughly-vertical edges, TL-BL and TR-BR), not an average of all 4
    sides -- so it lines up directly with "a fraction of the camera
    frame's height" rather than some blend of width and height."""
    best = None
    for detector in detectors.values():
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            continue
        for tag_corners in corners:
            points = tag_corners[0]
            cx, cy = points.mean(axis=0)
            left_edge = float(np.linalg.norm(points[3] - points[0]))
            right_edge = float(np.linalg.norm(points[2] - points[1]))
            size = (left_edge + right_edge) / 2.0
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


def perform_close_in_hop(dm):
    """Blind maneuver: turn toward the tag, drive forward blind, turn
    back. The tag is mounted on the car's side and faces away from the
    camera for the whole middle step, so there is nothing to see and
    nothing to correct -- this is pure open-loop dead reckoning, using
    the Double Motor's own IMU-checked turn command for the two turns.
    Blocking: doesn't return until all three steps finish."""
    turn_toward = le.MOVEMENT_TURN_DIRECTION_LEFT if CLOSE_IN_TURN_REVERSED else le.MOVEMENT_TURN_DIRECTION_RIGHT
    turn_back = le.MOVEMENT_TURN_DIRECTION_RIGHT if CLOSE_IN_TURN_REVERSED else le.MOVEMENT_TURN_DIRECTION_LEFT
    drive_direction = (
        le.MOVEMENT_MOVE_DIRECTION_BACKWARD if CLOSE_IN_DRIVE_REVERSED
        else le.MOVEMENT_MOVE_DIRECTION_FORWARD
    )

    try:
        print(f"  Hop: turning {CLOSE_IN_TURN_DEGREES} deg toward the tag...")
        dm.movement_turn_for_degrees(CLOSE_IN_TURN_DEGREES, direction=turn_toward, speed=CLOSE_IN_SPEED)

        print(f"  Hop: driving blind ({CLOSE_IN_DRIVE_DEGREES} deg)...")
        dm.movement_move_for_degrees(CLOSE_IN_DRIVE_DEGREES, direction=drive_direction, speed=CLOSE_IN_SPEED)

        print(f"  Hop: turning {CLOSE_IN_TURN_DEGREES} deg back...")
        dm.movement_turn_for_degrees(CLOSE_IN_TURN_DEGREES, direction=turn_back, speed=CLOSE_IN_SPEED)
    except Exception as exc:
        print(f"Warning: hop maneuver failed partway through: {exc}")
    finally:
        stop_motors(dm)
    # Note: this does NOT sleep here. The camera feed lags a bit behind
    # the motors physically stopping, so the caller holds the motors at
    # 0 for POST_HOP_SETTLE_SECONDS via the SETTLING state in run()
    # instead -- a real time.sleep() here would block the whole process,
    # freezing frame reads and the preview window along with it.


def perform_backup_nudge(dm):
    """Small blind nudge backward, used when giving up on a lost tag and
    falling back to the random walk. Most of the time the tag was lost
    because the car drove or turned slightly past it, so backing up a
    touch often brings it back into view without a full random search."""
    direction = (
        le.MOVEMENT_MOVE_DIRECTION_FORWARD if BACKUP_NUDGE_REVERSED
        else le.MOVEMENT_MOVE_DIRECTION_BACKWARD
    )
    try:
        print(f"  Backing up {BACKUP_NUDGE_DEGREES} deg before searching...")
        dm.movement_move_for_degrees(BACKUP_NUDGE_DEGREES, direction=direction, speed=BACKUP_NUDGE_SPEED)
    except Exception as exc:
        print(f"Warning: backup nudge failed: {exc}")
    finally:
        stop_motors(dm)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(cap, source_index, detectors, dm):
    """source_index is the position in CAMERA_SOURCES of the currently
    open `cap`, tracked here so 'c' knows which source to open next."""
    armed = False
    state = "SEARCHING"
    prev_state = "SEARCHING"  # so the PAUSED -> SEARCHING transition (grace
                                # period expiring) can be told apart from
                                # already being in SEARCHING
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

    # "Hop closer" bookkeeping.
    hop_count = 0
    close_in_done = False

    # While now < settle_until, the loop keeps reading/showing frames and
    # handling keys as normal, but holds the motors at 0 and skips
    # detection-driven control -- see the SETTLING check below. Set after
    # a hop finishes turning back, since the camera lags a bit behind the
    # motors actually stopping.
    settle_until = 0.0

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

        if now < settle_until:
            # Keep the camera loop alive -- frames still read and shown,
            # keys still handled -- instead of literally pausing the
            # whole script. Just hold the motors at 0 and skip detection-
            # driven control until the camera has caught up.
            state = "SETTLING"
            tag = None
            error_x = size = skew = 0.0
            left_speed, right_speed = 0, 0
        else:
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

                # Recorded now (before any early `continue` below from a hop)
                # so the next frame's PAUSED -> SEARCHING check has an
                # accurate previous state.
                prev_state = state

                # Once parked, decide whether to hop closer instead of just
                # sending the (zero) motor command below. The threshold is a
                # fraction of THIS frame's actual height, not a fixed pixel
                # count, since the real camera resolution can differ from
                # what FRAME_HEIGHT asked for.
                close_enough_size = CLOSE_ENOUGH_HEIGHT_FRACTION * height
                if state == "PARKED" and armed and not close_in_done:
                    if size >= close_enough_size:
                        close_in_done = True
                        print(f"Close enough (size={size:.1f}px >= {close_enough_size:.1f}px, "
                              f"{CLOSE_ENOUGH_HEIGHT_FRACTION:.0%} of {height}px frame height). Done hopping.")
                    elif hop_count >= MAX_CLOSE_IN_HOPS:
                        close_in_done = True
                        print(f"Reached MAX_CLOSE_IN_HOPS ({MAX_CLOSE_IN_HOPS}) without getting close enough; giving up on hopping.")
                    else:
                        hop_count += 1
                        print(f"Parked (size={size:.1f}px < {close_enough_size:.1f}px). Hop {hop_count}/{MAX_CLOSE_IN_HOPS}...")
                        perform_close_in_hop(dm)

                        # The tag was out of view for the whole hop; treat the
                        # next sighting as a fresh reacquisition rather than
                        # blending in now-meaningless smoothed state.
                        was_tracking = False
                        prev_turn_command = 0.0
                        next_search_reroll = 0.0
                        last_tag_time = None

                        # Frames queued up in the camera buffer during the
                        # blocking hop are stale; drop them before resuming.
                        for _ in range(POST_HOP_FRAME_FLUSH):
                            cap.read()

                        # Hold the motors at 0 for a bit -- see the
                        # SETTLING check above -- since the camera feed
                        # lags a bit behind the motors actually stopping.
                        settle_until = time.monotonic() + POST_HOP_SETTLE_SECONDS

                        continue
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
                elif prev_state == "PAUSED" and armed:
                    # Grace period just expired -- most of the time the tag
                    # was lost because the car drove/turned slightly past it,
                    # so back up a touch before searching; this alone often
                    # re-frames it without a full random walk. Recorded as
                    # already-SEARCHING so this doesn't fire again next frame,
                    # and `continue` for a fresh frame rather than showing/
                    # processing this now-stale one under the new state.
                    print("Grace period expired; backing up before searching...")
                    perform_backup_nudge(dm)
                    next_search_reroll = 0.0  # fresh random walk after the nudge
                    prev_state = "SEARCHING"
                    for _ in range(POST_HOP_FRAME_FLUSH):
                        cap.read()
                    continue
                else:
                    state = "SEARCHING"

                    if now >= next_search_reroll:
                        search_forward = random.uniform(*SEARCH_FORWARD_RANGE)
                        search_turn = random.uniform(*SEARCH_TURN_RANGE)
                        next_search_reroll = now + random.uniform(*SEARCH_STEP_SECONDS)

                    left_speed, right_speed = mix_tank(search_forward, search_turn)

                prev_state = state

        if now - last_command_time >= COMMAND_INTERVAL:
            last_command_time = now
            try:
                if not armed or state in ("PARKED", "PAUSED", "SETTLING"):
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
                    f"size={size:6.1f}px  hops={hop_count}  L={left_speed:+.0f}  R={right_speed:+.0f}"
                )
            else:
                print(f"[{state}] ({source_name}) no tag visible  L={left_speed:+.0f}  R={right_speed:+.0f}")

        # --- on-screen debug overlay ---
        colors = {"SEARCHING": (0, 165, 255), "PAUSED": (0, 0, 255),
                  "PARKING": (0, 255, 255), "PARKED": (0, 255, 0),
                  "SETTLING": (200, 200, 200)}
        status = state if armed else "DISARMED - press SPACE"
        if state == "PARKED" and close_in_done:
            status += "  (close-in done)"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colors.get(state, (255, 255, 255)), 2)
        cv2.putText(frame, f"L: {left_speed:+.0f}  R: {right_speed:+.0f}   hops: {hop_count}/{MAX_CLOSE_IN_HOPS}",
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
