"""Wraps MediaPipe's PoseLandmarker (Tasks API).

MACOS STABILITY NOTE: on this machine, mediapipe 1.0.1's Tasks API
hard-crashes the process (uncatchable SIGABRT, not a Python exception)
unless BOTH of these are used together:
    delegate = BaseOptions.Delegate.GPU
    image format = mp.ImageFormat.SRGBA   (4-channel)
This was already diagnosed for HandLandmarker in
"Hand control files/mediapipetest.py" (CPU delegate dies with "Check
failed: service_ Service is unavailable"; GPU + 3-channel SRGB dies with
"unsupported ImageFrame format: 1"). PoseLandmarker shares the same Tasks
API runtime, so the same workaround is applied here.
"""
import os
import time
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
    drawing_utils,
)

# Verified reachable (curl HEAD -> 200, ~5.8MB) before using this URL.
# "lite" is chosen over "full"/"heavy" for lower latency, since this needs
# to keep up with a fast swing in real time.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
MIN_MODEL_BYTES = 1_000_000  # a truncated download is smaller than this

# MediaPipe Pose's 33-point landmark topology.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS


def ensure_model_downloaded():
    """Download the model if missing, atomically so a partial file can't stick."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_MODEL_BYTES:
        return
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading pose-landmark model (one-time, ~5.8 MB)...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        if os.path.getsize(tmp_path) < MIN_MODEL_BYTES:
            raise RuntimeError("downloaded model is too small to be valid")
        os.replace(tmp_path, MODEL_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("Model downloaded.")


def create_landmarker():
    ensure_model_downloaded()
    return PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=BaseOptions.Delegate.GPU,  # required - see module docstring
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


class PoseTracker:
    """Feed it BGR frames (as from OpenCV); get back the dominant arm's
    landmarks in normalized (0-1) image coordinates, or None if no pose
    was detected in that frame.

    IMPORTANT: this expects frame_bgr to already be horizontally mirrored
    (cv2.flip(frame, 1)), which is what gives an intuitive on-screen view.
    MediaPipe's own LEFT_*/RIGHT_* landmark indices are anatomical, based
    on what it sees in the image - mirroring the image swaps which side is
    which from the model's perspective, so a mirrored frame's "left
    shoulder" landmark is actually the player's real right shoulder. The
    handedness mapping below is inverted to compensate, so from the
    caller's point of view handedness="right" still means the player's
    actual right arm.
    """

    def __init__(self, handedness="right"):
        self.handedness = handedness
        self.landmarker = create_landmarker()
        self._last_timestamp_ms = -1
        self._last_raw_landmarks = None  # kept for draw_debug()

    def set_handedness(self, handedness):
        self.handedness = handedness

    def rebuild(self):
        """Recreate the underlying landmarker (and its GPU delegate/Metal
        texture cache) from scratch. Works around a known mediapipe-on-
        macOS bug where the GPU delegate's texture cache is never recycled
        and eventually fills up, hard-crashing the process (SIGABRT, not a
        catchable Python exception) after a few minutes of continuous use -
        confirmed as the exact cause of a real crash during testing here,
        matching https://github.com/google-ai-edge/mediapipe/issues/5267.
        Call this periodically (see VisionWorker's REBUILD_INTERVAL_SECONDS)
        well before that point - CPU delegate isn't a usable workaround on
        this machine (see the module docstring)."""
        self.landmarker.close()
        self.landmarker = create_landmarker()
        self._last_timestamp_ms = -1
        self._last_raw_landmarks = None

    def process(self, frame_bgr):
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba))

        # detect_for_video() requires strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except ValueError:
            return None

        if not result.pose_landmarks:
            self._last_raw_landmarks = None
            return None

        landmarks = result.pose_landmarks[0]
        self._last_raw_landmarks = landmarks

        # Swapped relative to what you'd expect - see the mirroring note in
        # the class docstring.
        if self.handedness == "right":
            shoulder, elbow, wrist, hip = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP
            other_shoulder, other_wrist = RIGHT_SHOULDER, RIGHT_WRIST
        else:
            shoulder, elbow, wrist, hip = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP
            other_shoulder, other_wrist = LEFT_SHOULDER, LEFT_WRIST

        def point(index):
            lm = landmarks[index]
            return (lm.x, lm.y)

        return {
            "shoulder": point(shoulder),
            "elbow": point(elbow),
            "wrist": point(wrist),
            "hip": point(hip),
            "other_shoulder": point(other_shoulder),
            "other_wrist": point(other_wrist),  # for auto-detecting handedness later
        }

    def draw_debug(self, frame_bgr):
        """Draw the full skeleton on frame_bgr in place, if a pose was
        found in the most recent process() call."""
        if self._last_raw_landmarks is not None:
            drawing_utils.draw_landmarks(frame_bgr, self._last_raw_landmarks, POSE_CONNECTIONS)

    def close(self):
        self.landmarker.close()
