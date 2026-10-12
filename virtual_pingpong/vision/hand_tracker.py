"""Wraps MediaPipe's HandLandmarker (Tasks API), tracking the hand's palm
center as one signal toward the paddle's position (paired with the wrist
as a fallback in vision/paddle_detector.py).

MACOS STABILITY NOTE: same crash risk as vision/pose_tracker.py - see that
module's docstring for the full explanation. This exact model/delegate
combination was already proven working in "Hand control files/
mediapipetest.py".

Hands is noticeably less reliable than Pose at detecting a hand that's
actively gripping something: its fingers are curled/occluded, which the
model wasn't trained on much, so detection confidence drops and it often
goes missing while holding the paddle (confirmed in testing). That's why
this is meant to be used as a sometimes-absent primary signal with the
wrist as a fallback, not as the sole source.
"""
import os
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

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MIN_MODEL_BYTES = 1_000_000
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + base of each finger


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_MODEL_BYTES:
        return
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading hand-landmark model (one-time, ~8 MB)...")
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
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=BaseOptions.Delegate.GPU,  # required - see module docstring
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )


class HandTracker:
    """Feed it BGR frames; get back the palm center in normalized (0-1)
    image coordinates, or None if no hand was detected in that frame.
    """

    def __init__(self):
        self.landmarker = create_landmarker()
        self._last_timestamp_ms = -1
        self._last_raw_landmarks = None  # kept for draw_debug()

    def rebuild(self):
        """Recreate the underlying landmarker from scratch - see
        PoseTracker.rebuild()'s docstring for why (a known mediapipe-on-
        macOS GPU texture cache leak that otherwise hard-crashes the
        process after a few minutes)."""
        self.landmarker.close()
        self.landmarker = create_landmarker()
        self._last_timestamp_ms = -1
        self._last_raw_landmarks = None

    def process(self, frame_bgr):
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba))

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except ValueError:
            return None

        if not result.hand_landmarks:
            self._last_raw_landmarks = None
            return None

        landmarks = result.hand_landmarks[0]
        self._last_raw_landmarks = landmarks
        x = sum(landmarks[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
        y = sum(landmarks[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
        return (x, y)

    def draw_debug(self, frame_bgr):
        """Draw the hand skeleton on frame_bgr in place, if a hand was
        found in the most recent process() call."""
        if self._last_raw_landmarks is not None:
            drawing_utils.draw_landmarks(frame_bgr, self._last_raw_landmarks, HAND_CONNECTIONS)

    def close(self):
        self.landmarker.close()
