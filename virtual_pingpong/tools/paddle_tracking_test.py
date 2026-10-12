"""Standalone tool: compare four candidate "where is the paddle" methods
live, to decide which one to actually use for gameplay. Move/rotate the
paddle (showing red then black) and watch which colored dot stays most
consistently on it.

Methods shown simultaneously:
  cyan   - raw wrist (MediaPipe Pose) - the baseline, already confirmed
           reliable in isolation via vision_test.py
  green  - hand palm-center (MediaPipe Hands, averaging palm landmarks -
           the same technique proven in Hand control files/mediapipetest.py)
  red    - the existing color-based PaddleDetector (red/dark/forearm
           fusion) - shown for comparison, to see how it actually fails
  yellow - Hough-circle detection near the wrist (shape-only, no color -
           looks for a round object regardless of which paddle face shows)

Run with: python3 tools/paddle_tracking_test.py
Press 'h' to toggle handedness, 'q' to quit.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode as HandRunningMode

import config
from vision.camera import Camera
from vision.paddle_detector import PaddleDetector
from vision.pose_tracker import PoseTracker

# Same model + same macOS GPU/SRGBA requirement as
# "Hand control files/mediapipetest.py" - HandLandmarker hard-crashes
# (uncatchable SIGABRT) without both.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MIN_MODEL_BYTES = 1_000_000
PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + base of each finger

HOUGH_SEARCH_RADIUS_FRACTION = 0.15  # of frame width, centered on the wrist
HOUGH_DP = 1.2
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30  # lower = more (and more false-positive) circles detected

COLOR_WRIST = (255, 255, 0)
COLOR_HAND_PALM = (0, 255, 0)
COLOR_COLOR_DETECTOR = (0, 0, 255)
COLOR_HOUGH = (0, 255, 255)


def ensure_hand_model_downloaded():
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        return
    if HAND_MODEL_PATH.exists():
        HAND_MODEL_PATH.unlink()
    print("Downloading hand-landmark model (one-time, ~8 MB)...")
    tmp_path = HAND_MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(HAND_MODEL_URL, tmp_path)
    if tmp_path.stat().st_size < MIN_MODEL_BYTES:
        tmp_path.unlink()
        raise RuntimeError("downloaded hand-landmark model is too small to be valid")
    os.replace(tmp_path, HAND_MODEL_PATH)
    print("Model downloaded.")


def create_hand_landmarker():
    ensure_hand_model_downloaded()
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH),
                delegate=BaseOptions.Delegate.GPU,  # required on macOS - see module docstring
            ),
            running_mode=HandRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )


def hough_circle_near(frame_bgr, center_px, search_radius_px):
    """Look for a circular blob (shape only, no color) near center_px.
    Returns (x, y) in full-frame pixel coords, or None."""
    cx, cy = center_px
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, int(cx - search_radius_px)), min(w, int(cx + search_radius_px))
    y0, y1 = max(0, int(cy - search_radius_px)), min(h, int(cy + search_radius_px))
    if x1 <= x0 or y1 <= y0:
        return None

    roi_gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        roi_gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=search_radius_px,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=search_radius_px // 4, maxRadius=search_radius_px,
    )
    if circles is None:
        return None
    circle_x, circle_y, _ = circles[0][0]
    return (x0 + circle_x, y0 + circle_y)


def main():
    cfg = config.load()
    camera = Camera()
    if not camera.open():
        print("Error: could not open the webcam.")
        return
    camera.start()

    pose_tracker = PoseTracker(handedness="right")
    paddle_detector = PaddleDetector(cfg["paddle"])
    hand_landmarker = create_hand_landmarker()
    last_hand_timestamp_ms = -1

    print("cyan=wrist  green=hand-palm-center  red=color-detector  yellow=hough-circle")
    print("Press 'h' to toggle handedness, 'q' to quit.")

    last_time = time.monotonic()
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            pose = pose_tracker.process(frame)

            if pose is not None:
                wx, wy = pose["wrist"]
                wrist_px = (int(wx * width), int(wy * height))
                cv2.circle(frame, wrist_px, 10, COLOR_WRIST, 2)
                cv2.putText(frame, "wrist", (wrist_px[0] + 12, wrist_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WRIST, 1)

                paddle_pos, _ = paddle_detector.update(frame, pose, dt)
                paddle_px = (int(paddle_pos[0] * width), int(paddle_pos[1] * height))
                cv2.circle(frame, paddle_px, 10, COLOR_COLOR_DETECTOR, 2)
                cv2.putText(frame, f"color ({paddle_detector.last_source})",
                            (paddle_px[0] + 12, paddle_px[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COLOR_DETECTOR, 1)

                search_radius_px = int(HOUGH_SEARCH_RADIUS_FRACTION * width)
                circle = hough_circle_near(frame, wrist_px, search_radius_px)
                if circle is not None:
                    circle_px = (int(circle[0]), int(circle[1]))
                    cv2.circle(frame, circle_px, 10, COLOR_HOUGH, 2)
                    cv2.putText(frame, "hough", (circle_px[0] + 12, circle_px[1] + 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HOUGH, 1)

            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba))
            timestamp_ms = max(int(time.monotonic() * 1000), last_hand_timestamp_ms + 1)
            last_hand_timestamp_ms = timestamp_ms
            try:
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except ValueError:
                hand_result = None

            if hand_result is not None and hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                hx = sum(landmarks[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
                hy = sum(landmarks[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
                palm_px = (int(hx * width), int(hy * height))
                cv2.circle(frame, palm_px, 10, COLOR_HAND_PALM, 2)
                cv2.putText(frame, "hand-palm", (palm_px[0] + 12, palm_px[1] + 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAND_PALM, 1)

            cv2.imshow("Paddle tracking comparison", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                pose_tracker.set_handedness("left" if pose_tracker.handedness == "right" else "right")
    finally:
        camera.stop()
        pose_tracker.close()
        hand_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
