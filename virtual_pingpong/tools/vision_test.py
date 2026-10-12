"""Standalone tool: open the webcam, run pose tracking + paddle detection,
and overlay the skeleton, fused paddle position, and wrist game-space
coordinates - for validating the whole vision subsystem together before
wiring it into the game.
Run with: python3 tools/vision_test.py
Press 'h' to toggle handedness, 'q' to quit.
"""
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from vision.camera import Camera
from vision.coords import to_game_space
from vision.hand_tracker import HandTracker
from vision.paddle_detector import PaddleDetector
from vision.pose_tracker import PoseTracker

WINDOW_NAME = "Vision subsystem test"

SOURCE_COLORS = {
    "hand": (0, 255, 0),
    "wrist": (255, 255, 0),
}


def main():
    cfg = config.load()

    camera = Camera()
    print("Opening camera...")
    if not camera.open():
        print("Error: could not open the webcam.")
        return
    camera.start()

    print("Loading pose + hand landmarkers (downloads models on first run)...")
    tracker = PoseTracker(handedness="right")
    hand_tracker = HandTracker()
    paddle_detector = PaddleDetector(cfg["paddle"])
    print("Ready. Press 'h' to toggle handedness, 'q' to quit.")

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

            frame = cv2.flip(frame, 1)  # mirror for an intuitive view
            pose = tracker.process(frame)
            tracker.draw_debug(frame)
            hand_position = hand_tracker.process(frame)
            h, w = frame.shape[:2]

            if pose is not None:
                game_xy = to_game_space(pose, "wrist")
                if game_xy is not None:
                    gx, gy = game_xy
                    cv2.putText(frame, f"wrist game-space: ({gx:+.2f}, {gy:+.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                paddle_pos, velocity = paddle_detector.update(pose, hand_position, dt)
                px, py = int(paddle_pos[0] * w), int(paddle_pos[1] * h)
                color = SOURCE_COLORS[paddle_detector.last_source]
                cv2.circle(frame, (px, py), 14, color, 3)
                cv2.putText(frame, f"paddle source: {paddle_detector.last_source}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"paddle velocity: ({velocity[0]:+.2f}, {velocity[1]:+.2f})/s",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"handedness: {tracker.handedness}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                tracker.set_handedness("left" if tracker.handedness == "right" else "right")
    finally:
        camera.stop()
        tracker.close()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
