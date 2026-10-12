"""Standalone tool: hold up a printed AprilTag (see tools/generated_tags/)
and confirm the stable-hold level-selection flow works - progress ring
while holding, confirmation after ~1 second (it waits indefinitely
otherwise, no timeout/default), and the 1-4 keyboard fallback.
Run with: python3 tools/apriltag_test.py
Press 1-4 to pick a level directly, 'r' to restart selection, 'q' to quit.
"""
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from vision.apriltag import LevelSelector, create_detector, detect_tags
from vision.camera import Camera

WINDOW_NAME = "AprilTag level select test"
KEY_TO_LEVEL = {
    ord("1"): "beginner",
    ord("2"): "intermediate",
    ord("3"): "advanced",
    ord("4"): "expert",
}


def main():
    cfg = config.load()
    levels_cfg = cfg["levels"]
    tag_to_level = {int(k): v for k, v in levels_cfg["tags"].items()}

    camera = Camera()
    if not camera.open():
        print("Error: could not open the webcam.")
        return
    camera.start()

    detector = create_detector()
    selector = LevelSelector(
        tag_to_level=tag_to_level,
        confirm_hold_seconds=levels_cfg["confirm_hold_seconds"],
    )
    selector.start(now=time.monotonic())

    print("Hold up a tag to select a level (see tools/generated_tags/).")
    print("Keys: 1-4=pick level directly  r=restart selection  q=quit")

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            now = time.monotonic()

            # Detect on the raw (unmirrored) frame - AprilTags aren't
            # mirror-symmetric in their bit encoding, so running detection
            # on an already-flipped frame makes every real tag undecodable
            # (confirmed: the detector still finds a marker-shaped square,
            # but it fails to verify as a valid tag). Mirror only for
            # display, transforming detected corners into that mirrored
            # coordinate space so the overlay still lines up.
            tags = detect_tags(frame, detector)
            width = frame.shape[1]
            frame = cv2.flip(frame, 1)

            mirrored_tags = []
            for tag_id, corners in tags:
                mirrored_corners = corners.copy()
                mirrored_corners[:, 0] = width - 1 - mirrored_corners[:, 0]
                mirrored_tags.append((tag_id, mirrored_corners))
            tags = mirrored_tags

            for tag_id, corners in tags:
                pts = corners.astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                center = pts.mean(axis=0).astype(int)
                cv2.putText(frame, f"id={tag_id}", tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            confirmed = selector.update([tag_id for tag_id, _ in tags], now)

            if confirmed is not None:
                cv2.putText(frame, f"LEVEL: {confirmed.upper()}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                progress = selector.progress(now)
                if progress > 0 and tags:
                    tx, ty = tags[0][1].mean(axis=0).astype(int)
                    radius = 50
                    cv2.ellipse(frame, (tx, ty), (radius, radius), -90, 0,
                                360 * progress, (0, 255, 255), 4)
                cv2.putText(frame, "Hold a tag steady...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                selector.start(now=time.monotonic())
                print("Restarted selection.")
            elif key in KEY_TO_LEVEL:
                level = selector.select_by_key(KEY_TO_LEVEL[key])
                print(f"Selected by key: {level}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
