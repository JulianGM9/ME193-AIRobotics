"""
apriltagtest.py

Detect AprilTags with the webcam and print each tag's position and
apparent size to the terminal.

Position is the tag's center in pixel coordinates (0, 0 at the top-left of
the frame). Size is the tag's average side length in pixels -- there's no
camera calibration here, so it isn't a real-world distance, but it's a
reliable proxy: a bigger number means the tag is closer to the camera
(confirmed: a tag drawn at 80px detects smaller than one drawn at 200px).

Usage:
    .venv/bin/python apriltagtest.py

Controls:
    Press 'q' or close the preview window to quit.

Uses OpenCV's built-in cv2.aruco module, which ships the AprilTag
dictionaries directly -- no separate `apriltag`/`pupil-apriltags` package
needed. Checks every common AprilTag family each frame (see TAG_FAMILIES)
since a tag silently fails to decode if the family is wrong -- there's no
way to tell "wrong family" apart from "no tag visible" otherwise.
"""

import sys
import time

import cv2
import numpy as np

TAG_FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
CAMERA_INDEX = 0
PRINT_INTERVAL = 0.1  # seconds between console updates, so ~30fps detection
                      # doesn't flood the terminal
WINDOW_NAME = "AprilTag Detection (press 'q' to quit)"


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() or not cap.read()[0]:
        print(
            "Error: could not open the webcam.\n"
            "  - On macOS this is almost always a permissions problem. Grant\n"
            "    camera access to your terminal app in System Settings ->\n"
            "    Privacy & Security -> Camera, then fully quit and reopen it.\n"
            f"  - Otherwise try a different CAMERA_INDEX (currently {CAMERA_INDEX})."
        )
        return None
    return cap


def tag_center_and_size(tag_corners):
    """tag_corners is the 1x4x2 corner array OpenCV returns per detected tag."""
    points = tag_corners[0]
    center_x, center_y = points.mean(axis=0)
    side_lengths = [
        float(np.linalg.norm(points[i] - points[(i + 1) % 4])) for i in range(4)
    ]
    size = sum(side_lengths) / len(side_lengths)
    return center_x, center_y, size


def main():
    cap = open_camera()
    if cap is None:
        sys.exit(1)

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detectors = {
        family_name: cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(family_id), detector_params
        )
        for family_name, family_id in TAG_FAMILIES.items()
    }

    cv2.namedWindow(WINDOW_NAME)
    last_print = 0.0
    print(f"Looking for AprilTags (families: {', '.join(TAG_FAMILIES)}). Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: lost the camera feed.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            now = time.monotonic()
            should_print = now - last_print >= PRINT_INTERVAL

            for family_name, detector in detectors.items():
                corners, ids, _rejected = detector.detectMarkers(gray)
                if ids is None:
                    continue

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                if should_print:
                    last_print = now

                for tag_corners, tag_id in zip(corners, ids.flatten()):
                    cx, cy, size = tag_center_and_size(tag_corners)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    if should_print:
                        print(
                            f"tag {int(tag_id):3d} ({family_name})  "
                            f"x={cx:7.1f}  y={cy:7.1f}  size={size:6.1f}px"
                        )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
