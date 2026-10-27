"""
iphone_camera_test.py

Minimal troubleshooting tool for Continuity Camera: connects to a webcam
by index and just displays the live stream, with no AprilTag detection or
motor code involved -- for narrowing down which cv2 camera index is your
iPhone before using it in apriltagparking_multicam.py.

See apriltagparking_multicam.py's docstring for the Continuity Camera
setup steps (Wi-Fi + Bluetooth on, same Apple ID, Handoff enabled, etc.)
if the iPhone doesn't show up as a camera index at all.

Usage:
    .venv/bin/python iphone_camera_test.py [start_index]

Controls:
    n / right arrow   try the next camera index
    p / left arrow    try the previous camera index
    q                 quit
"""

import sys

import cv2

MAX_INDEX = 5  # highest index to cycle through (0..MAX_INDEX-1)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Camera Test (n/p to switch, q to quit)"


def try_open_camera(index):
    """Open `index` and confirm a frame actually arrives. Returns
    (cap, width, height) on success, or (None, 0, 0) on failure --
    doesn't exit, so the caller can just try a different index."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  index {index}: not available")
        cap.release()
        return None, 0, 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # isOpened() can lie on macOS: the real test is whether a frame arrives.
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {index}: OK, {w}x{h}")
            return cap, w, h

    print(f"  index {index}: opened but produced no frames")
    cap.release()
    return None, 0, 0


def find_first_working_index(start_index):
    """Starting at start_index, scan forward (wrapping around) for the
    first index that actually opens and produces a frame."""
    print("Scanning for a working camera...")
    for offset in range(MAX_INDEX):
        index = (start_index + offset) % MAX_INDEX
        cap, w, h = try_open_camera(index)
        if cap is not None:
            return cap, index, w, h
    return None, start_index, 0, 0


def main():
    start_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap, index, width, height = find_first_working_index(start_index)
    if cap is None:
        print(
            f"Error: no camera opened successfully on indices 0-{MAX_INDEX - 1}.\n"
            "  - On macOS this is almost always a permissions problem. Grant\n"
            "    camera access to your terminal app in System Settings ->\n"
            "    Privacy & Security -> Camera, then fully quit and reopen it.\n"
            "  - If you're trying to reach an iPhone, see the Continuity Camera\n"
            "    setup notes in apriltagparking_multicam.py's docstring."
        )
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)
    print(f"Showing index {index} ({width}x{height}). Press 'n'/'p' to switch, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"Warning: lost the feed from index {index}.")
                break

            cv2.putText(frame, f"index {index}  ({width}x{height})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "n/right = next   p/left = prev   q = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            step = 0
            if key in (ord("n"), 83):   # 'n' or right arrow
                step = 1
            elif key in (ord("p"), 81):  # 'p' or left arrow
                step = -1
            elif key == ord("q"):
                break

            if step != 0:
                next_index = (index + step) % MAX_INDEX
                print(f"Trying index {next_index}...")
                new_cap, new_w, new_h = try_open_camera(next_index)
                if new_cap is None:
                    print(f"Could not open index {next_index}; staying on index {index}.")
                else:
                    cap.release()
                    cap, index, width, height = new_cap, next_index, new_w, new_h
                    print(f"Now showing index {index} ({width}x{height}).")

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
