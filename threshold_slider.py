"""
threshold_slider.py

Display an image with a live slider that sets a brightness cutoff:
pixels brighter than the threshold are shown white, everything else black.
Erosion and dilation can be toggled on top of that, and a third toggle
subtracts the eroded/dilated result back off of the original thresholded
image (useful for pulling out the boundary erosion or dilation removed).

Usage:
    .venv/bin/python threshold_slider.py [image_path]

Controls:
    Threshold slider (0-255)  -- the black/white cutoff.
    Erosion / Dilation / Subtract Original -- 0 = off, 1 = on. This
        OpenCV build's GUI backend (Cocoa, not Qt) has no real checkbox
        widget, so on/off trackbars are the standard stand-in for
        checkboxes here.
    Press 'q' or close the window to quit.
"""

import sys

import cv2

IMAGE_PATH = "/Users/julianmoody/Desktop/IMG_1827.jpeg"
WINDOW_NAME = "Threshold (press 'q' to quit)"
MAX_DISPLAY_DIM = 1000  # shrink large photos so the window fits on screen
MORPH_KERNEL_SIZE = 3
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))


def load_grayscale(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: could not read image: {path}")
        sys.exit(1)

    height, width = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIM / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    gray = load_grayscale(path)

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Threshold", WINDOW_NAME, 127, 255, lambda _pos: None)
    cv2.createTrackbar("Erosion (0=off 1=on)", WINDOW_NAME, 0, 1, lambda _pos: None)
    cv2.createTrackbar("Dilation (0=off 1=on)", WINDOW_NAME, 0, 1, lambda _pos: None)
    cv2.createTrackbar("Subtract Original (0=off 1=on)", WINDOW_NAME, 0, 1, lambda _pos: None)

    while True:
        threshold = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        _, black_and_white = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        result = black_and_white
        if cv2.getTrackbarPos("Erosion (0=off 1=on)", WINDOW_NAME):
            result = cv2.erode(result, MORPH_KERNEL)
        if cv2.getTrackbarPos("Dilation (0=off 1=on)", WINDOW_NAME):
            result = cv2.dilate(result, MORPH_KERNEL)

        if cv2.getTrackbarPos("Subtract Original (0=off 1=on)", WINDOW_NAME):
            # Saturating subtract (clamps at 0) instead of plain "-" so this
            # can't wrap around to near-255 where the eroded/dilated image
            # is brighter than the original.
            result = cv2.subtract(black_and_white, result)

        cv2.imshow(WINDOW_NAME, result)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
