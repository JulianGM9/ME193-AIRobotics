"""
kernel_sharpen_animation.py

Grayscale the same photo, then animate a 3x3 sharpening kernel sliding
across it pixel by pixel, showing the convolution math build the
sharpened output live.

Usage:
    .venv/bin/python kernel_sharpen_animation.py [image_path]

Controls:
    "Steps/frame" slider -- how many pixels to compute between screen
        redraws. Higher = faster animation but less visible per-step detail.
    Press 'q' or close the window to quit.
"""

import sys

import cv2
import numpy as np

IMAGE_PATH = "/Users/julianmoody/Desktop/IMG_1827.jpeg"
WINDOW_NAME = "Kernel Sharpening Animation (press 'q' to quit)"

# Kept small on purpose: this animates every single pixel's convolution
# step, so the working image is shrunk first -- a full-resolution photo
# would mean millions of animation steps.
ANIM_MAX_DIM = 120
DISPLAY_SCALE = 8  # nearest-neighbor upscale so individual pixels are visible

SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype=np.int32)

HIGHLIGHT_COLOR = (0, 0, 255)  # BGR red
LABEL_COLOR = (0, 255, 0)      # BGR green


def load_small_grayscale(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: could not read image: {path}")
        sys.exit(1)

    height, width = image.shape[:2]
    scale = ANIM_MAX_DIM / max(height, width)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = cv2.resize(image, new_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_panel(gray_image, title):
    upscaled = cv2.resize(
        gray_image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST
    )
    bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2)
    return bgr


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    gray = load_small_grayscale(path)
    height, width = gray.shape

    # 1px replicate-border pad so the 3x3 kernel has a full window at every
    # output pixel, including the image edges.
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.int32)
    output = np.zeros_like(gray)

    positions = [(y, x) for y in range(height) for x in range(width)]
    total = len(positions)
    last_idx = 0

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Steps/frame", WINDOW_NAME, 5, 300, lambda _pos: None)

    i = 0
    while True:
        if i < total:
            steps = max(1, cv2.getTrackbarPos("Steps/frame", WINDOW_NAME))
            for _ in range(steps):
                if i >= total:
                    break
                y, x = positions[i]
                # padded[y:y+3, x:x+3] is centered on output pixel (y, x)
                # because of the 1px border added above.
                window = padded[y:y + 3, x:x + 3]
                value = int(np.sum(window * SHARPEN_KERNEL))
                output[y, x] = np.clip(value, 0, 255)
                last_idx = i
                i += 1

        hy, hx = positions[min(last_idx, total - 1)]

        original_panel = draw_panel(gray, "Original (kernel window highlighted)")
        top_left = ((hx - 1) * DISPLAY_SCALE, (hy - 1) * DISPLAY_SCALE)
        bottom_right = ((hx + 2) * DISPLAY_SCALE, (hy + 2) * DISPLAY_SCALE)
        cv2.rectangle(original_panel, top_left, bottom_right, HIGHLIGHT_COLOR, 2)

        output_panel = draw_panel(output, "Sharpened output (building up)")
        combined = np.hstack([original_panel, output_panel])
        cv2.imshow(WINDOW_NAME, combined)

        delay_ms = 1 if i < total else 30  # idle at low CPU once finished
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
