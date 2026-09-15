"""
grayscale_photo.py

Convert a photo to grayscale.

Usage:
    .venv/bin/python grayscale_photo.py [input_path] [output_path]

If input_path is omitted, IMAGE_PATH below is used. If output_path is
omitted, the result is saved next to the input with "_grayscale" appended
to the filename.
"""

import os
import sys

from PIL import Image

# Default input photo if no path is given on the command line.
IMAGE_PATH = "/Users/julianmoody/Desktop/IMG_1827.jpeg"


def grayscale_image(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_grayscale{ext}"

    with Image.open(input_path) as img:
        img.convert("L").save(output_path)

    print(f"Saved grayscale image to {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    grayscale_image(input_path, output_path)
