"""Standalone tool: generate printable AprilTag images (PNGs + one combined
PDF) for the levels defined in config.yaml's levels.tags mapping.
Run with: python3 tools/apriltag_generator.py
Outputs to tools/generated_tags/.
"""
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

OUTPUT_DIR = Path(__file__).parent / "generated_tags"
TAG_PIXELS = 600
MARGIN_PIXELS = 80  # AprilTag detection needs a quiet white zone around the tag
LABEL_HEIGHT = 80


def generate():
    cfg = config.load()
    tags = cfg["levels"]["tags"]  # {tag_id: level_name}
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    OUTPUT_DIR.mkdir(exist_ok=True)

    pages = []
    for tag_id, level_name in sorted(tags.items()):
        tag_img = aruco.generateImageMarker(dictionary, tag_id, TAG_PIXELS)
        tag_img = cv2.copyMakeBorder(
            tag_img, MARGIN_PIXELS, MARGIN_PIXELS, MARGIN_PIXELS, MARGIN_PIXELS,
            cv2.BORDER_CONSTANT, value=255,
        )

        full_size = (tag_img.shape[1], tag_img.shape[0] + LABEL_HEIGHT)
        page = Image.new("L", full_size, color=255)
        page.paste(Image.fromarray(tag_img), (0, 0))

        draw = ImageDraw.Draw(page)
        label = f"Tag {tag_id}: {level_name.capitalize()}"
        try:
            font = ImageFont.truetype("Arial.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
        text_width = draw.textlength(label, font=font)
        draw.text(((full_size[0] - text_width) / 2, tag_img.shape[0] + 15),
                   label, fill=0, font=font)

        png_path = OUTPUT_DIR / f"tag_{tag_id}_{level_name}.png"
        page.save(png_path)
        print(f"Wrote {png_path}")
        pages.append(page.convert("RGB"))

    if pages:
        pdf_path = OUTPUT_DIR / "all_tags.pdf"
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    generate()
