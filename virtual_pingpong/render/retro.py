"""8-bit visual foundation: a global pixelation post-process filter (so
everything - menus, HUD, the table, future characters - reads as chunky
retro pixel art without redrawing it all individually) and a free/open
pixel bitmap font ("Press Start 2P", SIL Open Font License) for all text,
downloaded and cached the same way the MediaPipe models are.
"""
import os
import urllib.request

import pygame

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/pressstart2p/PressStart2P-Regular.ttf"
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PressStart2P-Regular.ttf")
MIN_FONT_BYTES = 10_000  # a truncated download is far smaller than the real ~115KB file

_font_cache = {}


def ensure_retro_font_downloaded():
    if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) >= MIN_FONT_BYTES:
        return
    if os.path.exists(FONT_PATH):
        os.remove(FONT_PATH)

    print("Downloading retro pixel font (one-time, ~115KB)...")
    tmp_path = FONT_PATH + ".part"
    try:
        urllib.request.urlretrieve(FONT_URL, tmp_path)
        if os.path.getsize(tmp_path) < MIN_FONT_BYTES:
            raise RuntimeError("downloaded font is too small to be valid")
        os.replace(tmp_path, FONT_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("Font downloaded.")


def get_retro_font(size):
    """Cached by size, since pygame.font.Font() re-parses the font file
    each call - this will be called every frame for some sizes (HUD text),
    so caching matters."""
    if size not in _font_cache:
        ensure_retro_font_downloaded()
        _font_cache[size] = pygame.font.Font(FONT_PATH, size)
    return _font_cache[size]


def apply_pixelation(surface, pixel_scale):
    """Returns a new Surface the same size as `surface`, chunked into
    pixel_scale x pixel_scale blocks for a retro low-res look. Both the
    shrink and the enlarge use plain scale() (nearest-neighbor, no
    blending) - smoothscale's area-averaging on the shrink step looked
    blurry/soft rather than crisply blocky, since it blends each block
    toward a muddy average instead of just sampling it."""
    width, height = surface.get_size()
    small_size = (max(1, width // pixel_scale), max(1, height // pixel_scale))
    small = pygame.transform.scale(surface, small_size)
    return pygame.transform.scale(small, (width, height))
