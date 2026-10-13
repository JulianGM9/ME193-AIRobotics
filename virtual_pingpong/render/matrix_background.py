"""Matrix-style falling-code background for the secret Claude level.

Stateless and deterministic: entirely a function of `now`, so (like the
other backgrounds) there's no per-column state object to create/own/reset
- it just looks different each frame because `now` changed. Column speeds
and glyph choices come from a cheap integer hash rather than Python's
random module, for the same reason (no RNG instance to carry around).
"""
import pygame

BG_COLOR = (0, 0, 0)
GLYPH_COLOR_HEAD = (210, 255, 210)
GLYPH_COLOR_TRAIL = (0, 150, 60)
GLYPHS = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COLUMN_WIDTH = 20
ROW_HEIGHT = 20
TRAIL_LENGTH = 9
FALL_SPEED_PX_PER_SEC = 200
GLYPH_CHANGE_HZ = 4  # how often a given trail position's glyph flips

_glyph_cache = {}


def _pseudo_random(seed):
    """Deterministic float in [0, 1) from an integer seed (cheap hash, not
    a real RNG - this background has no state to seed a proper one with).
    Half-open (never exactly 1.0) since this is used to index into
    sequences like GLYPHS - dividing by 0xFFFF instead of 0x10000 let it
    land on exactly 1.0 and index one past the end."""
    x = (seed * 2654435761) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    return (x & 0xFFFF) / 0x10000


def _glyph_surface(font, glyph, fade_bucket, max_buckets=8):
    """Font rendering is too expensive to redo for every glyph every
    frame at 60fps - cache by (glyph, fade bucket) and blit instead."""
    key = (id(font), glyph, fade_bucket)
    cached = _glyph_cache.get(key)
    if cached is not None:
        return cached
    fade = fade_bucket / max_buckets
    color = tuple(int(GLYPH_COLOR_TRAIL[i] + (GLYPH_COLOR_HEAD[i] - GLYPH_COLOR_TRAIL[i]) * fade) for i in range(3))
    surf = font.render(glyph, True, color)
    _glyph_cache[key] = surf
    return surf


def draw_matrix_background(surface, now, font):
    surface.fill(BG_COLOR)
    width, height = surface.get_size()
    num_columns = width // COLUMN_WIDTH
    cycle_height = height + TRAIL_LENGTH * ROW_HEIGHT

    for col in range(num_columns):
        speed = FALL_SPEED_PX_PER_SEC * (0.6 + 0.8 * _pseudo_random(col * 97 + 11))
        offset = _pseudo_random(col * 53 + 7) * cycle_height
        head_y = (now * speed + offset) % cycle_height

        for trail_index in range(TRAIL_LENGTH):
            y = head_y - trail_index * ROW_HEIGHT
            if y < -ROW_HEIGHT or y > height:
                continue
            glyph_tick = int(now * GLYPH_CHANGE_HZ)
            glyph_seed = col * 1000 + trail_index * 31 + glyph_tick
            glyph = GLYPHS[int(_pseudo_random(glyph_seed) * len(GLYPHS))]
            fade_bucket = max(0, 8 - trail_index)
            glyph_surf = _glyph_surface(font, glyph, fade_bucket)
            surface.blit(glyph_surf, (col * COLUMN_WIDTH, y))
