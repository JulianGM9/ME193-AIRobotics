"""Claude's on-court presence for the secret training level.

This is an original starburst icon, NOT a reproduction of Anthropic's
actual (trademarked) Claude logo - reusing a real company's logo inside a
derivative hobby project isn't something to do casually, so this is an
original mark merely evoking an AI presence (warm orange/cream tones,
echoing Anthropic's branding, without copying the mark itself). It pulses
with idle "breathing" and flashes/brightens for hit/score/lose, rather
than using a humanoid character like the other 4 levels - fits a digital
training opponent better than another sprite-person, and sidesteps
needing a 5th full character design.
"""
import math

import pygame

from game.opponent_animator import HIT, LOSE, SCORE
from render.characters import draw_name_tag
from render.projection import project_point
from render.table_view import scaled_radius

ICON_UNIT_SIZE_M = 0.35
BASE_COLOR = (240, 180, 110)
BRIGHT_COLOR = (255, 225, 180)
DIM_COLOR = (140, 90, 50)
NUM_RAYS = 8


def _lerp_color(color_a, color_b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def _pulse_for_state(state, progress, now):
    """Returns (scale, brightness): scale is a size multiplier around 1.0;
    brightness is -1 (dim) .. 0 (base) .. 1 (bright). Idle breathing is
    always applied; hit/score/lose layer a brief flash on top."""
    scale = 1.0 + 0.08 * math.sin(now * 2.4)
    brightness = 0.0

    if state == HIT:
        flash = 1.0 - abs(progress * 2 - 1)
        scale += flash * 0.3
        brightness = flash
    elif state == SCORE:
        pulse = 1.0 - abs(progress * 2 - 1)
        scale += pulse * 0.6
        brightness = pulse
    elif state == LOSE:
        settle = 1.0 - progress
        scale -= settle * 0.25
        brightness = -settle

    return scale, brightness


def draw_claude_icon(surface, x, y, unit, state, progress, now):
    scale, brightness = _pulse_for_state(state, progress, now)
    radius = max(3, int(unit * 0.5 * scale))
    color = (
        _lerp_color(BASE_COLOR, BRIGHT_COLOR, brightness)
        if brightness >= 0
        else _lerp_color(BASE_COLOR, DIM_COLOR, -brightness)
    )

    # Several concentric rings fading outward read as a soft glow - a
    # single flat semi-transparent circle just looked like a muddy brown
    # disc against the Matrix background's black. Always warm/bright
    # regardless of current state color, so it doesn't itself go muddy
    # when dimmed (e.g. on a LOSE reaction).
    glow_radius = int(radius * 1.6)
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    base_alpha = 70 + int(60 * max(0.0, brightness))
    for ring in range(3, 0, -1):
        ring_radius = int(glow_radius * ring / 3)
        ring_alpha = int(base_alpha * (1 - ring / 4))
        pygame.draw.circle(glow_surf, (*BRIGHT_COLOR, ring_alpha), (glow_radius, glow_radius), ring_radius)
    surface.blit(glow_surf, (x - glow_radius, y - glow_radius))

    ray_length = radius * 1.8
    for i in range(NUM_RAYS):
        angle = (2 * math.pi / NUM_RAYS) * i + now * 0.3  # slow idle rotation
        dx, dy = math.cos(angle), math.sin(angle)
        start = (x + dx * radius * 0.5, y + dy * radius * 0.5)
        end = (x + dx * ray_length, y + dy * ray_length)
        pygame.draw.line(surface, color, start, end, width=max(2, radius // 8))

    pygame.draw.circle(surface, color, (int(x), int(y)), radius)
    pygame.draw.circle(surface, BRIGHT_COLOR, (int(x), int(y)), max(2, radius // 3))


def draw_claude(surface, camera, opponent_x, physics_cfg, animator, now, name_font, skill_font,
                 paddle_height=0.3):
    half_l = physics_cfg["table_length"] / 2
    projected = project_point(opponent_x, paddle_height, -half_l - 0.1, camera)
    if projected is None:
        return
    x, y, scale = projected
    unit = scaled_radius(ICON_UNIT_SIZE_M, scale, min_px=10)
    state, progress = animator.current_state(now)
    draw_claude_icon(surface, x, y, unit, state, progress, now)
    draw_name_tag(surface, name_font, skill_font, "CLAUDE", "Q-LEARNING", (255, 180, 100),
                  center_x=x, bottom_y=y - unit * 2.8)
