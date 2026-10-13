import pygame
import pytest

from game.opponent_animator import HIT, IDLE, LOSE, SCORE, OpponentAnimator
from render.claude_icon import _pulse_for_state, draw_claude, draw_claude_icon


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


def test_idle_scale_oscillates_gently_around_one():
    for now in (0.0, 0.5, 1.0, 1.5, 2.0):
        scale, brightness = _pulse_for_state(IDLE, 0.0, now)
        assert 0.85 <= scale <= 1.15
        assert brightness == 0.0


def test_hit_pulse_peaks_at_the_midpoint():
    start = _pulse_for_state(HIT, 0.0, now=0.0)
    middle = _pulse_for_state(HIT, 0.5, now=0.0)
    end = _pulse_for_state(HIT, 1.0, now=0.0)
    assert middle[1] > start[1]
    assert middle[1] > end[1]
    assert middle[1] == pytest.approx(1.0)


def test_score_pulse_is_brighter_and_bigger_than_hit():
    hit_scale, hit_brightness = _pulse_for_state(HIT, 0.5, now=0.0)
    score_scale, score_brightness = _pulse_for_state(SCORE, 0.5, now=0.0)
    assert score_scale > hit_scale
    assert score_brightness >= hit_brightness


def test_lose_dims_and_shrinks_then_settles():
    early_scale, early_brightness = _pulse_for_state(LOSE, 0.0, now=0.0)
    late_scale, late_brightness = _pulse_for_state(LOSE, 0.9, now=0.0)
    assert early_brightness < 0  # dimmed
    assert early_scale < 1.0  # shrunk
    assert late_brightness > early_brightness  # settling back toward 0
    assert late_scale > early_scale


def test_draw_claude_icon_does_not_crash_in_any_state():
    surface = pygame.Surface((200, 200))
    for state in (IDLE, HIT, SCORE, LOSE):
        draw_claude_icon(surface, 100, 100, unit=40, state=state, progress=0.5, now=1.0)


def test_draw_claude_icon_actually_draws_something():
    surface = pygame.Surface((200, 200))
    surface.fill((1, 2, 3))
    draw_claude_icon(surface, 100, 100, unit=40, state=IDLE, progress=0.0, now=0.0)
    pixels_changed = any(
        surface.get_at((x, y))[:3] != (1, 2, 3)
        for x in range(70, 130) for y in range(70, 130)
    )
    assert pixels_changed


def test_draw_claude_handles_a_point_behind_the_camera_gracefully():
    camera = {"z": 0.0, "height": 1.0, "focal_length": 700, "near_clip": 0.05,
              "screen_center_x": 480, "screen_center_y": 300}
    surface = pygame.Surface((960, 600))
    animator = OpponentAnimator()
    font = pygame.font.Font(None, 16)
    physics_cfg = {"table_length": 2.74}
    draw_claude(surface, camera, 0.0, physics_cfg, animator, now=0.0, name_font=font, skill_font=font)
