import pygame
import pytest

from game.opponent_animator import HIT, IDLE, LOSE, SCORE, OpponentAnimator
from render.characters import (
    CHARACTER_SCHEMES,
    _pose_for_state,
    draw_character,
    draw_name_tag,
    draw_opponent,
)

LEVELS = ["beginner", "intermediate", "advanced", "expert"]


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


# --- CHARACTER_SCHEMES ---

@pytest.mark.parametrize("level_name", LEVELS)
def test_every_level_has_a_complete_character_scheme(level_name):
    scheme = CHARACTER_SCHEMES[level_name]
    for key in ("name", "skill_label", "skin_color", "body_color", "accent_color", "accessory", "tag_color"):
        assert key in scheme


def test_every_level_has_a_distinct_name():
    names = {CHARACTER_SCHEMES[level]["name"] for level in LEVELS}
    assert len(names) == len(LEVELS)


# --- _pose_for_state (pure logic) ---

def test_idle_pose_has_no_event_offsets():
    pose = _pose_for_state(IDLE, 0.0, sway_deg=0.0)
    assert pose["arms_up"] == 0.0
    assert pose["head_dy"] == 0.0
    assert pose["body_dy"] == 0.0


def test_idle_pose_leans_with_sway():
    pose = _pose_for_state(IDLE, 0.0, sway_deg=5.0)
    assert pose["lean_x"] > 0


def test_hit_pose_peaks_at_the_midpoint():
    start = _pose_for_state(HIT, 0.0, sway_deg=0.0)
    middle = _pose_for_state(HIT, 0.5, sway_deg=0.0)
    end = _pose_for_state(HIT, 1.0, sway_deg=0.0)
    assert middle["arm_swing"] > start["arm_swing"]
    assert middle["arm_swing"] > end["arm_swing"]
    assert middle["arm_swing"] == pytest.approx(1.0)


def test_score_pose_raises_both_arms_and_bounces_up():
    pose = _pose_for_state(SCORE, 0.5, sway_deg=0.0)
    assert pose["arms_up"] == 1.0
    assert pose["body_dy"] < 0  # negative = up, screen-space y
    assert pose["head_dy"] < 0


def test_score_pose_keeps_arms_up_throughout():
    assert _pose_for_state(SCORE, 0.0, sway_deg=0.0)["arms_up"] == 1.0
    assert _pose_for_state(SCORE, 1.0, sway_deg=0.0)["arms_up"] == 1.0


def test_lose_pose_slumps_down_and_settles_by_the_end():
    early = _pose_for_state(LOSE, 0.0, sway_deg=0.0)
    late = _pose_for_state(LOSE, 0.9, sway_deg=0.0)
    assert early["head_dy"] > 0  # positive = down, screen-space y
    assert early["body_dy"] > 0
    assert late["head_dy"] < early["head_dy"]
    assert late["body_dy"] < early["body_dy"]


# --- Drawing smoke tests (this project doesn't unit-test pixel output of
# draw_* functions beyond "it runs and visibly draws something" - see
# tests/test_backgrounds.py for the same convention) ---

@pytest.mark.parametrize("level_name", LEVELS)
def test_draw_character_does_not_crash_for_every_level(level_name):
    surface = pygame.Surface((200, 200))
    scheme = CHARACTER_SCHEMES[level_name]
    pose = _pose_for_state(IDLE, 0.0, sway_deg=0.0)
    draw_character(surface, 100, 150, unit=40, scheme=scheme, pose=pose)


@pytest.mark.parametrize("state", [IDLE, HIT, SCORE, LOSE])
def test_draw_character_does_not_crash_in_any_animation_state(state):
    surface = pygame.Surface((200, 200))
    scheme = CHARACTER_SCHEMES["expert"]
    pose = _pose_for_state(state, 0.5, sway_deg=3.0)
    draw_character(surface, 100, 150, unit=40, scheme=scheme, pose=pose)


def test_draw_character_actually_draws_something():
    surface = pygame.Surface((200, 200))
    surface.fill((1, 2, 3))
    scheme = CHARACTER_SCHEMES["beginner"]
    pose = _pose_for_state(IDLE, 0.0, sway_deg=0.0)
    draw_character(surface, 100, 150, unit=40, scheme=scheme, pose=pose)
    pixels_changed = any(
        surface.get_at((x, y))[:3] != (1, 2, 3)
        for x in range(60, 140) for y in range(60, 180)
    )
    assert pixels_changed


def test_draw_name_tag_does_not_crash():
    surface = pygame.Surface((200, 200))
    font = pygame.font.Font(None, 16)
    draw_name_tag(surface, font, font, "TEST NAME", "LV.1 TEST", (255, 0, 0), center_x=100, bottom_y=100)


def test_draw_opponent_handles_a_point_behind_the_camera_gracefully():
    # project_point returns None when behind the near clip plane - this
    # must not raise, just skip drawing.
    camera = {"z": 0.0, "height": 1.0, "focal_length": 700, "near_clip": 0.05,
              "screen_center_x": 480, "screen_center_y": 300}
    surface = pygame.Surface((960, 600))
    animator = OpponentAnimator()
    font = pygame.font.Font(None, 16)
    physics_cfg = {"table_length": 2.74}
    draw_opponent(surface, camera, 0.0, physics_cfg, "beginner", animator, now=0.0,
                  name_font=font, skill_font=font)
