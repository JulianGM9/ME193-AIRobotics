"""Opponent character rendering: a procedural pixel-art humanoid (simple
rects/circles, consistent with this project's no-asset-pipeline approach),
one per difficulty level with its own name, skill tag, color scheme and
accessory - posed each frame from game/opponent_animator.py's state
(idle sway, hit swing, score celebration, lose slump).
"""
import pygame

from game.opponent_animator import HIT, LOSE, SCORE
from render.projection import project_point
from render.table_view import MAX_RADIUS_PX, scaled_radius

CHARACTER_UNIT_SIZE_M = 0.4  # roughly head-to-waist, real-world scale for perspective sizing

CHARACTER_SCHEMES = {
    "beginner": {
        "name": "ROOKIE ROBBIE",
        "skill_label": "LV.1 BEGINNER",
        "skin_color": (230, 190, 150),
        "body_color": (70, 120, 210),
        "accent_color": (210, 200, 70),
        "accessory": None,
        "tag_color": (120, 200, 255),
    },
    "intermediate": {
        "name": "GYM TEACHER GERALD",
        "skill_label": "LV.2 INTERMEDIATE",
        "skin_color": (210, 160, 120),
        "body_color": (230, 140, 30),
        "accent_color": (255, 220, 60),
        "accessory": "headband",
        "tag_color": (255, 180, 80),
    },
    "advanced": {
        "name": "ACE MCSLAM",
        "skill_label": "LV.3 ADVANCED",
        "skin_color": (180, 130, 100),
        "body_color": (130, 60, 170),
        "accent_color": (230, 230, 230),
        "accessory": "visor",
        "tag_color": (200, 120, 255),
    },
    "expert": {
        "name": "THE MACHINE",
        "skill_label": "LV.4 EXPERT",
        "skin_color": (170, 175, 185),
        "body_color": (60, 65, 75),
        "accent_color": (220, 40, 40),
        "accessory": "antenna",
        "tag_color": (255, 70, 70),
    },
}


def _pose_for_state(state, progress, sway_deg):
    """Returns pixel-space pose offsets, derived from the animator's
    current (state, progress) plus the always-on idle sway angle. Kept
    separate from drawing so the "what does each animation look like"
    mapping is in one obvious place.

    lean_x: horizontal shimmy (idle sway, in sway-degree units treated as
        a small px shift - there's no real rotation here, just a cheap
        side-to-side read as swaying).
    arm_swing: -1 (back) .. 1 (forward extended) for the paddle arm.
    arms_up: 0-1, both arms raised (score celebration).
    head_dy / body_dy: vertical offsets (score = bounce up, lose = slump
        down).
    """
    pose = {"lean_x": sway_deg * 0.6, "arm_swing": sway_deg * 0.05, "arms_up": 0.0, "head_dy": 0.0, "body_dy": 0.0}

    if state == HIT:
        # Swings forward then back over the course of the animation.
        swing = 1.0 - abs(progress * 2 - 1)  # 0 -> 1 -> 0
        pose["arm_swing"] = swing
        pose["lean_x"] = swing * 2.0
    elif state == SCORE:
        # A quick upward bounce with both arms thrown up.
        bounce = 1.0 - abs(progress * 2 - 1)
        pose["arms_up"] = 1.0
        pose["body_dy"] = -bounce * 6
        pose["head_dy"] = -bounce * 6
    elif state == LOSE:
        # A slump that settles back by the end of the animation.
        settle = 1.0 - progress
        pose["head_dy"] = settle * 9
        pose["body_dy"] = settle * 6
        pose["arm_swing"] = -settle * 0.8

    return pose


def _draw_accessory(surface, kind, head_rect, accent_color):
    if kind == "headband":
        band_rect = pygame.Rect(head_rect.x, head_rect.centery - 2, head_rect.width, 4)
        pygame.draw.rect(surface, accent_color, band_rect)
    elif kind == "visor":
        visor_rect = pygame.Rect(head_rect.x, head_rect.centery - 3, head_rect.width, 6)
        pygame.draw.rect(surface, accent_color, visor_rect, border_radius=2)
    elif kind == "antenna":
        top = (head_rect.centerx, head_rect.top)
        tip = (head_rect.centerx, head_rect.top - head_rect.height // 2)
        pygame.draw.line(surface, (120, 120, 130), top, tip, width=2)
        pygame.draw.circle(surface, accent_color, tip, 3)


def draw_character(surface, x, y, unit, scheme, pose):
    """Draws the humanoid centered at screen position (x, y) (feet level),
    `unit` pixels tall-ish (a rough head-to-waist scale), per `scheme`
    (see CHARACTER_SCHEMES) posed per `pose` (see _pose_for_state)."""
    body_color = scheme["body_color"]
    skin_color = scheme["skin_color"]
    accent_color = scheme["accent_color"]

    lean_x = pose["lean_x"]
    body_dy = pose["body_dy"]
    head_dy = pose["head_dy"] + body_dy
    arm_swing = pose["arm_swing"]
    arms_up = pose["arms_up"]

    leg_h = unit * 0.5
    body_h = unit * 0.7
    head_r = max(2, int(unit * 0.3))

    feet_y = y
    body_bottom = feet_y - leg_h + body_dy
    body_top = body_bottom - body_h
    head_cy = body_top - head_r + head_dy

    # Legs.
    leg_w = max(2, int(unit * 0.16))
    for side in (-1, 1):
        leg_x = x + side * unit * 0.14
        pygame.draw.rect(surface, body_color, (leg_x - leg_w // 2, body_bottom, leg_w, leg_h))

    # Torso.
    body_w = max(3, int(unit * 0.55))
    body_rect = pygame.Rect(0, 0, body_w, int(body_h))
    body_rect.midtop = (x + lean_x, body_top)
    pygame.draw.rect(surface, body_color, body_rect, border_radius=max(1, body_w // 5))

    # Non-paddle arm: raised with the paddle arm during a score
    # celebration, otherwise hangs loosely at the side.
    arm_w = max(2, int(unit * 0.14))
    other_arm_y = body_rect.top + (0 if arms_up > 0 else int(body_h * 0.3))
    other_arm_rect = pygame.Rect(0, 0, arm_w, int(unit * (0.5 if arms_up > 0 else 0.35)))
    other_arm_rect.midtop = (body_rect.left - arm_w // 2, other_arm_y)
    pygame.draw.rect(surface, skin_color, other_arm_rect, border_radius=2)

    # Paddle arm: extends forward/out on a hit swing, raised on a score.
    paddle_arm_len = int(unit * (0.55 if arms_up > 0 else 0.4))
    paddle_arm_rect = pygame.Rect(0, 0, arm_w, paddle_arm_len)
    if arms_up > 0:
        paddle_arm_rect.midbottom = (body_rect.right + arm_w // 2, body_rect.top + 4)
    else:
        extend = int(arm_swing * unit * 0.35)
        paddle_arm_rect.midtop = (body_rect.right + arm_w // 2 + extend, body_rect.top + int(body_h * 0.25))
    pygame.draw.rect(surface, skin_color, paddle_arm_rect, border_radius=2)

    # Paddle, at the paddle-hand end.
    paddle_center = paddle_arm_rect.midtop if arms_up > 0 else paddle_arm_rect.midbottom
    paddle_radius = max(2, int(unit * 0.18))
    pygame.draw.circle(surface, (220, 40, 40), paddle_center, paddle_radius)
    pygame.draw.circle(surface, (20, 20, 20), paddle_center, paddle_radius, width=1)

    # Head.
    head_rect = pygame.Rect(0, 0, head_r * 2, head_r * 2)
    head_rect.center = (int(x + lean_x), int(head_cy))
    pygame.draw.circle(surface, skin_color, head_rect.center, head_r)
    _draw_accessory(surface, scheme["accessory"], head_rect, accent_color)


def draw_name_tag(surface, name_font, skill_font, name, skill_label, tag_color, center_x, bottom_y):
    name_surf = name_font.render(name, True, (255, 255, 255))
    skill_surf = skill_font.render(skill_label, True, tag_color)
    width = max(name_surf.get_width(), skill_surf.get_width()) + 16
    height = name_surf.get_height() + skill_surf.get_height() + 10
    tag_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    tag_surf.fill((0, 0, 0, 170))
    pygame.draw.rect(tag_surf, tag_color, tag_surf.get_rect(), width=2)
    tag_surf.blit(name_surf, name_surf.get_rect(centerx=width // 2, y=4))
    tag_surf.blit(skill_surf, skill_surf.get_rect(centerx=width // 2, y=4 + name_surf.get_height() + 2))
    rect = tag_surf.get_rect(midbottom=(center_x, bottom_y))
    surface.blit(tag_surf, rect.topleft)


def draw_opponent(surface, camera, opponent_x, physics_cfg, level_name, animator, now,
                   name_font, skill_font, paddle_height=0.3):
    """Replaces render/table_view.py's plain-rectangle draw_opponent with
    the full animated character + floating nameplate."""
    half_l = physics_cfg["table_length"] / 2
    projected = project_point(opponent_x, paddle_height, -half_l - 0.1, camera)
    if projected is None:
        return
    x, y, scale = projected
    unit = scaled_radius(CHARACTER_UNIT_SIZE_M, scale, min_px=10)
    unit = min(unit, MAX_RADIUS_PX)

    scheme = CHARACTER_SCHEMES.get(level_name, CHARACTER_SCHEMES["beginner"])
    state, progress = animator.current_state(now)
    sway_deg = animator.sway_offset(now)
    pose = _pose_for_state(state, progress, sway_deg)

    draw_character(surface, x, y, unit, scheme, pose)
    # 2.8x clears the "Ready? N" countdown text, which renders centered on
    # the screen - 1.9x put the tag's bottom edge almost exactly there.
    draw_name_tag(surface, name_font, skill_font, scheme["name"], scheme["skill_label"], scheme["tag_color"],
                  center_x=x, bottom_y=y - unit * 2.8)
