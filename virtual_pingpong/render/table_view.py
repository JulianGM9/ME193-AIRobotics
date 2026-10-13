"""Draws the table, net, ball (with drop shadow), opponent, and a
translucent player-paddle marker, using render/projection.py's camera
math. Pure pygame drawing - no physics or game-state logic here.
"""
import pygame

from render.projection import project_point

TABLE_COLOR = (20, 90, 160)
TABLE_LINE_COLOR = (230, 230, 230)
NET_COLOR = (50, 50, 50)
BALL_COLOR = (255, 220, 0)
SHADOW_COLOR = (0, 0, 0)
OPPONENT_COLOR = (200, 60, 60)
PADDLE_FACE_COLOR = (0, 200, 80)  # resting color
PADDLE_FLASH_COLOR = (255, 255, 120)  # blended in on a detected swing
PADDLE_HANDLE_COLOR = (90, 60, 30)
PADDLE_OUTLINE_COLOR = (20, 20, 20)

# Real-world sizes in meters - project_point's `scale` is pixels-per-meter
# at a given depth (focal_length / depth), so multiplying a REAL size by
# it gives correctly-perspective-scaled pixels. (A previous version used
# raw pixel counts here, which produced radii in the thousands of pixels
# and crashed pygame's circle rasterizer - see MAX_RADIUS_PX below, kept
# as a defensive clamp against any future version of this same mistake.)
BALL_RADIUS_M = 0.05  # real size is ~0.02 (40mm diameter) - exaggerated for visibility, cosmetic only,
# doesn't affect physics (game/physics.py has no notion of ball radius at all) or hit detection
# (game/hit_detection.py's hit_radius is a separate, pre-existing difficulty-tuned gameplay setting)
PADDLE_FACE_RADIUS_M = 0.08
PADDLE_HANDLE_LENGTH_M = 0.06
PADDLE_HANDLE_WIDTH_M = 0.03
OPPONENT_SIZE_M = 0.15
MAX_RADIUS_PX = 300  # sanity clamp - no on-screen marker should ever be bigger than this


def _lerp_color(color_a, color_b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def scaled_radius(real_size_m, scale, min_px=2):
    return max(min_px, min(MAX_RADIUS_PX, int(real_size_m * scale)))


def draw_table(surface, camera, physics_cfg):
    half_w = physics_cfg["table_width"] / 2
    half_l = physics_cfg["table_length"] / 2
    corners_3d = [
        (-half_w, 0, half_l), (half_w, 0, half_l),
        (half_w, 0, -half_l), (-half_w, 0, -half_l),
    ]
    points = []
    for x, y, z in corners_3d:
        projected = project_point(x, y, z, camera)
        if projected is None:
            return  # part of the table is behind the camera - skip drawing it this frame
        points.append((projected[0], projected[1]))
    pygame.draw.polygon(surface, TABLE_COLOR, points)
    pygame.draw.polygon(surface, TABLE_LINE_COLOR, points, width=2)

    near = project_point(0, 0, half_l, camera)
    far = project_point(0, 0, -half_l, camera)
    if near and far:
        pygame.draw.line(surface, TABLE_LINE_COLOR, near[:2], far[:2], width=1)


def draw_net(surface, camera, physics_cfg):
    half_w = physics_cfg["table_width"] / 2
    net_h = physics_cfg["net_height"]
    corners_3d = [
        (-half_w, 0, 0), (half_w, 0, 0),
        (half_w, net_h, 0), (-half_w, net_h, 0),
    ]
    points = []
    for x, y, z in corners_3d:
        projected = project_point(x, y, z, camera)
        if projected is None:
            return
        points.append((projected[0], projected[1]))
    pygame.draw.polygon(surface, NET_COLOR, points)


def draw_ball(surface, camera, ball):
    shadow = project_point(ball.x, 0, ball.z, camera)
    if shadow is not None:
        sx, sy, sscale = shadow
        # Shadow shrinks and fades as the ball rises off the table - this
        # is what gives the ball's height a sense of depth.
        height_factor = max(0.0, 1.0 - ball.y / 0.5)
        shadow_radius = scaled_radius(BALL_RADIUS_M * 0.8, sscale, min_px=1)
        if height_factor > 0:
            shadow_surf = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
            alpha = int(140 * height_factor)
            pygame.draw.ellipse(shadow_surf, (*SHADOW_COLOR, alpha),
                                 (0, shadow_radius // 2, shadow_radius * 2, shadow_radius))
            surface.blit(shadow_surf, (sx - shadow_radius, sy - shadow_radius))

    projected = project_point(ball.x, ball.y, ball.z, camera)
    if projected is None:
        return
    x, y, scale = projected
    radius = scaled_radius(BALL_RADIUS_M, scale)
    pygame.draw.circle(surface, BALL_COLOR, (int(x), int(y)), radius)


def draw_opponent(surface, camera, opponent_x, physics_cfg, paddle_height=0.3):
    half_l = physics_cfg["table_length"] / 2
    projected = project_point(opponent_x, paddle_height, -half_l - 0.1, camera)
    if projected is None:
        return
    x, y, scale = projected
    size = scaled_radius(OPPONENT_SIZE_M, scale, min_px=4)
    rect = pygame.Rect(0, 0, size, int(size * 1.5))
    rect.center = (int(x), int(y))
    pygame.draw.rect(surface, OPPONENT_COLOR, rect, border_radius=4)


def draw_paddle(surface, camera, x, y, z, swing_intensity=0.0):
    """Draws a paddle-shaped marker (round face + handle) at the tracked
    position. swing_intensity in [0, 1]: 0 = resting translucent green,
    1 = a bright flash + expanding glow ring, for immediate visual
    confirmation that a swing was just detected - independent of whether
    it ends up counting as a hit.
    """
    projected = project_point(x, y, z, camera)
    if projected is None:
        return
    sx, sy, scale = projected

    face_radius = scaled_radius(PADDLE_FACE_RADIUS_M, scale, min_px=4)
    handle_length = scaled_radius(PADDLE_HANDLE_LENGTH_M, scale, min_px=2)
    handle_width = scaled_radius(PADDLE_HANDLE_WIDTH_M, scale, min_px=2)

    if swing_intensity > 0:
        glow_radius = min(MAX_RADIUS_PX, int(face_radius * (1.0 + 1.5 * swing_intensity)))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_alpha = int(180 * swing_intensity)
        glow_width = max(2, glow_radius // 6)
        pygame.draw.circle(glow_surf, (*PADDLE_FLASH_COLOR, glow_alpha),
                            (glow_radius, glow_radius), glow_radius, width=glow_width)
        surface.blit(glow_surf, (sx - glow_radius, sy - glow_radius))

    width_px = face_radius * 2
    height_px = face_radius * 2 + handle_length
    marker_surf = pygame.Surface((width_px, height_px), pygame.SRCALPHA)
    center_x, center_y = face_radius, face_radius

    handle_rect = pygame.Rect(0, 0, handle_width * 2, handle_length + 2)
    handle_rect.midtop = (center_x, center_y + face_radius - 2)
    pygame.draw.rect(marker_surf, (*PADDLE_HANDLE_COLOR, 220), handle_rect,
                      border_radius=max(1, handle_width // 2))

    face_color = _lerp_color(PADDLE_FACE_COLOR, PADDLE_FLASH_COLOR, swing_intensity)
    face_alpha = 160 + int(95 * swing_intensity)
    pygame.draw.circle(marker_surf, (*face_color, face_alpha), (center_x, center_y), face_radius)
    pygame.draw.circle(marker_surf, (*PADDLE_OUTLINE_COLOR, 220), (center_x, center_y), face_radius, width=2)

    surface.blit(marker_surf, (sx - center_x, sy - center_y))
