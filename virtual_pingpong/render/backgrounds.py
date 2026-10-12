"""Per-level backgrounds, reflecting the difficulty progression as a venue
progression (a casual basement up to a packed stadium) - procedurally
drawn with plain pygame shapes (consistent with the rest of this project's
no-asset-pipeline approach), composed to look good once the retro
pixelation filter chunks them up.

Pure 2D set-dressing: drawn directly in screen coordinates (not run
through render/projection.py's 3D camera), since the camera never moves -
only the table/net/ball/paddle/opponent need real 3D projection. Each
draw_* function fills the whole surface before adding detail, so callers
don't need a separate screen.fill() first.
"""
import pygame

HORIZON_FRACTION = 0.5  # where the wall meets the floor, as a fraction of screen height


def _horizon_y(surface):
    return int(surface.get_height() * HORIZON_FRACTION)


def _draw_wall_and_floor(surface, wall_color, floor_color):
    width, height = surface.get_size()
    horizon = _horizon_y(surface)
    surface.fill(wall_color, (0, 0, width, horizon))
    surface.fill(floor_color, (0, horizon, width, height - horizon))
    return horizon


def draw_beginner_background(surface):
    """A casual basement rec room: wood-paneled wall, a single bare
    hanging bulb, a few stacked boxes in the corners."""
    width, height = surface.get_size()
    horizon = _draw_wall_and_floor(surface, (92, 64, 42), (70, 70, 72))

    # Wood paneling seams.
    for x in range(0, width, 70):
        pygame.draw.line(surface, (76, 52, 34), (x, 0), (x, horizon), width=2)

    # A bare bulb on a cord, hanging from the top center.
    bulb_x = width // 2
    pygame.draw.line(surface, (40, 40, 40), (bulb_x, 0), (bulb_x, 70), width=2)
    glow = pygame.Surface((90, 90), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 235, 150, 70), (45, 45), 45)
    surface.blit(glow, (bulb_x - 45, 70 - 25))
    pygame.draw.circle(surface, (255, 240, 180), (bulb_x, 80), 12)

    # Stacked cardboard boxes, bottom corners.
    for base_x in (30, width - 110):
        pygame.draw.rect(surface, (160, 120, 75), (base_x, horizon - 70, 70, 70))
        pygame.draw.rect(surface, (120, 88, 52), (base_x, horizon - 70, 70, 70), width=3)
        pygame.draw.rect(surface, (160, 120, 75), (base_x + 14, horizon - 120, 55, 55))
        pygame.draw.rect(surface, (120, 88, 52), (base_x + 14, horizon - 120, 55, 55), width=3)


def draw_intermediate_background(surface):
    """A school/rec-center gym: pale cinderblock wall, a basketball hoop,
    a couple of tall windows, honey-colored wood floor."""
    width, height = surface.get_size()
    horizon = _draw_wall_and_floor(surface, (196, 188, 160), (176, 140, 84))

    # Floor planks.
    for y in range(horizon + 14, height, 16):
        pygame.draw.line(surface, (156, 120, 68), (0, y), (width, y), width=1)

    # Windows.
    for win_x in (60, width - 160):
        pygame.draw.rect(surface, (140, 190, 210), (win_x, 40, 100, 130))
        pygame.draw.rect(surface, (90, 90, 90), (win_x, 40, 100, 130), width=4)
        pygame.draw.line(surface, (90, 90, 90), (win_x + 50, 40), (win_x + 50, 170), width=3)
        pygame.draw.line(surface, (90, 90, 90), (win_x, 105), (win_x + 100, 105), width=3)

    # Basketball hoop, centered above the table.
    hoop_x = width // 2
    pygame.draw.rect(surface, (200, 60, 60), (hoop_x - 45, 20, 90, 60))
    pygame.draw.circle(surface, (230, 140, 40), (hoop_x, 90), 22, width=4)
    pygame.draw.rect(surface, (210, 210, 210), (hoop_x - 22, 86, 44, 3))


def draw_advanced_background(surface):
    """A proper sports club: deep blue walls, hanging pennant banners,
    a few rows of bleacher seating along the back, dark polished floor."""
    width, height = surface.get_size()
    horizon = _draw_wall_and_floor(surface, (26, 40, 70), (35, 35, 42))

    # Pennant banners.
    for banner_x in range(40, width - 20, 130):
        color = (220, 190, 60) if (banner_x // 130) % 2 == 0 else (210, 70, 70)
        pygame.draw.line(surface, (200, 200, 200), (banner_x, 10), (banner_x, 40), width=2)
        pygame.draw.polygon(surface, color, [(banner_x - 18, 40), (banner_x + 18, 40), (banner_x, 70)])

    # Bleacher rows along the back wall.
    bleacher_top = 110
    for row in range(4):
        y = bleacher_top + row * 16
        pygame.draw.rect(surface, (60, 60, 70), (40, y, width - 80, 12))
        for seat_x in range(50, width - 50, 20):
            pygame.draw.rect(surface, (90 + (row * 10) % 50, 90, 110), (seat_x, y, 10, 10))

    # Floor sheen.
    pygame.draw.polygon(surface, (50, 50, 60), [
        (width * 0.2, height), (width * 0.8, height), (width * 0.6, horizon), (width * 0.4, horizon),
    ])


def draw_expert_background(surface):
    """A packed stadium at night: dark sky, tall light towers, a dense
    crowd silhouette, a big scoreboard, dramatically lit dark floor."""
    width, height = surface.get_size()
    horizon = _draw_wall_and_floor(surface, (8, 10, 24), (18, 18, 24))

    # Light towers with glowing clusters.
    for tower_x in (70, width - 70):
        pygame.draw.rect(surface, (50, 50, 55), (tower_x - 4, 20, 8, 90))
        for dx, dy in ((-14, 10), (14, 10), (0, -6)):
            glow = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 255, 220, 160), (20, 20), 18)
            surface.blit(glow, (tower_x + dx - 20, 20 + dy - 20))
            pygame.draw.circle(surface, (255, 255, 230), (tower_x + dx, 20 + dy), 6)

    # Crowd silhouette band, packed with small varied-color blobs.
    crowd_top, crowd_bottom = 120, 165
    pygame.draw.rect(surface, (12, 14, 20), (0, crowd_top, width, crowd_bottom - crowd_top))
    rng_colors = [(90, 70, 70), (70, 80, 95), (95, 90, 70), (75, 95, 80), (85, 75, 95)]
    i = 0
    for y in range(crowd_top + 4, crowd_bottom, 8):
        offset = 6 if (y // 8) % 2 == 0 else 0
        for x in range(offset, width, 12):
            pygame.draw.circle(surface, rng_colors[i % len(rng_colors)], (x, y), 4)
            i += 1

    # Scoreboard.
    board_w, board_h = 160, 50
    board_x, board_y = width // 2 - board_w // 2, crowd_top - board_h - 8
    pygame.draw.rect(surface, (10, 10, 10), (board_x, board_y, board_w, board_h))
    pygame.draw.rect(surface, (255, 60, 60), (board_x, board_y, board_w, board_h), width=3)


_BACKGROUND_BY_LEVEL = {
    "beginner": draw_beginner_background,
    "intermediate": draw_intermediate_background,
    "advanced": draw_advanced_background,
    "expert": draw_expert_background,
}


def draw_background(surface, level_name):
    draw_fn = _BACKGROUND_BY_LEVEL.get(level_name, draw_beginner_background)
    draw_fn(surface)
