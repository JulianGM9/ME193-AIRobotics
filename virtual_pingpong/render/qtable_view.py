"""Live Q-table heatmap for the Claude level: one cell per (ball-position
bucket, paddle-target bucket), colored green for a learned-good action and
red for a learned-bad one, intensity scaled by how confident the learned
value is relative to the rest of the table.
"""
import pygame

BORDER_COLOR = (255, 255, 255)
NEUTRAL_COLOR = (35, 35, 40)
POSITIVE_COLOR = (40, 220, 90)
NEGATIVE_COLOR = (220, 50, 50)


def _lerp_color(color_a, color_b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def cell_color(value, max_magnitude):
    """Pure color-mapping logic, pulled out of the drawing loop so it's
    unit-testable without a Surface."""
    if max_magnitude <= 0:
        return NEUTRAL_COLOR
    t = min(abs(value) / max_magnitude, 1.0)
    return _lerp_color(NEUTRAL_COLOR, POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR, t)


def draw_q_table(surface, rect, q_table, font, label="Q-TABLE"):
    rows = len(q_table)
    cols = len(q_table[0]) if rows else 0
    if rows == 0 or cols == 0:
        return

    max_magnitude = max((abs(v) for row in q_table for v in row), default=0.0)
    cell_w = rect.width / cols
    cell_h = rect.height / rows

    for r, row in enumerate(q_table):
        for c, value in enumerate(row):
            color = cell_color(value, max_magnitude)
            cell_rect = pygame.Rect(rect.x + c * cell_w, rect.y + r * cell_h, cell_w + 1, cell_h + 1)
            pygame.draw.rect(surface, color, cell_rect)

    pygame.draw.rect(surface, BORDER_COLOR, rect, width=2)
    label_surf = font.render(label, True, BORDER_COLOR)
    surface.blit(label_surf, (rect.x, rect.y - label_surf.get_height() - 4))

    axis_x = font.render("PADDLE TARGET ->", True, BORDER_COLOR)
    surface.blit(axis_x, (rect.x, rect.bottom + 4))
