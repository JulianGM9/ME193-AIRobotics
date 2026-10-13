"""Reusable interactive UI widgets - currently just a horizontal slider,
for the Claude level's live learning-rate/exploration-rate controls.

Slider's value/dragging state is pure logic (no pygame drawing), so the
drag math is unit-testable; draw_slider() does the actual rendering.
"""
import pygame


class Slider:
    def __init__(self, rect, min_value, max_value, value, label=""):
        self.rect = pygame.Rect(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = min(max(value, min_value), max_value)
        self.label = label
        self.dragging = False

    def _set_from_x(self, screen_x):
        fraction = (screen_x - self.rect.x) / self.rect.width
        fraction = min(max(fraction, 0.0), 1.0)
        self.value = self.min_value + fraction * (self.max_value - self.min_value)

    def handle_mouse_down(self, pos):
        """Returns True if this click landed on the slider (and started a
        drag), so the caller can know not to treat it as a click on
        whatever's behind it."""
        if self.rect.collidepoint(pos):
            self.dragging = True
            self._set_from_x(pos[0])
            return True
        return False

    def handle_mouse_up(self):
        self.dragging = False

    def handle_mouse_motion(self, pos):
        if self.dragging:
            self._set_from_x(pos[0])

    def fraction(self):
        span = self.max_value - self.min_value
        if span <= 0:
            return 0.0
        return (self.value - self.min_value) / span


TRACK_COLOR = (60, 60, 70)
FILL_COLOR = (90, 200, 120)
HANDLE_COLOR = (255, 255, 255)
LABEL_COLOR = (220, 220, 220)


def draw_slider(surface, font, slider):
    pygame.draw.rect(surface, TRACK_COLOR, slider.rect, border_radius=3)
    fill_width = int(slider.rect.width * slider.fraction())
    if fill_width > 0:
        fill_rect = pygame.Rect(slider.rect.x, slider.rect.y, fill_width, slider.rect.height)
        pygame.draw.rect(surface, FILL_COLOR, fill_rect, border_radius=3)
    handle_x = slider.rect.x + fill_width
    pygame.draw.circle(surface, HANDLE_COLOR, (handle_x, slider.rect.centery), slider.rect.height)

    label_text = f"{slider.label} {slider.value:.2f}"
    label_surf = font.render(label_text, True, LABEL_COLOR)
    surface.blit(label_surf, (slider.rect.x, slider.rect.y - label_surf.get_height() - 4))
