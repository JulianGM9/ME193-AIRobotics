"""Debug overlay (toggled with 'D'): webcam picture-in-picture (with
skeleton/paddle contour already drawn on it by the vision modules) plus a
live IMU swing-intensity bar.
"""
import cv2
import numpy as np
import pygame

BAR_BG_COLOR = (60, 60, 60)
BAR_BORDER_COLOR = (255, 255, 255)
BAR_FILL_COLOR = (0, 200, 0)
BAR_FILL_HIGH_COLOR = (220, 50, 50)
HIGH_INTENSITY_FRACTION = 0.8


def frame_to_surface(frame_bgr):
    """Convert an OpenCV BGR frame to a pygame Surface."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # pygame surfaces are indexed (x, y), numpy arrays (row=y, col=x) -
    # transpose the first two axes to match.
    rgb = np.transpose(rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(rgb)


def draw_webcam_pip(surface, frame_bgr, rect):
    """Blit frame_bgr (any cv2 overlays already drawn on it) into `rect`
    (a pygame.Rect) as a picture-in-picture, scaled to fit."""
    cam_surface = frame_to_surface(frame_bgr)
    scaled = pygame.transform.smoothscale(cam_surface, (rect.width, rect.height))
    surface.blit(scaled, rect.topleft)
    pygame.draw.rect(surface, BAR_BORDER_COLOR, rect, width=2)


def intensity_fraction(intensity, max_intensity):
    """Clamp intensity into [0, 1] of max_intensity - pulled out of the
    drawing code so the clamping math is unit-testable without pygame."""
    if max_intensity <= 0:
        return 0.0
    return max(0.0, min(1.0, intensity / max_intensity))


def draw_swing_intensity_bar(surface, rect, intensity, max_intensity=3.0):
    pygame.draw.rect(surface, BAR_BG_COLOR, rect)
    fraction = intensity_fraction(intensity, max_intensity)
    fill_rect = pygame.Rect(rect.x, rect.y, int(rect.width * fraction), rect.height)
    color = BAR_FILL_HIGH_COLOR if fraction >= HIGH_INTENSITY_FRACTION else BAR_FILL_COLOR
    pygame.draw.rect(surface, color, fill_rect)
    pygame.draw.rect(surface, BAR_BORDER_COLOR, rect, width=1)
