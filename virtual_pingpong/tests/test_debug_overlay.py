import numpy as np
import pygame
import pytest

from render.debug_overlay import frame_to_surface, intensity_fraction


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


# --- intensity_fraction (pure logic) ---

def test_zero_intensity_is_zero_fraction():
    assert intensity_fraction(0, max_intensity=3.0) == 0.0


def test_intensity_at_max_is_one():
    assert intensity_fraction(3.0, max_intensity=3.0) == 1.0


def test_intensity_above_max_is_clamped_to_one():
    assert intensity_fraction(10.0, max_intensity=3.0) == 1.0


def test_negative_intensity_is_clamped_to_zero():
    assert intensity_fraction(-5.0, max_intensity=3.0) == 0.0


def test_zero_max_intensity_does_not_divide_by_zero():
    assert intensity_fraction(1.0, max_intensity=0.0) == 0.0


def test_halfway_intensity_is_half_fraction():
    assert intensity_fraction(1.5, max_intensity=3.0) == 0.5


# --- frame_to_surface ---

def test_frame_to_surface_preserves_dimensions():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # OpenCV shape: (height, width, channels)
    surface = frame_to_surface(frame)
    assert surface.get_width() == 640
    assert surface.get_height() == 480


def test_frame_to_surface_preserves_pixel_color():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)  # pure red in BGR
    surface = frame_to_surface(frame)
    assert surface.get_at((5, 5))[:3] == (255, 0, 0)  # pure red in pygame's RGB
