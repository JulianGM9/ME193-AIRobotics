import pygame
import pytest

from render.matrix_background import _pseudo_random, draw_matrix_background


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


def test_pseudo_random_is_deterministic():
    assert _pseudo_random(42) == _pseudo_random(42)


def test_pseudo_random_is_in_the_half_open_unit_range():
    # Must never reach exactly 1.0 - this value is used to index into
    # GLYPHS (int(value * len(GLYPHS))), and 1.0 indexes one past the end.
    for seed in range(2000):
        value = _pseudo_random(seed)
        assert 0.0 <= value < 1.0


def test_pseudo_random_varies_across_seeds():
    values = {_pseudo_random(seed) for seed in range(20)}
    assert len(values) > 10  # not all collapsing to the same/few values


def test_draw_matrix_background_does_not_crash():
    surface = pygame.Surface((960, 600))
    font = pygame.font.Font(None, 16)
    draw_matrix_background(surface, now=0.0, font=font)


def test_draw_matrix_background_fills_the_background():
    surface = pygame.Surface((960, 600))
    surface.fill((1, 2, 3))
    font = pygame.font.Font(None, 16)
    draw_matrix_background(surface, now=0.0, font=font)
    assert surface.get_at((0, 0))[:3] != (1, 2, 3)


def test_draw_matrix_background_changes_over_time():
    surface_a = pygame.Surface((960, 600))
    surface_b = pygame.Surface((960, 600))
    font = pygame.font.Font(None, 16)
    draw_matrix_background(surface_a, now=0.0, font=font)
    draw_matrix_background(surface_b, now=5.0, font=font)
    a_pixels = pygame.image.tostring(surface_a, "RGB")
    b_pixels = pygame.image.tostring(surface_b, "RGB")
    assert a_pixels != b_pixels
