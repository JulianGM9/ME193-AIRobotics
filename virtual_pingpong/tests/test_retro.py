import pygame
import pytest

from render.retro import apply_pixelation


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


def test_pixelation_preserves_surface_dimensions():
    surface = pygame.Surface((960, 600))
    result = apply_pixelation(surface, pixel_scale=4)
    assert result.get_size() == (960, 600)


def test_pixelation_makes_fine_detail_blocky_not_blended():
    # A checkerboard of alternating black/white single pixels should
    # collapse into solid, uniform blocks once pixelated (nearest-neighbor
    # sampling - no averaging), not a blurry gray - that crisp blockiness
    # is the whole point versus a soft blur.
    surface = pygame.Surface((40, 40))
    for y in range(40):
        for x in range(40):
            surface.set_at((x, y), (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0))
    result = apply_pixelation(surface, pixel_scale=10)
    # Every pixel within one 10x10 block must be identical (either pure
    # black or pure white) - never a blended gray.
    for y in range(10):
        for x in range(10):
            color = result.get_at((x, y))[:3]
            assert color in ((255, 255, 255), (0, 0, 0))
    corner_block_colors = {result.get_at((x, y))[:3] for y in range(10) for x in range(10)}
    assert len(corner_block_colors) == 1  # the whole block is one solid color


def test_pixelation_on_a_uniform_surface_stays_uniform():
    surface = pygame.Surface((100, 60))
    surface.fill((60, 120, 200))
    result = apply_pixelation(surface, pixel_scale=5)
    assert result.get_at((0, 0))[:3] == (60, 120, 200)
    assert result.get_at((99, 59))[:3] == (60, 120, 200)


def test_pixelation_handles_scale_larger_than_the_surface():
    surface = pygame.Surface((3, 3))
    surface.fill((10, 20, 30))
    result = apply_pixelation(surface, pixel_scale=100)  # would floor to 0 without the max(1, ...) guard
    assert result.get_size() == (3, 3)
