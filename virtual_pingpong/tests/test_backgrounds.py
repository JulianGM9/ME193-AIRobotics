import pygame
import pytest

from render.backgrounds import draw_background

LEVELS = ["beginner", "intermediate", "advanced", "expert"]


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


@pytest.mark.parametrize("level_name", LEVELS)
def test_draw_background_does_not_crash_for_every_known_level(level_name):
    surface = pygame.Surface((960, 600))
    draw_background(surface, level_name)  # must not raise


def test_draw_background_fills_the_whole_surface():
    surface = pygame.Surface((960, 600))
    surface.fill((1, 2, 3))  # a color none of the backgrounds would plausibly produce
    draw_background(surface, "beginner")
    corners = [(0, 0), (959, 0), (0, 599), (959, 599)]
    assert all(surface.get_at(p)[:3] != (1, 2, 3) for p in corners)


def test_unknown_level_falls_back_to_beginner_rather_than_crashing():
    surface = pygame.Surface((960, 600))
    draw_background(surface, "not_a_real_level")  # must not raise, must not no-op either
    surface_beginner = pygame.Surface((960, 600))
    draw_background(surface_beginner, "beginner")
    assert surface.get_at((10, 10)) == surface_beginner.get_at((10, 10))


def test_different_levels_produce_visually_different_wall_colors():
    sampled_colors = set()
    for level_name in LEVELS:
        surface = pygame.Surface((960, 600))
        draw_background(surface, level_name)
        sampled_colors.add(surface.get_at((5, 5))[:3])
    assert len(sampled_colors) == len(LEVELS)  # each level's wall color is distinct
