import pygame
import pytest

from render.qtable_view import NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR, cell_color, draw_q_table


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


def test_zero_value_is_neutral():
    assert cell_color(0.0, max_magnitude=1.0) == NEUTRAL_COLOR


def test_all_zero_table_is_neutral_everywhere():
    assert cell_color(0.0, max_magnitude=0.0) == NEUTRAL_COLOR


def test_max_positive_value_is_fully_positive_color():
    assert cell_color(1.0, max_magnitude=1.0) == POSITIVE_COLOR


def test_max_negative_value_is_fully_negative_color():
    assert cell_color(-1.0, max_magnitude=1.0) == NEGATIVE_COLOR


def test_halfway_positive_value_is_between_neutral_and_positive():
    color = cell_color(0.5, max_magnitude=1.0)
    assert color != NEUTRAL_COLOR
    assert color != POSITIVE_COLOR
    for channel, neutral, positive in zip(color, NEUTRAL_COLOR, POSITIVE_COLOR):
        assert min(neutral, positive) <= channel <= max(neutral, positive)


def test_value_beyond_max_magnitude_is_clamped_to_full_color():
    assert cell_color(5.0, max_magnitude=1.0) == POSITIVE_COLOR
    assert cell_color(-5.0, max_magnitude=1.0) == NEGATIVE_COLOR


def test_draw_q_table_does_not_crash_on_a_real_table():
    surface = pygame.Surface((300, 300))
    font = pygame.font.Font(None, 16)
    q_table = [[0.0] * 9 for _ in range(9)]
    q_table[3][5] = 0.8
    q_table[1][2] = -0.6
    draw_q_table(surface, pygame.Rect(10, 10, 180, 180), q_table, font)


def test_draw_q_table_handles_an_empty_table_without_crashing():
    surface = pygame.Surface((300, 300))
    font = pygame.font.Font(None, 16)
    draw_q_table(surface, pygame.Rect(10, 10, 180, 180), [], font)


def test_draw_q_table_actually_draws_something():
    surface = pygame.Surface((300, 300))
    surface.fill((1, 2, 3))
    font = pygame.font.Font(None, 16)
    q_table = [[0.0] * 9 for _ in range(9)]
    draw_q_table(surface, pygame.Rect(10, 10, 180, 180), q_table, font)
    assert surface.get_at((15, 15))[:3] != (1, 2, 3)
