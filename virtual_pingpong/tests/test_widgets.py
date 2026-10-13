import pygame
import pytest

from render.widgets import Slider, draw_slider


@pytest.fixture(autouse=True, scope="module")
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


def make_slider(value=0.5):
    return Slider(rect=(100, 50, 200, 10), min_value=0.0, max_value=1.0, value=value)


def test_initial_value_is_clamped_into_range():
    slider = Slider(rect=(0, 0, 100, 10), min_value=0.0, max_value=1.0, value=5.0)
    assert slider.value == 1.0
    slider = Slider(rect=(0, 0, 100, 10), min_value=0.0, max_value=1.0, value=-5.0)
    assert slider.value == 0.0


def test_fraction_reflects_value_within_range():
    slider = make_slider(value=0.25)
    assert slider.fraction() == pytest.approx(0.25)


def test_mouse_down_on_the_track_starts_dragging_and_sets_value():
    slider = make_slider()
    hit = slider.handle_mouse_down((100, 55))  # left edge of the track -> min value
    assert hit is True
    assert slider.dragging is True
    assert slider.value == pytest.approx(0.0)


def test_mouse_down_off_the_track_does_not_start_dragging():
    slider = make_slider()
    hit = slider.handle_mouse_down((500, 500))
    assert hit is False
    assert slider.dragging is False


def test_mouse_down_at_the_right_edge_sets_max_value():
    slider = make_slider()
    # x=300 is the rect's right edge, but pygame.Rect.collidepoint treats
    # the right/bottom edge as exclusive - click one pixel inside instead.
    slider.handle_mouse_down((299, 55))
    assert slider.value == pytest.approx(1.0, abs=0.01)


def test_mouse_down_at_the_middle_sets_the_midpoint_value():
    slider = make_slider()
    slider.handle_mouse_down((200, 55))  # rect spans x=100..300, midpoint x=200
    assert slider.value == pytest.approx(0.5)


def test_motion_while_dragging_updates_the_value():
    slider = make_slider()
    slider.handle_mouse_down((100, 55))
    slider.handle_mouse_motion((250, 55))
    assert slider.value == pytest.approx(0.75)


def test_motion_without_dragging_does_nothing():
    slider = make_slider(value=0.5)
    slider.handle_mouse_motion((250, 55))  # never pressed down
    assert slider.value == pytest.approx(0.5)


def test_mouse_up_stops_dragging():
    slider = make_slider()
    slider.handle_mouse_down((100, 55))
    slider.handle_mouse_up()
    assert slider.dragging is False
    slider.handle_mouse_motion((300, 55))  # should be ignored now
    assert slider.value == pytest.approx(0.0)


def test_dragging_past_the_track_edge_clamps_rather_than_extrapolates():
    slider = make_slider()
    slider.handle_mouse_down((100, 55))
    slider.handle_mouse_motion((9999, 55))
    assert slider.value == pytest.approx(1.0)
    slider.handle_mouse_motion((-9999, 55))
    assert slider.value == pytest.approx(0.0)


def test_respects_a_non_zero_to_one_range():
    slider = Slider(rect=(0, 0, 100, 10), min_value=0.0, max_value=0.9, value=0.0)
    slider.handle_mouse_down((99, 5))  # one pixel inside the right edge (see note above)
    assert slider.value == pytest.approx(0.9, abs=0.01)


def test_draw_slider_does_not_crash():
    surface = pygame.Surface((400, 100))
    font = pygame.font.Font(None, 16)
    slider = make_slider()
    draw_slider(surface, font, slider)
