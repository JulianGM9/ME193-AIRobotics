import pygame
import pytest

from audio.sounds import Sounds


@pytest.fixture(autouse=True, scope="module")
def pygame_audio_session():
    pygame.mixer.init()
    yield
    pygame.mixer.quit()


class FakeChannel:
    """Stands in for pygame.mixer.Channel (a C type whose methods can't be
    monkeypatched directly) - just records play()/stop() calls."""

    def __init__(self):
        self.calls = []

    def play(self, sound, loops=0):
        self.calls.append(("play", sound, loops))

    def stop(self):
        self.calls.append(("stop",))


@pytest.fixture
def sounds():
    s = Sounds()
    s._music_channel = FakeChannel()
    return s


def test_play_does_nothing_when_muted(sounds):
    sounds.toggle_mute()
    assert sounds.muted is True
    sounds.play(sounds.hit)  # must not raise, and should be a no-op


def test_toggle_mute_flips_the_flag(sounds):
    assert sounds.muted is False
    sounds.toggle_mute()
    assert sounds.muted is True
    sounds.toggle_mute()
    assert sounds.muted is False


def test_play_music_tracks_which_track_is_current(sounds):
    sounds.play_music("menu")
    assert sounds._current_music_name == "menu"
    sounds.play_music("action")
    assert sounds._current_music_name == "action"


def test_play_music_same_track_again_is_a_cheap_no_op(sounds):
    sounds.play_music("menu")
    sounds._music_channel.calls.clear()
    # Calling play_music with the SAME name again must not touch the
    # channel at all (that would restart/stutter an already-looping track).
    sounds.play_music("menu")
    assert sounds._music_channel.calls == []


def test_play_music_switching_tracks_does_touch_the_channel(sounds):
    sounds.play_music("menu")
    sounds._music_channel.calls.clear()
    sounds.play_music("action")
    assert len(sounds._music_channel.calls) == 1
    assert sounds._music_channel.calls[0] == ("play", sounds.action_theme, -1)


def test_muting_stops_the_music_channel(sounds):
    sounds.play_music("menu")
    sounds._music_channel.calls.clear()
    sounds.toggle_mute()
    assert sounds._music_channel.calls == [("stop",)]


def test_unmuting_resumes_whatever_track_was_current(sounds):
    sounds.play_music("action")
    sounds.toggle_mute()  # mute
    sounds._music_channel.calls.clear()
    sounds.toggle_mute()  # unmute - should resume "action"
    assert sounds._music_channel.calls == [("play", sounds.action_theme, -1)]


def test_unmuting_with_no_music_ever_started_does_not_play_anything(sounds):
    sounds.toggle_mute()
    sounds._music_channel.calls.clear()
    sounds.toggle_mute()
    assert sounds._music_channel.calls == []


def test_play_music_while_muted_does_not_touch_the_channel(sounds):
    sounds.toggle_mute()
    sounds._music_channel.calls.clear()  # drop the stop() call toggle_mute() itself made
    sounds.play_music("menu")
    assert sounds._music_channel.calls == []
    assert sounds._current_music_name == "menu"  # still tracked, just not audible
