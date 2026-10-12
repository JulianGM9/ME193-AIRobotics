import math

import numpy as np
import pygame
import pytest

from audio.synth import apply_envelope, note_to_freq, render_melody, square_sweep, square_wave


@pytest.fixture(autouse=True, scope="module")
def pygame_audio_session():
    pygame.mixer.init()
    yield
    pygame.mixer.quit()


def test_note_to_freq_a4_is_440():
    assert note_to_freq("A4") == pytest.approx(440.0)


def test_note_to_freq_one_octave_up_doubles_frequency():
    assert note_to_freq("A5") == pytest.approx(880.0)


def test_note_to_freq_one_octave_down_halves_frequency():
    assert note_to_freq("A3") == pytest.approx(220.0)


def test_note_to_freq_c4_is_middle_c():
    assert note_to_freq("C4") == pytest.approx(261.63, abs=0.01)


def test_note_to_freq_rest_is_none():
    assert note_to_freq("R") is None


def test_note_to_freq_sharps_and_flats_are_enharmonic():
    assert note_to_freq("C#4") == pytest.approx(note_to_freq("Db4"))


def test_square_wave_length_matches_duration():
    wave = square_wave(440, 0.5, sample_rate=1000)
    assert len(wave) == 500


def test_square_wave_only_takes_two_values():
    wave = square_wave(100, 0.1, sample_rate=44100, volume=0.4)
    uniques = set(np.round(wave, 6))
    assert uniques <= {0.4, -0.4}


def test_square_wave_none_frequency_is_silence():
    wave = square_wave(None, 0.1, sample_rate=1000)
    assert len(wave) == 100
    assert np.all(wave == 0)


def test_square_wave_duty_cycle_changes_the_high_fraction():
    wave = square_wave(10, 1.0, sample_rate=10000, volume=1.0, duty_cycle=0.25)
    high_fraction = np.mean(wave > 0)
    assert high_fraction == pytest.approx(0.25, abs=0.02)


def test_square_sweep_starts_near_start_frequency():
    # Over a short enough window, a sweep starting at a low frequency
    # shouldn't complete many cycles yet - check it hasn't already finished
    # a full period's worth of oscillation almost instantly.
    wave = square_sweep(100, 100, 1.0, sample_rate=10000, volume=1.0)
    # A constant 100Hz sweep (degenerate case) should look just like a
    # fixed-frequency square wave: 100 cycles over 1 second.
    sign_changes = np.sum(np.diff(np.sign(wave)) != 0)
    assert 190 <= sign_changes <= 210  # ~2 sign changes per cycle


def test_square_sweep_zero_duration_is_empty():
    wave = square_sweep(100, 500, 0.0, sample_rate=44100)
    assert len(wave) == 0


def test_apply_envelope_starts_and_ends_at_zero():
    wave = np.ones(1000)
    enveloped = apply_envelope(wave, attack=0.01, release=0.01, sample_rate=1000)
    assert enveloped[0] == pytest.approx(0.0, abs=1e-6)
    assert enveloped[-1] == pytest.approx(0.0, abs=1e-6)


def test_apply_envelope_leaves_the_middle_untouched():
    wave = np.full(1000, 0.7)
    enveloped = apply_envelope(wave, attack=0.01, release=0.01, sample_rate=1000)
    assert enveloped[500] == pytest.approx(0.7)


def test_apply_envelope_handles_empty_wave():
    assert len(apply_envelope(np.array([]))) == 0


# --- render_melody ---

def test_render_melody_total_duration_matches_note_beats():
    # get_length() depends on the mixer's actual configured frequency, not
    # the sample_rate passed in here - check raw sample count instead,
    # which is what render_melody's own math actually controls.
    notes = [("C4", 1), ("E4", 1), ("G4", 2)]  # 4 beats total
    sound = render_melody(notes, bpm=60, sample_rate=1000)  # 1 beat = 1 second at 60bpm
    assert len(pygame.sndarray.array(sound)) == pytest.approx(4000, abs=10)


def test_render_melody_handles_rests():
    notes = [("C4", 1), ("R", 1)]
    sound = render_melody(notes, bpm=60, sample_rate=1000)
    assert len(pygame.sndarray.array(sound)) == pytest.approx(2000, abs=10)


def test_render_melody_without_decay_sustains_at_constant_volume():
    # Default behavior (decay=None) - a note's envelope-adjusted samples
    # should hold near-constant amplitude through most of its duration,
    # not fade out mid-note.
    notes = [("A4", 1)]
    sound = render_melody(notes, bpm=60, sample_rate=44100, volume=0.5)
    samples = pygame.sndarray.array(sound)[:, 0]
    early = np.abs(samples[5000:5100]).mean()
    late = np.abs(samples[-5100:-5000]).mean()
    assert late == pytest.approx(early, rel=0.15)


def test_render_melody_with_decay_fades_out_within_the_note():
    notes = [("A4", 1)]
    sound = render_melody(notes, bpm=60, sample_rate=44100, volume=0.5, decay=10.0)
    samples = pygame.sndarray.array(sound)[:, 0]
    early = np.abs(samples[2000:2100]).mean()
    late = np.abs(samples[-2100:-2000]).mean()
    assert late < early * 0.5  # meaningfully quieter by the end of the note
