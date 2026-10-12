"""Chiptune synthesis: square/pulse waves (the classic 8-bit timbre),
note-name-to-frequency conversion, and stitching a sequence of notes into
one looping melody. All pure numpy - the only pygame touchpoint is the
final make_sound() call, so the waveform math is unit-testable without an
audio device.
"""
import numpy as np
import pygame

SAMPLE_RATE = 44100

_NOTE_SEMITONES_FROM_A = {
    "C": -9, "C#": -8, "Db": -8, "D": -7, "D#": -6, "Eb": -6, "E": -5,
    "F": -4, "F#": -3, "Gb": -3, "G": -2, "G#": -1, "Ab": -1, "A": 0,
    "A#": 1, "Bb": 1, "B": 2,
}


def note_to_freq(note):
    """'A4' -> 440.0, 'C#5' -> ..., 'R' (rest) -> None. Octave 4 is the
    octave containing A440, per scientific pitch notation - equal
    temperament, computed rather than looked up so any note name/octave
    combination works without a giant table."""
    if note == "R":
        return None
    name, octave = note[:-1], int(note[-1])
    semitones = _NOTE_SEMITONES_FROM_A[name] + (octave - 4) * 12
    return 440.0 * (2.0 ** (semitones / 12.0))


def apply_envelope(wave, attack=0.005, release=0.01, sample_rate=SAMPLE_RATE):
    """Fades the start/end of `wave` in/out linearly, to avoid an audible
    click at a note's boundaries (a raw square wave's hard edges pop if cut
    off mid-cycle)."""
    n = len(wave)
    if n == 0:
        return wave
    attack_n = min(int(attack * sample_rate), n // 2)
    release_n = min(int(release * sample_rate), n // 2)
    envelope = np.ones(n)
    if attack_n > 0:
        envelope[:attack_n] = np.linspace(0, 1, attack_n)
    if release_n > 0:
        envelope[n - release_n:] = np.linspace(1, 0, release_n)
    return wave * envelope


def square_wave(frequency, duration, sample_rate=SAMPLE_RATE, volume=0.3, duty_cycle=0.5):
    """A fixed-frequency pulse wave, volume in [-1, 1] float samples (not
    yet int16/Sound) - the square/pulse wave is what gives classic 8-bit
    consoles their timbre, versus a smooth sine tone."""
    num_samples = int(sample_rate * duration)
    if frequency is None or num_samples <= 0:
        return np.zeros(max(num_samples, 0))
    t = np.linspace(0, duration, num_samples, endpoint=False)
    phase = (t * frequency) % 1.0
    return np.where(phase < duty_cycle, 1.0, -1.0) * volume


def square_sweep(start_freq, end_freq, duration, sample_rate=SAMPLE_RATE, volume=0.3, duty_cycle=0.5):
    """Like square_wave, but linearly sweeps frequency from start_freq to
    end_freq over the duration - classic "boop" (rising) / "womp"
    (falling) 8-bit sound-effect sweeps. Phase is the running integral of
    instantaneous frequency (not just t * freq, which is only valid for a
    constant frequency) so the sweep doesn't warp pitch incorrectly.
    """
    num_samples = int(sample_rate * duration)
    if num_samples <= 0:
        return np.zeros(0)
    freq_t = np.linspace(start_freq, end_freq, num_samples)
    phase = np.cumsum(freq_t) / sample_rate
    return np.where((phase % 1.0) < duty_cycle, 1.0, -1.0) * volume


def _to_sound(wave):
    samples = np.clip(wave, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([samples, samples])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_sfx_sound(wave):
    """Wraps a single pre-built waveform array (e.g. from square_wave or
    square_sweep, after apply_envelope) into a playable Sound."""
    return _to_sound(wave)


def render_melody(notes, bpm, sample_rate=SAMPLE_RATE, volume=0.25, duty_cycle=0.5, decay=None):
    """notes: a list of (note_name_or_frequency_or_"R", beats) pairs,
    where one beat is a quarter note at the given bpm. Returns a playable
    (loopable) Sound of the whole sequence concatenated together.

    decay: if given, multiplies each note's wave by exp(-decay * t) over
    its own duration - a plucked/harpsichord-like fade rather than a
    sustained organ-like tone at constant volume for the note's full
    length. None (the default) keeps the old sustained behavior.
    """
    beat_seconds = 60.0 / bpm
    chunks = []
    for note, beats in notes:
        duration = beats * beat_seconds
        freq = note_to_freq(note) if isinstance(note, str) else note
        wave = square_wave(freq, duration, sample_rate, volume, duty_cycle)
        if decay is not None and len(wave) > 0:
            t = np.linspace(0, duration, len(wave), endpoint=False)
            wave = wave * np.exp(-decay * t)
        chunks.append(apply_envelope(wave, sample_rate=sample_rate))
    full = np.concatenate(chunks) if chunks else np.zeros(1)
    return _to_sound(full)
