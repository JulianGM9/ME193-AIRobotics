"""All of the game's audio: retro square-wave SFX and two looping chiptune
tracks (a calmer menu theme, a more driving in-game "action" theme),
plus the Sounds class that plays/mutes/switches between them.
"""
import pygame

from audio.synth import apply_envelope, make_sfx_sound, render_melody, square_wave, square_sweep

MUSIC_CHANNEL_ID = 7  # reserved for looping music, separate from one-shot SFX channels


def _hit_sfx():
    # A short, bright upward "boop" - the satisfying paddle-contact sound.
    return make_sfx_sound(square_sweep(600, 1100, 0.08, volume=0.35))


def _bounce_sfx():
    # A very short, low tick for the ball bouncing on the table.
    return make_sfx_sound(apply_envelope(square_wave(300, 0.04, volume=0.25)))


def _miss_sfx():
    # A descending "womp" - the classic 8-bit "you messed up" sound.
    return make_sfx_sound(square_sweep(500, 120, 0.25, volume=0.3))


def _score_sfx():
    # A quick triumphant 3-note ascending arpeggio for winning a point.
    return render_melody([("C5", 0.5), ("E5", 0.5), ("G5", 0.75)], bpm=300, volume=0.3)


# Menu theme: the opening of Bach's Prelude No. 1 in C major, BWV 846
# (Well-Tempered Clavier, Book I) - public domain. Transcribed from
# memory/by ear rather than against a score, so it's a best-effort
# approximation of the famous progression and arpeggio shape, not a
# guaranteed note-perfect transcription.
#
# The piece's signature texture is a continuous 16th-note arpeggio: each
# bar outlines a 5-note chord (root, 3rd, 5th, octave, octave+3rd) in the
# same up-then-partly-down shape, played twice per bar, with the harmony
# changing underneath. _BACH_CHORDS is the famous opening progression
# (I-ii-V-I-vi-II-V-I); _ARPEGGIO_SHAPE is that repeating up/down pattern
# as indices into each 5-note chord tuple.
_BACH_CHORDS = [
    ("C3", "E3", "G3", "C4", "E4"),    # I
    ("D3", "F3", "A3", "D4", "F4"),    # ii
    ("G3", "B3", "D4", "G4", "B4"),    # V
    ("C3", "E3", "G3", "C4", "E4"),    # I
    ("A3", "C4", "E4", "A4", "C5"),    # vi
    ("D3", "F#3", "A3", "D4", "F#4"),  # II (secondary dominant)
    ("G3", "B3", "D4", "G4", "B4"),    # V
    ("C3", "E3", "G3", "C4", "E4"),    # I - loops cleanly back to the start
]
_ARPEGGIO_SHAPE = (0, 1, 2, 3, 4, 3, 2, 1)


def _build_bach_prelude_notes():
    notes = []
    for chord in _BACH_CHORDS:
        half_bar = [(chord[i], 0.25) for i in _ARPEGGIO_SHAPE]  # 8 sixteenth notes = 2 beats
        notes.extend(half_bar)
        notes.extend(half_bar)  # same 8-note group again, filling the bar (4 beats)
    return notes


MENU_THEME_NOTES = _build_bach_prelude_notes()
MENU_THEME_BPM = 66  # roughly the prelude's traditional quarter-note tempo
MENU_THEME_DUTY_CYCLE = 0.3  # narrower pulse than a plain 50% square - brighter, more plucked/harpsichord-like
MENU_THEME_DECAY = 8.0  # per-note amplitude decay, for a plucked-string feel instead of a sustained drone

ACTION_THEME_NOTES = [
    # Phrase A: driving low-register eighth notes.
    ("E4", 0.5), ("E4", 0.5), ("G4", 0.5), ("E4", 0.5),
    ("D4", 0.5), ("D4", 0.5), ("F4", 0.5), ("D4", 0.5),
    ("C4", 0.5), ("C4", 0.5), ("E4", 0.5), ("G4", 0.5),
    ("A4", 0.5), ("G4", 0.5), ("E4", 0.5), ("C4", 0.5),
    # Phrase B: answers up an octave, same rhythm - a 3-second loop felt
    # too repetitive on its own, this doubles the loop length to ~6s and
    # gives it a bit of call-and-response shape instead of just repeating.
    ("E5", 0.5), ("E5", 0.5), ("G5", 0.5), ("E5", 0.5),
    ("D5", 0.5), ("D5", 0.5), ("F5", 0.5), ("D5", 0.5),
    ("C5", 0.5), ("G4", 0.5), ("E4", 0.5), ("G4", 0.5),
    ("A4", 0.5), ("G4", 0.5), ("E4", 0.5), ("C4", 1.0),
]
ACTION_THEME_BPM = 160


class Sounds:
    def __init__(self):
        self.muted = False
        self.hit = _hit_sfx()
        self.bounce = _bounce_sfx()
        self.miss = _miss_sfx()
        self.score = _score_sfx()
        self.menu_theme = render_melody(MENU_THEME_NOTES, MENU_THEME_BPM, volume=0.18,
                                         duty_cycle=MENU_THEME_DUTY_CYCLE, decay=MENU_THEME_DECAY)
        self.action_theme = render_melody(ACTION_THEME_NOTES, ACTION_THEME_BPM, volume=0.18)
        self._music_channel = pygame.mixer.Channel(MUSIC_CHANNEL_ID)
        self._current_music_name = None  # which track *should* be playing, independent of mute

    def play(self, sound):
        if not self.muted:
            sound.play()

    def play_music(self, name):
        """name: "menu" or "action". Switching to the track already
        playing is a no-op (not a restart) so calling this every frame
        with the current state's track doesn't stutter the loop."""
        if name == self._current_music_name:
            return
        self._current_music_name = name
        if not self.muted:
            track = self.menu_theme if name == "menu" else self.action_theme
            self._music_channel.play(track, loops=-1)

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self._music_channel.stop()
        elif self._current_music_name is not None:
            track = self.menu_theme if self._current_music_name == "menu" else self.action_theme
            self._music_channel.play(track, loops=-1)
