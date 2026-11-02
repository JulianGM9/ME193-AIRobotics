"""The recorder's automatic bout end (scripts/record.py run_bout), replayed on a scripted signal."""

import importlib.util
from pathlib import Path

import numpy as np

from stepnet.protocols import PROTOCOLS

spec = importlib.util.spec_from_file_location("record", Path(__file__).resolve().parents[1] / "scripts" / "record.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)

GYRO_LSB = 16.4


class ScriptedStream:
    """Plays back a gyro timeline: each since() call reveals the next 0.2 s, like real time."""

    def __init__(self, walking, already_received):
        self.walking = walking  # function seq -> bool
        self.latest = already_received

    def _samples(self, seqs):
        out = np.zeros((len(seqs), 8), dtype=np.int64)
        out[:, 0] = seqs
        swing = np.array([250 * np.sin(2 * np.pi * s / 100) if self.walking(s) else 0.0 for s in seqs])
        out[:, 7] = np.round(swing * GYRO_LSB)
        return out

    def latest_seq(self):
        return self.latest

    def since(self, seq):
        self.latest += 20
        return self._samples(np.arange(max(0, seq), self.latest + 1))


def run(walking, already_received, monkeypatch):
    monkeypatch.setattr(record.time, "sleep", lambda s: None)
    stream = ScriptedStream(walking, already_received)
    step = PROTOCOLS["A"][0]  # any step: only .say is used
    return record.run_bout(stream, step, [0.0, 0.0, 0.0], record.Voice(False))


def test_walking_before_go_and_a_pause_do_not_end_the_bout(monkeypatch):
    # Still walking to the stairs while the instruction plays (seq < 300), a pause at the
    # bottom of the stairs (300-700), then the real climb (700-1200).
    walking = lambda s: s < 300 or 700 <= s < 1200
    start, go, end, timed_out = run(walking, already_received=300, monkeypatch=monkeypatch)
    assert not timed_out
    assert end > 1200  # the bug ended the bout during the pause, at ~450
    assert start <= 340
    assert start <= go < 700  # Go comes after the instruction, before the climb


def test_normal_bout_ends_after_walking_stops(monkeypatch):
    walking = lambda s: 400 <= s < 900
    start, go, end, timed_out = run(walking, already_received=300, monkeypatch=monkeypatch)
    assert not timed_out
    assert 900 < end < 1150
