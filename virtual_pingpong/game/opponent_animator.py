"""Opponent character animation state: pure logic, no pygame or drawing
here (same injectable-clock pattern as render/hud.py's FeedbackBanner/
SwingFlash/Countdown), so it's unit-testable without a display.

Idle sway is continuous and always computable from `now` alone - no
trigger needed. The three event animations (hit/score/lose) are one-shot:
triggering one starts a timed window: current_state() reports it until its
duration elapses, then automatically falls back to "idle". Triggering a
new event while one is already playing simply replaces it (the most
recent event always wins - e.g. a hit animation interrupted by a score
reaction when the point immediately resolves).
"""
import math

IDLE = "idle"
HIT = "hit"
SCORE = "score"
LOSE = "lose"

_EVENT_STATES = (HIT, SCORE, LOSE)


class OpponentAnimator:
    def __init__(self, sway_period_seconds=2.4, sway_amplitude_deg=5.0,
                 hit_duration_seconds=0.25, score_duration_seconds=0.9, lose_duration_seconds=0.9):
        self.sway_period = sway_period_seconds
        self.sway_amplitude_deg = sway_amplitude_deg
        self._durations = {HIT: hit_duration_seconds, SCORE: score_duration_seconds, LOSE: lose_duration_seconds}
        self._event = None
        self._event_started_at = None

    def trigger_hit(self, now):
        self._event, self._event_started_at = HIT, now

    def trigger_score(self, now):
        self._event, self._event_started_at = SCORE, now

    def trigger_lose(self, now):
        self._event, self._event_started_at = LOSE, now

    def current_state(self, now):
        """Returns (state, progress). state is one of IDLE/HIT/SCORE/LOSE;
        progress is 0-1 through the event's duration (0.0 for IDLE)."""
        if self._event is None:
            return IDLE, 0.0
        elapsed = now - self._event_started_at
        duration = self._durations[self._event]
        if elapsed >= duration:
            self._event = None
            return IDLE, 0.0
        return self._event, elapsed / duration

    def sway_offset(self, now):
        """Idle sway angle in degrees - continuous ambient motion,
        independent of whatever event is also currently playing."""
        return math.sin(now * (2 * math.pi / self.sway_period)) * self.sway_amplitude_deg
