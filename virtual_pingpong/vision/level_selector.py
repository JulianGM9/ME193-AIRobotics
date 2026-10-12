"""LevelSelector: pure logic, no cv2 dependency, kept separate from
apriltag.py (which imports cv2.aruco at module level) so code that only
needs the selection state machine - e.g. main.py in --mock-vision mode -
never pulls in cv2.
"""


class LevelSelector:
    """Tracks which level's tag is currently being held up, requiring it
    to be seen stably for confirm_hold_seconds before confirming a
    selection. Waits indefinitely otherwise - there's no timeout/default
    fallback, so it won't start a game until the player explicitly picks a
    level, either by tag or immediately via select_by_key() (keyboard).

    Takes `now` as an explicit argument everywhere rather than reading the
    clock itself, so this can be unit-tested with a fake clock.
    """

    def __init__(self, tag_to_level, confirm_hold_seconds):
        self.tag_to_level = tag_to_level
        self.confirm_hold_seconds = confirm_hold_seconds
        self._candidate_level = None
        self._candidate_since = None
        self.confirmed_level = None

    def start(self, now):
        self._candidate_level = None
        self._candidate_since = None
        self.confirmed_level = None

    def update(self, detected_tag_ids, now):
        """Call every frame with the set of tag IDs currently visible.
        Returns the confirmed level once decided, else None."""
        if self.confirmed_level is not None:
            return self.confirmed_level

        level = None
        for tag_id in detected_tag_ids:
            if tag_id in self.tag_to_level:
                level = self.tag_to_level[tag_id]
                break  # first recognized tag wins if several are visible

        if level != self._candidate_level:
            self._candidate_level = level
            self._candidate_since = now

        if level is not None and (now - self._candidate_since) >= self.confirm_hold_seconds:
            self.confirmed_level = level
            return level

        return None

    def select_by_key(self, level):
        """Immediately confirm a level, e.g. from a keyboard shortcut."""
        self.confirmed_level = level
        return level

    def current_candidate(self):
        """The level currently being held up (even if not yet confirmed),
        or None if nothing is currently recognized. For UI feedback like
        "You are selecting X... keep holding!" while progress builds."""
        return self._candidate_level

    def progress(self, now):
        """How far through the confirm-hold the current candidate is, in
        [0, 1]; 0 if nothing is currently being held up."""
        if self._candidate_level is None or self._candidate_since is None:
            return 0.0
        elapsed = now - self._candidate_since
        return min(1.0, elapsed / self.confirm_hold_seconds)
