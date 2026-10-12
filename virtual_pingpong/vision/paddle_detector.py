"""Tracks the paddle's on-screen position using the hand's palm center
(vision/hand_tracker.py) as the primary signal, falling back to the wrist
(vision/pose_tracker.py, always detected) when the hand isn't confidently
found - which happens often while gripping the paddle, since the hand
model is mostly trained on open hands with visible fingers.

This replaces an earlier color-based (HSV red-face) approach: it was
fragile to lighting changes and, worse, its dark-blob fallback would
frequently lock onto a dark patch of clothing instead of the paddle
(confirmed in testing). Pure landmark tracking has no color dependency at
all, trading a little positional precision (palm center vs. the actual
paddle face) for much better reliability.
"""


class PaddleDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.smoothed_pos = None  # normalized image coords
        self.velocity = (0.0, 0.0)
        self.last_source = None  # "hand" or "wrist", for the debug overlay

    def _detect_raw(self, pose, hand_position):
        if hand_position is not None:
            return hand_position, "hand"
        return pose["wrist"], "wrist"

    def update(self, pose, hand_position, dt):
        """Run detection + smoothing for one frame. pose is a PoseTracker
        result dict (for its "wrist" fallback); hand_position is a
        HandTracker result (normalized (x, y), or None if not detected).
        Returns (smoothed_position, velocity), both in normalized image
        coordinates; velocity is in units/second.
        """
        detected_point, source = self._detect_raw(pose, hand_position)
        self.last_source = source

        if self.smoothed_pos is None:
            self.smoothed_pos = detected_point
            self.velocity = (0.0, 0.0)
            return self.smoothed_pos, self.velocity

        alpha = self.cfg["smoothing_alpha"]
        prev = self.smoothed_pos
        new_smoothed = (
            prev[0] + alpha * (detected_point[0] - prev[0]),
            prev[1] + alpha * (detected_point[1] - prev[1]),
        )
        if dt > 0:
            self.velocity = ((new_smoothed[0] - prev[0]) / dt, (new_smoothed[1] - prev[1]) / dt)
        self.smoothed_pos = new_smoothed
        return self.smoothed_pos, self.velocity
