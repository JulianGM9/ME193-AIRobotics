"""AI opponent: moves toward the predicted ball-landing x position with a
reaction delay and max speed, then decides whether to return the ball or
miss. Built as a plain class with tunable parameters (from a level's
config entry) so future opponent "characters" can each use their own cfg
without changing this logic.
"""
from game.physics import predict_landing_x


class Opponent:
    def __init__(self, cfg):
        """cfg: a level's opponent_* settings, e.g.
        {"max_speed": 1.5, "reaction_delay": 0.25, "miss_probability": 0.15,
         "return_speed_range": (2.5, 3.5), "return_target_spread": 0.6}
        """
        self.cfg = cfg
        self.x = 0.0
        self._target_x = None
        self._reaction_elapsed = 0.0
        self._tracking_new_ball = False

    def notice_ball_incoming(self):
        """Call once when the ball starts heading toward the opponent
        (e.g. right after the player's hit), to begin the reaction delay."""
        self._reaction_elapsed = 0.0
        self._tracking_new_ball = True
        self._target_x = None

    def update(self, ball, dt, opponent_end_z):
        """Move toward the predicted landing x, respecting reaction_delay
        and max_speed. opponent_end_z is physics_cfg's -table_length/2."""
        if self._tracking_new_ball:
            self._reaction_elapsed += dt
            if self._reaction_elapsed >= self.cfg["reaction_delay"]:
                self._target_x = predict_landing_x(ball, target_z=opponent_end_z)
                self._tracking_new_ball = False

        if self._target_x is not None:
            max_step = self.cfg["max_speed"] * dt
            delta = self._target_x - self.x
            if abs(delta) <= max_step:
                self.x = self._target_x
            else:
                self.x += max_step if delta > 0 else -max_step

    def decide_outcome(self, rng):
        """rng: a random.Random instance (injected for testability).
        Returns "miss" or "return"."""
        return "miss" if rng.random() < self.cfg["miss_probability"] else "return"

    def choose_return(self, rng):
        """Returns {"speed": ..., "target_x": ...} for the return shot."""
        speed_lo, speed_hi = self.cfg["return_speed_range"]
        spread = self.cfg["return_target_spread"]
        return {
            "speed": rng.uniform(speed_lo, speed_hi),
            "target_x": rng.uniform(-spread, spread),
        }
