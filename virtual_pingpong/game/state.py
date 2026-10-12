"""Rally state machine and streak/record scoring.

A rally requires the ball to bounce on the player's side (per real
ping-pong rules) before a player hit is allowed to count - bounced flags
are reset at each new rally and after each player hit, since the ball
must bounce again on its way back before the next swing counts.
"""
SERVING = "serving"
RALLY = "rally"
POINT_OVER = "point_over"

WINNING_SCORE = 11  # first to reach this wins the match


class GameState:
    def __init__(self):
        self.current_streak = 0
        self.record_streak = 0
        self.phase = SERVING
        self.bounced_on_player_side = False
        self.player_score = 0
        self.opponent_score = 0

    def start_rally(self):
        self.phase = RALLY
        self.bounced_on_player_side = False

    def note_bounce(self, side):
        if side == "player":
            self.bounced_on_player_side = True

    def register_player_hit(self):
        self.current_streak += 1
        self.record_streak = max(self.record_streak, self.current_streak)
        self.bounced_on_player_side = False  # must bounce again before the next hit

    def register_player_miss(self):
        self.current_streak = 0
        self.phase = POINT_OVER

    def ready_for_next_rally(self):
        self.phase = SERVING

    def register_point(self, winner):
        """winner: "player" or "opponent". A rally's outcome (whichever
        side faulted) always awards the other side a point, separately
        from current_streak (which only tracks the player's own
        consecutive successful hits, via register_player_hit/
        register_player_miss above)."""
        if winner == "player":
            self.player_score += 1
        else:
            self.opponent_score += 1

    @property
    def match_winner(self):
        """"player" or "opponent" once either has reached WINNING_SCORE,
        else None."""
        if self.player_score >= WINNING_SCORE:
            return "player"
        if self.opponent_score >= WINNING_SCORE:
            return "opponent"
        return None
