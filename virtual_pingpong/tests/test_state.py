from game.state import RALLY, SERVING, WINNING_SCORE, GameState


def test_starts_in_serving_phase_with_zero_streaks():
    state = GameState()
    assert state.phase == SERVING
    assert state.current_streak == 0
    assert state.record_streak == 0


def test_hits_increment_current_and_record_streak():
    state = GameState()
    state.register_player_hit()
    state.register_player_hit()
    state.register_player_hit()
    assert state.current_streak == 3
    assert state.record_streak == 3


def test_miss_resets_current_streak_but_keeps_record():
    state = GameState()
    state.register_player_hit()
    state.register_player_hit()
    state.register_player_miss()
    assert state.current_streak == 0
    assert state.record_streak == 2


def test_record_streak_keeps_the_best_across_multiple_rallies():
    state = GameState()
    state.register_player_hit()
    state.register_player_hit()
    state.register_player_hit()  # streak of 3
    state.register_player_miss()
    state.ready_for_next_rally()
    state.start_rally()
    state.register_player_hit()  # only gets to 1 this time
    state.register_player_miss()
    assert state.current_streak == 0
    assert state.record_streak == 3  # unaffected by the shorter second rally


def test_miss_sets_point_over_phase():
    state = GameState()
    state.start_rally()
    state.register_player_miss()
    assert state.phase == "point_over"


def test_ready_for_next_rally_returns_to_serving():
    state = GameState()
    state.start_rally()
    state.register_player_miss()
    state.ready_for_next_rally()
    assert state.phase == SERVING


def test_start_rally_resets_bounce_flag():
    state = GameState()
    state.start_rally()
    state.note_bounce("player")
    assert state.bounced_on_player_side is True
    state.start_rally()
    assert state.bounced_on_player_side is False


def test_bounce_on_opponent_side_does_not_set_player_bounce_flag():
    state = GameState()
    state.start_rally()
    state.note_bounce("opponent")
    assert state.bounced_on_player_side is False


def test_player_hit_clears_bounce_flag_so_a_second_bounce_is_required():
    state = GameState()
    state.start_rally()
    state.note_bounce("player")
    state.register_player_hit()
    assert state.bounced_on_player_side is False


def test_starts_with_zero_match_score_and_no_winner():
    state = GameState()
    assert state.player_score == 0
    assert state.opponent_score == 0
    assert state.match_winner is None


def test_register_point_increments_the_right_side():
    state = GameState()
    state.register_point("player")
    state.register_point("player")
    state.register_point("opponent")
    assert state.player_score == 2
    assert state.opponent_score == 1


def test_match_winner_is_none_below_the_winning_score():
    state = GameState()
    for _ in range(WINNING_SCORE - 1):
        state.register_point("player")
    assert state.match_winner is None


def test_player_reaching_winning_score_wins_the_match():
    state = GameState()
    for _ in range(WINNING_SCORE):
        state.register_point("player")
    assert state.match_winner == "player"


def test_opponent_reaching_winning_score_wins_the_match():
    state = GameState()
    for _ in range(WINNING_SCORE):
        state.register_point("opponent")
    assert state.match_winner == "opponent"


def test_match_score_is_independent_of_rally_streak_tracking():
    state = GameState()
    state.start_rally()
    state.register_player_hit()
    state.register_player_miss()  # resets current_streak, but is a separate call from scoring
    state.register_point("opponent")
    assert state.current_streak == 0
    assert state.opponent_score == 1
    assert state.player_score == 0
