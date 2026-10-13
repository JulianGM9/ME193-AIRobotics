import random

import pytest

from game.q_learning import QLearningOpponent


class FixedRng:
    """Deterministic stand-in for random.Random: explore_roll controls
    the random() call (used for the epsilon-greedy explore/exploit
    decision), choice_index controls which of a tie-broken list
    rng.choice() picks."""

    def __init__(self, explore_roll=1.0, random_action=0, choice_index=0):
        self.explore_roll = explore_roll
        self.random_action = random_action
        self.choice_index = choice_index

    def random(self):
        return self.explore_roll

    def randrange(self, n):
        return self.random_action

    def choice(self, seq):
        return seq[self.choice_index]


def test_starts_with_an_all_zero_q_table():
    bot = QLearningOpponent(num_ball_buckets=5, num_paddle_buckets=5)
    assert bot.q_table == [[0.0] * 5 for _ in range(5)]


def test_episodes_starts_at_zero():
    bot = QLearningOpponent()
    assert bot.episodes == 0


def test_ball_at_center_maps_to_the_middle_bucket():
    bot = QLearningOpponent(num_ball_buckets=9, half_width=0.9)
    state, _, _ = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=1.0))
    assert state == 4  # middle of 9 buckets (0-8)


def test_ball_at_left_edge_maps_to_bucket_zero():
    bot = QLearningOpponent(num_ball_buckets=9, half_width=0.9)
    state, _, _ = bot.choose_action(ball_x=-0.9, rng=FixedRng(explore_roll=1.0))
    assert state == 0


def test_ball_at_right_edge_maps_to_the_last_bucket():
    bot = QLearningOpponent(num_ball_buckets=9, half_width=0.9)
    state, _, _ = bot.choose_action(ball_x=0.9, rng=FixedRng(explore_roll=1.0))
    assert state == 8


def test_ball_beyond_the_table_edge_clamps_to_the_nearest_bucket():
    bot = QLearningOpponent(num_ball_buckets=9, half_width=0.9)
    state, _, _ = bot.choose_action(ball_x=5.0, rng=FixedRng(explore_roll=1.0))
    assert state == 8
    state, _, _ = bot.choose_action(ball_x=-5.0, rng=FixedRng(explore_roll=1.0))
    assert state == 0


def test_exploration_roll_below_epsilon_picks_a_random_action():
    bot = QLearningOpponent(num_ball_buckets=5, num_paddle_buckets=5, exploration_rate=0.5)
    # explore_roll (0.1) < exploration_rate (0.5) -> explore
    state, action, target_x = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=0.1, random_action=3))
    assert action == 3


def test_exploration_roll_above_epsilon_picks_the_best_known_action():
    bot = QLearningOpponent(num_ball_buckets=5, num_paddle_buckets=5, exploration_rate=0.5)
    bot.q_table[2][4] = 0.9  # best action for state 2 is action 4
    state, action, target_x = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=0.9))
    # ball_x=0.0 with 5 buckets over default half_width -> state 2 (middle)
    assert state == 2
    assert action == 4


def test_ties_among_best_actions_are_broken_by_rng_choice():
    bot = QLearningOpponent(num_ball_buckets=5, num_paddle_buckets=5, exploration_rate=0.0)
    # All zero -> every action is tied for best; choice_index=2 should pick the 3rd candidate (index 2).
    _, action, _ = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=1.0, choice_index=2))
    assert action == 2


def test_target_x_is_the_center_of_the_chosen_action_bucket():
    bot = QLearningOpponent(num_ball_buckets=5, num_paddle_buckets=5, half_width=1.0)
    bot.q_table[2][0] = 1.0  # force action 0 to be best
    _, action, target_x = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=1.0))
    assert action == 0
    # 5 buckets over [-1, 1]: bucket 0 spans [-1.0, -0.6], center -0.8
    assert target_x == pytest.approx(-0.8)


def test_update_moves_the_q_value_toward_the_reward():
    bot = QLearningOpponent(learning_rate=0.5)
    bot.update(state=0, action=0, reward=1.0)
    assert bot.q_table[0][0] == pytest.approx(0.5)  # halfway from 0 toward 1, at lr=0.5
    bot.update(state=0, action=0, reward=1.0)
    assert bot.q_table[0][0] == pytest.approx(0.75)  # halfway again, from 0.5 toward 1


def test_update_handles_negative_reward():
    bot = QLearningOpponent(learning_rate=0.5)
    bot.update(state=0, action=0, reward=-1.0)
    assert bot.q_table[0][0] == pytest.approx(-0.5)


def test_update_tracks_hit_and_miss_counts():
    bot = QLearningOpponent()
    bot.update(state=0, action=0, reward=1.0)
    bot.update(state=0, action=0, reward=-1.0)
    bot.update(state=1, action=1, reward=1.0)
    assert bot.hits == 2
    assert bot.misses == 1
    assert bot.episodes == 3


def test_learning_rate_and_exploration_rate_are_mutable_at_runtime():
    # These need to be live-adjustable via the in-game sliders, not fixed
    # at construction time.
    bot = QLearningOpponent(learning_rate=0.1, exploration_rate=0.9)
    bot.learning_rate = 0.8
    bot.exploration_rate = 0.0
    bot.update(state=0, action=0, reward=1.0)
    assert bot.q_table[0][0] == pytest.approx(0.8)
    _, action, _ = bot.choose_action(ball_x=0.0, rng=FixedRng(explore_roll=0.5))
    # exploration_rate is now 0.0, so even a low roll must not trigger exploration.
    assert action == bot.q_table[bot._bucket_for_x(0.0, bot.num_ball_buckets)].index(
        max(bot.q_table[bot._bucket_for_x(0.0, bot.num_ball_buckets)])
    )


def test_works_with_the_real_random_module_end_to_end():
    # Sanity check against the real random module (not just the fake),
    # since that's what production code actually passes.
    bot = QLearningOpponent()
    rng = random.Random(0)
    state, action, target_x = bot.choose_action(ball_x=0.1, rng=rng)
    bot.update(state, action, reward=1.0)
    assert bot.episodes == 1
