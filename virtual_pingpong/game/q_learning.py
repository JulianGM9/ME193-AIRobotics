"""Tabular Q-learning opponent for the secret "Claude" level.

State = which bucket the incoming ball's predicted landing x falls into.
Action = which bucket to move the paddle to. Reward is a single
immediate +1 (the paddle ended up close enough to the ball's actual
arrival x) or -1 (it didn't) - there's no meaningful "next state" to chain
rewards through (each incoming ball is an independent trial, not a step in
a longer sequence), so this is really a contextual bandit dressed as
Q-learning with no discount term, which is exactly what the classic
"Q-table for Pong" tutorials implement under that name.

Pure logic, no pygame/rendering here, so it's unit-testable - mirrors the
rest of this project's game/ modules (Opponent, GameState, etc.).
"""
import random


class QLearningOpponent:
    def __init__(self, num_ball_buckets=9, num_paddle_buckets=9, half_width=0.7625,
                 learning_rate=0.3, exploration_rate=0.2):
        self.num_ball_buckets = num_ball_buckets
        self.num_paddle_buckets = num_paddle_buckets
        self.half_width = half_width
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = [[0.0] * num_paddle_buckets for _ in range(num_ball_buckets)]
        self.hits = 0
        self.misses = 0

    @property
    def episodes(self):
        return self.hits + self.misses

    def _bucket_for_x(self, x, num_buckets):
        """Maps a table-space x in [-half_width, +half_width] to a bucket
        index in [0, num_buckets), clamped at the edges."""
        fraction = (x + self.half_width) / (2 * self.half_width)
        fraction = min(max(fraction, 0.0), 0.999999)
        return int(fraction * num_buckets)

    def _bucket_center_x(self, bucket, num_buckets):
        fraction = (bucket + 0.5) / num_buckets
        return fraction * 2 * self.half_width - self.half_width

    def choose_action(self, ball_x, rng=random):
        """Epsilon-greedy over the ball's predicted-landing bucket.
        Returns (state, action, target_x) - state/action are bucket
        indices (pass both back into update()); target_x is where to
        actually move the paddle, in the same table-space units as
        ball_x.
        """
        state = self._bucket_for_x(ball_x, self.num_ball_buckets)
        if rng.random() < self.exploration_rate:
            action = rng.randrange(self.num_paddle_buckets)
        else:
            row = self.q_table[state]
            best_value = max(row)
            best_actions = [i for i, value in enumerate(row) if value == best_value]
            # Ties (e.g. an untrained all-zero row) broken randomly rather
            # than always picking action 0 - otherwise the bot would
            # always start by camping at one edge instead of exploring
            # evenly before it's learned anything.
            action = rng.choice(best_actions)
        target_x = self._bucket_center_x(action, self.num_paddle_buckets)
        return state, action, target_x

    def update(self, state, action, reward):
        """reward: +1 (hit) or -1 (miss). No discount/next-state term -
        see the module docstring for why."""
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
        if reward > 0:
            self.hits += 1
        else:
            self.misses += 1
