"""HUD: streak/record/level/connection status, transient hit-feedback
banner, and the between-rally "Ready?" countdown.

FeedbackBanner and Countdown hold pure, clock-driven state (no pygame),
so their timing logic is unit-testable; the draw_* functions below do the
actual pygame rendering and aren't tested directly.
"""
import pygame

TEXT_COLOR = (255, 255, 255)
STREAK_COLOR = (255, 220, 0)
OFFLINE_COLOR = (220, 50, 50)


class FeedbackBanner:
    """A transient on-screen message (e.g. "Too early!") that fades out
    over duration_seconds. Takes `now` explicitly everywhere."""

    def __init__(self, duration_seconds=1.2):
        self.duration = duration_seconds
        self.message = None
        self.shown_at = None

    def show(self, message, now):
        self.message = message
        self.shown_at = now

    def alpha(self, now):
        """0-255, linearly fading out over duration_seconds; 0 once expired
        or if nothing has been shown."""
        if self.message is None or self.shown_at is None:
            return 0
        elapsed = now - self.shown_at
        if elapsed >= self.duration:
            return 0
        return int(255 * (1.0 - elapsed / self.duration))

    def is_visible(self, now):
        return self.alpha(now) > 0


class SwingFlash:
    """Tracks a brief "a swing was just detected" visual pulse, decoupled
    from hit/miss outcome - fires the instant a swing is seen, regardless
    of whether it ends up counting as a hit. Returns a 0-1 intensity that
    decays to 0 over duration_seconds, for fading a flash/glow effect."""

    def __init__(self, duration_seconds=0.3):
        self.duration = duration_seconds
        self.triggered_at = None

    def trigger(self, now):
        self.triggered_at = now

    def intensity(self, now):
        if self.triggered_at is None:
            return 0.0
        elapsed = now - self.triggered_at
        if elapsed >= self.duration:
            return 0.0
        return 1.0 - elapsed / self.duration


class Countdown:
    """A simple "Ready?" countdown between rallies."""

    def __init__(self, duration_seconds=3.0):
        self.duration = duration_seconds
        self.started_at = None

    def start(self, now, duration=None):
        """duration: overrides this instance's normal duration for just
        this run (e.g. the Claude level using a much shorter pause between
        rallies than the regular levels, from the same shared timer)."""
        self.started_at = now
        if duration is not None:
            self.duration = duration

    def remaining(self, now):
        if self.started_at is None:
            return 0.0
        return max(0.0, self.duration - (now - self.started_at))

    def is_active(self, now):
        return self.started_at is not None and self.remaining(now) > 0.0

    def is_done(self, now):
        return self.started_at is not None and self.remaining(now) <= 0.0


def draw_hud(surface, font, current_streak, record_streak, player_score, opponent_score, level_name, mqtt_connected):
    streak_text = font.render(f"Streak: {current_streak}  (best: {record_streak})", True, STREAK_COLOR)
    surface.blit(streak_text, (10, 10))

    score_text = font.render(f"You {player_score} - {opponent_score} Opponent", True, TEXT_COLOR)
    surface.blit(score_text, score_text.get_rect(centerx=surface.get_width() // 2, y=10))

    level_text = font.render(level_name.capitalize(), True, TEXT_COLOR)
    surface.blit(level_text, (surface.get_width() - level_text.get_width() - 10, 10))

    if not mqtt_connected:
        offline_text = font.render("MQTT offline", True, OFFLINE_COLOR)
        surface.blit(offline_text, (10, surface.get_height() - offline_text.get_height() - 10))


def draw_game_over(surface, title_font, subtitle_font, winner, player_score, opponent_score):
    """A big full-screen result display once a match ends (first to
    WINNING_SCORE in game/state.py). winner: "player" or "opponent"."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surface.blit(overlay, (0, 0))

    center_x, center_y = surface.get_width() // 2, surface.get_height() // 2
    title_text = "YOU WIN!" if winner == "player" else "YOU LOSE!"
    title_color = (80, 220, 80) if winner == "player" else (220, 70, 70)

    title = title_font.render(title_text, True, title_color)
    surface.blit(title, title.get_rect(center=(center_x, center_y - 40)))

    score_text = subtitle_font.render(f"Final score: {player_score} - {opponent_score}", True, TEXT_COLOR)
    surface.blit(score_text, score_text.get_rect(center=(center_x, center_y + 20)))

    hint_text = subtitle_font.render("Press Enter for level select", True, TEXT_COLOR)
    surface.blit(hint_text, hint_text.get_rect(center=(center_x, center_y + 55)))


def draw_feedback_banner(surface, font, banner, now):
    alpha = banner.alpha(now)
    if alpha <= 0:
        return
    text_surf = font.render(banner.message, True, TEXT_COLOR)
    text_surf.set_alpha(alpha)
    rect = text_surf.get_rect(center=(surface.get_width() // 2, surface.get_height() // 3))
    surface.blit(text_surf, rect)


def draw_countdown(surface, font, countdown, now):
    if not countdown.is_active(now):
        return
    seconds_left = int(countdown.remaining(now)) + 1
    text_surf = font.render(f"Ready? {seconds_left}", True, TEXT_COLOR)
    rect = text_surf.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))
    surface.blit(text_surf, rect)
