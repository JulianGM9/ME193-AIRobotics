"""Main menu and pause menu overlays. Esc/P toggle pause, L returns to
level select, T toggles the tracking demo view, and Q quits; Enter or a
click on the begin button starts from the main menu - all handled by
main.py's event loop, this module only draws them."""
import pygame

OVERLAY_COLOR = (0, 0, 0, 160)
TITLE_COLOR = (255, 255, 255)
SUBTITLE_COLOR = (220, 220, 220)

MAIN_MENU_BG_COLOR = (12, 12, 28)
MAIN_MENU_TITLE_COLOR = (255, 220, 60)
MAIN_MENU_TITLE_SHADOW_COLOR = (120, 90, 0)
BUTTON_FILL_COLOR = (40, 180, 90)
BUTTON_BORDER_COLOR = (255, 255, 255)
BUTTON_TEXT_COLOR = (20, 20, 20)


def draw_main_menu(surface, title_font, subtitle_font, begin_button_rect):
    surface.fill(MAIN_MENU_BG_COLOR)

    center_x = surface.get_width() // 2
    title_y = surface.get_height() // 2 - 80

    # A simple drop-shadow (same text, offset, in a darker color, drawn
    # first) for a bit of retro-title depth without needing an actual
    # sprite.
    shadow = title_font.render("VIRTUAL PING PONG", True, MAIN_MENU_TITLE_SHADOW_COLOR)
    surface.blit(shadow, shadow.get_rect(center=(center_x + 4, title_y + 4)))
    title = title_font.render("VIRTUAL PING PONG", True, MAIN_MENU_TITLE_COLOR)
    surface.blit(title, title.get_rect(center=(center_x, title_y)))

    pygame.draw.rect(surface, BUTTON_FILL_COLOR, begin_button_rect)
    pygame.draw.rect(surface, BUTTON_BORDER_COLOR, begin_button_rect, width=3)
    label = subtitle_font.render("BEGIN", True, BUTTON_TEXT_COLOR)
    surface.blit(label, label.get_rect(center=begin_button_rect.center))

    hint = subtitle_font.render("Press Enter or click to begin", True, SUBTITLE_COLOR)
    surface.blit(hint, hint.get_rect(center=(center_x, begin_button_rect.bottom + 40)))


def draw_pause_overlay(surface, title_font, subtitle_font):
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill(OVERLAY_COLOR)
    surface.blit(overlay, (0, 0))

    center_x, center_y = surface.get_width() // 2, surface.get_height() // 2

    title = title_font.render("PAUSED", True, TITLE_COLOR)
    surface.blit(title, title.get_rect(center=(center_x, center_y - 40)))

    subtitle = subtitle_font.render("P / Esc to resume - Q to quit", True, SUBTITLE_COLOR)
    surface.blit(subtitle, subtitle.get_rect(center=(center_x, center_y)))

    menu_hint = subtitle_font.render("L to return to level select", True, SUBTITLE_COLOR)
    surface.blit(menu_hint, menu_hint.get_rect(center=(center_x, center_y + 35)))

    tracking_hint = subtitle_font.render("T to toggle the tracking view", True, SUBTITLE_COLOR)
    surface.blit(tracking_hint, tracking_hint.get_rect(center=(center_x, center_y + 70)))
