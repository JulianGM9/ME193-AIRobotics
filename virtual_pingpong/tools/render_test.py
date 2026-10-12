"""Standalone tool: visually validate render/projection.py +
render/table_view.py by running real ball physics (no camera/IMU needed)
with a repeating serve, plus keyboard controls to move the paddle marker
around and trigger its swing-flash glow. This is the render subsystem's
isolation test, same pattern as vision_test.py/swing_test.py for their
subsystems.
Run with: python3 tools/render_test.py
Keys: SPACE = re-serve the ball, F = trigger the swing flash,
      arrow keys = move paddle marker, q = quit.
"""
import sys
import time
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from game.physics import Ball, step
from render.hud import SwingFlash
from render.projection import make_camera
from render.table_view import draw_ball, draw_net, draw_opponent, draw_paddle, draw_table


def serve_ball(physics_cfg):
    half_l = physics_cfg["table_length"] / 2
    return Ball(x=0.0, y=0.4, z=half_l - 0.1, vx=0.3, vy=0.0, vz=-3.0)


def main():
    cfg = config.load()
    physics_cfg = cfg["physics"]
    render_cfg = cfg["render"]
    camera = make_camera(physics_cfg, render_cfg)

    pygame.init()
    screen = pygame.display.set_mode((render_cfg["screen_width"], render_cfg["screen_height"]))
    pygame.display.set_caption("Render subsystem test")
    clock = pygame.time.Clock()

    ball = serve_ball(physics_cfg)
    paddle_x, paddle_y = 0.0, 0.3
    opponent_x = 0.0
    swing_flash = SwingFlash(duration_seconds=0.3)

    print("Keys: SPACE = re-serve the ball, F = trigger swing flash, arrow keys = move paddle marker, q = quit")

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.monotonic()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                swing_flash.trigger(now)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False
        if keys[pygame.K_SPACE]:
            ball = serve_ball(physics_cfg)
        if keys[pygame.K_LEFT]:
            paddle_x -= 1.0 * dt
        if keys[pygame.K_RIGHT]:
            paddle_x += 1.0 * dt
        if keys[pygame.K_UP]:
            paddle_y += 1.0 * dt
        if keys[pygame.K_DOWN]:
            paddle_y = max(0.0, paddle_y - 1.0 * dt)

        ball, result = step(ball, dt, physics_cfg)
        if result.out_of_bounds:
            ball = serve_ball(physics_cfg)

        screen.fill((30, 30, 40))
        draw_table(screen, camera, physics_cfg)
        draw_net(screen, camera, physics_cfg)
        draw_opponent(screen, camera, opponent_x, physics_cfg)
        draw_paddle(screen, camera, paddle_x, paddle_y, physics_cfg["table_length"] / 2 - 0.1,
                    swing_intensity=swing_flash.intensity(now))
        draw_ball(screen, camera, ball)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
