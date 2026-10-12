"""A simple radial particle burst for celebratory effects (e.g. scoring a
point). Screen-space only (no 3D physics) - the burst is triggered at a
fixed 2D screen position and particles fly outward from there.

Position/fade timing logic is pure and takes `now` explicitly (same
pattern as render/hud.py's timers), so it's unit-testable; draw_particles()
does the actual pygame rendering and isn't tested directly. trigger()
takes an injectable `rng` (defaults to the real `random` module, same
pattern as game/opponent.py) so tests can get deterministic particle
positions instead of random ones.
"""
import math
import random

import pygame

PARTICLE_COLOR = (255, 220, 0)


class ParticleBurst:
    def __init__(self, num_particles=24, lifetime_seconds=0.6, speed_range=(80, 220)):
        self.num_particles = num_particles
        self.lifetime = lifetime_seconds
        self.speed_range = speed_range
        self.particles = []  # each: {"x", "y", "vx", "vy"}
        self.triggered_at = None

    def trigger(self, x, y, now, rng=random):
        self.triggered_at = now
        self.particles = []
        for _ in range(self.num_particles):
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(*self.speed_range)
            self.particles.append({
                "x": x, "y": y,
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
            })

    def is_active(self, now):
        return self.triggered_at is not None and (now - self.triggered_at) < self.lifetime

    def fade(self, now):
        """0-1, 1.0 right when triggered, fading linearly to 0 by lifetime."""
        if self.triggered_at is None:
            return 0.0
        elapsed = now - self.triggered_at
        if elapsed >= self.lifetime:
            return 0.0
        return 1.0 - elapsed / self.lifetime

    def positions(self, now):
        """Current (x, y) of each particle at time `now`, or [] if the
        burst isn't active (not triggered yet, or already expired)."""
        if not self.is_active(now):
            return []
        elapsed = now - self.triggered_at
        return [(p["x"] + p["vx"] * elapsed, p["y"] + p["vy"] * elapsed) for p in self.particles]


def draw_particles(surface, burst, now):
    fade = burst.fade(now)
    if fade <= 0:
        return
    alpha = int(255 * fade)
    radius = max(1, int(5 * fade) + 1)
    particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(particle_surf, (*PARTICLE_COLOR, alpha), (radius, radius), radius)
    for x, y in burst.positions(now):
        surface.blit(particle_surf, (x - radius, y - radius))
