"""Animated stick figure for the gait UI (MQTT add-on, part 3).

FigureAnimator.update(dt, activity, cadence_hz) advances the motion;
FigureAnimator.draw(surf, rect, color) draws one frame inside rect.

The figure faces right. Joint angles are in degrees from straight down,
positive = toward the front. One "phase" cycle = one full stride (each leg
swings once) or one pedal revolution, and it advances at the cadence the
board measures, so the figure moves at your pace.
"""
import math

import pygame

DEFAULT_CADENCE = {"standing": 0.25, "walking": 0.9, "running": 1.4, "biking": 1.0}  # Hz
CADENCE_RANGE = {"walking": (0.5, 1.4), "running": (1.0, 2.2), "biking": (0.4, 2.0)}
GROUND_SPEED = {"walking": 1.6, "running": 3.2}  # figure-heights per stride, for the scrolling ground

# Body proportions, as fractions of total figure height
HEAD_R, NECK, TORSO = 0.07, 0.04, 0.30
THIGH, SHIN = 0.25, 0.25
UPPER_ARM, FOREARM = 0.16, 0.15


def _pt(origin, angle_deg, length):
    """End point of a limb hanging from origin at angle_deg from straight down (positive = forward)."""
    a = math.radians(angle_deg)
    return origin[0] + length * math.sin(a), origin[1] + length * math.cos(a)


def _ik(root, target, l1, l2, bend_forward=True):
    """Two-link inverse kinematics: middle joint (knee/elbow) between root and target."""
    dx, dy = target[0] - root[0], target[1] - root[1]
    d = max(1e-6, min(math.hypot(dx, dy), l1 + l2 - 1e-6))
    a = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    base = math.atan2(dy, dx)
    ang = base - a if bend_forward else base + a
    return root[0] + l1 * math.cos(ang), root[1] + l1 * math.sin(ang)


class FigureAnimator:
    def __init__(self):
        self.phase = 0.0  # stride / pedal cycles, keeps growing
        self.ground = 0.0  # scroll offset in figure-heights
        self.activity = "standing"

    def update(self, dt, activity, cadence_hz):
        self.activity = activity if activity in DEFAULT_CADENCE else "standing"
        hz = DEFAULT_CADENCE[self.activity]
        if self.activity in CADENCE_RANGE and cadence_hz:
            lo, hi = CADENCE_RANGE[self.activity]
            hz = min(hi, max(lo, cadence_hz))
        self.phase += hz * dt
        self.ground += GROUND_SPEED.get(self.activity, 0.0) * hz * dt

    # ------------------------------------------------------------ poses

    def _upright_pose(self):
        """Joint angles for standing / walking / running at the current phase."""
        p = 2 * math.pi * self.phase
        if self.activity == "walking":
            hip = [25 * math.sin(p), 25 * math.sin(p + math.pi)]
            knee = [5 + 35 * max(0.0, math.cos(p)), 5 + 35 * max(0.0, math.cos(p + math.pi))]
            shoulder = [-22 * math.sin(p), -22 * math.sin(p + math.pi)]
            elbow, lean = [15, 15], 3
            bob = -0.012 * abs(math.cos(p))
        elif self.activity == "running":
            hip = [40 * math.sin(p), 40 * math.sin(p + math.pi)]
            knee = [20 + 80 * max(0.0, math.cos(p)), 20 + 80 * max(0.0, math.cos(p + math.pi))]
            shoulder = [-38 * math.sin(p), -38 * math.sin(p + math.pi)]
            elbow, lean = [85, 85], 12
            bob = -0.035 * abs(math.sin(p))
        else:  # standing: weight shifts and breathing
            hip, knee = [4, -4], [0, 0]
            shoulder, elbow, lean = [6, -4], [8, 8], 0
            bob = 0.004 * math.sin(p)
        return hip, knee, shoulder, elbow, lean, bob

    # ------------------------------------------------------------ drawing

    def draw(self, surf, rect, color, line=(70, 75, 95)):
        size = rect.height * 0.78  # total figure height in pixels
        width = max(3, int(size * 0.035))
        ground_y = rect.bottom - rect.height * 0.08
        self._draw_ground(surf, rect, ground_y, size, line)
        if self.activity == "biking":
            self._draw_biker(surf, rect, ground_y, size, color, line, width)
        else:
            self._draw_upright(surf, rect, ground_y, size, color, width)

    def _draw_ground(self, surf, rect, y, size, line):
        pygame.draw.line(surf, line, (rect.x + 10, y), (rect.right - 10, y), 2)
        spacing = size * 0.35
        offset = (self.ground * size) % spacing
        x = rect.right - 10 - offset
        while x > rect.x + 10:
            pygame.draw.line(surf, line, (x, y + 6), (x - spacing * 0.3, y + 6), 2)
            x -= spacing

    def _limb(self, surf, color, pts, width):
        pygame.draw.lines(surf, color, False, [(round(x), round(y)) for x, y in pts], width)
        for x, y in pts[1:]:
            pygame.draw.circle(surf, color, (round(x), round(y)), width // 2)

    def _draw_upright(self, surf, rect, ground_y, size, color, width):
        hip_a, knee_a, sh_a, el_a, lean, bob = self._upright_pose()
        leg_len = (THIGH + SHIN) * size
        hip = (rect.centerx, ground_y - leg_len * 0.97 + bob * size)
        # Keep the lowest foot on the ground.
        feet = []
        for h, k in zip(hip_a, knee_a):
            knee = _pt(hip, h, THIGH * size)
            feet.append(_pt(knee, h - k, SHIN * size)[1])
        hip = (hip[0], hip[1] + ground_y - max(feet))

        neck = _pt(hip, 180 - lean, TORSO * size)
        head = _pt(neck, 180 - lean, (NECK + HEAD_R) * size)
        shoulder = _pt(hip, 180 - lean, TORSO * size * 0.92)

        far, near = tuple(c // 2 + 40 for c in color), color  # far-side limbs drawn darker, behind
        for i, shade in ((1, far), (0, near)):
            knee = _pt(hip, hip_a[i], THIGH * size)
            foot = _pt(knee, hip_a[i] - knee_a[i], SHIN * size)
            toe = (foot[0] + size * 0.05, foot[1])
            elbow = _pt(shoulder, sh_a[i], UPPER_ARM * size)
            hand = _pt(elbow, sh_a[i] + el_a[i], FOREARM * size)
            self._limb(surf, shade, [hip, knee, foot, toe], width)
            self._limb(surf, shade, [shoulder, elbow, hand], width)
            if i == 1:  # torso and head between the far and near limbs
                self._limb(surf, near, [hip, neck], width)
                pygame.draw.circle(surf, near, (round(head[0]), round(head[1])), round(HEAD_R * size))

    def _draw_biker(self, surf, rect, ground_y, size, color, line, width):
        p = 2 * math.pi * self.phase
        wheel_r = size * 0.24
        rear = (rect.centerx - size * 0.42, ground_y - wheel_r)
        front = (rect.centerx + size * 0.42, ground_y - wheel_r)
        crank = (rect.centerx - size * 0.02, ground_y - wheel_r * 0.95)
        seat = (rect.centerx - size * 0.16, ground_y - size * 0.58)
        bar = (rect.centerx + size * 0.30, ground_y - size * 0.66)
        bike = (150, 155, 175)

        # Wheels with spinning spokes
        spin = p * 2.2
        for c in (rear, front):
            pygame.draw.circle(surf, bike, (round(c[0]), round(c[1])), round(wheel_r), max(2, width - 1))
            for k in range(4):
                a = spin + k * math.pi / 4
                dx, dy = math.cos(a) * wheel_r * 0.9, math.sin(a) * wheel_r * 0.9
                pygame.draw.line(surf, line, (c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy), 1)
        # Frame
        for a, b in ((rear, crank), (crank, seat), (seat, rear), (seat, bar), (crank, bar), (bar, front)):
            pygame.draw.line(surf, bike, a, b, max(2, width - 1))

        crank_r = size * 0.09
        hip = (seat[0], seat[1] - size * 0.03)
        shoulder = _pt(hip, 180 - 55, TORSO * size)  # leaning forward over the bars
        neck = _pt(hip, 180 - 55, TORSO * size * 1.08)
        head = _pt(neck, 180 - 35, (NECK + HEAD_R) * size)
        grip = (bar[0], bar[1] - size * 0.02)

        far, near = tuple(c // 2 + 40 for c in color), color
        for i, shade in ((1, far), (0, near)):
            a = p + i * math.pi
            pedal = (crank[0] + crank_r * math.cos(a), crank[1] + crank_r * math.sin(a))
            knee = _ik(hip, pedal, THIGH * size, SHIN * size, bend_forward=True)  # knee toward the bars
            self._limb(surf, shade, [hip, knee, pedal], width)
            pygame.draw.line(surf, bike, crank, pedal, 2)
            if i == 1:
                self._limb(surf, near, [hip, neck], width)
                pygame.draw.circle(surf, near, (round(head[0]), round(head[1])), round(HEAD_R * size))
        elbow = _ik(shoulder, grip, UPPER_ARM * size, FOREARM * size, bend_forward=False)
        self._limb(surf, near, [shoulder, elbow, grip], width)

