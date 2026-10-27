"""Laptop UI for the gait detector (MQTT add-on, parts 2-3).

    python3 tools/gait_ui.py

Shows the activity the board is detecting with an animated figure doing it
(paced by the measured cadence), which model it's using, both models'
raw/smoothed predictions and a 60 s history strip. Click Model 1 /
Model 2 (or press 1 / 2) to switch the board's model over MQTT.

Talks to python/live.py on the board over ME193/Julian/GaitDetection.
"""
import json
import queue
import time

import paho.mqtt.client as mqtt
import pygame

from gait_figure import FigureAnimator

BROKER = "broker.emqx.io"  # must match python/live.py
TOPIC = "ME193/Julian/GaitDetection"
STALE_S = 3.0  # no status for this long -> show the board as silent
HISTORY_S = 60

W, H = 960, 840
BG = (18, 20, 28)
PANEL = (30, 33, 45)
TEXT = (230, 232, 240)
MUTED = (140, 145, 165)
ACTIVITY_COLORS = {
    "standing": (76, 114, 176),
    "walking": (85, 168, 104),
    "running": (196, 78, 82),
    "biking": (129, 114, 178),
}
ACTIVITIES = list(ACTIVITY_COLORS)

incoming = queue.Queue()  # filled by the MQTT thread, drained by the UI loop


# ---------------------------------------------------------------- MQTT

def on_connect(client, userdata, flags, reason_code, properties):
    incoming.put(("connected", str(reason_code)))
    client.subscribe(TOPIC)


def on_disconnect(client, userdata, flags, reason_code, properties):
    incoming.put(("disconnected", str(reason_code)))


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    if data.get("type") == "status":
        incoming.put(("status", data))


def start_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    try:
        client.connect(BROKER)  # same pattern as the other ME193 MQTT scripts
    except OSError as exc:
        incoming.put(("disconnected", f"can't reach {BROKER}: {exc}"))
        client.connect_async(BROKER)  # keep retrying in the background; the window still opens
    client.loop_start()
    return client


def send_model(client, model):
    client.publish(TOPIC, json.dumps({"type": "set_model", "model": model}))


# ---------------------------------------------------------------- drawing helpers

class Fonts:
    def __init__(self):
        self.huge = pygame.font.SysFont("helveticaneue,arial", 84, bold=True)
        self.big = pygame.font.SysFont("helveticaneue,arial", 32, bold=True)
        self.mid = pygame.font.SysFont("helveticaneue,arial", 22)
        self.small = pygame.font.SysFont("helveticaneue,arial", 16)


def text(surf, font, s, color, pos, anchor="topleft"):
    img = font.render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def panel(surf, rect, title=None, fonts=None):
    pygame.draw.rect(surf, PANEL, rect, border_radius=12)
    if title:
        text(surf, fonts.small, title.upper(), MUTED, (rect.x + 16, rect.y + 12))


class Button:
    def __init__(self, rect, label, model):
        self.rect, self.label, self.model = pygame.Rect(rect), label, model

    def draw(self, surf, fonts, active, pending, hover):
        if active:
            fill, border, color = (70, 120, 220), (120, 165, 255), TEXT
        else:
            fill, border, color = ((44, 48, 64) if hover else (36, 39, 52)), (70, 75, 95), MUTED
        pygame.draw.rect(surf, fill, self.rect, border_radius=10)
        pygame.draw.rect(surf, (255, 200, 80) if pending else border, self.rect, 2, border_radius=10)
        text(surf, fonts.big, self.label, color, self.rect.center, "center")


# ---------------------------------------------------------------- app state

class State:
    def __init__(self):
        self.connected = False
        self.conn_msg = f"connecting to {BROKER}..."
        self.status = None  # last status message
        self.last_status_t = 0.0
        self.pending_model = None  # model we asked for, until the board confirms it
        self.history = []  # (receive time, shown activity)

    def handle(self, kind, payload):
        if kind == "connected":
            self.connected, self.conn_msg = True, f"connected to {BROKER}"
        elif kind == "disconnected":
            self.connected, self.conn_msg = False, f"disconnected ({payload}) - retrying"
        elif kind == "status":
            now = time.time()
            self.status, self.last_status_t = payload, now
            self.history.append((now, payload["activity"]))
            self.history = [h for h in self.history if now - h[0] <= HISTORY_S]
            if self.pending_model == payload.get("model"):
                self.pending_model = None

    @property
    def live(self):
        return self.status is not None and time.time() - self.last_status_t < STALE_S


# ---------------------------------------------------------------- screens

def draw(surf, fonts, st, buttons, mouse, figure):
    surf.fill(BG)
    now = time.time()

    # Header + connection
    text(surf, fonts.big, "Gait Detector", TEXT, (28, 22))
    dot = (90, 200, 120) if st.connected and st.live else (230, 180, 60) if st.connected else (220, 80, 80)
    board = "board live" if st.live else ("waiting for board..." if st.status is None
                                         else f"board silent for {now - st.last_status_t:.0f} s")
    pygame.draw.circle(surf, dot, (W - 300, 40), 7)
    text(surf, fonts.small, st.conn_msg, MUTED, (W - 285, 24))
    text(surf, fonts.small, board, MUTED, (W - 285, 42))

    # Current activity (left)
    cur = pygame.Rect(28, 80, 560, 300)
    panel(surf, cur, "Detected activity", fonts)
    if st.status:
        act = st.status["activity"]
        color = ACTIVITY_COLORS.get(act, MUTED) if st.live else MUTED
        text(surf, fonts.huge, act.upper(), color, (cur.centerx, cur.y + 120), "center")
        text(surf, fonts.mid, f"Model {st.status['model']}   ·   cadence {st.status['cadence_hz']:.2f} Hz",
             MUTED, (cur.centerx, cur.y + 190), "center")
    else:
        text(surf, fonts.big, "waiting for the board...", MUTED, cur.center, "center")

    # Model selector (right)
    sel = pygame.Rect(612, 80, 320, 300)
    panel(surf, sel, "Model shown on the board", fonts)
    active_model = st.status["model"] if st.status else None
    for b in buttons:
        b.draw(surf, fonts, b.model == active_model, b.model == st.pending_model, b.rect.collidepoint(mouse))
    text(surf, fonts.small, "Model 1: peak shin accel (1 feature)", MUTED, (sel.x + 20, sel.bottom - 58))
    text(surf, fonts.small, "Model 2: full IMU (18 features)", MUTED, (sel.x + 20, sel.bottom - 36))
    if st.pending_model:
        text(surf, fonts.small, f"asking board for Model {st.pending_model}...", (255, 200, 80),
             (sel.centerx, sel.y + 196), "center")

    # Animated figure (full width)
    anim = pygame.Rect(28, 400, 904, 240)
    panel(surf, anim, None, fonts)
    act = st.status["activity"] if st.status else "standing"
    figure.draw(surf, anim.inflate(-24, -16), ACTIVITY_COLORS.get(act, MUTED) if st.live else MUTED)

    # Both models (bottom left)
    both = pygame.Rect(28, 660, 560, 172)
    panel(surf, both, "Both models", fonts)
    for i, m in enumerate(("m1", "m2")):
        y = both.y + 50 + i * 52
        text(surf, fonts.mid, f"Model {i + 1}", TEXT, (both.x + 20, y))
        if st.status:
            for j, kind in enumerate(("raw", "smoothed")):
                a = st.status[m][kind]
                x = both.x + 150 + j * 200
                pygame.draw.rect(surf, ACTIVITY_COLORS.get(a, MUTED), (x, y + 4, 14, 14), border_radius=3)
                text(surf, fonts.mid, a, TEXT, (x + 22, y))
                text(surf, fonts.small, kind, MUTED, (x + 22, y + 26))

    # History strip (bottom right)
    hist = pygame.Rect(612, 660, 320, 172)
    panel(surf, hist, f"Last {HISTORY_S} s", fonts)
    strip = pygame.Rect(hist.x + 16, hist.y + 46, hist.width - 32, 40)
    pygame.draw.rect(surf, (24, 27, 37), strip, border_radius=6)
    surf.set_clip(strip)
    bar_w = strip.width / HISTORY_S + 1  # one status per second; +1 px so neighbors touch
    for t, a in st.history:
        x = strip.right - (now - t) / HISTORY_S * strip.width
        pygame.draw.rect(surf, ACTIVITY_COLORS.get(a, MUTED), (x, strip.y, bar_w, strip.h))
    surf.set_clip(None)
    for i, a in enumerate(ACTIVITIES):
        x, y = hist.x + 16 + (i % 2) * 150, hist.y + 104 + (i // 2) * 28
        pygame.draw.rect(surf, ACTIVITY_COLORS[a], (x, y + 3, 12, 12), border_radius=3)
        text(surf, fonts.small, a, MUTED, (x + 20, y))


def main():
    pygame.init()
    surf = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Gait Detector")
    fonts = Fonts()
    st = State()
    buttons = [Button((632, 130, 135, 90), "Model 1", 1), Button((777, 130, 135, 90), "Model 2", 2)]
    client = start_mqtt()
    clock = pygame.time.Clock()
    figure = FigureAnimator()

    def choose(model):
        st.pending_model = model
        send_model(client, model)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.unicode in ("1", "2"):
                choose(int(ev.unicode))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in buttons:
                    if b.rect.collidepoint(ev.pos):
                        choose(b.model)
        while not incoming.empty():
            st.handle(*incoming.get())
        dt = clock.tick(60) / 1000.0
        if st.live:
            figure.update(dt, st.status["activity"], st.status.get("cadence_hz", 0.0))
        else:
            figure.update(dt, "standing", 0.0)  # board silent: idle instead of freezing mid-stride
        draw(surf, fonts, st, buttons, pygame.mouse.get_pos(), figure)
        pygame.display.flip()

    client.loop_stop()
    client.disconnect()
    pygame.quit()


if __name__ == "__main__":
    main()
