"""Virtual Ping-Pong: entry point and state machine.

State flow: MAIN_MENU -> CONNECTING -> CALIBRATING -> LEVEL_SELECT ->
COUNTDOWN -> PLAYING -> (RALLY_OVER -> COUNTDOWN -> PLAYING ...) ->
GAME_OVER once either side reaches 11, with PAUSED overlaying PLAYING. Q
quits from anywhere; from PAUSED, L abandons the match back to
LEVEL_SELECT; from GAME_OVER, Enter does the same (fresh 0-0 match once a
level's reselected). T toggles a big webcam + skeleton view for demoing
the tracking, from any state. MAIN_MENU's Begin (Enter or click) starts
the CONNECTING/CALIBRATING hardware hookup, same as it always ran, just
now gated behind an actual title screen instead of happening immediately
on launch.

Run with:
    python3 main.py
    python3 main.py --mock-imu --mock-vision --no-mqtt   (no hardware needed)
"""
import argparse
import math
import random
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from audio.sounds import Sounds
from game.hit_detection import FEEDBACK_MESSAGES, evaluate_hit
from game.opponent import Opponent
from game.opponent_animator import OpponentAnimator
from game.physics import Ball, predict_landing_x, step
from game.q_learning import QLearningOpponent
from game.state import GameState
from imu.motor_connection import MotorConnection
from imu.swing_detector import Calibration, SwingDetector, SwingEvent, calibrate as calibrate_swing, run_in_thread
from net.mqtt_publisher import MqttPublisher
from render.backgrounds import draw_background
from render.characters import draw_opponent
from render.claude_icon import draw_claude
from render.hud import (
    Countdown, FeedbackBanner, SwingFlash, draw_countdown, draw_feedback_banner, draw_game_over, draw_hud,
)
from render.matrix_background import draw_matrix_background
from render.menus import draw_main_menu, draw_pause_overlay
from render.particles import ParticleBurst, draw_particles
from render.projection import make_camera, project_point
from render.qtable_view import draw_q_table
from render.retro import apply_pixelation, get_retro_font
from render.table_view import draw_ball, draw_net, draw_paddle, draw_table
from render.widgets import Slider, draw_slider
from vision.coords import shoulder_width, to_game_space
from vision.level_selector import LevelSelector

# cv2 (and therefore vision.camera / vision.pose_tracker / vision.paddle_detector /
# vision.apriltag's detector / render.debug_overlay) bundles its own SDL2,
# which collides with pygame's bundled SDL2 at the Objective-C runtime
# level on macOS ("Class X is implemented in both ... this may cause
# spurious casting failures and mysterious crashes" - confirmed present
# in testing, though it didn't crash in that run). Importing cv2
# unconditionally would pull that risk into --mock-vision runs that have
# no actual need for it, defeating the point of a hardware-free mock mode
# - so everything that imports cv2 is deferred to inside VisionWorker's
# own methods and the debug-overlay render block, loaded only when real
# vision or the debug overlay is actually used.

STATE_MAIN_MENU = "main_menu"
STATE_CONNECTING = "connecting"
STATE_CALIBRATING = "calibrating"
STATE_LEVEL_SELECT = "level_select"
STATE_COUNTDOWN = "countdown"
STATE_PLAYING = "playing"
STATE_PAUSED = "paused"
STATE_RALLY_OVER = "rally_over"
STATE_GAME_OVER = "game_over"

SWING_SEARCH_WINDOW_MS = 500  # how far from ball arrival to look for a swing at all
# Must stay comfortably above the widest per-level timing_window_ms
# (currently beginner's 400ms) - otherwise this grace period would force
# an early "no_swing" resolution before a legitimately late-but-in-window
# swing ever got the chance to arrive, silently capping the timing
# tolerance regardless of how generous timing_window_ms is set.
HIT_GRACE_MS = 600
COUNTDOWN_SECONDS = 3.0
RALLY_OVER_SECONDS = 1.5
# The Claude level has no scoring to linger on and wants episodes as fast
# as possible for training - these replace the two above for just that
# level (see reset_rally_without_scoring and the countdown.start() calls),
# splitting a combined ~1 second pause between the two stages rather than
# the normal ~4.5s (1.5 + 3.0).
CLAUDE_RALLY_OVER_SECONDS = 0.5
CLAUDE_COUNTDOWN_SECONDS = 0.5
FEEDBACK_DURATION_SECONDS = 1.2
LEVEL_CONFIRM_DISPLAY_SECONDS = 1.5  # how long "X mode selected!" shows before play starts
SWING_FLASH_SECONDS = 0.3  # how long the paddle glows after any detected swing
SWING_POLL_WINDOW_MS = 100  # how recent a swing must be to count as "just happened" for the flash


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Ping-Pong")
    parser.add_argument("--mock-imu", action="store_true", help="spacebar triggers a swing instead of the real motor")
    parser.add_argument("--mock-vision", action="store_true", help="mouse position stands in for the paddle")
    parser.add_argument("--no-mqtt", action="store_true", help="don't publish score over MQTT")
    parser.add_argument("--recalibrate", action="store_true",
                         help="ignore any saved IMU calibration in config.yaml and run it fresh")
    return parser.parse_args()


# --- Mock IMU: spacebar stands in for a real swing ---

class MockSwingSource:
    def __init__(self):
        self._last_swing = None

    def trigger(self, now):
        self._last_swing = SwingEvent(timestamp=now, peak_intensity=1.5, direction="forehand", dominant_axis="x")

    def recent_swing(self, window_ms):
        if self._last_swing is None:
            return None
        if (time.monotonic() - self._last_swing.timestamp) * 1000 <= window_ms:
            return self._last_swing
        return None


# --- Vision worker: runs camera + pose + paddle (+ tags, when asked) on
# its own thread, so the main loop never blocks on frame processing. ---

# How often to recreate the pose/hand landmarkers, to work around a known
# mediapipe-on-macOS GPU delegate bug: its Metal texture cache is never
# recycled and eventually fills up, hard-crashing the process (SIGABRT)
# after a few minutes of continuous use - confirmed as the exact cause of
# a real crash during testing (see PoseTracker.rebuild()'s docstring).
# Running two landmarkers (pose + hand) roughly doubles the fill rate
# versus a single one, so this is set well under the "few minutes" window
# that bug reports describe for one landmarker alone.
LANDMARKER_REBUILD_INTERVAL_SECONDS = 30.0


class VisionWorker:
    def __init__(self, handedness="right"):
        from vision.apriltag import create_detector as create_tag_detector
        from vision.camera import Camera

        self.camera = Camera()
        self.pose_tracker = None
        self.hand_tracker = None
        self.paddle_detector = None
        self.tag_detector = create_tag_detector()
        self.handedness = handedness

        self._lock = threading.Lock()
        self._paddle_position = None  # game-space (x, y)
        self._paddle_velocity = (0.0, 0.0)
        self._debug_frame = None
        self._detected_tags = []
        self._detect_tags_enabled = False

        self._stop = threading.Event()
        self._thread = None

    def start(self, paddle_cfg):
        from vision.hand_tracker import HandTracker
        from vision.paddle_detector import PaddleDetector
        from vision.pose_tracker import PoseTracker

        if not self.camera.open():
            raise RuntimeError("Could not open the webcam.")
        self.camera.start()
        self.pose_tracker = PoseTracker(handedness=self.handedness)
        self.hand_tracker = HandTracker()
        self.paddle_detector = PaddleDetector(paddle_cfg)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_handedness(self, handedness):
        self.handedness = handedness
        if self.pose_tracker is not None:
            self.pose_tracker.set_handedness(handedness)

    def set_detect_tags(self, enabled):
        self._detect_tags_enabled = enabled

    def _loop(self):
        import cv2

        from vision.apriltag import detect_tags

        last_time = time.monotonic()
        last_rebuild_time = last_time
        while not self._stop.is_set():
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if now - last_rebuild_time >= LANDMARKER_REBUILD_INTERVAL_SECONDS:
                self.pose_tracker.rebuild()
                self.hand_tracker.rebuild()
                last_rebuild_time = now

            # Detect tags on the RAW (unmirrored) frame first - AprilTags
            # aren't mirror-symmetric in their bit encoding, so detecting
            # on an already-flipped frame makes every real tag
            # undecodable (same issue diagnosed and fixed in
            # tools/apriltag_test.py; this call site was missed then).
            tags = []
            if self._detect_tags_enabled:
                detected = detect_tags(frame, self.tag_detector)
                tags = [tag_id for tag_id, _ in detected]

            frame = cv2.flip(frame, 1)
            pose = self.pose_tracker.process(frame)
            self.pose_tracker.draw_debug(frame)
            hand_position = self.hand_tracker.process(frame)
            self.hand_tracker.draw_debug(frame)

            paddle_position, paddle_velocity = None, (0.0, 0.0)
            if pose is not None:
                paddle_norm, velocity_norm = self.paddle_detector.update(pose, hand_position, dt)
                game_xy = to_game_space({**pose, "wrist": paddle_norm}, "wrist")
                if game_xy is not None:
                    width = shoulder_width(pose)
                    paddle_position = game_xy
                    if width > 1e-6:
                        paddle_velocity = (velocity_norm[0] / width, velocity_norm[1] / width)

            with self._lock:
                self._paddle_position = paddle_position
                self._paddle_velocity = paddle_velocity
                self._debug_frame = frame
                self._detected_tags = tags

    def get_paddle_state(self):
        with self._lock:
            return self._paddle_position, self._paddle_velocity

    def get_debug_frame(self):
        with self._lock:
            return None if self._debug_frame is None else self._debug_frame.copy()

    def get_detected_tags(self):
        with self._lock:
            return list(self._detected_tags)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.camera.stop()
        if self.pose_tracker is not None:
            self.pose_tracker.close()
        if self.hand_tracker is not None:
            self.hand_tracker.close()


# --- Mock vision: mouse position stands in for the paddle ---

class MockVision:
    def __init__(self):
        self._last_mouse = None
        self._last_time = time.monotonic()
        self.position = (0.0, 0.0)
        self.velocity = (0.0, 0.0)

    def update(self, screen_size):
        now = time.monotonic()
        dt = max(1e-6, now - self._last_time)
        self._last_time = now

        mx, my = pygame.mouse.get_pos()
        width, height = screen_size
        gx = (mx - width / 2) / (width / 4)
        gy = (height / 2 - my) / (height / 4)

        if self._last_mouse is not None:
            self.velocity = ((gx - self._last_mouse[0]) / dt, (gy - self._last_mouse[1]) / dt)
        self._last_mouse = (gx, gy)
        self.position = (gx, gy)

    def get_paddle_state(self):
        return self.position, self.velocity

    def get_detected_tags(self):
        return []  # level select falls back to keyboard (1-4) in mock mode


# --- Ball launch helpers ---

BALL_SPEED_MULTIPLIER = 3.5  # m/s per 1.0 of a level's ball_speed; tune here to re-pace the whole game
LAUNCH_VY_SIM_DT = 1 / 60  # nominal frame time used to search for a safe vy; matches the game's target framerate
LAUNCH_VY_BUFFER = 0.3  # m/s added on top of the bare minimum that clears the net, for real-world frame jitter


def compute_launch_vy(y0, z0, vz, physics_cfg):
    """Minimum vy (plus a small safety buffer) that clears the net, given
    the ball's current height y0, z-position z0 (net is at z=0), and its
    signed horizontal velocity vz.

    A single fixed vy shared across the serve, player hits, and opponent
    returns broke down once a shot's horizontal speed diverged from the
    serve's: a faster vz reaches z=0 sooner, so less gravity has acted by
    then, meaning a *lower* vy is actually needed to clear the net - reusing
    the slow serve's high fixed vy on a fast opponent return instead gave it
    way too much hang time, so it stayed airborne (never bounced) all the
    way past the player.

    This is solved by directly bisecting against the game's own discrete
    step() rather than an analytic continuous-parabola formula: a fixed
    per-shot height margin on the continuous formula doesn't scale right
    across this game's 4x range of horizontal speeds (semi-implicit Euler's
    undershoot from the true parabola grows with airborne time, so slow
    serves need much more margin than fast returns) - bisecting against the
    real step() sidesteps that entirely and is exact for whatever dt the
    game actually runs at.
    """
    from game.physics import Ball, step

    if vz == 0:
        return 0.0

    def clears_net(vy):
        ball = Ball(x=0.0, y=y0, z=z0, vx=0.0, vy=vy, vz=vz)
        t = 0.0
        while t < 5.0:
            prev_z = ball.z
            ball, result = step(ball, LAUNCH_VY_SIM_DT, physics_cfg)
            t += LAUNCH_VY_SIM_DT
            if result.net_hit or result.bounced_side is not None:
                # Either an explicit net fault, or - just as much a failure
                # here - a bounce on the *launching* side before the ball
                # even reached the net (too low a vy causes this too, and
                # it's equally not "clearing the net").
                return False
            if (prev_z < 0 <= ball.z) or (prev_z > 0 >= ball.z):
                return True  # crossed z=0 while still airborne, no fault
        return False  # never even reached the net - vy/vz combination is broken

    lo, hi = 0.0, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if clears_net(mid):
            hi = mid
        else:
            lo = mid
    return hi + LAUNCH_VY_BUFFER


def serve_ball(physics_cfg, level_cfg):
    half_l = physics_cfg["table_length"] / 2
    speed = BALL_SPEED_MULTIPLIER * level_cfg["ball_speed"]
    y0 = 0.4
    z0 = -half_l + 0.1
    vy = compute_launch_vy(y0, z0, speed, physics_cfg)
    return Ball(x=0.0, y=y0, z=z0, vx=0.0, vy=vy, vz=speed)


def apply_player_hit(ball, quality, level_cfg, physics_cfg):
    base_speed = BALL_SPEED_MULTIPLIER * level_cfg["ball_speed"] * quality["speed_factor"]
    vy = compute_launch_vy(ball.y, ball.z, -base_speed, physics_cfg)
    # has_bounced resets: this is a new flight leg, which should get its
    # own fresh net-clearance check (see game/physics.py's step()).
    # angle_offset is already a usable vx (see compute_quality in
    # game/hit_detection.py) - no extra scaling needed here.
    return replace(ball, vx=quality["angle_offset"], vy=vy, vz=-base_speed, has_bounced=False)


def apply_opponent_return(ball, choice, physics_cfg):
    vy = compute_launch_vy(ball.y, ball.z, choice["speed"], physics_cfg)
    return replace(ball, vx=(choice["target_x"] - ball.x), vy=vy, vz=choice["speed"], has_bounced=False)


def trigger_score_celebration(ball, camera, particle_burst, feedback_banner, now):
    """Call when a rally ends in the player's favor (opponent faulted):
    bursts particles from the ball's last on-screen position and shows a
    "Score!" banner, so a point win is visually obvious instead of the
    ball just silently freezing mid-air."""
    feedback_banner.show("Score!", now)
    projected = project_point(ball.x, ball.y, ball.z, camera)
    if projected is not None:
        screen_x, screen_y, _ = projected
        particle_burst.trigger(screen_x, screen_y, now)


def publish_status(publisher, game_state):
    """Broadcasts current streak, best streak, and score together - called
    wherever any of those three actually change, so MQTT subscribers get a
    prompt update rather than waiting on the periodic heartbeat alone."""
    if publisher is not None:
        publisher.publish(game_state.current_streak, game_state.record_streak,
                           game_state.player_score, game_state.opponent_score)


def resolve_point(game_state, winner, rally_over_timer, now):
    """Awards a point to "player" or "opponent" and returns the state to
    transition to: STATE_GAME_OVER if that point just won the match,
    otherwise STATE_RALLY_OVER (with its timer started) to continue on to
    the next rally. Always explicit about RALLY_OVER_SECONDS (rather than
    relying on whatever duration the shared rally_over_timer last had) -
    otherwise a prior Claude-level rally's shorter CLAUDE_RALLY_OVER_SECONDS
    would silently stick around into a regular level's next point."""
    game_state.register_point(winner)
    if game_state.match_winner is not None:
        return STATE_GAME_OVER
    rally_over_timer.start(now, duration=RALLY_OVER_SECONDS)
    return STATE_RALLY_OVER


def reset_rally_without_scoring(rally_over_timer, now):
    """The Claude level's equivalent of resolve_point(): there's no
    scoring or match end for that level (the point is training, not
    winning), so every rally just goes straight to STATE_RALLY_OVER (and
    from there, the normal COUNTDOWN -> PLAYING flow serves a fresh ball)
    regardless of which side "won" the exchange. Uses the short
    CLAUDE_RALLY_OVER_SECONDS pause instead of the regular levels'
    RALLY_OVER_SECONDS, for faster training iterations."""
    rally_over_timer.start(now, duration=CLAUDE_RALLY_OVER_SECONDS)
    return STATE_RALLY_OVER


def main():
    args = parse_args()
    cfg = config.load()
    physics_cfg = cfg["physics"]
    render_cfg = cfg["render"]
    player_cfg = cfg["player"]
    levels_cfg = cfg["levels"]
    swing_cfg = cfg["swing_detector"]

    pygame.init()
    screen = pygame.display.set_mode((render_cfg["screen_width"], render_cfg["screen_height"]))
    pygame.display.set_caption("Virtual Ping-Pong")
    clock = pygame.time.Clock()
    # Press Start 2P (retro pixel font) renders much "wider"/denser per
    # character than a normal font at the same point size, so these sizes
    # are well below the old (None, N) defaults for a comparable footprint.
    # 14 was too small to survive pixelation legibly at this font+pixel_scale
    # combo (a "B" reliably misreads as a "D") - confirmed repeatedly across
    # several different labels, so this is sized up globally rather than
    # case-by-case.
    hud_font = get_retro_font(16)
    feedback_font = get_retro_font(20)
    title_font = get_retro_font(28)
    game_over_font = get_retro_font(48)
    camera_world = make_camera(physics_cfg, render_cfg)
    begin_button_rect = pygame.Rect(0, 0, 220, 60)
    begin_button_rect.center = (render_cfg["screen_width"] // 2, render_cfg["screen_height"] // 2 + 60)
    sounds = Sounds()

    # --- Secret Q-learning "Claude" level (unlocked via P on level select -
    # see the event loop below) ---
    claude_bot = None  # a QLearningOpponent, created once the level's picked
    claude_x = 0.0  # its paddle position - snaps straight to its chosen target, no smooth movement
    claude_cfg = levels_cfg["definitions"]["claude"]
    learning_rate_slider = Slider(rect=(20, render_cfg["screen_height"] - 70, 200, 10),
                                   min_value=0.01, max_value=1.0, value=claude_cfg["q_learning_rate"],
                                   label="LEARNING RATE")
    exploration_rate_slider = Slider(rect=(20, render_cfg["screen_height"] - 30, 200, 10),
                                      min_value=0.0, max_value=1.0, value=claude_cfg["q_exploration_rate"],
                                      label="EXPLORATION RATE")
    q_table_rect = pygame.Rect(20, 60, 180, 180)

    # --- IMU setup ---
    motor_connection = None
    swing_detector = None
    swing_source = None
    if args.mock_imu:
        swing_source = MockSwingSource()
    else:
        motor_connection = MotorConnection(card_color=cfg["motor"]["card_color"], card_serial=cfg["motor"]["card_serial"])

    # --- Vision setup ---
    vision = MockVision() if args.mock_vision else VisionWorker()

    # --- MQTT setup ---
    publisher = None if args.no_mqtt else MqttPublisher(cfg["mqtt"])
    if publisher is not None:
        publisher.start()

    state = STATE_MAIN_MENU
    status_message = "Connecting..."
    game_state = GameState()
    feedback_banner = FeedbackBanner(duration_seconds=FEEDBACK_DURATION_SECONDS)
    particle_burst = ParticleBurst()
    swing_flash = SwingFlash(duration_seconds=SWING_FLASH_SECONDS)
    last_flashed_swing_time = None
    countdown = Countdown(duration_seconds=COUNTDOWN_SECONDS)
    rally_over_timer = Countdown(duration_seconds=RALLY_OVER_SECONDS)
    level_confirm_timer = Countdown(duration_seconds=LEVEL_CONFIRM_DISPLAY_SECONDS)
    level_confirm_shown = False
    tag_selector = None
    level_name = levels_cfg["default"]
    level_cfg = levels_cfg["definitions"][level_name]
    opponent = None
    opponent_animator = OpponentAnimator()
    rng = random.Random()
    ball = None
    paused_from = None
    show_debug = False
    show_tracking_view = False  # big webcam + skeleton overlay, for demoing the tracking to someone
    awaiting_hit = False
    ball_arrival_time = None
    ball_arrival_x_gs = None
    swing_baseline_time = 0.0
    # Which side (if any) the current flight leg has already bounced on -
    # reset to None at the start of every new leg (serve/player hit/
    # opponent return). Used to tell "my shot went out without ever
    # landing" (my fault) apart from "my shot landed in on their side and
    # they just didn't return it" (their fault, a point for me) once it
    # later nets or goes out.
    last_bounce_side = None
    opponent_end_z = -physics_cfg["table_length"] / 2
    player_end_z = physics_cfg["table_length"] / 2

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.monotonic()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if state == STATE_MAIN_MENU and event.button == 1 and begin_button_rect.collidepoint(event.pos):
                    state = STATE_CONNECTING
                elif level_name == "claude" and event.button == 1:
                    learning_rate_slider.handle_mouse_down(event.pos)
                    exploration_rate_slider.handle_mouse_down(event.pos)
                    if claude_bot is not None:
                        claude_bot.learning_rate = learning_rate_slider.value
                        claude_bot.exploration_rate = exploration_rate_slider.value
            elif event.type == pygame.MOUSEBUTTONUP:
                if level_name == "claude" and event.button == 1:
                    learning_rate_slider.handle_mouse_up()
                    exploration_rate_slider.handle_mouse_up()
            elif event.type == pygame.MOUSEMOTION:
                if level_name == "claude":
                    learning_rate_slider.handle_mouse_motion(event.pos)
                    exploration_rate_slider.handle_mouse_motion(event.pos)
                    if claude_bot is not None:
                        claude_bot.learning_rate = learning_rate_slider.value
                        claude_bot.exploration_rate = exploration_rate_slider.value
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif state == STATE_MAIN_MENU and event.key == pygame.K_RETURN:
                    state = STATE_CONNECTING
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_t:
                    show_tracking_view = not show_tracking_view
                elif event.key == pygame.K_m:
                    sounds.toggle_mute()
                elif event.key == pygame.K_SPACE and args.mock_imu:
                    swing_source.trigger(now)
                elif state == STATE_LEVEL_SELECT and event.key == pygame.K_p and tag_selector is not None:
                    # Placeholder unlock for the secret Q-learning "Claude"
                    # level - not tied to any tag, just this key for now
                    # until a real unlock condition is decided.
                    tag_selector.select_by_key("claude")
                elif event.key in (pygame.K_ESCAPE, pygame.K_p):
                    if state == STATE_PAUSED:
                        state = paused_from
                    elif state not in (STATE_MAIN_MENU, STATE_CONNECTING, STATE_CALIBRATING, STATE_GAME_OVER):
                        paused_from = state
                        state = STATE_PAUSED
                elif state == STATE_PAUSED and event.key == pygame.K_l:
                    # Abandon the current match and return to level select,
                    # rather than being stuck playing out whatever level was
                    # chosen earlier in the session.
                    if tag_selector is not None:
                        tag_selector.start(now)
                    if isinstance(vision, VisionWorker):
                        vision.set_detect_tags(True)
                    level_confirm_shown = False
                    state = STATE_LEVEL_SELECT
                elif state == STATE_GAME_OVER and event.key == pygame.K_RETURN:
                    if tag_selector is not None:
                        tag_selector.start(now)
                    if isinstance(vision, VisionWorker):
                        vision.set_detect_tags(True)
                    level_confirm_shown = False
                    state = STATE_LEVEL_SELECT
                elif state == STATE_LEVEL_SELECT and event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    keyed_level = {pygame.K_1: "beginner", pygame.K_2: "intermediate",
                                   pygame.K_3: "advanced", pygame.K_4: "expert"}[event.key]
                    tag_selector.select_by_key(keyed_level)

        if not running:
            break

        # --- State machine ---
        if state == STATE_MAIN_MENU:
            pass  # waits for Enter or a click on begin_button_rect (handled in the event loop above)

        elif state == STATE_CONNECTING:
            try:
                if motor_connection is not None:
                    status_message = "Connecting to Double Motor..."
                    motor_connection.connect()
                    motor_connection.start_watchdog()
                if isinstance(vision, VisionWorker):
                    status_message = "Opening camera..."
                    vision.start(cfg["paddle"])
                state = STATE_CALIBRATING
            except Exception as exc:
                status_message = f"Connection failed: {exc}"

        elif state == STATE_CALIBRATING:
            existing = None if args.recalibrate else swing_cfg.get("calibration")
            if args.mock_imu:
                state = STATE_LEVEL_SELECT
            elif existing is not None:
                calibration = Calibration(**existing)
                swing_detector = SwingDetector(
                    smoothing_alpha=swing_cfg["smoothing_alpha"],
                    gravity_ema_alpha=swing_cfg["gravity_ema_alpha"],
                    refractory_ms=swing_cfg["refractory_ms"],
                    threshold_fraction=swing_cfg["threshold_fraction"],
                    calibration=calibration,
                )
                _, stop_event = run_in_thread(motor_connection, swing_detector, poll_interval_ms=swing_cfg["poll_interval_ms"])
                state = STATE_LEVEL_SELECT
            else:
                status_message = "Calibrating - hold the paddle still, then swing when prompted..."

                def read_sample():
                    from imu.swing_detector import IMUSample
                    imu = motor_connection.motor.imu_device
                    return IMUSample(time.monotonic(), (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ),
                                      (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ))

                calibration = calibrate_swing(read_sample, hold_seconds=swing_cfg["calibration_hold_seconds"],
                                               num_swings=swing_cfg["calibration_num_swings"], prompt=print,
                                               poll_interval_ms=swing_cfg["poll_interval_ms"])
                swing_cfg["calibration"] = {
                    "gyro_noise_floor": calibration.gyro_noise_floor, "gyro_avg_peak": calibration.gyro_avg_peak,
                    "accel_noise_floor": calibration.accel_noise_floor, "accel_avg_peak": calibration.accel_avg_peak,
                }
                config.save(cfg)
                swing_detector = SwingDetector(
                    smoothing_alpha=swing_cfg["smoothing_alpha"], gravity_ema_alpha=swing_cfg["gravity_ema_alpha"],
                    refractory_ms=swing_cfg["refractory_ms"], threshold_fraction=swing_cfg["threshold_fraction"],
                    calibration=calibration,
                )
                run_in_thread(motor_connection, swing_detector, poll_interval_ms=swing_cfg["poll_interval_ms"])
                state = STATE_LEVEL_SELECT

        elif state == STATE_LEVEL_SELECT:
            if tag_selector is None:
                tag_selector = LevelSelector(
                    tag_to_level={int(k): v for k, v in levels_cfg["tags"].items()},
                    confirm_hold_seconds=levels_cfg["confirm_hold_seconds"],
                )
                tag_selector.start(now)
                level_confirm_shown = False
                if isinstance(vision, VisionWorker):
                    vision.set_detect_tags(True)
            tag_selector.update(vision.get_detected_tags(), now)

            if tag_selector.confirmed_level is not None:
                if not level_confirm_shown:
                    feedback_banner.show(f"{tag_selector.confirmed_level.capitalize()} mode selected!", now)
                    level_confirm_timer.start(now)
                    level_confirm_shown = True

                if level_confirm_timer.is_done(now):
                    level_name = tag_selector.confirmed_level
                    level_cfg = levels_cfg["definitions"][level_name]
                    if level_name == "claude":
                        opponent = None
                        # Created once and then reused across repeated
                        # visits to this level within the same run (not
                        # saved to disk - see game/q_learning.py) - picking
                        # beginner then coming back to Claude later
                        # shouldn't wipe out training progress.
                        if claude_bot is None:
                            claude_bot = QLearningOpponent(
                                num_ball_buckets=level_cfg["q_num_ball_buckets"],
                                num_paddle_buckets=level_cfg["q_num_paddle_buckets"],
                                half_width=physics_cfg["table_width"] / 2,
                                learning_rate=learning_rate_slider.value,
                                exploration_rate=exploration_rate_slider.value,
                            )
                        claude_x = 0.0
                    else:
                        claude_bot = None
                        opponent = Opponent({
                            "max_speed": level_cfg["opponent_max_speed"],
                            "reaction_delay": level_cfg["opponent_reaction_delay"],
                            "miss_probability": level_cfg["opponent_miss_probability"],
                            "return_speed_range": level_cfg["opponent_return_speed_range"],
                            "return_target_spread": level_cfg["opponent_return_target_spread"],
                        })
                    if isinstance(vision, VisionWorker):
                        vision.set_detect_tags(False)
                    game_state = GameState()
                    countdown.start(now, duration=CLAUDE_COUNTDOWN_SECONDS if level_name == "claude"
                                     else COUNTDOWN_SECONDS)
                    state = STATE_COUNTDOWN

        elif state == STATE_COUNTDOWN:
            if countdown.is_done(now):
                game_state.start_rally()
                ball = serve_ball(physics_cfg, level_cfg)
                if opponent is not None:
                    opponent.x = 0.0
                claude_x = 0.0
                awaiting_hit = False
                last_bounce_side = None
                state = STATE_PLAYING

        elif state == STATE_PLAYING:
            # Captured before step(): on a net hit, step() zeroes the
            # returned ball's vz to 0 (caught by the net) - checking the
            # post-step ball.vz for fault direction below would then always
            # read as "not < 0" and misattribute every net fault (even the
            # player's own) as an opponent fault.
            incoming_vz = ball.vz
            ball, result = step(ball, dt, physics_cfg)

            if result.bounced_side is not None:
                sounds.play(sounds.bounce)
                game_state.note_bounce(result.bounced_side)
                last_bounce_side = result.bounced_side
                if result.bounced_side == "player":
                    awaiting_hit = True
                    ball_arrival_time = now
                    ball_arrival_x_gs = ball.x / player_cfg["reach_scale"]
                    swing_baseline_time = now

            if awaiting_hit:
                # The ball already legitimately bounced on the player's
                # side - this is their chance to hit it, so anything that
                # happens to it next (including flying into the net or out
                # of bounds before they swing) is on them, not the
                # opponent. This must be checked before the net_hit/
                # out_of_bounds block below: that block's direction-based
                # attribution doesn't apply here, since incoming_vz still
                # reflects the opponent's original (perfectly legal) shot,
                # not whatever happens afterward because the player hasn't
                # reacted yet - checking it here first was incorrectly
                # scoring some "player ran out of time" misses as points.
                swing_event = None
                source = swing_source if args.mock_imu else swing_detector
                if source is not None:
                    candidate = source.recent_swing(SWING_SEARCH_WINDOW_MS)
                    if candidate is not None and candidate.timestamp >= swing_baseline_time:
                        swing_event = candidate

                paddle_position, _ = vision.get_paddle_state()
                # The ball getting away entirely (net/out of bounds) forces
                # immediate resolution, same as the grace period expiring.
                grace_expired = (result.net_hit or result.out_of_bounds
                                 or (now - ball_arrival_time) * 1000 > HIT_GRACE_MS)

                if swing_event is not None or grace_expired:
                    if paddle_position is None:
                        result_hit = None  # no pose/paddle detected at all - treat as a miss below
                    else:
                        result_hit = evaluate_hit(
                            paddle_position, (ball_arrival_x_gs, 0.0), ball_arrival_time,
                            swing_event, hit_radius=level_cfg["hit_radius"],
                            timing_window_ms=level_cfg["timing_window_ms"],
                            swing_search_window_ms=SWING_SEARCH_WINDOW_MS,
                        )

                    awaiting_hit = False
                    if paddle_position is not None and result_hit.outcome == "hit":
                        sounds.play(sounds.hit)
                        game_state.register_player_hit()
                        ball = apply_player_hit(ball, result_hit.quality, level_cfg, physics_cfg)
                        last_bounce_side = None  # fresh leg, needs its own bounce before landing counts
                        if opponent is not None:
                            opponent.notice_ball_incoming()
                        if motor_connection is not None:
                            motor_connection.haptic_pulse()
                        publish_status(publisher, game_state)
                    else:
                        reason = "no_swing" if paddle_position is None else result_hit.reason
                        feedback_banner.show(FEEDBACK_MESSAGES[reason], now)
                        sounds.play(sounds.miss)
                        game_state.register_player_miss()
                        opponent_animator.trigger_score(now)
                        if level_name == "claude":
                            state = reset_rally_without_scoring(rally_over_timer, now)
                        else:
                            state = resolve_point(game_state, "opponent", rally_over_timer, now)
                            publish_status(publisher, game_state)

            elif result.net_hit or result.out_of_bounds:
                # The ball hasn't bounced on the player's side yet (that
                # case is handled above), so this is either: the player's
                # own shot going wide/into the net before ever landing on
                # the opponent's side (their fault), or a shot that DID
                # already land on the opponent's side this leg and is now
                # just continuing on past the table/into the net because
                # the opponent never returned it (same real-ping-pong rule
                # as the player's own side above: once it's legitimately
                # in, whatever happens next is on whoever failed to return
                # it, not whoever hit it) - last_bounce_side distinguishes
                # these two cases. incoming_vz's sign still separately
                # covers the opponent's own serve/return faulting before
                # ever reaching either side at all. Uses incoming_vz
                # (captured before step(), above), not ball.vz, since a net
                # hit specifically zeroes the ball's vz - checking post-step
                # would make every net fault look like incoming_vz >= 0
                # regardless of which way it was actually headed.
                if last_bounce_side == "opponent" or incoming_vz >= 0:
                    trigger_score_celebration(ball, camera_world, particle_burst, feedback_banner, now)
                    sounds.play(sounds.score)
                    opponent_animator.trigger_lose(now)
                    if level_name == "claude":
                        state = reset_rally_without_scoring(rally_over_timer, now)
                    else:
                        state = resolve_point(game_state, "player", rally_over_timer, now)
                        publish_status(publisher, game_state)
                else:
                    feedback_banner.show("Miss!", now)
                    sounds.play(sounds.miss)
                    game_state.register_player_miss()
                    opponent_animator.trigger_score(now)
                    if level_name == "claude":
                        state = reset_rally_without_scoring(rally_over_timer, now)
                    else:
                        state = resolve_point(game_state, "opponent", rally_over_timer, now)
                        publish_status(publisher, game_state)

            if state == STATE_PLAYING and ball.z < 0:
                half_width = physics_cfg["table_width"] / 2
                near_back = ball.z <= opponent_end_z + 0.1
                # A sharply-angled shot can exit past the side of the table
                # well before it ever reaches the back wall - without this,
                # the catch-check below (gated only on near_back) would
                # never fire for those, so the opponent never got a chance
                # to return anything but straight-down-the-back shots, and
                # the ball would just go out of bounds untouched.
                near_side = abs(ball.x) >= half_width - 0.1
                # Also requires last_bounce_side == "opponent": a shot that
                # reaches either zone without ever having landed on the
                # opponent's side first (e.g. a smash hit long, or one that
                # goes straight off the side) never had a legal landing to
                # return in the first place - real table tennis doesn't let
                # you volley it, and it should be an automatic miss for
                # whoever hit it (handled by the net_hit/out_of_bounds
                # fault-attribution above) rather than something the
                # opponent can "catch".
                catchable = (near_back or near_side) and ball.vz < 0 and last_bounce_side == "opponent"

                if level_name == "claude":
                    # No reaction-delay/max_speed tracking here - Claude's
                    # "movement" is just snapping straight to wherever its
                    # policy says, the instant the ball becomes catchable
                    # (fits the digital-presence visual, and keeps the
                    # state/action/reward all lined up to the same instant).
                    if catchable:
                        predicted_x = predict_landing_x(ball, target_z=opponent_end_z)
                        if predicted_x is None:
                            predicted_x = ball.x
                        q_state, q_action, target_x = claude_bot.choose_action(predicted_x, rng)
                        claude_x = target_x
                        if abs(target_x - ball.x) <= level_cfg["q_hit_radius"]:
                            claude_bot.update(q_state, q_action, reward=1.0)
                            opponent_animator.trigger_hit(now)
                            choice = {
                                "speed": rng.uniform(*level_cfg["opponent_return_speed_range"]),
                                "target_x": rng.uniform(-level_cfg["opponent_return_target_spread"],
                                                         level_cfg["opponent_return_target_spread"]),
                            }
                            ball = apply_opponent_return(ball, choice, physics_cfg)
                            last_bounce_side = None  # fresh leg, needs its own bounce before landing counts
                        else:
                            claude_bot.update(q_state, q_action, reward=-1.0)
                            opponent_animator.trigger_lose(now)
                            state = reset_rally_without_scoring(rally_over_timer, now)
                else:
                    opponent.update(ball, dt, opponent_end_z)
                    if catchable:
                        # The opponent only has a chance to return the ball
                        # if it actually got close enough (its on-screen
                        # movement is now load-bearing, not cosmetic, and
                        # it's free to move past the table's side edge to
                        # reach a wide ball the same way it reaches a deep
                        # one). This used to decide in_position once, on
                        # the very first frame the ball entered this zone,
                        # and lock in an immediate miss if it wasn't - but
                        # the opponent is still converging on the ball
                        # every frame via Opponent.update(), so a
                        # near-miss on that first frame could still have
                        # closed the gap moments later. Now it only checks
                        # "not in position yet, so do nothing" - the
                        # opponent keeps trying every subsequent frame
                        # while the ball's still catchable. If it's
                        # genuinely never reached in time, the ball just
                        # goes out of bounds / into the net on its own,
                        # which the net_hit/out_of_bounds block above
                        # already scores as a point for the player (since
                        # last_bounce_side == "opponent" by then) - so
                        # there's no need to force that outcome here.
                        in_position = abs(opponent.x - ball.x) <= level_cfg["opponent_hit_radius"]
                        if in_position:
                            # Decided once, right here: miss_probability is
                            # a separate "skill" factor on top of actually
                            # reaching the ball in time.
                            outcome = opponent.decide_outcome(rng)
                            if outcome == "miss":
                                # The AI reached it but still whiffed - the
                                # rally ends in the player's favor, but
                                # current_streak counts player hits
                                # specifically (see game/state.py), so it's
                                # left untouched here: not incremented (the
                                # player didn't just hit the ball) and not
                                # reset (the player didn't miss either). It
                                # carries over into the next rally unchanged.
                                trigger_score_celebration(ball, camera_world, particle_burst, feedback_banner, now)
                                sounds.play(sounds.score)
                                opponent_animator.trigger_lose(now)
                                state = resolve_point(game_state, "player", rally_over_timer, now)
                                publish_status(publisher, game_state)
                            else:
                                choice = opponent.choose_return(rng)
                                ball = apply_opponent_return(ball, choice, physics_cfg)
                                last_bounce_side = None  # fresh leg, needs its own bounce before landing counts
                                opponent_animator.trigger_hit(now)

            if publisher is not None:
                publisher.maybe_heartbeat(game_state.current_streak, game_state.record_streak,
                                          game_state.player_score, game_state.opponent_score, now)

        elif state == STATE_RALLY_OVER:
            if rally_over_timer.is_done(now):
                game_state.ready_for_next_rally()
                countdown.start(now, duration=CLAUDE_COUNTDOWN_SECONDS if level_name == "claude"
                                 else COUNTDOWN_SECONDS)
                state = STATE_COUNTDOWN

        elif state == STATE_GAME_OVER:
            pass  # waits for the player to press Enter (handled in the event loop above)

        elif state == STATE_PAUSED:
            pass

        # --- Mock-vision needs an explicit per-frame update (no own thread) ---
        if isinstance(vision, MockVision):
            vision.update(screen.get_size())

        # Background music: the calmer menu theme for anything that isn't
        # mid-rally (including PAUSED, so pausing mid-match also drops back
        # to it), the more driving action theme once a point is actually
        # live. play_music() no-ops if this is already the track playing,
        # so calling it every frame doesn't restart/stutter the loop.
        if state in (STATE_COUNTDOWN, STATE_PLAYING, STATE_RALLY_OVER):
            sounds.play_music("action")
        else:
            sounds.play_music("menu")

        # Swing flash: fires the instant any swing is detected, independent
        # of hit-evaluation/game phase, so the player gets immediate visual
        # confirmation "yes, that registered" even if it's not currently
        # being judged against a ball (e.g. swinging early, or just testing).
        active_swing_source = swing_source if args.mock_imu else swing_detector
        if active_swing_source is not None:
            candidate = active_swing_source.recent_swing(SWING_POLL_WINDOW_MS)
            if candidate is not None and candidate.timestamp != last_flashed_swing_time:
                swing_flash.trigger(now)
                last_flashed_swing_time = candidate.timestamp

        # --- Render ---
        screen.fill((30, 30, 40))

        if state == STATE_MAIN_MENU:
            draw_main_menu(screen, title_font, feedback_font, begin_button_rect)
        elif state in (STATE_CONNECTING, STATE_CALIBRATING):
            text = title_font.render(status_message, True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)))
        else:
            if level_name == "claude":
                draw_matrix_background(screen, now, hud_font)
            else:
                draw_background(screen, level_name)
            draw_table(screen, camera_world, physics_cfg)
            draw_net(screen, camera_world, physics_cfg)
            if opponent is not None:
                draw_opponent(screen, camera_world, opponent.x, physics_cfg, level_name, opponent_animator, now,
                              feedback_font, hud_font)
            elif level_name == "claude" and claude_bot is not None:
                draw_claude(screen, camera_world, claude_x, physics_cfg, opponent_animator, now,
                            feedback_font, hud_font)
            paddle_position, _ = vision.get_paddle_state()
            if paddle_position is not None:
                table_x = paddle_position[0] * player_cfg["reach_scale"]
                table_y = player_cfg["base_height"] + paddle_position[1] * player_cfg["reach_scale"]
                draw_paddle(screen, camera_world, table_x, table_y, player_end_z - 0.1,
                            swing_intensity=swing_flash.intensity(now))
            if ball is not None and state in (STATE_PLAYING, STATE_RALLY_OVER):
                draw_ball(screen, camera_world, ball)
            draw_particles(screen, particle_burst, now)

            if state == STATE_LEVEL_SELECT and tag_selector is not None and tag_selector.confirmed_level is None:
                candidate = tag_selector.current_candidate()
                if candidate is not None:
                    remaining = tag_selector.confirm_hold_seconds * (1.0 - tag_selector.progress(now))
                    prompt_text = f"You are selecting {candidate.capitalize()}... keep holding! ({remaining:.1f}s)"
                    prompt_color = (0, 255, 0)
                else:
                    prompt_text = "Hold up an AprilTag, or press 1-4 to choose a level"
                    prompt_color = (255, 255, 0)
                prompt = hud_font.render(prompt_text, True, prompt_color)
                screen.blit(prompt, (10, screen.get_height() - 30))
                progress = tag_selector.progress(now)
                if progress > 0:
                    pygame.draw.rect(screen, (0, 255, 0), (10, screen.get_height() - 50, int(200 * progress), 10))
            # Once confirmed, feedback_banner's "X mode selected!" message
            # (shown in the state-handling block above) carries the UI for
            # the brief pause before play begins.

            if level_name == "claude" and claude_bot is not None:
                # No score/streak HUD here - this level doesn't keep one
                # (see reset_rally_without_scoring) - training stats and
                # the live Q-table/sliders take its place instead.
                win_rate = claude_bot.hits / claude_bot.episodes if claude_bot.episodes > 0 else 0.0
                stats_text = (f"EPISODES: {claude_bot.episodes}  HITS: {claude_bot.hits}  "
                              f"MISSES: {claude_bot.misses}  WIN RATE: {win_rate:.0%}")
                stats_surf = hud_font.render(stats_text, True, (220, 220, 220))
                screen.blit(stats_surf, (screen.get_width() - stats_surf.get_width() - 10, 10))
                draw_q_table(screen, q_table_rect, claude_bot.q_table, hud_font)
                draw_slider(screen, hud_font, learning_rate_slider)
                draw_slider(screen, hud_font, exploration_rate_slider)
            else:
                draw_hud(screen, hud_font, game_state.current_streak, game_state.record_streak,
                         game_state.player_score, game_state.opponent_score, level_name,
                         mqtt_connected=(publisher.connected if publisher is not None else True))
            draw_feedback_banner(screen, feedback_font, feedback_banner, now)
            if state == STATE_COUNTDOWN:
                draw_countdown(screen, feedback_font, countdown, now)
            if state == STATE_PAUSED:
                draw_pause_overlay(screen, title_font, hud_font)
            if state == STATE_GAME_OVER:
                draw_game_over(screen, game_over_font, feedback_font, game_state.match_winner,
                               game_state.player_score, game_state.opponent_score)

        # Pixelation applies to everything above (main menu, connecting/
        # calibrating status text, and the table/HUD/menus), regardless of
        # which branch ran - but not the debug/tracking webcam overlays
        # below, since those show real camera footage (for debugging
        # detection or demoing the tracking to someone) and chunking that
        # up along with everything else would defeat the point of showing
        # it clearly.
        screen.blit(apply_pixelation(screen, render_cfg["pixel_scale"]), (0, 0))

        if show_debug and not args.mock_vision:
            # Deferred import - see the note at the top of this file on
            # why cv2 (pulled in by render.debug_overlay) is never
            # loaded for a pure --mock-vision run.
            from render.debug_overlay import draw_swing_intensity_bar, draw_webcam_pip

            debug_frame = vision.get_debug_frame() if isinstance(vision, VisionWorker) else None
            if debug_frame is not None:
                pip_rect = pygame.Rect(screen.get_width() - render_cfg["pip_width"] - 10, 10,
                                        render_cfg["pip_width"], render_cfg["pip_height"])
                draw_webcam_pip(screen, debug_frame, pip_rect)
            intensity = 0.0
            source = swing_source if args.mock_imu else swing_detector
            if source is not None:
                recent = source.recent_swing(300)
                if recent is not None:
                    intensity = recent.peak_intensity
            bar_rect = pygame.Rect(10, screen.get_height() - 70, 200, 20)
            draw_swing_intensity_bar(screen, bar_rect, intensity)

        if show_tracking_view and not args.mock_vision:
            # A big, clean webcam + skeleton view for demoing the
            # tracking to someone - separate from the small developer
            # debug PIP above (which also crowds in the swing bar and
            # other debug info), toggled independently with 'T'.
            from render.debug_overlay import draw_webcam_pip

            demo_frame = vision.get_debug_frame() if isinstance(vision, VisionWorker) else None
            if demo_frame is not None:
                demo_w = render_cfg["tracking_view_width"]
                demo_h = render_cfg["tracking_view_height"]
                demo_rect = pygame.Rect((screen.get_width() - demo_w) // 2,
                                         (screen.get_height() - demo_h) // 2, demo_w, demo_h)
                draw_webcam_pip(screen, demo_frame, demo_rect)
                label = hud_font.render("Tracking view (T to hide)", True, (255, 255, 255))
                screen.blit(label, label.get_rect(midtop=(screen.get_width() // 2, demo_rect.bottom + 6)))

        pygame.display.flip()

    if motor_connection is not None:
        motor_connection.disconnect()
    if isinstance(vision, VisionWorker):
        vision.stop()
    if publisher is not None:
        publisher.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
