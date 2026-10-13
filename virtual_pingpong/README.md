# Virtual Ping-Pong

Play virtual ping-pong against a computer opponent using a real paddle: a
webcam tracks your hand/arm, while a LEGO Education Double Motor (45651)
taped to the paddle detects your swing via its IMU and buzzes back when
you land a hit. The whole game runs in a retro 8-bit style - a global
pixelation filter, a chiptune soundtrack (including a chiptune
arrangement of Bach's Prelude No. 1 in C major for the menus), and four
animated, named opponents (one per difficulty) in their own themed venue,
from a basement rec room up to a packed stadium. A hidden 5th level lets
you train a Q-learning agent ("Claude") live, with its learned table and
hyperparameters visible and adjustable on screen. Your current/best
streak and score are published live over MQTT.

## Installation

```bash
cd virtual_pingpong
python3 -m venv .venv   # or reuse an existing project venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.11. Everything (BLE motor connection, MediaPipe pose
tracking, OpenCV, pygame rendering, MQTT) runs in a single process/Python
version - no separate processes or Python version juggling needed.

The first run also downloads two small one-time assets automatically (and
caches them locally - no need to re-download on later runs): MediaPipe's
pose/hand landmark models (~12MB total) and the "Press Start 2P" pixel
font (~115KB, SIL Open Font License) used for all in-game text.

### Bluetooth pairing & permissions

- **macOS**: the first run will prompt for Bluetooth and Camera access for
  whichever app hosts your terminal (Terminal, iTerm, VS Code). Grant both
  in System Settings -> Privacy & Security, then fully quit and reopen
  that host app - permission is granted to the app, not to Python, so a
  restart is required.
- **Windows/Linux**: ensure Bluetooth is enabled and your user has
  permission to access BLE devices (on Linux this sometimes requires
  running with elevated permissions or adding your user to a `bluetooth`
  group, depending on distro).
- Power on the Double Motor before running. If connection fails with a
  firmware/RPC version mismatch, update either this project's
  `legoeducation` package (`pip install --upgrade legoeducation`) or the
  motor's firmware via LEGO Education's Coding Canvas, so the two match.

### Attaching the motor to the paddle

Tape or strap the Double Motor hub flat against the back of the paddle
handle/shaft, oriented consistently (e.g. the hub's "front" pointing
toward the paddle head), so swings produce a clean, repeatable motion
signature. The exact orientation doesn't matter for detection (calibration
adapts to whatever you do), but a loose or rotating mount will add noise.
The same motor also drives the haptic feedback - a brief back-and-forth
jiggle - every time you successfully return the ball.

## Printing the AprilTags (level select)

```bash
python3 tools/apriltag_generator.py
```

Generates PNGs and a combined `tools/generated_tags/all_tags.pdf` for the
4 regular levels (beginner/intermediate/advanced/expert, tag IDs 0-3).
Print the PDF and cut out the tags, or just display the PNGs on a
phone/tablet screen with a plain border around them. There's no tag for
the secret 5th level - see [The secret Claude level](#the-secret-claude-level-q-learning-training)
below.

## Calibration

The first time you run the real game (no `--mock-imu`), it will:
1. Ask you to hold the paddle still for a few seconds (measures the
   resting noise floor).
2. Ask you to do a few practice swings (measures a typical swing's peak).

This calibration is unit-agnostic by design (see `imu/swing_detector.py`'s
module docstring for why) and is saved into `config.yaml` under
`swing_detector.calibration`, so **it's only needed once** - subsequent
runs reuse the saved calibration automatically. Delete that section from
`config.yaml` (or just the `calibration:` value, set to `null`) to force
recalibration, e.g. after remounting the motor differently.

## Controls

| Key / input | Action |
|---|---|
| `Enter` | begin (main menu) / return to level select (game-over screen) |
| mouse click on BEGIN | same as `Enter` on the main menu |
| `1`-`4` | pick a level directly (beginner/intermediate/advanced/expert) - works anytime during level select, as a fallback if you don't have a printed tag handy |
| `P` (on level select) | unlock the secret Claude (Q-learning) level - see below |
| `D` | toggle the small developer debug overlay (webcam picture-in-picture + swing intensity bar) |
| `T` | toggle a large, clean webcam + skeleton tracking view (for demoing how the tracking works to someone) |
| `M` | mute/unmute all audio (music + sound effects) |
| `Esc` / `P` (elsewhere) | pause / resume |
| `L` (while paused) | abandon the current match and return to level select |
| `Q` | quit |
| `Space` | (only with `--mock-imu`) trigger a swing |
| mouse | (only with `--mock-vision`) stands in for the paddle position |
| mouse drag on the sliders | (Claude level only) adjust learning rate / exploration rate live |

`P` is intentionally overloaded: on the level-select screen it's the
secret-level unlock; everywhere else (including level select once that
stops being relevant) it's the pause toggle, same as `Esc`.

## Running

```bash
python3 main.py                              # full game, real hardware
python3 main.py --mock-imu                   # spacebar instead of the real motor
python3 main.py --mock-vision                # mouse instead of the real camera/paddle
python3 main.py --mock-imu --mock-vision --no-mqtt   # no hardware needed at all
```

The mock flags are meant to be mixed and matched - e.g. test real swing
detection with mock vision, or vice versa, before combining both. With
`--mock-vision`, OpenCV (`cv2`) is never imported at all, which also
avoids a real native-library conflict: `cv2` and `pygame` each bundle
their own copy of SDL2, which collide at the macOS Objective-C runtime
level when both are loaded in the same process (confirmed during
development - see the comment at the top of `main.py`). This hasn't been
observed to actually crash in testing, but `--mock-vision` sidesteps the
risk entirely for hardware-free demos/development.

**Note (macOS, real `--mock-vision`-free runs only):** MediaPipe's GPU
delegate has a known upstream bug where its Metal texture cache is never
recycled and eventually fills up, hard-crashing the process after several
minutes of continuous tracking. This game works around it automatically
(`VisionWorker` periodically rebuilds the pose/hand landmarkers before
that happens - see `LANDMARKER_REBUILD_INTERVAL_SECONDS` in `main.py`), so
you shouldn't need to do anything, but it's worth knowing about if you
ever see a `SIGABRT`/"GetStorageForViewOrDie" crash after a long session.

## Match play

A normal (non-secret) match is first to 11 points - whoever reaches 11
first gets a win/lose screen. Press `Enter` there (or `L` from pause
mid-match) to go back to level select for a fresh 0-0 match.

## The secret Claude level (Q-learning training)

Press `P` on the level-select screen to unlock a 5th, hidden level: your
opponent is "Claude", a tabular Q-learning agent you train live by
playing against it, set in a Matrix-style digital-rain scene.

- **No scoring, no win condition** - every rally (hit or miss, either
  side) just resets straight to the next serve as fast as possible
  (~1 second, vs. the normal levels' ~4.5 second pause), since the point
  is maximizing training episodes, not winning.
- **State** = which of 9 buckets the ball's predicted landing x falls
  into; **action** = which of 9 buckets to move the paddle to; **reward**
  = +1 if that position ended up close enough to the ball's actual
  arrival, else -1. See `game/q_learning.py` for the full logic.
- The **live Q-table** (top-left) is a 9x9 heatmap - green cells are
  learned-good actions, red is learned-bad, intensity scaled to how
  confident the learned value is.
- Two **sliders** (bottom-left, draggable with the mouse at any time,
  including mid-pause) control the learning rate and exploration rate
  live. Defaults (0.5 / 0.3) are tuned for fast visible convergence on
  this small table - see the chat history (or just ask) for the fuller
  reasoning if you want to tune further. Once it looks reasonably
  trained, lowering exploration toward ~0.05-0.1 makes it rely on what
  it's learned instead of still trying random spots fairly often.
- **Persistence is in-memory only** (not saved to disk) - the Q-table
  survives switching to another level and back within the same run, but
  resets if you quit and relaunch.
- Claude's icon is an original starburst mark, not a reproduction of
  Anthropic's actual (trademarked) logo.

## Standalone subsystem tools (`tools/`)

Each hardware-touching subsystem has its own isolated test/calibration
tool, useful for debugging one piece without running the whole game:

| Tool | Purpose |
|---|---|
| `imu_recorder.py` | record raw IMU samples to CSV for offline threshold tuning |
| `swing_test.py` | run IMU calibration + live swing detection, printing each detected swing |
| `vision_test.py` | webcam + pose + hand + paddle detection, with a live overlay |
| `paddle_tracking_test.py` | compare paddle-tracking methods side by side (wrist / hand-palm / Hough circle) |
| `apriltag_test.py` | test the level-select tag detection + hold-to-confirm flow |
| `apriltag_generator.py` | generate the printable level tags |
| `render_test.py` | visually test the 3D table/ball/paddle rendering with real physics, no camera/IMU needed |
| `mqtt_listener.py` | subscribe to the status topic and print every message, to verify publishing |

## Config (`config.yaml`)

All tunable values live here, grouped by subsystem:

- `mqtt`: broker/port/topic/QoS/heartbeat interval
- `motor`: Connection Card color/serial to target a specific Double Motor
  (leave `null` to connect to the first one found)
- `swing_detector`: poll rate, smoothing, refractory period, calibration
  (auto-filled after the first calibration run)
- `paddle`: smoothing for the paddle-position signal (hand palm-center
  when detected, else the wrist - see `vision/paddle_detector.py`)
- `levels`: tag-to-level mapping, selection timing, and per-level tuning
  (ball speed, hit radius, timing window, opponent skill/return shot) -
  includes a `claude` entry (not in `tags`, so it's only reachable via the
  `P` secret unlock) configuring the Q-learning side instead
- `physics`: real table dimensions (meters), gravity, restitution
- `render`: window size, camera placement/zoom, picture-in-picture sizes,
  and `pixel_scale` (the retro pixelation filter's chunk size - higher is
  blockier, lower is closer to the original crisp rendering)
- `player`: maps the camera's normalized game-space coordinates to real
  table positions (for drawing the paddle marker) - doesn't affect
  hit-detection itself, which stays in game-space units

## Tests

```bash
python3 -m pytest
```

Covers every subsystem's pure logic without requiring actual hardware or
a display: swing detection, coordinate math, paddle detection on
synthetic images, AprilTag detection on generated tags, physics, hit
detection (including the paddle-contact-position-based directional
angle), opponent AI (both the regular difficulty curve and the
Q-learning agent), scoring/match state, the animation state machines
(opponent idle/hit/score/lose, Claude's pulse), rendering math (including
the retro pixelation filter and procedural backgrounds/characters),
chiptune audio synthesis, and MQTT payload formatting. Rendering/drawing
functions themselves are smoke-tested (they run and visibly draw
something) rather than asserting exact pixel output, consistent with how
the rest of the suite treats pygame drawing code.

The suite runs headless (no real display/audio device needed) - pytest
doesn't require any special flags or a connected webcam/motor.

## Verifying the whole thing actually works end to end

Beyond the automated tests, two quick manual checks are worth doing after
any change, especially to the game loop or hardware integration:

```bash
# 1. A short hardware-free run, watching for any startup error or crash:
python3 main.py --mock-imu --mock-vision --no-mqtt
# Press Enter at the main menu, 1-4 at level select, and play a bit with
# the mouse standing in for your paddle (Space triggers a mock swing).

# 2. The real thing, once the above looks clean:
python3 main.py
```

If you're testing a change to a specific subsystem, the matching tool in
`tools/` (see the table above) isolates it without needing a full game
session - e.g. `swing_test.py` to sanity-check the IMU without touching
vision at all, or `render_test.py` to check rendering changes without
needing the camera or motor connected.

## Troubleshooting

- **"Could not open the webcam"**: almost always a macOS permissions
  issue (see above) or another app already using the camera. Try a
  different `CAMERA_INDEX` in `vision/camera.py`'s `Camera(...)` call if
  you have multiple webcams.
- **"Connecting to Double Motor..." hangs**: confirm the motor is powered
  on and `config.yaml`'s `motor.card_color`/`card_serial` (if set) match
  its actual Connection Card.
- **AprilTags not detected**: make sure there's a plain white margin
  around the printed/displayed tag - the detector needs that quiet zone
  to find the tag's boundary. Also check lighting isn't washing out the
  tag's contrast.
- **Paddle tracking looks wrong/jumpy**: run `tools/paddle_tracking_test.py`
  to see live which signal (hand palm-center vs. wrist) is driving it and
  how stable each looks for your setup/lighting.
- **Hits never register / always "Wrong spot!"**: your actual reach (in
  meters) may not match `player.reach_scale`'s assumption, or
  `levels.<level>.hit_radius` may be too tight for your current
  calibration - these only affect feel/difficulty, not correctness, so
  it's safe to adjust them directly in `config.yaml`.
- **Haptic buzz stops happening after a while**: the motor connection
  self-heals from a stuck/hung BLE command after a couple seconds (see
  `imu/motor_connection.py`'s `HAPTIC_STUCK_TIMEOUT_SECONDS`) - if it's
  still not buzzing well after that, check the motor's still connected
  (the HUD/status message would show a dropped connection) rather than
  assuming it's permanently broken.
- **Some text looks like the wrong letter (e.g. "B" reading as "D")**:
  the pixel font can lose fine details at very small sizes once the
  retro pixelation filter chunks it up - this has been fixed everywhere
  it was found, but if you add new UI text at a small font size, test it
  through the real pipeline (pixelation included) before assuming it's
  legible.
- **No sound**: check you haven't muted with `M`; sound effects and music
  are synthesized programmatically (no audio files to go missing) via
  `pygame.mixer`, which requires a working audio output device.
