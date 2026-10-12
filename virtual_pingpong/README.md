# Virtual Ping-Pong

Play virtual ping-pong against a computer opponent using a real paddle: a
webcam tracks your body/arm and the paddle's color, while a LEGO Education
Double Motor (45651) taped to the paddle detects your swing via its IMU.
Current best consecutive-hit streak is published live over MQTT.

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

## Printing the AprilTags (level select)

```bash
python3 tools/apriltag_generator.py
```

Generates PNGs and a combined `tools/generated_tags/all_tags.pdf` for the
4 levels (beginner/intermediate/advanced/expert, tag IDs 0-3). Print the
PDF and cut out the tags, or just display the PNGs on a phone/tablet
screen with a plain border around them.

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

| Key | Action |
|---|---|
| `D` | toggle debug overlay (webcam picture-in-picture + swing intensity bar) |
| `M` | mute/unmute sound effects |
| `P` / `Esc` | pause / resume |
| `Q` | quit |
| `1`-`4` | pick a level directly (beginner/intermediate/advanced/expert) - works anytime during level select, as a fallback if you don't have a printed tag handy |
| `Space` | (only with `--mock-imu`) trigger a swing |
| mouse | (only with `--mock-vision`) stands in for the paddle position |

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
| `mqtt_listener.py` | subscribe to the score topic and print every value, to verify publishing |

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
  (ball speed, hit radius, timing window, opponent skill)
- `physics`: real table dimensions (meters), gravity, restitution
- `render`: window size, camera placement/zoom, picture-in-picture size
- `player`: maps the camera's normalized game-space coordinates to real
  table positions (for drawing the paddle marker) - doesn't affect
  hit-detection itself, which stays in game-space units

## Tests

```bash
python3 -m pytest tests/
```

Covers every subsystem's pure logic (swing detection, coordinate math,
paddle detection on synthetic images, AprilTag detection on generated
tags, physics, hit detection, opponent AI, scoring, rendering math, MQTT
payload formatting) - everything that doesn't require actual hardware or
a display.

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
- **No sound**: check you haven't muted with `M`; sound effects are
  generated programmatically (no audio files to go missing) via
  `pygame.mixer`, which requires a working audio output device.
