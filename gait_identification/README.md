# Gait Identification

An ankle-worn activity classifier built on an **Arduino UNO Q** and an **MPU-6050** IMU. It recognizes four activities: **standing, walking, running and biking**. Air-biking while lying on your back stands in for real cycling.

There are two k-nearest-neighbor models, both written by hand (no neural networks):

| Model | Features | Held-out accuracy |
|---|---|---|
| Model 1 | Peak acceleration along the shin, with gravity removed (1 feature) | 91.5% |
| Model 2 | 18 motion features from all 6 IMU channels | 99.1% |

Full results are in [`reports/report.md`](reports/report.md).

**How it works:** the UNO Q's microcontroller reads the IMU at 100 Hz and streams the samples to the Linux side over the Bridge. Python on the Linux side classifies each 2 s window every second and shows the result on the LED matrix. It also publishes the result over MQTT, so a laptop UI can show it with an animated figure and switch models.

## Folder layout

```
sketch/sketch.ino      MCU: read MPU-6050 @ 100 Hz, send to Python, draw LED matrix
python/                board app (Linux side)
  main.py              MODE = "live" or "collect"
  live.py              live classification + LED + MQTT
  collect.py           guided recording of one labeled session
  gaitlib.py           shared windowing, features, KNN (used by board and laptop)
  models/              trained models (model1.npz, model2.npz)
tools/                 laptop scripts: train, evaluate, plot, UI, MQTT helpers
data/                  recorded sessions (session_NN.csv + .json)
reports/               evaluation report + confusion matrices
tests/                 unit tests
```

## 1. Hardware

Wire the MPU-6050 to the UNO Q's header with short jumper wires:

| MPU-6050 | UNO Q |
|---|---|
| VCC | 3.3V (not 5V) |
| GND | GND |
| SDA | SDA (next to AREF) |
| SCL | SCL (next to AREF) |

Strap the board and IMU to the outside of the lower shin, just above the ankle bone.

**Axis mapping:** the code assumes the board is mounted with shin = −x, forward = −y, lateral = +z. If you mount it differently, update `AXIS_MAP` in `python/live.py` and `python/collect.py`.

## 2. Setup

**On the laptop**, from the repo root:
```
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib scikit-learn paho-mqtt pygame pytest
brew install android-platform-tools     # provides adb, for copying files to/from the board
```

**On the board**, in Arduino App Lab:
1. Create an app, for example `gaitdetector`.
2. Copy the files onto the board over USB:
   ```
   cd gait_identification
   APP=/home/arduino/ArduinoApps/gaitdetector
   adb push sketch/sketch.ino $APP/sketch/
   adb push python/main.py python/live.py python/collect.py python/gaitlib.py python/requirements.txt $APP/python/
   adb push python/models $APP/python/
   ```

App Lab's editor only shows `main.py` and the sketch. The other Python files are still on the board and get used when the app runs. Edit those on the laptop and push them again.

## 3. Record training data

1. Set `MODE = "collect"` in `python/main.py` and push it (or edit it in App Lab).
2. Press **Run** in App Lab and follow the cues in the console. You'll do 90 s each of standing, walking, running and air-biking, then 30 s lying still. Each run records one session, about 8 minutes. To do a short dry run first, set `PROTOCOL = "test"` in `collect.py`; test sessions are skipped during training.
3. Re-strap the board between sessions. Three sessions is enough.
4. Copy the sessions to the laptop:
   ```
   adb pull $APP/python/data/. data/
   .venv/bin/python tools/plot_session.py data/session_02.json    # optional: check a session
   ```

If something went wrong in part of a session, set that segment's `ok` to `false` in its JSON. It will then be left out of training.

## 4. Train and evaluate

Run from `gait_identification/`:
```
../.venv/bin/python tools/train.py       # trains both models on all sessions -> python/models/
../.venv/bin/python tools/evaluate.py    # holds out the last session -> reports/report.md
../.venv/bin/python -m pytest tests      # unit tests
```

Then push the new models to the board:
```
adb push python/models $APP/python/
```

## 5. Run it live

1. Set `MODE = "live"` in `python/main.py`. `MODEL` in `live.py` sets which model is shown at startup.
2. Press **Run** in App Lab and stand still for the 3 s calibration.
3. The board then classifies every second:
   - The **LED matrix** shows S, W, R or B plus the model number.
   - The **console** shows both models' predictions.
   - Every prediction is logged to `data/live_*.csv`.

**Laptop UI**, which shows the live activity with an animated figure and has buttons to switch models:
```
../.venv/bin/python tools/gait_ui.py
```

The board and the UI talk over MQTT on topic `ME193/Julian/GaitDetection`, using broker `broker.emqx.io` port 1883. To watch the raw messages, run `tools/mqtt_monitor.py`. If the UI never connects, run `tools/mqtt_probe.py` to find a broker your network allows, then change `BROKER` in `python/live.py`, `tools/gait_ui.py` and `tools/mqtt_monitor.py`.
