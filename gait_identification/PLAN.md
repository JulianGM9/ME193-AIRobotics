# Ankle-Worn Gait Classifier — Design Plan

**Status:** Plan approved 2026-10-26. Stages 1–4 done. Stage 5 (live mode) is written and tested by replaying session 04 on the laptop; next it gets deployed to the board.
**Deadline:** the dataset gets collected and trained **today**. The aim is a classifier that works reasonably well, not a perfect one.
**Hardware:** Arduino UNO Q + MPU-6050, worn on the ankle.

## Goal

Classify the wearer's activity into four classes from an ankle-mounted IMU:

| # | Class | Notes |
|---|---|---|
| 1 | Standing still | |
| 2 | Walking | |
| 3 | Running | |
| 4 | Biking | Includes lying on your back and "air-biking" as a stand-in for real cycling |

## Hard requirements

- The classifier is **k-nearest neighbors only**. No neural networks of any kind (no MLPs, CNNs, RNNs or TinyML).
- There are **two models**, either as two scripts or as two modes you can switch between on the device:
  - **Model 1 (single-axis):** uses only peak acceleration in one direction.
  - **Model 2 (full IMU):** uses all six MPU-6050 channels (3-axis accelerometer and 3-axis gyro).
- The project includes the full dataset workflow: a collection tool and protocol, labeling, storage format, and training and evaluation.

## Platform facts

### Verified from Arduino / Zephyr documentation

| Item | Detail |
|---|---|
| Linux side (MPU) | Qualcomm Dragonwing QRB2210, quad-core Cortex-A53 @ 2.0 GHz, runs Debian Linux |
| Microcontroller side (MCU) | STM32U585 Cortex-M33 @ 160 MHz, with floating-point unit, 2 MB flash, 786 KB SRAM. Runs Arduino sketches on Zephyr OS |
| Link between the two | Arduino Bridge (remote procedure calls). App Lab builds one app containing `python/` (Linux) and `sketch/` (MCU), the same layout as `minifig_detection/Minifig-UnoQ-Blue` |
| Logic level | All GPIO is **3.3 V**. Third-party guides say the pins are not 5 V tolerant, so we treat them as 3.3 V only |
| I2C | SDA/SCL on the UNO header, plus a **Qwiic connector (3.3 V only) on the second I2C bus, `Wire1`** |
| On-board output | 13×8 LED matrix and 2 RGB LEDs, both driven by the MCU |
| Wireless | Wi-Fi 5 (2.4 and 5 GHz) and Bluetooth 5.1 |
| Power input | USB-C, which requests **5 V / 3 A** (no higher-voltage USB PD profiles). Alternatively, the VIN pin takes 7–24 V DC |

### Not yet verified — must be checked on the hardware

- **U1. ✅ Verified 2026-10-26:** 10 batches/s of 10 samples gives 99.2–99.3 Hz with 0 gaps.
- **U2.** How much power the board actually draws (the Linux side is the main load). This sets the battery size and run time.
- **U3.** Whether an App Lab app starts on its own at power-up without a computer attached. Needed if the device runs untethered.
- **U4.** Header SDA/SCL (D20/D21, next to AREF) are on `Wire`, according to third-party pinout guides; the first sensor test confirms it. Still unknown: whether the sketch can read a user button.
- **U5. Partly verified:**
  - The app writes to `/app/python/data/`, and `/app` is the app folder on the board, `/home/arduino/ArduinoApps/gaitdetector/`. The path inside the app comes from stage 1; the board-side path comes from the build error paths.
  - Still open: how to copy files to the laptop, probably `adb pull` or `scp`.

## Decisions made

### D1. Division of work between the processors
- **Microcontroller (sketch):** reads the MPU-6050 on a fixed timer and sends the samples to Linux in batches over the Bridge. It also drives the LED matrix when Python tells it to.
- **Linux (Python):** windowing, features, KNN, logging, data collection and mode switching.
- The same Python feature and KNN code is used for training and for live classification.
- **Risk:** Bridge speed (U1). Build stage 1 measures it before anything is built on top.
- **Rejected:** running everything on the microcontroller. It would need the feature code written twice (C++ on the device, Python for training), with the two versions kept identical.

### D2. Physical layout (Q1.1) — revised 2026-10-26
- The **UNO Q and the MPU-6050 are both on the ankle**, in one unit.
- **Power** comes from the power bank **at the waist**, through a USB-C cable down the leg. Only power goes through that cable, so its length doesn't matter.
- **Why it changed:** the IMU is now connected with short jumper wires, which removes the long-I2C-cable risk entirely.
- **Today:** the board stays plugged into the laptop by USB for collecting data, training and testing (D13). The power bank comes later.
  - **Risk:** walking and running on a USB cable. Options: carry the laptop in a backpack, or use a long cable and do laps. Running on the spot is a last resort, because it isn't the same motion as running.
- **Earlier version (replaced):** board and power bank at the waist, with an I2C cable down to an ankle-mounted IMU.

### D3. Hardware on hand (Q1.2, Q1.3)
- You have all the parts, including a USB-C power bank.
- The board's real current draw is still measured in stage 1 (U2) to work out run time.

### D4. IMU wiring (Q1.5) — revised 2026-10-26
Short **jumper wires**, from the MPU-6050 breakout to the UNO Q headers:

| MPU-6050 | UNO Q |
|---|---|
| VCC | 3.3V (power header) |
| GND | GND |
| SDA | SDA (D20, next to AREF) |
| SCL | SCL (D21, next to AREF) |

- That's I2C bus `Wire`, at 100 kHz.
- The AD0, INT, XDA and XCL pins are left unconnected (address 0x68).
- **Secure the jumpers** with tape or hot glue at both ends, because running shakes loose push-on jumper connections.
- **The software still detects dropouts:** a failed read is logged as a gap and the sensor is reinitialized.
- **Earlier version (replaced):** the Qwiic port (`Wire1`) with a long twisted cable down the leg.

### D5. Mounting spot and enclosure (Q1.4)
- The sensor goes on the **outside of the lower shin, just above the ankle bone (lateral malleolus)**, not on the shoe.
- The breakout sits in a **small rigid case**, held with a **tight velcro strap** so it can't wobble against the leg.

### D6. Axis convention and the Model 1 axis (Q2.1, Q2.2)
- **Axis convention:** one axis **along the shin** (pointing up when standing), one **forward** (direction of travel), one **lateral**.
- **Measured mapping (2026-10-26, stage 1):** **S = −x**, **F = −y**, **L = +z**.
  - **S:** standing still gave x = −0.97 g.
  - **F:** tilting the shin forward about 50° took y from −0.11 to +0.73 g (and x to −0.60 g, consistent with that tilt).
  - **L:** completes a right-handed frame, so L points to the **wearer's left**. It reads +0.31 g when standing, meaning the board is tilted about 18° sideways.
  - **This mapping only holds for this mounting.** Recheck it if the board gets remounted differently.
- **Model 1 starting axis:** along the shin. Gravity is removed before taking the peak (see D11). Expected peak values on this axis:
  - standing: near 0
  - walking: moderate heel-strike peaks
  - running: large impact peaks
  - air-biking on your back: low to moderate peaks
- **All six raw channels are always recorded**, so the axis choice is never locked in. Evaluation runs Model 1 on each of the three axes and reports which one wins. That result is the data-backed justification for the final axis.

### D7. Sampling (Q3.1–Q3.4)
- **Sample rate: 100 Hz**, timed on the microcontroller. Samples go to Linux in batches of about 10.
  - 50 Hz was rejected because it can miss the top of a sharp impact peak.
  - 200 Hz was rejected: twice the Bridge load for no expected accuracy gain.
- **Ranges:** accelerometer **±16 g**, gyro **±2000°/s**, so running impacts and fast leg swings never get clipped.
- **Gyro offset:** stand still for about 2 s at the start of each recording. The average gyro reading over that time is subtracted from all later readings.
- **Accelerometer:** no calibration, and **gravity is kept** in the signal. Factory offsets are small compared with the differences between activities.
- **Filtering:** the MPU-6050's built-in low-pass filter at about **44 Hz**, and no other filtering. Features are calculated from the raw readings, so training and live classification see the same signal.
- **Timestamps:** the microcontroller stamps each sample, so Python can detect dropped samples, including I2C dropouts.

### D8. Windowing (Q4.1–Q4.3)
- **Window:** 2 s (200 samples), with **50% overlap**, so a new prediction comes every 1 s.
- **Smoothing:** majority vote over the **last 3 raw predictions**. The shown activity changes only when a new class has 2 of the 3 votes; on a tie, the current activity stays.
- **Logging and evaluation:** both raw and smoothed predictions are logged.
  - Model 1 vs Model 2 is compared on **raw per-window** results, so smoothing can't hide differences between the models.
  - Smoothed accuracy is reported as a separate number.
- **Training data cleanup:** drop the first and last **3 s** of each labeled recording, and drop any window that crosses a label change.

### D9. Guiding principle: classify motion, not posture (Q5.3)
- Air-biking on your back stands in for **real cycling**.
- So the models must recognize biking from the **pedaling motion**, not from the fact that you're lying down.
- Any feature that mainly reveals which way gravity points (body posture) is left out.

### D10. Model 2 features (Q5.2, Q5.3) — "Option B, motion only"
18 features per window. S, F and L are the shin, forward and lateral axes from D6.

| Group | Features | Count |
|---|---|---|
| Per channel (3 accelerometer + 3 gyro) | standard deviation, range (max − min) | 12 |
| Gyro channels | mean | 3 |
| Overall acceleration (combined size of all 3 axes) | peak, standard deviation | 2 |
| Cadence | main repetition rate of the L-axis gyro (the leg's forward-back swing) | 1 |

- **Accelerometer means are excluded.** They mostly measure which way gravity points, which is posture.
- The overall acceleration size doesn't depend on how the sensor is tilted, so it's allowed.
- **Feature check:** the evaluation removes one feature group at a time to show what each group contributes.
- **Posture check:** a short recording of **lying still on your back** is held out of training. It should be classified as standing still. If it comes out as biking, the model has learned posture after all.
- If you can manage a **real bike ride**, record it as a held-out test as well.

### D11. Model 1 feature (Q5.1, Q5.4)
- **Feature:** `peak = max(a_S - mean(a_S))` over the window. That's one number: how hard the shin was jolted in its +S direction.
- **Gravity removed:** subtracting the window's average takes gravity out, so the feature is **posture-free**, following D9. Both models therefore follow the same rule, and the Model 1 vs Model 2 comparison isolates "one axis vs all six."
- **Comparison only:** a 95th-percentile version, `p95(a_S - mean(a_S))`, is calculated in the evaluation as a robustness check. It's not the main Model 1.
- **Evaluation:** the same feature is also run on the F and L axes, as decided in D6.

### D12. KNN design (Q6.1–Q6.3)
- **Implementation:** written by hand in numpy, about 30 lines. This hand-written version is the one that runs on the device.
  - **Check:** a unit test runs it alongside scikit-learn's `KNeighborsClassifier` on the same data, and the two must agree.
  - scikit-learn's metrics functions (confusion matrix, precision/recall) are used for evaluation on the laptop. They aren't neural networks.
- **Scaling:** standardize each feature (subtract its mean, divide by its standard deviation). Those values come from the **training data only**, are saved with the model, and are reused unchanged at test time and on the device.
- **Distance:** ordinary straight-line (Euclidean) distance.
- **k:** odd values from 1 to 15, chosen by **leave-one-session-out** cross-validation on the training sessions. Start from k = 5. Model 1 and Model 2 each get their own k.
- **Ties:** the class whose neighbors are closer in total wins.
- **Memory and speed:** not a constraint. KNN runs on the Linux side (D1), and a few thousand windows × 18 features is well under 1 MB, at less than 1 ms per prediction.

### D13. Dataset: a minimal version for today (Q7.1, Q7.2)
- **Sessions:** **3 sessions** recorded today, from one person (you). Take the sensor off and strap it back on between sessions; that's the one source of variety worth keeping.
- **Each session:**
  - about 2 s standing still at the start, for the gyro calibration (D7)
  - **90 s of each activity**: standing, walking, running, air-biking
  - **30 s lying still on your back** for the posture check (D10). Never used for training.
  - Vary the pace a little within each activity if it's easy.
  - Takes about 8 minutes per session including breaks, so about 25 minutes in total.
- **Expected data:** 3 sessions × 4 activities × ~85 windows ≈ **1,000 windows**. That's enough for KNN.
- **Labeling:** a **guided session script** run from the laptop while the board is plugged in by USB.
  - The terminal and the LED matrix show the current activity letter (S, W, R, B, L) and a countdown, and you follow along.
  - Labels come from the schedule automatically. The 3 s trimming (D8) absorbs the changes between activities.
  - At the end you can mark any stretch as bad, and it gets excluded.
- **File format:** each session saves two files.
  - `session_NN.csv`: timestamp in ms plus the raw sensor readings `ax, ay, az, gx, gy, gz`.
  - `session_NN.json`: who wore it, the date, mounting notes, the axis mapping, the gyro offset, and the labeled stretches as `[label, start_ms, end_ms, ok]`.
  - Features are always recalculated from the raw CSVs, so nothing needs re-recording if a feature changes.

### D14. Train/test split (Q7.3)
- **Always split by whole session**, never by window.
- With 3 sessions, two sets aside would leave only one to train on, so the earlier plan is scaled down:
  - **Set aside the last session (session 3)** before training anything. It's only used for the final evaluation.
  - **Choose k** with leave-one-session-out on sessions 1–2. That's only two rounds, so if the choice is unclear, just use k = 5.
  - **Report final scores** from training on sessions 1–2 and testing on session 3.
  - **Extra, if time allows:** leave-one-session-out across all 3 sessions with a fixed k, to show how much results vary between sessions.

### D15. Evaluation (Q8.1)
For each model, on the set-aside session:
- **Core results:**
  - overall accuracy
  - precision and recall for each class
  - **confusion matrix**, saved as a PNG
  - both raw per-window results and smoothed results (D8)
- **Model 1 vs Model 2:** a table comparing the two side by side.
- **Model 1 axis comparison:** accuracy using the S, F and L axes (D6).
- **Posture check:** what the lying-still recording gets classified as (D10).
- **If time allows:** the 95th-percentile robustness check for Model 1 (D11), and the Model 2 feature check that removes one group at a time (D10).
- **Where it runs:** on the laptop, using the CSVs copied off the board and scikit-learn's metrics functions.

### D16. Switching models and showing the result (Q9.1, Q9.2)
- **Both models classify every window at the same time.**
- **Switching:** you choose which model's answer is *shown*, with a `MODEL = 1` or `2` setting in the live app.
  - This counts as two modes on one device, so it meets the requirement.
  - A live button toggle can be added later, which depends on U4.
- **LED matrix:** the shown activity's letter (S, W, R or B) plus a small `1` or `2` marking the active model.
- **Terminal:** one line per second with the raw and smoothed predictions from both models.
- **Live log:** a CSV of every prediction, which allows live checks such as "walk for 1 minute, count how many seconds were labeled correctly."

### D17. Project layout, testing and build order (Q10.1)
```
gait_identification/
├── PLAN.md
├── sketch/sketch.ino          # MCU: read MPU-6050 on Wire1 @ 100 Hz, batch over Bridge, LED matrix
├── python/main.py             # board: live mode (both models, display, logging) and collection mode
├── python/                    # the board app's Python side
│   ├── main.py                #   MODE = "live" or "collect"
│   ├── live.py                #   live classification: both models, LED matrix, console, CSV log
│   ├── collect.py             #   guided recording of one labeled session
│   ├── gaitlib.py             #   shared: loading, windowing, features, hand-written KNN, smoothing
│   ├── models/                #   model1.npz, model2.npz (written by tools/train.py)
│   └── requirements.txt       #   numpy
├── tools/                     # laptop-side
│   ├── common.py              #   dataset extraction + leave-one-session-out k selection
│   ├── plot_session.py        #   plot one session with its labeled segments
│   ├── train.py               #   build both models on all sessions -> models/model1.npz, model2.npz
│   └── evaluate.py            #   hold out last session: metrics, confusion matrices, comparisons -> reports/
├── data/                      # session_NN.csv + .json copied off the board
├── reports/
└── tests/                     # pytest: features on synthetic signals, KNN vs scikit-learn, windowing
```
- **Testing:**
  - unit tests on fake signals with known answers (for example, a sine wave at 1 Hz → cadence 1 Hz)
  - a test checking the hand-written KNN against scikit-learn
  - a hardware check at each stage before moving on
- **Build stages.** Each stage ends with a short check with you before the next one starts:
  1. **Sensor link:** the sketch reads the MPU-6050 and Python receives samples. Measure the actual sample rate, the Bridge speed (U1) and dropouts. Also work out where files are saved and how to copy them off the board (U5). Map the breakout's X/Y/Z to S/F/L (D6).
  2. **Collection tool:** the guided session script writes the CSV and JSON. Do a 1-minute test recording and look at the plots.
  3. **Collect data:** record 3 sessions (you).
  4. **Train and evaluate on the laptop:** features, KNN, tests, then both models and the evaluation report.
  5. **Live mode:** both models running on the board, with the LED matrix and logging. Then a live check: do each activity for a minute and see how often it's labeled correctly.

## Results (2026-10-26, stage 4)
Full report: `reports/report.md`. Confusion matrices: `reports/confusion_model1.png`, `reports/confusion_model2.png`.

**Data:** sessions 02–04 (protocol "full"), 0 gaps. 988 training windows, about 247 per class.

**Held-out test:** trained on sessions 2–3, tested on session 4 (329 windows).

| Model | k | Raw accuracy | Smoothed accuracy |
|---|---|---|---|
| Model 1 (peak shin accel) | 13 | 91.5% | 93.9% |
| Model 2 (18 motion features) | 1 | 99.1% | 99.1% |

**Findings:**
- **Model 1's mistakes are mostly walking vs biking.** Biking recall is 76%, and walking precision is 77%. Standing is perfect.
- **The shin axis (S) is the best single axis:** S 91.5%, F 84.5%, L 79.0%. That's the data-backed answer to D6.
- **Using the true peak matters.** The 95th percentile does much worse (S: 64.1%).
- **No single Model 2 feature group is critical:** removing any one group changes accuracy by less than 1%.
- **Posture check passed.** All 68 lying-still windows came out as standing for both models, so neither model learned posture (D9).
- **Leave-one-session-out across all 3 sessions (k = 5):**
  - Model 1: 76.6–86.9%
  - Model 2: 95.4–99.7%

**Deployed models** (`models/`, trained on all 3 sessions):
- Model 1: k = 13 (cross-validated accuracy 83.1%)
- Model 2: k = 1 (cross-validated accuracy 99.1%; k = 5 gets 98.3%)

## Add-on: MQTT link and laptop UI (requested 2026-10-26)
- **Topic:** `ME193/Julian/GaitDetection`. One topic carries JSON messages tagged with `type`:
  - board → UI: `status`, sent every second with the shown activity, the active model, cadence, and both models' raw and smoothed predictions
  - UI → board: `set_model`, which switches the model shown on the LED and in status messages
- **Broker:** `broker.emqx.io`, port 1883.
  - `test.mosquitto.org` (used by the other ME193 scripts) timed out on every port on 2026-10-26.
  - `tools/mqtt_probe.py` found EMQX working, with a round trip of about 1.7 s.
- **Build parts:**
  1. MQTT link, tested with `tools/mqtt_monitor.py`
  2. UI window that shows the activity and selects the model
  3. Animated figure doing the detected activity
- **Status:** parts 1–2 tested by you on the board, with an MQTT lag of 0.13–0.34 s. Part 3 is built and checked in off-screen renders: `tools/gait_figure.py` plus a full-width panel in `tools/gait_ui.py`.

## Approval and later changes
- **Plan approved 2026-10-26.**
- **Possible later change:** set up SSH over Wi-Fi on the board so recording doesn't need the USB cable. Only if you ask for it.

## Open questions

None. Every question has a recommended answer above. Sign-off needed on D13–D17, which were proposed together to fit today's deadline.

## Sources

- [Arduino UNO Q — Arduino Docs](https://docs.arduino.cc/hardware/uno-q/)
- [UNO Q User Manual — Arduino Docs](https://docs.arduino.cc/tutorials/uno-q/user-manual/)
- [Arduino UNO Q — Zephyr Project docs](https://docs.zephyrproject.org/latest/boards/arduino/uno_q/doc/index.html)
- [Arduino UNO Q Pinout Guide & Power Requirements — eTechnophiles](https://www.etechnophiles.com/arduino-uno-q-pinout-guide/)
