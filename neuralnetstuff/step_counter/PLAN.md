# Ankle IMU Step Counter: Design Plan

**Status:** Plan approved 2026-11-02. M0–M4 and Guided sessions 1 + 3 done (10 sessions; training S001–S008 = 578 usable strides; locked test S009/S010 = 179). **Deadline 2026-11-03: scope trimmed (see the milestone table).** M5 in progress.
**Hardware:** Arduino UNO Q + MPU-6050 worn on the ankle. Battery is strapped to the thigh.
**Environment:** repo root `.venv` (Python 3.13, torch 2.14). Run from `neuralnetstuff/step_counter/`.
**How we work:** small milestones, each one verified before the next. After each guided session we review the results and decide what to try next. Feature ideas from the [ideas backlog](#ideas-backlog) are built only when Julian picks them.

---

## 1. Platform facts (researched 2026-11-02)

| Item | Detail | Confidence |
|---|---|---|
| Linux side (MPU) | Qualcomm QRB2210, 4× Cortex-A53 @ 2.0 GHz, 2 or 4 GB RAM, Debian 13 | Official docs |
| Microcontroller side (MCU) | STM32U585 Cortex-M33 @ 160 MHz, 786 kB SRAM, Zephyr | Official docs |
| Logic level | Header I/O is 3.3 V. Sources disagree on which pins are 5 V tolerant, so we treat **all** of them as 3.3 V only | Mixed |
| I2C | Header SDA/SCL (D20/D21) is `Wire`. Qwiic is `Wire1` | Official manual |
| Bridge | MessagePack remote procedure calls (RPC) over a 115200-baud UART through the `arduino-router` service. `notify` is fire-and-forget; `call` blocks | Community write-up |
| Broker rate limit | broker.emqx.io drops messages above about 8/s per client, whatever their size. At 10/s about 3% are lost, and with status + pongs added about 20% are lost | Measured 2026-11-02 (`scripts/listen.py` + a probe) |
| Bridge bandwidth | About 11.5 kB/s. ASCII batches of 10 samples at 100 Hz are about 4 kB/s (35%). The gait project logged 0 gaps, even though one source claims a 256 B message cap | Measured in the gait project |
| Python on the board | App Lab runs it in a Docker container. Packages go in `requirements.txt` (installed with uv). apt packages can't be added | Forum and tutorials |
| PyTorch on the board | **Unverified.** aarch64 wheels exist on PyPI, but nobody has reported them working in App Lab. We don't depend on it | — |

## 2. Decisions

| ID | Decision |
|---|---|
| D1 | **Board = thin relay, laptop = everything else.** Board app: a sketch reads the MPU-6050 at 100 Hz and sends batches over the Bridge; Python publishes them over MQTT and **also logs them locally** (a gap-free backup, pulled with `adb`). Laptop: recording, labeling, training, inference, uncertainty, UI. |
| D2 | **MQTT** like the other ME193 projects: broker `broker.emqx.io:1883`, topics under `ME193/Julian/StepCounter/...`, JSON messages tagged with `type`. Every sample carries a `seq` number, so the laptop can detect gaps. **The public broker drops messages above about 8/s per client** (measured 2026-11-02), so the relay groups the MCU's 10 batches/s into **4 MQTT messages/s** (about 25 samples, +0.25 s latency). |
| D3 | **Wiring:** reuse the gait setup (VCC→3.3 V, GND, SDA, SCL on the header `Wire` bus). **Mount:** rigid, outside of the **right** ankle, with an arrow toward the knee. Axis map: S (up the shin) = −x, F (forward) = −y, L = +z = **toward your left (inner side of the right ankle)**, so the frame is right-handed. Every session starts with a 5 s standing calibration that measures gyro bias and checks that gravity points the expected way (orientation check: within 20° of +S, and within 10° of the last session's mount). A **one-time six-pose accelerometer calibration** (`scripts/calibrate_accel.py` → `data/calibration/accel.json`, offset + scale per axis) is applied in preprocessing, and each session keeps a copy in its meta.json. |
| D4 | **Sampling:** 100 Hz, ±16 g, ±2000 °/s, MPU digital low-pass filter about 44 Hz. Reuse the gait sketch's ASCII batches of 10 samples. |
| D5 | **Preprocessing:** convert to g and °/s → subtract gyro bias → rotate into the canonical shin/forward/lateral frame using the calibration's gravity direction → normalize per channel using **training-set** statistics. Don't normalize each window separately, because absolute magnitude matters. |
| D6 | **Counting target:** one event per **right-foot stride**. Forward steps are marked at the **mid-swing minimum of the L (sagittal) gyro**. **Turning, side and backward steps** are marked at the **peak of the total angular speed |ω|**: they barely swing the shin forward, so they make no L dip (found in S008, where the S/F bursts matched the counts exactly). The labeler proposes |ω| peaks only inside counted blocks. Reported steps = **2 × strides**. The doubling error δ ∈ {−1, 0, +1} per bout goes into the uncertainty. |
| D7 | **Ground truth from the IMU only, with no extra sensors.** (a) Julian counts **total steps (both feet)** during fixed-target bouts. (b) A semi-automatic labeler: a loose detector proposes candidates, and Julian accepts, rejects or moves them. A bout can't be saved unless \|2·strides − counted steps\| ≤ 1. Labels are never raw detector output, otherwise the network would only learn to copy the baseline. |
| D8 | **Protocol:** sessions of about 6 minutes, alternating two versions. **A** (movement): walking slow/normal/fast with turns, running, stairs, shuffling. **B** (hard non-steps): standing, fidgeting, seated leg swings, foot taps, plus some walking. Both include calibration and start/stop bouts. Target **10 sessions** (about 1,200–1,500 strides). **Start from scratch**: no gait data reused. Record the hallway length once and log laps, for distance estimation later. **Revised after M4:** sessions 6–10 are C, B, C, A, C. **Protocol C (everyday movement, about 4.5 min)** uses timed *counted* blocks (pottering, turning on the spot, side and back steps, sit-to-stand, shuffle-and-tap) plus very slow and normal walking bouts. Counted blocks don't end at a pause, and every step is counted afterwards. |
| D9 | **Splits by session.** The locked test sessions are **S009 (protocol A) and S010 (protocol C)**. S001–S008 are for training, validation and cross-conformal calibration (LOSO). Evaluation refuses to touch test sessions unless given `--final`. |
| D10 | **Model output:** a dense per-timestep probability that a stride happens there. Input window 6×256 (2.56 s). Target = a Gaussian bump (σ ≈ 50 ms) at each labeled stride. Loss = binary cross-entropy (BCE) with extra weight on positive samples. Count = peak-picking. Live inference: 0.5 s hop with overlap blending. |
| D11 | **Baselines:** B1 = peak detection on the low-pass filtered sagittal gyro (height, prominence, minimum gap of about 0.4 s). B2 = B1 plus a **heel-strike gate**: each swing must be followed, 0.05–0.4/0.6 s later, by an impact |a| ≥ a threshold. (Revised 2026-11-02 from "foot planted between strides": on an ankle mount the shin keeps rotating at ~35 °/s during stance, and seated leg swings are stiller than that, so that gate would have backfired. Seated swings peak at ~1.3 g, real strides at ≥ 2.2 g.) Both are grid-search tuned on the training sessions. |
| D12 | **Model order:** A = dilated 1D CNN (about 25k params, receptive field > 1.5 s) **first** → C = causal TCN → B = small GRU. Live latency of about 1.3 s for the CNN is acceptable. |
| D13 | **Augmentation:** random crops, ±10° rotation, ±10% amplitude scaling. Time-warp is deferred. |
| D14 | **Metrics:** stride event F1 (±100 ms tolerance), count error per block and per session, false strides per minute in no-step blocks, with bootstrap CIs over blocks. **Success criterion (revised after M4, where B2 came within ~0.2 steps per bout of the oracle):** on the locked test sessions the network must (1) be **no worse than B2** on protocol A data (count error and false/min CIs overlap or are better); (2) be **better than B2** on the hard subset (protocol C counted blocks, shuffles, foot taps): lower mean \|count error\| per block, with non-overlapping CIs; and (3) have **calibrated uncertainty**, with 90% intervals reaching about 90% coverage. B2 gives no intervals at all. |
| D15 | **Uncertainty:** 5-model deep ensemble (mean probabilities → p_k; spread → Var_ens) + Poisson-binomial Var_PB = Σ p_k(1−p_k) + doubling term B̂·σ_δ² (σ_δ measured from Julian's step counts). σ̂² = sum of the three. **Cross-conformal** on LOSO predictions over the non-test sessions: score = \|N_true − N̂\| / σ̂ over bouts and random spans → quantile q → report **N̂ ± q·σ̂**. MC Dropout is used as a comparison experiment and a UI visual. |
| D16 | **Calibration checks:** reliability diagram + expected calibration error (ECE) for p_k (temperature scaling if needed); coverage vs. nominal at 50/80/90/95% next to interval width; coverage per activity. Default level 90%, with a UI toggle for 50/80/95. |
| D17 | **UI:** a browser UI with a FastAPI + WebSocket backend on the laptop. Frontend is plain JS modules (no build step) + uPlot + Canvas/D3, copied into the repo so it works offline. Clean dark "instrument panel" look. Six views: Live, Record, Label, Datasets, Train, Models/Compare. **Replay mode** feeds a recorded session through the live pipeline. Activations are captured with forward hooks and sent compressed to 8 bits. |
| D18 | **Pipeline:** raw sessions never change. Labels are versioned. Dataset versions are manifests named by their content hash. YAML configs are checked against dataclasses and can be overridden from the command line. Each run folder holds the resolved config, git commit + dirty flag, dataset hash, seed, per-epoch metrics, checkpoints and plots. Good runs are promoted to `models/registry.json`, which groups ensemble members. No MLflow or W&B. |
| D19 | **Git:** track code, `data/` (small), the registered models in `models/`, and PLAN.md. Ignore `runs/`. |

## 3. System and data flow

```
MPU-6050 ─I2C─> STM32 sketch (100 Hz, batches of 10) ─Bridge─> board Python relay
                                                              ├─ local CSV log (backup)
                                                              └─ MQTT ──> laptop backend (FastAPI)
                                                                          ├─ recorder / protocol runner
                                                                          ├─ preprocess → ensemble → peaks
                                                                          │   → uncertainty → N̂ ± q·σ̂
                                                                          ├─ training jobs, registry
                                                                          └─ WebSocket ──> browser UI
```

## 4. Uncertainty math (reference)

- Per-model candidate peaks k with calibrated probability p_k: E[S] = Σ p_k, Var_PB = Σ p_k(1−p_k).
- Ensemble: Var_ens = variance of the 5 members' counts.
- Doubling: within a bout N ∈ {2S−1, 2S, 2S+1}, so Var_δ = B̂·σ_δ². Bouts are split at gaps of more than 2 s between strides.
- N̂ = 2·E[S], σ̂² = 4(Var_PB + Var_ens) + B̂·σ_δ².
- Correlated errors make σ̂ too small (random errors grow like √N, a systematic bias grows like N). The conformal factor q corrects for this.

## 5. Repo layout

```
neuralnetstuff/step_counter/
  PLAN.md  README.md  requirements.txt  pytest.ini
  board/       sketch/sketch.ino, python/{main.py, requirements.txt}   (App Lab app; main.py is the relay, so it can be edited in App Lab)
  stepnet/     laptop package: io/ data/ models/ eval/ uncertainty/
               preprocess.py baselines.py losses.py train.py infer.py tracking.py
  server/      FastAPI app, WebSocket, routes
  web/         index.html, JS modules, CSS, vendor/ (uPlot, D3)
  configs/     YAML experiment configs
  scripts/     command-line tools (listen, record, train, evaluate, replay)
  tests/       pytest
  data/        raw/ labels/ datasets/        (tracked)
  models/      registry.json + promoted checkpoints (tracked)
  runs/        experiment runs (gitignored)
```

## 6. Testing strategy

- **Unit:** rotation with known gravity, Poisson-binomial vs. brute-force enumeration, conformal coverage on synthetic data, event-matching F1 on hand-built cases, overlap-add reconstruction, stable manifest hashes, no session in two splits.
- **Model:** a gradient-based receptive-field test (∂out[t]/∂in is non-zero only inside the designed receptive field); a causality test (the TCN's gradient with respect to future inputs is exactly 0); overfitting one batch drives the loss to about 0.
- **Synthetic gait generator** with known strides, for end-to-end pipeline tests before real data exists.
- **Integration:** replay mode, a fake MQTT publisher.
- **Hardware self-checks:** measured sample rate, gap count, a 90° tilt test.

## 7. Milestones

| # | Milestone | Verification | Status |
|---|---|---|---|
| M0 | Scaffold, dependencies, test harness | `pytest` passes | done 2026-11-02 |
| M1 | Board app (sketch + MQTT relay + local log) + laptop listener | 100 ± 0.5 Hz, 0 gaps over 5 min, tilt test | done 2026-11-02: 299.9 s, 99.88 Hz, 0 missing, 0 restarts. Tilt: S +0.98, F +1.01, L −0.84 with the outer side up (correct for L = left) |
| M2 | Recorder + protocol runner + orientation check (command line), six-pose accelerometer calibration, 400 kHz I2C | Dry-run session recorded and plotted | done 2026-11-02 (hardware dry runs T001/T002; bout start fixed to the start of the instruction) |
| M3 | Backend/web skeleton + Label view | Dry run labeled, bout checks pass | done 2026-11-02 (T002 labeled; all controls verified by Julian) |
| S1 | **Guided session 1:** record and label about 5 sessions | | |
| M4 | Baselines B1/B2 + evaluation harness | Results walkthrough | built 2026-11-02 (`scripts/evaluate_baselines.py` → `reports/m4_baselines.md`); done 2026-11-02: results walked through, and the criterion revised (D14) |
| M5 | Configs, dataset versions, tracking + CNN (synthetic → real) | **Guided session 2:** CNN vs. baselines | in progress |
| S3 | **Guided session 3:** about 5 more sessions, lock the test set | | |
| M6 | Ensemble + uncertainty stack + calibration reports | **Guided session 4:** coverage study | |
| M7 | Live engine + Live view + replay | Replayed session, then a live walk | slimmed: IMU + stride markers, count ± interval, CNN activations |
| M8 | Datasets / Train / Compare views | Train and compare from the UI | **cut** (deadline): the command-line scripts cover this |
| M9 | TCN and GRU experiments | **Guided session 5:** architecture comparison | **cut** (deadline); one quick TCN run only if time allows |
| M10 | One-time final evaluation on the test set | Final report | |
| M11 | ~~Stretch: distance estimation (stride-length regression / ZUPT / calibration walks; separate mini-planning)~~ | | **cut** (deadline) |

## Ideas backlog

These need Julian's approval before anything is built.

- On-board inference (ONNX or numpy on the UNO Q). First test whether torch installs in the App Lab container.
- Activity classification head trained alongside the stride output.
- Active learning in the labeler (label the model's most uncertain segments first).
- Gait metrics: cadence, swing/stance ratio, stride-time variability.
- Time-warp augmentation (deferred by D13).
- int8 quantization.
- Phone companion view.

## End-of-project deliverables

These happen only after sign-off.

1. Animation prompt with the real architecture, layer sizes and hyperparameters.
2. Crash course with quizzes.
3. A three-paragraph explanation: Julian drafts it, then we refine it together.

## Change log

- 2026-11-02: Plan approved.
- 2026-11-02 (M1): MPU internal rate changed to 1 kHz (SMPLRT_DIV = 0), so the MCU's 100 Hz reads never repeat or skip an internal sample. Relay publishes grouped messages 4/s because of the broker rate limit (D2).
- 2026-11-02 (M1): **The MCU can get stuck after App Lab re-flashes it.** The first runs got no data from the MCU, and afterwards neither did the known-good gait app. The arduino-router log shows the MCU's 37-character startup string ("MCU starting RPC Bridge communication", logged as 37 "invalid packet" errors) every time a sketch starts, and that string was missing on every failed run. So the sketch never reached `Bridge.begin()`. Most likely cause: the flash reset the MCU in the middle of an I2C read, and the MPU-6050 (which stays powered) holds SDA low, so the core hangs while setting up its devices, before `setup()`. After the MPU VCC jumper was unplugged, every later flash worked, with both the gait sketch and ours. The undefined symbols of both builds are identical (checked with nm), which rules out a sketch linking problem. **Recovery:** unplug MPU VCC for 2 s or power-cycle the board, then Run. `listen.py` prints this hint when MQTT works but no IMU data arrives.
- 2026-11-02 (M1 tilt test): L = +z points to the left (the inner side of the right ankle). Only the label changed; the data and axis map did not. The L axis reads about +0.2 g too high (0.20 and 0.25 g where it should read 0, and 0.84 where it should read 1.00).
- 2026-11-02 (M2): I2C clock raised to 400 kHz, so a re-flash is about 4× less likely to land mid-read. Recorder design: spoken cues (macOS `say`); bouts end automatically after 1.5 s of gyro stillness, so the laptop is never touched mid-session; step counts are confirmed at the end. Segment boundaries are seq ranges, with a +30-sample (0.3 s) lag correction for cues given on the laptop. Each protocol has a "walk to the start" transition after calibration. Dry runs are saved as T###, real sessions as S###. `STEPNET_BASE_TOPIC` gives tests a private MQTT topic.
- 2026-11-02 (M2 hardware): accelerometer calibration offsets x +0.029, y −0.006, z +0.190 g; scales 1.007/1.006/1.029; all six poses read 1.000 g after correction. Stride signature on this mount: **the L-axis (sagittal) gyro dips to about −150 to −300 deg/s once per right-foot stride**, which is the mid-swing event D6 labels. T001 found a recorder bug: the walking started during the spoken instruction, before "Go", so the bout segment missed its first strides. Bouts now start when the instruction starts.
- 2026-11-02 (M3): Label view. The server proposes candidates with a loose detector (L-gyro minima: height 60 deg/s, prominence 40, gap 0.35 s). The user accepts, rejects, nudges, snaps (to the L-gyro minimum within ±0.25 s) or adds strides (⌥-click). Saving is refused unless every bout and no-step block still in use passes |2·strides − steps| ≤ 1 with no undecided candidates. A block can be excluded instead (for example T001 #3, where the counted 10 steps don't match the ~16 in the signal). Labels are saved as `data/labels/<session>/v###.json`, a new version on every save. Candidates outside blocks are optional and dimmed. uPlot 1.6.31 is copied into `web/vendor/` (MIT).
- 2026-11-02: hallway measured at 16 ft = 4.88 m. It is saved as `hallway_m` in each session's meta (`record.py --hallway-m` to override). A 40-step walking bout covers about 6 hallway lengths, with about 5 turns.
- 2026-11-02: **Training and evaluation windows are drawn only from the counted part of bout and no-step blocks that are in use** (a bout from `go_seq`, see the next entry). Transitions and gaps between blocks are unlabeled (they may contain real strides with no count), so they are never treated as "no stride". Candidates there are optional in the labeler.
- 2026-11-02 (Guided session 1, first try): a stairs bout ended before the climb. Walking to the stairs during the spoken instruction counted as the bout starting, and the pause at the bottom then ended it. Fix: only movement after "Go" starts a bout, a bout needs at least 1 s of movement before stillness can end it, and the to_stairs transition is now 30 s and says to wait. Regression test: `tests/test_record.py`.
- 2026-11-02 (Guided session 1 review): 6 bouts failed the count check. In 4 of them the extra strides all came **before "Go"**: walking back to the hallway after the stairs, turning at the top of the stairs, or repositioning before a run. Counted from Go, they match Julian's counts exactly. Fixes: (1) record.py saves `go_seq` for each bout; (2) a bout's counted range, used by both the count check and training windows, is `[go_seq, end_seq)` (`stepnet.labels.counted_range`, mirrored in label.js), and the labeler shades the pre-Go part faintly and marks Go with a dashed line; (3) `scripts/add_go_seq.py` estimated `go_seq` for S001–S005 from the measured `say` length of each instruction + 0.2 s, flagged `go_seq_estimated` (T001/T002 predate bouts starting at the instruction, so their start_seq already is Go); (4) protocol A gets `turn_at_top` (8 s) and `to_hallway` (30 s) transitions. Of the other two failures, S003 stairs_down falls inside a 2.6 s MQTT gap (to be filled from the board log), and S003 shuffle is a real miscount (−2) and stays excluded.
- 2026-11-02 (M4): leave-one-session-out on S001–S005 (43 blocks), 95% bootstrap CIs over blocks. **B1:** F1 0.88, mean |bout error| 2.77 steps, 13.1 false strides/min in no-step blocks (59 on seated leg swings). It has to trade missing stairs and shuffles against counting leg swings, so its tuned parameters jump between folds. **B2:** F1 0.989, 0.23 steps/bout, 0.2 false/min, net −0.2%, with the same parameters in 4 of 5 folds (1.6 g impact, 0.4 s window). Its remaining errors are shuffle (−4 steps in one bout: gentle landings miss the impact gate), foot taps (2 false), and stride timing on stairs up. The oracle is 0 because every usable block passes the ±1 count check and the bouts have even targets. **Implication:** on protocol data B2 is near the floor, so D14's "beat B2 with non-overlapping CIs" is probably unreachable here, and the criterion needs revisiting (see the discussion with Julian).
- 2026-11-02: Julian chose options A + B: add protocol C (without adding sessions) and revise D14. Deadline is approaching, so the remaining scope will be trimmed if needed.
- 2026-11-02 (Guided session 3): S008's turning and side/back steps had no L-gyro dips, so the event definition was extended (D6). The labeler now shows |ω| (categorical slot 4), proposes |ω| peaks in counted blocks (6 Hz low-pass, height 80 deg/s, prominence 50, gap 0.35 s, dropped if within 0.2 s of an L candidate), snaps with `w` to the |ω| peak (`s` still snaps to the L minimum), and ⌥-click picks the more salient one. Label files now store the candidates that were shown, so new or unreviewed candidates reload as undecided instead of rejected. B2 still uses only the L gyro, so it is blind to these steps by design: that is where the network can win (D14).
- 2026-11-02: deadline is tomorrow, so the remaining order is M5 → M6 → M10 (final test) → slim M7 → deliverables; M8, M9 and M11 are cut. Test sessions are now locked in code (`real_sessions()` leaves out S009/S010 unless `include_test=True`). M4 re-run on the 8 training sessions: B2 F1 0.963, 1.33 steps/block, net −4.2% (protocol C made it harder).
