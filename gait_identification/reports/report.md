# Gait classifier evaluation

Training sessions: [2, 3] (659 windows). Held-out test session: 4 (329 windows).

## Model 1 vs Model 2 (held-out session)

| Model | k | Raw accuracy | Smoothed accuracy |
|---|---|---|---|
| Model 1 - peak shin accel (1 feature) | 13 | 91.5% | 93.9% |
| Model 2 - full IMU (18 features) | 1 | 99.1% | 99.1% |

## Per-class precision / recall (raw)

| Class | M1 precision | M1 recall | M2 precision | M2 recall | test windows |
|---|---|---|---|---|---|
| standing | 100% | 100% | 100% | 100% | 83 |
| walking | 77% | 94% | 98% | 99% | 82 |
| running | 100% | 96% | 99% | 98% | 82 |
| biking | 93% | 76% | 100% | 100% | 82 |

Confusion matrices: `confusion_model1.png`, `confusion_model2.png`.

## Model 1: which axis? (D6, D11)

| Axis | Peak (max) accuracy | 95th-percentile accuracy |
|---|---|---|
| S | 91.5% | 64.1% |
| F | 84.5% | 77.5% |
| L | 79.0% | 71.1% |

## Model 2: drop one feature group at a time (D10)

| Feature group removed | Accuracy | Change |
|---|---|---|
| per-channel std | 99.1% | +0.0% |
| per-channel range | 98.8% | -0.3% |
| gyro means | 99.7% | +0.6% |
| accel magnitude | 98.8% | -0.3% |
| cadence | 98.8% | -0.3% |

## Posture check: lying still on your back (D10)

Never used for training. Should come out as **standing** (no motion), not biking.

| Model | standing | walking | running | biking |
|---|---|---|---|---|
| Model 1 | 100% | 0% | 0% | 0% |
| Model 2 | 100% | 0% | 0% | 0% |

(68 lying-still windows from all sessions.)

## Leave-one-session-out across all sessions (k fixed at 5)

| Test session | Model 1 | Model 2 |
|---|---|---|
| 2 | 76.6% | 95.4% |
| 3 | 80.0% | 99.7% |
| 4 | 86.9% | 99.7% |
