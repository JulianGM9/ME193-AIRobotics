# M5: CNN vs B2 (leave-one-session-out on the training sessions)

Generated 2026-11-02 18:17 by `scripts/train.py --loso`, run `20261102_180135_cnn_loso`, dataset `006899f3fae0`. Each held-out session is scored with a CNN trained, and a B2 tuned, on the other sessions. Values: point estimate [95% bootstrap CI over blocks]. The **hard subset** is every protocol C counted block plus shuffles and foot taps (D14).

## All blocks (64 blocks)

| metric | CNN | B2 | CNN − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.968 [0.940, 0.990] | 0.963 [0.929, 0.988] | 0.005 [-0.007, 0.021] |
| mean abs count error per block with steps | 0.26 [0.05, 0.49] | 1.33 [0.42, 2.46] | -1.08 [-2.15, -0.23] |
| net count error (%) | +0.2 [-0.6, +1.0] | -4.2 [-8.5, -1.0] | +4.3 [+1.1, +8.9] |
| false strides / min (no-step blocks) | 0.3 [0.0, 0.6] | 0.0 [0.0, 0.0] | 0.3 [0.0, 0.6] |

## Scripted (not hard) (54 blocks)

| metric | CNN | B2 | CNN − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.987 [0.973, 0.998] | 0.991 [0.977, 0.999] | -0.004 [-0.010, 0.002] |
| mean abs count error per block with steps | 0.13 [0.00, 0.33] | 0.19 [0.00, 0.42] | -0.06 [-0.21, 0.00] |
| net count error (%) | +0.0 [-0.6, +0.6] | -0.2 [-0.9, +0.4] | +0.2 [+0.0, +0.7] |
| false strides / min (no-step blocks) | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.5] |

## Hard subset (10 blocks)

| metric | CNN | B2 | CNN − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.818 [0.600, 0.955] | 0.692 [0.353, 0.886] | 0.127 [0.023, 0.373] |
| mean abs count error per block with steps | 0.75 [0.00, 1.60] | 5.75 [2.50, 9.43] | -5.00 [-8.89, -1.75] |
| net count error (%) | +1.5 [-3.6, +8.2] | -35.4 [-65.9, -16.4] | +36.9 [+15.6, +68.8] |
| false strides / min (no-step blocks) | 1.0 [0.0, 2.0] | 0.0 [0.0, 0.0] | 1.0 [0.0, 2.0] |

## Per activity

| activity | kind | blocks | CNN abs err | B2 abs err | CNN false | B2 false |
|---|---|---|---|---|---|---|
| stand | no-step | 14 | 0.00 | 0.00 | 0 | 0 |
| walk_slow | bout | 6 | 0.00 | 0.00 | 0 | 0 |
| walk_normal | bout | 11 | 0.00 | 0.00 | 2 | 0 |
| walk_fast | bout | 3 | 0.00 | 0.00 | 0 | 0 |
| run | bout | 3 | 0.67 | 0.67 | 1 | 1 |
| stairs_up | bout | 3 | 0.00 | 0.00 | 3 | 3 |
| stairs_down | bout | 3 | 0.00 | 0.00 | 0 | 0 |
| fidget_standing | no-step | 3 | 0.00 | 0.00 | 0 | 0 |
| leg_swing_seated | no-step | 3 | 0.67 | 0.00 | 1 | 0 |
| foot_tap_seated | no-step | 2 | 1.00 | 0.00 | 1 | 0 |
| sit_still | no-step | 3 | 0.00 | 0.00 | 0 | 0 |
| shuffle | bout | 1 | 2.00 | 4.00 | 0 | 0 |
| potter | counted | 2 | 1.50 | 4.50 | 2 | 0 |
| side_and_back | counted | 2 | 0.50 | 8.50 | 7 | 5 |
| walk_very_slow | bout | 2 | 1.00 | 2.00 | 0 | 0 |
| sit_to_stand | counted | 2 | 0.00 | 0.00 | 0 | 0 |
| turn_in_place | counted | 1 | 0.00 | 16.00 | 3 | 0 |

## Training loss per fold

| held out | first epoch | last epoch |
|---|---|---|
| S001 | 1.006 | 0.521 |
| S002 | 1.106 | 0.606 |
| S003 | 0.990 | 0.508 |
| S004 | 1.090 | 0.592 |
| S005 | 0.925 | 0.479 |
| S006 | 1.052 | 0.594 |
| S007 | 1.083 | 0.594 |
| S008 | 1.022 | 0.556 |

## Config

```
name: cnn
data:
  window: 256
  target_sigma: 5.0
  windows_per_epoch: 3000
  rotate_deg: 10.0
  scale: 0.1
model:
  channels:
  - 16
  - 32
  - 32
  - 32
  - 32
  dilations:
  - 1
  - 2
  - 4
  - 8
  - 16
  kernel: 7
  dropout: 0.1
train:
  epochs: 30
  batch: 64
  lr: 0.001
  weight_decay: 0.0001
  pos_weight: 20.0
  seed: 0
detect:
  threshold: 0.5
  min_gap_s: 0.3
  hop: 64
```
