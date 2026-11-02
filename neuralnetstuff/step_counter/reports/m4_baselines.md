# M4: Classical baselines (leave-one-session-out)

Generated 2026-11-02 18:01 by `scripts/evaluate_baselines.py`. 8 sessions (S001, S002, S003, S004, S005, S006, S007, S008), 64 usable blocks, 578 labeled strides. Each session's numbers come from parameters tuned on the other sessions. Values: point estimate [95% bootstrap CI over blocks].

| metric | B1 peak detection | B2 + heel-strike gate | oracle (labels) |
|---|---|---|---|
| stride F1 | 0.854 [0.752, 0.933] | 0.963 [0.929, 0.988] | 1.000 [1.000, 1.000] |
| recall | 0.867 [0.783, 0.934] | 0.943 [0.894, 0.981] | 1.000 [1.000, 1.000] |
| precision | 0.842 [0.685, 0.988] | 0.984 [0.963, 0.998] | 1.000 [1.000, 1.000] |
| mean abs count error per bout (steps) | 3.69 [1.83, 5.86] | 1.33 [0.42, 2.46] | 0.05 [0.00, 0.13] |
| net count error, all bouts (%) | -12.5 [-20.7, -5.8] | -4.2 [-8.5, -1.0] | +0.0 [-0.3, +0.3] |
| false strides / min in no-step blocks | 12.7 [0.2, 25.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## Per activity (held-out)

| activity | kind | blocks | B1 abs err / bout | B2 abs err / bout | B1 false/min | B2 false/min | B1 recall | B2 recall |
|---|---|---|---|---|---|---|---|---|
| stand | no-step | 14 | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0.00 |
| walk_slow | bout | 6 | 1.00 | 0.00 | 0.0 | 0.0 | 0.97 | 1.00 |
| walk_normal | bout | 11 | 0.36 | 0.00 | 0.2 | 0.0 | 0.99 | 1.00 |
| walk_fast | bout | 3 | 0.00 | 0.00 | 0.9 | 0.0 | 0.98 | 1.00 |
| run | bout | 3 | 0.00 | 0.67 | 0.0 | 1.1 | 1.00 | 1.00 |
| stairs_up | bout | 3 | 9.33 | 0.00 | 1.3 | 3.8 | 0.58 | 0.92 |
| stairs_down | bout | 3 | 2.00 | 0.00 | 2.7 | 0.0 | 0.86 | 1.00 |
| fidget_standing | no-step | 3 | 0.67 | 0.00 | 0.7 | 0.0 | 0.00 | 0.00 |
| leg_swing_seated | no-step | 3 | 58.67 | 0.00 | 58.5 | 0.0 | 0.00 | 0.00 |
| foot_tap_seated | no-step | 2 | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0.00 |
| sit_still | no-step | 3 | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0.00 |
| shuffle | bout | 1 | 30.00 | 4.00 | 0.0 | 0.0 | 0.00 | 0.87 |
| potter | counted | 2 | 8.50 | 4.50 | 0.0 | 0.0 | 0.64 | 0.80 |
| side_and_back | counted | 2 | 15.50 | 8.50 | 0.0 | 6.2 | 0.12 | 0.24 |
| walk_very_slow | bout | 2 | 3.00 | 2.00 | 0.0 | 0.0 | 0.85 | 0.90 |
| sit_to_stand | counted | 2 | 0.00 | 0.00 | 0.0 | 0.0 | 0.00 | 0.00 |
| turn_in_place | counted | 1 | 16.00 | 16.00 | 0.0 | 0.0 | 0.00 | 0.00 |

## Per session: net count error over all bouts (held-out)

| session | counted steps | B1 | B2 | oracle |
|---|---|---|---|---|
| S001 | 238 | -16 (-6.7%) | +0 (+0.0%) | +0 (+0.0%) |
| S002 | 70 | +58 (+82.9%) | +0 (+0.0%) | +0 (+0.0%) |
| S003 | 238 | -14 (-5.9%) | +0 (+0.0%) | +0 (+0.0%) |
| S004 | 70 | +62 (+88.6%) | +0 (+0.0%) | +0 (+0.0%) |
| S005 | 268 | -42 (-15.7%) | -2 (-0.7%) | +0 (+0.0%) |
| S006 | 50 | -26 (-52.0%) | -18 (-36.0%) | +0 (+0.0%) |
| S007 | 72 | +58 (+80.6%) | +0 (+0.0%) | +0 (+0.0%) |
| S008 | 50 | -46 (-92.0%) | -28 (-56.0%) | +0 (+0.0%) |

## Parameters

Tuned on all five sessions (for later live use):

- B1: `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`
- B2: `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`

Per fold (tuned without that session):

- S001: B1 `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S002: B1 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 80, "min_gap_s": 0.5}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S003: B1 `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S004: B1 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 80, "min_gap_s": 0.5}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S005: B1 `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S006: B1 `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S007: B1 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 80, "min_gap_s": 0.5}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`
- S008: B1 `{"cutoff_hz": 8.0, "height_dps": 180, "prominence_dps": 20, "min_gap_s": 0.3}`, B2 `{"cutoff_hz": 3.0, "height_dps": 40, "prominence_dps": 20, "min_gap_s": 0.5, "impact_g": 1.6, "impact_window_s": 0.6}`

![per activity](m4_per_activity.png)

![heel-strike gate](m4_heel_strike_gate.png)
