# M10: Final test on the locked sessions (run once)

Generated 2026-11-02 18:43 by `scripts/final_test.py --final`. Test sessions: S009, S010 (16 usable blocks). Ensemble `cnn_ensemble_v1` (5 CNNs, trained on S001, S002, S003, S004, S005, S006, S007, S008), conformal q90 = 1.64. B2 tuned on the same training sessions.

## Scripted blocks (12 blocks)

| metric | ensemble | B2 | ensemble − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.993 [0.975, 1.000] | 0.972 [0.910, 1.000] | 0.021 [-0.016, 0.085] |
| mean abs count error per block with steps | 0.00 [0.00, 0.00] | 1.78 [0.00, 4.80] | -1.78 [-4.80, 0.00] |
| net count error (%) | +0.0 [+0.0, +0.0] | -2.8 [-13.3, +3.8] | +2.8 [-3.8, +13.3] |
| false strides / min | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## Hard subset (4 blocks)

| metric | ensemble | B2 | ensemble − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.754 [0.500, 0.867] | 0.721 [0.286, 0.867] | 0.032 [0.000, 0.214] |
| mean abs count error per block with steps | 0.50 [0.00, 1.50] | 4.50 [0.00, 9.00] | -4.00 [-9.00, 0.00] |
| net count error (%) | -2.9 [-10.0, +0.0] | -25.7 [-60.0, +0.0] | +22.9 [+0.0, +60.0] |
| false strides / min | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## All blocks (16 blocks)

| metric | ensemble | B2 | ensemble − B2 (paired 95% CI) |
|---|---|---|---|
| stride F1 | 0.947 [0.872, 0.992] | 0.928 [0.832, 0.984] | 0.019 [-0.011, 0.071] |
| mean abs count error per block with steps | 0.15 [0.00, 0.50] | 2.62 [0.44, 5.14] | -2.46 [-5.00, -0.33] |
| net count error (%) | -0.6 [-2.0, +0.0] | -7.3 [-19.1, +1.2] | +6.7 [-1.3, +18.3] |
| false strides / min | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## D14 verdict

| criterion | result | evidence |
|---|---|---|
| 1. no worse than B2 on scripted blocks | ✅ pass | count error diff CI [-4.80, 0.00], false/min diff CI [0.00, 0.00] |
| 2. better than B2 on the hard subset | ❌ fail | count error diff -4.00 CI [-9.00, 0.00] |
| 3. calibrated 90% intervals | ✅ pass | 16/16 = 100% covered, 95% CI [81%, 100%] |

## Per block: counted steps vs the ensemble's 90% interval

| session | block | kind | counted | estimate ± 90% | inside? |
|---|---|---|---|---|---|
| S009 | #3 stand | no-step | 0 | 0 ± 0.8 | ✓ |
| S009 | #4 walk_slow | bout | 30 | 30 ± 2.2 | ✓ |
| S009 | #5 walk_normal | bout | 40 | 40 ± 2.1 | ✓ |
| S009 | #6 walk_fast | bout | 40 | 40 ± 2.0 | ✓ |
| S009 | #7 run | bout | 40 | 40 ± 2.3 | ✓ |
| S009 | #8 shuffle | bout | 30 | 30 ± 4.5 | ✓ |
| S009 | #10 stairs_up | bout | 24 | 24 ± 1.7 | ✓ |
| S009 | #12 stairs_down | bout | 24 | 24 ± 2.7 | ✓ |
| S009 | #14 walk_normal_2 | bout | 40 | 40 ± 2.1 | ✓ |
| S009 | #15 stand | no-step | 0 | 0 ± 0.8 | ✓ |
| S010 | #3 potter | counted | 20 | 18 ± 4.4 | ✓ |
| S010 | #4 turn_in_place | counted | 20 | 20 ± 5.3 | ✓ |
| S010 | #6 walk_very_slow | bout | 20 | 20 ± 2.9 | ✓ |
| S010 | #8 sit_to_stand | counted | 0 | 0 ± 0.8 | ✓ |
| S010 | #10 walk_normal | bout | 30 | 30 ± 2.0 | ✓ |
| S010 | #11 stand | no-step | 0 | 0 ± 0.8 | ✓ |

## Per session totals (usable blocks)

| session | counted | ensemble | B2 |
|---|---|---|---|
| S009 | 268 | 268 (+0.0%) | 272 (+1.5%) |
| S010 | 90 | 88 (-2.2%) | 60 (-33.3%) |

## Post-hoc note (written after the result; the verdict above stands)

Hard test blocks, per block (descriptive only, nothing retuned):

| block | counted | ensemble | B2 |
|---|---|---|---|
| S009 #8 shuffle | 30 | 30 | 30 |
| S010 #3 potter | 20 | 18 | 14 |
| S010 #4 turn_in_place | 20 | 20 | 8 |
| S010 #8 sit_to_stand | 0 | 0 | 0 |

The ensemble is as good or better on every hard block. Criterion 2 failed because only 4 hard blocks survived labeling, and 2 of them are exact ties. A bootstrap resample of 4 blocks consists only of ties with probability (2/4)^4 = 6.25%, which is more than the 2.5% tail, so the CI's upper bound is exactly 0 whatever the model does on the other two. With this test set, criterion 2 was not passable as written. Lesson for next time: power the hard subset (more hard blocks in the test sessions), or pre-register a test that suits tiny samples (e.g. a paired sign test).
