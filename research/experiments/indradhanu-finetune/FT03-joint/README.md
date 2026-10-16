# FT03 - Joint reconstruction + segmentation

**Status:** revised and launched 2026-10-17 on the user's go ("Go"); prepared 2026-10-16 with one-sensor batches,
FT02 init and seg_weight 1.0 - all three changed, below.

## Question
Trained together, does the segmentation head beat the frozen probe (FT01) while reconstruction and detection hold?

**Expected, written before launch:** water and cloud IoU at or above FT01's (0.82 / 0.80) - the encoder can now adapt
to the labels; thin cloud stays low (label ceiling, FT01 README); per-sensor reconstruction within 0.1 deg SAM of
FT02c; detection unchanged within noise (the bench has not separated any stage-2 model).

## Setup
| | |
|---|---|
| config | [config.json](config.json): `mode: joint`, `mixed_batches: true`, init FT02c `best.pt` (epoch 10), `seg_head_from` FT01 `best.pt`, `seg_weight 0.1`, `sam_ramp_epochs 0`, no edge term, 10 epochs, LR 1e-4 cosine, shares 0.40 / 0.35 / 0.25 |
| segmentation pass | `FinetuneTrainer._mixed_seg_loss`: one unmasked `segment_mixed` pass over the whole mixed batch (BatchNorm sees every sensor), masked BCE + Dice on the EnMAP part (the only labelled sensor), with the 2026-10-16 label policy |
| why seg_weight 0.1 | laptop smoke: segmentation loss ~0.4 against reconstruction ~0.05 (L1 + 0.5 SAM). At 1.0 segmentation would dominate the shared encoder; 0.1 makes them comparable, as the proposal's "MAE dominant" intends |
| why FT01's head | a trained start rather than random weights; FT01 learned on F001 features and the encoder has moved since, so it is a warm start, not a fit |
| validation | per-sensor reconstruction + EnMAP per-class IoU every epoch |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept; data staged |

## Pass criteria
1. Water and cloud IoU >= FT01 (0.82 / 0.80) on EnMAP test; thin cloud, shadow and snow reported only.
2. Per-sensor validation SAM within 0.1 deg of FT02c (3.93 / 4.21 / 5.41).
3. Detection not below FT02c beyond noise: PRISMA bench (all measures) and the implanted-target benchmark by tier.

## Results
## Observations
