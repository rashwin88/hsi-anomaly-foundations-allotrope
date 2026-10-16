# FT03 - Joint reconstruction + segmentation

**Status:** done 2026-10-16 19:49 - 22:48 UTC (3.0 h, ~$8), code ea4652c, exit 0, best.pt = epoch 10. Revised and launched on the user's go ("Go"); prepared 2026-10-16 with one-sensor batches,
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

Per epoch (EnMAP test IoU; per-sensor validation SAM deg):

| epoch | water | cloud | thin cloud | shadow | SAM EnMAP / PRISMA / AVIRIS-NG |
|---|---|---|---|---|---|
| 1 | 0.819 | 0.785 | 0.063 | 0.133 | 4.03 / 4.49 / 6.27 |
| 4 | 0.833 | 0.811 | 0.059 | 0.174 | 3.96 / 4.30 / 5.80 |
| 7 | 0.838 | 0.802 | 0.102 | 0.185 | 3.94 / 4.21 / 5.51 |
| **10 (best.pt)** | **0.841** | **0.815** | 0.108 | 0.205 | **3.92 / 4.21 / 5.48** |

Criterion 1 (water / cloud >= 0.82 / 0.80): **met**. Criterion 2 (SAM within 0.1 deg of FT02c 3.93 / 4.21 / 5.41):
**met** (AVIRIS-NG +0.07; it was +0.46 at epoch 6 and recovered). Snow 0.0 throughout (1 test tile).

Criterion 3, detection, residual Mahalanobis - **met** (unchanged from FT02c):

| | implant bench: all / easy / medium caught top 1% | easy pAUC | medium pAUC | PRISMA bench: pAUC <= 1% | caught top 1% / 0.1% |
|---|---|---|---|---|---|
| FT02c ep10 | 27% / 50% / 21% | 0.371 | 0.124 | 0.159 | 413 / 131 |
| **FT03 ep10** | 27% / 50% / 21% | 0.371 | 0.121 | 0.158 | 413 / 130 |

**Verdict: passed all three.** FT03 is the stage-3 model: the segmentation head improved on the frozen probe (water
0.841, cloud 0.815) while reconstruction and detection held.

## Observations

- Joint training changed detection by nothing measurable, as expected with seg_weight 0.1: the head learned on top of
  an encoder that kept reconstructing as before. The head's value comes in stage 5 - masks for `scene_segmentation`
  and class-aware scoring - not from joint training lifting detection by itself.
- Thin cloud rose 0.06 -> 0.11 and shadow 0.13 -> 0.21 as the encoder adapted; both remain low (label quality, rarity).
