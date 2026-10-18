# FT04 - the final joint run: masks on all three sensors

**Status:** prepared 2026-10-18 on the user's go ("Go"). Curriculum stage 4 (docs/proposals/fine-tuning-and-segmentation.md).

## Question
FT03's segmentation head learned only from EnMAP's labels. Taught also from its own confident calls on PRISMA and
AVIRIS-NG, with rare classes weighted up, do the masks improve - and does detection, with cloud exclusion using
FT04's own masks, hold or improve?

**Expected, written before launch:** EnMAP IoU about unchanged (water / cloud ~0.84 / 0.82); shadow up from the 3x
weight; detection on the real bench unchanged with the plain score and at least as good with cloud exclusion on
FT04's masks as on FT03's.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT03's (joint, mixed batches, seg_weight 0.1, SAM 0.5 throughout), init FT03 `best.pt`, 10 epochs, `seg_class_weights` [1, 1, 1, 3, 3] (water, cloud, thin cloud, shadow, snow), 2 single-thread loader workers per source |
| pseudo-labels | `scripts/pseudo_label.py` with FT03 over the PRISMA and AVIRIS-NG training shards -> `<tile key>.npy` (5 x 128 x 128 uint8 probabilities); the loader keeps confident pixels only (p > 0.9 or < 0.1), teaches water and cloud only, and turns a confident water with NIR (800-900 nm) > 0.15 into a known not-water (`targets.pseudo_targets`) |
| checked before launch | on 150 EnMAP tiles, the guarded pseudo-labels against EnMAP's own labels: water IoU 0.875; the teacher's raw confident cloud 0.82. A bright-and-flat cloud guard was tried and dropped: it rejected 69% of labelled clouds, and set4's "11% cloud" it targeted is haze, not desert (README log, 2026-10-18) |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept; training data staged |

## Pass criteria (fixed before launch)
1. **Masks:** EnMAP test IoU not below FT03 by more than 0.01 (water 0.841, cloud 0.815).
2. **Detection (real PRISMA bench):** plain residual Mahalanobis not below FT03 (pAUC 0.158, TPR @ 0.1% 0.058, caught
   top 0.1% 14.3%); cloud exclusion with FT04's masks at least as good as with FT03's (pAUC 0.166).
3. **Reconstruction:** per-sensor validation SAM within 0.1 deg of FT03 (3.92 / 4.21 / 5.48).

## Results
## Observations
