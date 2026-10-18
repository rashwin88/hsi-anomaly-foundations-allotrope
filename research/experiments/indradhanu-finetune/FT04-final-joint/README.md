# FT04 - the final joint run: masks on all three sensors

**Status:** done 2026-10-18 21:38 UTC, code 25c4069, exit 0, best.pt = epoch 10. **Passed all three criteria.** Launched on the user's go ("Go"). Curriculum stage 4 (docs/proposals/fine-tuning-and-segmentation.md).

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

**Masks and reconstruction (epoch 10):** IoU water 0.839, cloud 0.810 (FT03 0.841 / 0.815 - within 0.01), shadow
0.223 (0.205), thin cloud 0.110; SAM 3.88 / 4.11 / 5.46 deg (FT03 3.92 / 4.21 / 5.48 - better on all three).

**Detection** (`configs/eval/ft04.json`, C:/es/eval/ft04), real PRISMA bench:

| | pAUC <= 1% | TPR @ 0.1% | caught top 0.1% | caught top 1% | ROC-AUC |
|---|---|---|---|---|---|
| v0.2.0, default (stage-3 run) | 0.159 | 0.058 | 14.1% | 44.2% | 0.854 |
| FT03, default | 0.158 | 0.058 | 14.3% | 45.4% | 0.853 |
| FT04, default | 0.162 | 0.061 | 14.9% | 45.7% | 0.853 |
| FT03 + cloud excluded (FT03 masks) | 0.166 | 0.064 | 15.8% | 46.1% | 0.850 |
| FT04 + cloud excluded (FT04 masks) | **0.170** | **0.065** | 15.8% | 46.0% | 0.852 |

bench-v2.1: FT04 35.1% caught top 1% (FT03 34.9%), with cloud exclusion 36.2% (36.0%).

Criteria: 1 met; 2 met (default at or above FT03 on all three measures; exclusion with FT04's masks as good as with
FT03's, with less ROC-AUC cost - its masks exclude less); 3 met.

## Observations

- **Release candidate: FT04 + cloud exclusion with its own masks.** Against the shipped v0.2.0 on the real bench:
  pAUC 0.159 -> 0.170, TPR @ 0.1% 0.058 -> 0.065. Small, on 4 scenes; most of it is the exclusion (the model alone
  +0.003 pAUC). On bench-v2.1 v0.2.0 and the fine-tuned models tie.
- Pseudo-labels did not cost EnMAP IoU; the 3x rare-class weight lifted shadow by 0.02.
