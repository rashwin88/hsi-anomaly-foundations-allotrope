# Scoring gains A1-A3 - better scores from the same reconstruction

**Status:** done 2026-10-19 on the laptop (C:/es/eval/scoring_gains, ~55 min, full report in [report.md](report.md)) (no training, no GPU instance). Started on the user's "Yes please" to
"Shall I start A1-A3 and write up the uncertainty-head design?". Config: [configs/eval/scoring_gains.json](../../../configs/eval/scoring_gains.json).

## Question
Residual Mahalanobis (RM) absorbs model gains: v0.2.0, FT03 and FT04 tie on detection once scored with it. Can the
*score* gain instead - with FT04's reconstruction fixed?

| | idea | why it might help | code |
|---|---|---|---|
| A1 | **spatial matched filter**: blur the residual (Gaussian, sigma = 1.0 / 1.4 px) over valid pixels before whitening | a target spans a PSF footprint (sigma ~1.4 on both sensors' benches); its residual adds coherently across the footprint while uncorrelated noise averages down | `spatial_sigma` |
| A1+ | rank mean of RM and RM blurred | keeps single-pixel targets that blurring dilutes | `fuse` |
| A2 | **RM + PCA-RX fusion** by rank (mean, max) | PCA-RX wins where the target is spectrally foreign to the scene (bench-v1), RM where it is foreign to the model; they fail on different objects | `fuse` |
| A3 | **mixture of backgrounds**: k-means (k = 6, 10) on the cube's whitened top-10 PCs, then RM whitened per type with a same-type ring | one global covariance mixes land, water and urban; each type's own normal is tighter | `mixture` |

## Pass criteria (fixed before the run)
A scorer is a **gain** if, against RM (baseline) on FT04:
1. real PRISMA bench: pAUC <= 1% at least +0.005 **and** TPR @ 0.1% FPR not lower;
2. bench-v2.1: objects caught at top 1% not lower by more than 0.5 points.

The real bench is the judge. bench-v2.1 implants are PSF-blurred with sigma 1.4 by construction, so A1's result there
is partly circular and counts only as a guard (criterion 2).

**Expected, written before the run:** A1 a small gain on the real bench for medium/large objects and a larger one on
bench-v2.1 (circular); A2 rank-mean a small gain, rank-max a loss (it adds both scorers' false alarms); A3 a gain on
scenes with several surfaces (set2/set4) and a near-tie elsewhere.

## Results

FT04, scores on the same reconstruction. Real PRISMA bench (4 scenes, 909 objects) and bench-v2.1 (18 hosts):

| scorer | real pAUC | real TPR @ 0.1% | real caught top 1% / 0.1% | v2.1 caught top 1% | criteria |
|---|---|---|---|---|---|
| RM (baseline) | 0.162 | 0.061 | **45.7% / 14.9%** | 35.1% | - |
| A1 RM blur 1.0 | 0.177 | 0.064 | 40.7% / 12.0% | 33.4% | 1 met, 2 failed |
| A1 RM blur 1.4 | 0.177 | 0.069 | 36.3% / 10.2% | 29.4% | 1 met, 2 failed |
| A1+ RM + RM blur 1.4 | **0.180** | 0.064 | 43.0% / 12.8% | 35.0% | **both met** - but see below |
| A2 RM + PCA-RX, rank mean | 0.162 | 0.057 | 43.8% / 14.0% | 34.9% | failed |
| A2 RM + PCA-RX, rank max | 0.138 | 0.046 | 42.7% / 11.4% | 34.8% | failed |
| A3 mixture k=6 | 0.111 | 0.045 | 39.4% / 11.4% | **41.2%** | failed (1) |
| A3 mixture k=10 | 0.096 | 0.034 | 34.4% / 8.1% | **41.9%** | failed (1) |
| PCA-RX (no model) | 0.128 | 0.029 | 39.7% / 8.1% | 33.2% | - |

Real-bench objects caught at top 1%, by object size (561 / 170 / 158 / 20 objects):

| scorer | 1-2 px | 3-5 px | 6-20 px | > 20 px |
|---|---|---|---|---|
| RM (baseline) | **36.5%** | **52.4%** | **66.5%** | 80.0% |
| A1 RM blur 1.0 | 33.3% | 45.3% | 57.6% | 75.0% |
| A1+ RM + RM blur 1.4 | 35.1% | 48.2% | 60.8% | 80.0% |
| A3 mixture k=6 | 32.6% | 43.5% | 54.4% | 75.0% |

## Observations

- **No scorer beats RM on the measure an analyst feels: distinct objects found.** RM catches the most real objects
  at both cut-offs and at every object size.
- **A1 (spatial matched filter) raises pixel measures and lowers object measures.** pAUC 0.162 -> 0.177 and ROC-AUC
  0.853 -> 0.891, yet objects caught fall at every size, even > 20 px. Blurring lights up more of the pixels of
  objects already found (pixel measures count each object pixel) and pulls isolated small targets and target edges
  down into their neighbours. A1+ (rank mean with plain RM) **formally meets both pre-set criteria** - pAUC +0.018,
  TPR not lower, bench-v2.1 -0.1 pt - **but loses 2.7 points of real objects at top 1% and 2.1 at top 0.1%.** The
  criteria were pixel-level and missed this; not adopted. Lesson: future pass criteria include real objects caught.
- **A2 (fusion with PCA-RX) adds nothing.** Rank mean ties on pAUC and loses objects; rank max adds both scorers'
  false alarms, as expected. PCA-RX's wins on bench-v1 do not carry to real targets.
- **A3 (mixture of backgrounds) is the bench-v2.1 winner and the real-bench loser** (v2.1 +6 points; real pAUC
  0.162 -> 0.111). Real targets are not spread over the scene: 60-70% of target pixels sit in two or three small
  background types (~5% of the scene each; set4: one type holds 60% of target pixels on 5.2% of the scene) - the
  built-up surfaces where targets live. Whitening such a type on its own pixels makes its targets normal. Implants
  are placed across all surfaces, so per-type whitening only sharpens them. **A second bias of bench-v2.1** (after
  bench-v1's cube-level oddity): implant placement is not conditioned on where real targets occur.
- Expectations: A1 "small gain on the real bench" - right on pixels, wrong on objects; A2 right; A3 wrong in
  direction (predicted a gain on multi-surface scenes).
- The run exposed a cache bug: float16 overflow on blurred / mixture squared distances (> 65,504) turned cached top
  scores into inf. Metrics of this run were computed before the cast and are unaffected; out-of-range maps are now
  cached float32 and the 13 overflowed cache files were deleted.

**Verdict: RM stays the score.** The gains question goes to the model side - B4, the uncertainty head
(docs/proposals/uncertainty-head.md), whose pass criteria now include real objects caught.
