# FT06 - the uncertainty head: score the error the model did not expect

**Status:** done 2026-10-20 11:02 UTC, exit 0, best.pt = epoch 3. **Failed criterion 2 (detection); not adopted.** Ran on `i-0a3afdd4efbcafe92` (code bundle 5d634bf = 6ae345c's working tree plus the
head). Launched on the user's "Yeah start training." Design: [docs/proposals/uncertainty-head.md](../../../../docs/proposals/uncertainty-head.md).

## Question
Residual Mahalanobis (RM) whitens with one covariance per scene, so a large error where the model is always
unsure (field edges, texture, glint, shoreline) looks as surprising as the same error where it is usually right.
Score-side fixes did not help (research/experiments/scoring-gains/). Can the model predict its own expected error
b(p) at each hidden pixel - and does RM on residual / b find more real objects?

**Expected, written before launch:** the head learns a clearly spatial b (Spearman with the actual error 0.5-0.7:
edges and texture high, flat land and water low). Detection: a small gain on the real bench (pAUC +0.005 to +0.015);
objects caught roughly unchanged; most of the gain on medium and subtle objects.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT04's (mixed batches, same sources and shares), `mode: "scale"`, `scale_decoder_dim` 64, init FT04 `best.pt`, 3 epochs, lr 5e-4 cosine |
| trained | the scale head only (~0.1 M parameters); FT04's backbone, reconstruction and segmentation frozen and in eval mode (BatchNorm statistics kept) |
| loss | Laplace NLL on hidden, valid pixels: mean of e/b + log b, e = mean over valid bands of FT04's \|normalised residual\| (detached) |
| validation | per sensor: NLL and Spearman(b, e) on hidden pixels (`finetune_metrics.csv`: `<sensor>_val_scale_nll`, `<sensor>_val_scale_spearman`) |
| scoring | `scaled_rm` (app/evaluation/scorers.py): RM on residual / b, b floored at its 5th percentile per scene |

## Pass criteria (fixed before launch)
1. **The head predicts its own error:** validation Spearman(b, e) >= 0.5 on every sensor.
2. **Detection (real PRISMA bench), against FT04 + RM, both with cloud excluded on FT04's masks:** pAUC <= 1% at
   least +0.010 (0.170 -> 0.180), TPR @ 0.1% not lower, **and objects caught at top 1% and 0.1% not lower**.
3. **Guard (bench-v2.1):** caught top 1% not lower by more than 0.5 points.

Reconstruction and masks cannot change (the backbone is frozen); checked by FT04's numbers reproducing under
`scaled_rm`'s sibling `residual_mahalanobis` on the FT06 checkpoint.

## Results

**The head predicts its own error** (criterion 1, met): validation Spearman(b, e) per epoch -

| epoch | EnMAP | PRISMA | AVIRIS-NG | val NLL |
|---|---|---|---|---|
| 1 | 0.584 | 0.620 | 0.607 | -1.331 |
| 2 | 0.600 | 0.631 | 0.625 | -1.338 |
| 3 | 0.604 | 0.636 | 0.640 | -1.341 (best) |

**Detection** (`configs/eval/ft06.json`, C:/es/eval/ft06, full report [eval_report.md](eval_report.md)), real PRISMA
bench, everything with cloud excluded on FT04's masks:

| | pAUC <= 1% | TPR @ 0.1% | caught top 1% | caught top 0.1% | ROC-AUC |
|---|---|---|---|---|---|
| FT04 + RM (the bar) | **0.170** | 0.065 | **46.0%** | 15.8% | **0.852** |
| FT06 + RM | 0.170 | 0.065 | 46.0% | 15.8% | 0.852 |
| FT06 + scaled RM (residual / b, as designed) | 0.125 | 0.064 | 37.8% | 13.2% | 0.708 |
| FT06 + scaled RM, residual / sqrt(b) | 0.163 | **0.072** | 44.8% | **16.6%** | 0.805 |
| *epoch 1* + residual / sqrt(b) | 0.176 | 0.079 | 46.1% | 17.3% | 0.822 |

bench-v2.1 caught top 1% (cloud excluded): FT04 36.2%; FT06 scaled 33.9%, sqrt 36.2%.

FT06 + RM reproduces FT04 exactly: the backbone, reconstruction and masks are untouched (frozen, as designed).

## Observations

- **Criterion 2 failed; not adopted.** As designed (residual / b) the score loses a quarter of its pAUC and 8 points of
  objects. The softer residual / sqrt(b) - **chosen after seeing epoch-1 results on the real bench**, so it carries
  selection risk - lifts the very top (TPR @ 0.1% 0.065 -> 0.072, caught top 0.1% 15.8% -> 16.6%) but loses overall
  (pAUC 0.170 -> 0.163, caught top 1% 46.0% -> 44.8%).
- **More training, worse detection.** Epoch 1 with sqrt scaling looked like a gain (pAUC 0.176, caught top 0.1%
  17.3%); epoch 3 is below FT04. The head got better at predicting its error (Spearman up every epoch) and worse for
  detection - the two are not the same goal.
- **Why: the head expects large errors exactly where targets are** ([b_at_targets.py](b_at_targets.py), sets 1, 2, 4):
  target pixels sit at the 87th-90th percentile of b across the scene (built-up surfaces - the same finding as the
  mixture-of-backgrounds scorer in research/experiments/scoring-gains/), and **b on a target exceeds b on its own
  3-8 px surroundings** - ratio 1.05 / 1.12 / 1.04 at epoch 1, 1.07 / 1.19 / 1.13 at epoch 3. Hiding the pixel's 4 x 4
  token does not hide the target: its PSF halo and any part larger than one token are visible in the neighbouring
  tokens (the FT02c caveat), so the head learns "something odd is next to here, expect a big error" - and dividing
  by b explains the anomaly away. set2, where the ratio is largest (1.19), loses most (pAUC 0.128 -> 0.097).
- Expectation was a small gain mostly on medium / subtle objects; the opposite happened (subtle tier caught top 1%
  40.3% -> 37.7% with sqrt).
- What might rescue it, not tried: hide a wider context than the pixel's token when training the head (e.g. 3 x 3
  tokens around each predicted one), so the head sees no halo; or predict b from the reconstruction branch's
  features of a *heavily* masked pass only. Both cost a retrain; neither is recommended before a cheaper idea.
- Evidence files here: `finetune_metrics.csv` (per-epoch Spearman / NLL), `eval_per_scene.csv` (per-scene pAUC, incl.
  set2 0.128 -> 0.097), `eval_epoch1_variants.md` (the epoch-1 sqrt row), `b_at_targets_output.txt` (the percentile and
  ratio figures; the ring is 4-8 dilation steps, city-block distance).
- Practical: the evaluation ran ~2.3 h, not ~1 h - two models plus the scale pass filled the laptop GPU's 20 GB and
  the last AVIRIS-NG hosts ran ~8x slower in shared memory.
