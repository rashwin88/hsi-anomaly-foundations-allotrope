# Gated scene adaptation - learn this scene's background, not its targets

**Status:** done 2026-10-20 on the laptop (C:/es/eval/scene_adapt, ~1.5 h). **Failed all criteria; not adopted.** Started on the user's "Gating is something we can use" (after the
GT-HAD discussion). Code: `app/foundation_models/scene_adaptation.py`, evaluator `ModelSpec.adapt_*`.

## Question
Every way of learning "normal" from a scene's context so far also learned its targets (mixture of backgrounds,
research/experiments/scoring-gains/; the uncertainty head, FT06). GT-HAD, reproduced on our four PRISMA scenes
(research/experiments/gt-had-prisma/), trains on the scene itself and was best after 5 epochs rather than 150 - the
network learns the common background before the rare targets - and gates likely anomalies out of what it learns.

Does FT04, adapted briefly to each scene with the pixels that already look anomalous left out of the loss, find more
real objects?

## Method
Per scene, starting from FT04 each time:
1. score with FT04 + residual Mahalanobis (RM);
2. **gate**: the top 1% of that score, dilated by 2 px (the halo FT06 showed leaks a target into its neighbours), is
   left out of the loss;
3. adapt N steps: the fine-tuning loss (L1 + 0.5 SAM on hidden pixels, 65% token masking), batches of 8 random
   128 px tiles from the scene, eval mode (stored BatchNorm statistics), segmentation head out of the loss;
4. reconstruct, score RM with cloud excluded on **unadapted** FT04's masks; restore FT04's weights.

Hyperparameters are chosen without ground truth: the learning rate from the adaptation loss on one scene
(`adapt_probe.py`); N by **leave-one-scene-out** - for each scene, N is the one with the best mean pAUC on the other
three - so no scene's number comes from settings tuned on it. Variants: N = 50, 200, 1000 gated; N = 200 ungated
(gate 0%) to isolate the gate.

**Learning rate (probe, set1, no ground truth):** 300 steps, 1.6 steps/s on the laptop GPU; gate covers 4.55% of the
scene (top 1% + 2 px halo); adaptation loss first 50 / steps 100-150 / last 50 steps -

| lr | first 50 | 100-150 | last 50 |
|---|---|---|---|
| 1e-5 | 0.0495 | 0.0409 | 0.0389 |
| 3e-5 | 0.0459 | 0.0382 | 0.0368 |
| 1e-4 | 0.0461 | 0.0365 | 0.0354 |

Chosen 3e-5: as fast as 1e-4 early, a third of the step size - the aim is to learn background without reaching the
targets. Config: [configs/eval/scene_adapt.json](../../../configs/eval/scene_adapt.json).

## Pass criteria (fixed before any detection result)
Against FT04 + RM with cloud excluded (0.170 / 46.0% / 15.8%), real PRISMA bench, leave-one-scene-out headline:
1. pAUC <= 1% at least +0.005;
2. objects caught at top 1% and at top 0.1% not lower;
3. guard: bench-v2.1 caught top 1% not lower by more than 0.5 points at the chosen N.

**Expected, written before the run:** a gain that peaks at small N and decays as N grows (GT-HAD's curve); ungated
adaptation worse than gated at the same N, because it learns the targets.

## Results

Real PRISMA bench ([eval_report.md](eval_report.md), per scene in [eval_per_scene.csv](eval_per_scene.csv)), cloud
excluded on unadapted FT04's masks:

| | pAUC <= 1% | caught top 1% | caught top 0.1% | TPR @ 0.1% |
|---|---|---|---|---|
| FT04 (the bar) | **0.170** | 46.0% | **15.8%** | **0.065** |
| gated, N = 50 | 0.160 | 46.8% | 15.2% | 0.059 |
| gated, N = 200 | 0.159 | 46.9% | 14.9% | 0.059 |
| gated, N = 1000 | 0.161 | 46.5% | 15.0% | 0.057 |
| ungated, N = 200 | 0.159 | 46.9% | 14.9% | 0.059 |

pAUC per scene (set1 / set2 / set3 / set4): FT04 0.246 / 0.128 / 0.084 / 0.220; N = 50 0.223 / 0.142 / 0.084 / 0.192;
N = 1000 0.240 / 0.139 / 0.083 / 0.181.

Leave-one-scene-out (each scene scored at the N with the best mean pAUC on the other three): picks N = 50 / 1000 /
1000 / 1000 for set1-4, headline pAUC **0.157** vs FT04's 0.170 (computed from eval_per_scene.csv).
bench-v2.1 (criterion 3) was not run.

## Observations

- **Failed; not adopted.** Adaptation costs ~0.01 pAUC and 0.6-0.9 points of objects at the top 0.1%, at every N; it
  gains ~0.5-0.9 points at the top 1% - a shift of hits from the very top into the top 1%, not more finds.
- **The gate made no difference.** Gated and ungated N = 200 agree to three decimals on every measure. Adaptation is
  not learning the targets (they are < 0.1% of pixels and 1000 steps of a 5.9 M-parameter model barely see them); it
  changes the residual for everything. Expectation 2 (ungated worse) was wrong because the failure mode it guards
  against never happened.
- **No GT-HAD curve.** The damage is already there at N = 50 and does not grow with N - no early window. GT-HAD's
  5-epoch optimum was relative to its own from-scratch training on the scene; it never reached FT04 (its PD at 1e-3
  was 5.5% at best vs FT04's 6.5%). A pretrained model starts past the point that early stopping finds.
- set2 is the one scene that gains at every N (0.128 -> 0.139-0.142); set1 and set4 lose. Four scenes cannot say whether
  that is signal.
- Expectation 1 (peak at small N then decay) was wrong: flat in N, below the baseline.
