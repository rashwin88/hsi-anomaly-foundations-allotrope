# Bench resolution - which differences on the real PRISMA bench are real?

**Status:** done 2026-10-21 on the laptop. Tool: `app/evaluation/significance.py` + `scripts/compare_scorers.py`
(paired bootstrap over labelled objects from an evaluation's cached score maps; 1000 draws).

## Question
The real bench is 4 PRISMA scenes and 909 objects; the gains we chase are ~0.005-0.01 pAUC. Which of the differences
measured so far would survive resampling the objects?

## Method
Resample each scene's objects with replacement - the same draw for both scorers (paired) - and recompute
mean-over-scenes pAUC <= 1% and objects caught; the 2.5-97.5 percentiles of B - A are its 95% interval. Background
is not resampled. Scene-level uncertainty (4 scenes) is not covered and is larger: this interval is a lower bound on
the real uncertainty. Point values reproduce the evaluator's (0.1621 / 0.1695, 14.9% / 15.8%).

## Results (A -> B, B - A, 95% interval)

| comparison | pAUC <= 1% | caught top 0.1% |
|---|---|---|
| FT04 RM -> + cloud excluded | 0.162 -> 0.170, **+0.0074 [+0.0047, +0.0101]** | 14.9 -> 15.8%, **+1.0 pts [+0.4, +1.7]** |
| FT04 (cloud excl.) -> gated adaptation N = 50 | 0.170 -> 0.160, **-0.0091 [-0.0176, -0.0028]** | -0.7 pts [-1.7, +0.2] |
| FT04 (cloud excl.) -> FT06 residual / sqrt b | 0.170 -> 0.163, -0.0067 [-0.0174, +0.0037] | +0.8 pts [-0.4, +2.0] |

The scoring-gains comparisons (A1+) could not be rerun: that run's cached score maps are keyed by the ScorerSpec JSON
of the time, and fields added since (scale_power, scale_floor_pct) change every key (docs/lld/evaluation.md).

## Observations
- **Resolution on objects: about +-0.005 to +-0.01 pAUC and +-1 to +-1.5 points of objects caught at the top 0.1%**
  (interval half-widths). A single-change gain below ~0.005 pAUC is indistinguishable from object sampling noise on
  this bench, before scene-level noise is counted.
- **Cloud exclusion is a real gain**, interval clear of zero on both measures - the one adopted change so far.
- **Gated adaptation's loss is real** (pAUC interval clear of zero).
- **FT06's sqrt variant is a tie**: neither its pAUC loss nor its top-0.1% gain is resolved.
- Consequence for pass criteria: "+0.005 pAUC" is at the edge of what the bench can resolve. Future runs report the
  bootstrap interval against FT04 alongside the point values, and a gain counts only if its interval clears zero.
