# 59 - The scoreboard

> **The one thing this part teaches:** every Indradhanu version on every benchmark fits in four tables,
> and they read correctly only with chance and the model-free PCA-RX row beside them: pretraining was the
> big step, fine-tuning on the real sensors is what beats the classical detector, joint training and the
> cloud mask move the top of the list by a point or two, and three results are still unexplained.

## Where we are

Parts 43 to 58 introduced the models one run at a time, each judged on whatever benchmark existed that
day. This part puts them side by side, on 2026-10-18, so you can see the whole curve and know which
numbers to trust. Nothing here is new measurement; every number comes from one of these files:

| table | source |
|---|---|
| real PRISMA bench | `C:/es/eval/stage3/report.md`, `C:/es/eval/cloud_only/report.md`, `C:/es/eval/ft05_ep5/report.md` (dev laptop) |
| bench-v1 | `C:/es/eval/stage3/report.md`; `research/experiments/implant-bench/README.md` |
| bench-v2 | `C:/es/eval/v2/report.md`, `cloud_only/report.md`, `ft05_ep5/report.md`; the implant-bench README, "bench-v2" |
| reconstruction and masks | the run READMEs under `research/experiments/indradhanu-finetune/`; the FT05 README's Results (epoch 5) |

The models, in training order: **v0.2.0** (shipped; parts 01-30), **F001** epoch 109 (SpectralEarth
pretraining; part 43), **FT02c** epoch 10 (three sensors, mixed batches, edge term; parts 50-53), **FT03**
epoch 10 (joint with the segmentation head; part 55), **FT05** (implant-denoising; part 58). FT01 trained
only a head, and FT02 / FT02b are FT02c's ancestors (their bench-v1 rows are in part 56). Every row below
uses the default score, residual Mahalanobis (d = 30, window 21 / 61; parts 45-46), unless it says
otherwise.

> **Analogy.** A league table means little on its own. You also need the score a team gets by turning
> up and doing nothing (chance), and the score of the amateur side down the road (PCA-RX). A professional
> team that cannot beat the amateurs has not earned its wages, whatever its position.

## Table 1 - the real PRISMA bench (909 labelled objects, 4 scenes)

Chance: 10.2% of objects at the top 1%, 1.1% at the top 0.1% (anomaly course, part 19's analytic chance).
FT05's rows are from `C:/es/eval/ft05_ep5/report.md`, at epoch 5 of 10.

| model | caught top 1% | caught top 0.1% | pAUC to 1% | TPR at 0.1% | ROC-AUC |
|---|---|---|---|---|---|
| v0.2.0 | 44.2% | 14.1% | 0.159 | 0.058 | 0.854 |
| F001 ep109 | 44.0% | 13.2% | 0.148 | 0.056 | 0.861 |
| FT02c ep10 | 45.4% | 14.4% | 0.159 | 0.059 | 0.855 |
| FT03 ep10 | 45.4% | 14.3% | 0.158 | 0.058 | 0.853 |
| FT03, cloud excluded | 46.1% | 15.8% | **0.166** | 0.064 | 0.850 |
| PCA-RX (no model) | 39.7% | 8.1% | 0.128 | 0.029 | 0.839 |
| FT05 ep5 | 44.8% | 14.3% | 0.158 | 0.059 | 0.853 |
| FT05 ep5, cloud excluded | 45.8% | 15.6% | 0.166 | 0.066 | 0.850 |

Read it: every model is far above chance (`14.3 / 1.1 = 13` times at the top 0.1% for FT03) and above
PCA-RX, but **the models tie each other** - the four scenes cannot separate them (part 50's finding,
unchanged). The cloud mask is the only thing that moves the row, and it is not a model (anomaly course,
part 21).

## Table 2 - bench-v1 (3,336 implants, 18 held-out hosts) - superseded

Chance 10.0% / 1.1%.

| model | caught top 1% | caught top 0.1% | pAUC to 1% | easy pAUC |
|---|---|---|---|---|
| v0.2.0 | 19.0% | 7.7% | 0.123 | 0.257 |
| F001 ep109 | 25.2% | 9.5% | 0.163 | 0.342 |
| FT02c ep10 | 26.8% | 10.0% | 0.173 | 0.371 |
| FT03 ep10 | 26.9% | 9.9% | 0.172 | 0.371 |
| PCA-RX (no model) | **30.2%** | **10.5%** | **0.209** | **0.423** |

The models line up in training order, but PCA-RX beats them all - the opposite of Table 1. That is why
bench-v1 was replaced (part 57). Keep it for history; do not rank anything on it.

## Table 3 - bench-v2 (2,913 implants, 18 held-out hosts) - the implant benchmark now

Chance 10.1% / 1.1%. "host source" is the third of implants made from a pixel elsewhere in the same scene,
never used in FT05's training (part 58).

| model | caught top 1% | top 0.1% | pAUC to 1% | host source | EnMAP hosts | PRISMA hosts |
|---|---|---|---|---|---|---|
| v0.2.0 | 22.1% | 8.6% | 0.118 | 20.1% | 16.1% | 16.5% |
| F001 ep109 | 29.2% | 10.2% | 0.145 | 27.2% | 15.6% | 35.2% |
| FT02c ep10 | **32.8%** | 11.4% | **0.166** | 30.0% | 15.8% | **40.2%** |
| FT03 ep10 | 32.3% | 11.4% | 0.164 | 29.4% | 15.7% | 39.4% |
| FT03, cloud excluded | 33.4% | 12.1% | 0.169 | 30.3% | 18.6% | 39.8% |
| PCA-RX (no model) | 30.2% | 9.1% | 0.154 | 25.6% | 14.2% | 36.0% |
| FT05 ep5 | 32.8% | 11.4% | 0.167 | 29.6% | 16.0% | 39.5% |
| FT05 ep5, cloud excluded | 33.8% | 11.9% | 0.172 | 30.4% | 18.6% | 40.0% |

The fine-tuned models beat PCA-RX, as on the real bench - the test that let bench-v2 be used (anomaly
course, part 20). F001 does not. So **pretraining alone is not enough to beat the classical detector;
fine-tuning on the real sensors is.**

![Figure 59.1 - Caught at the top 1%, on three benchmarks](figures/59-scoreboard.svg)

*Figure 59.1 - Share of objects caught at the top 1% for each model and PCA-RX, on the real bench (grey),
bench-v1 (amber outline) and bench-v2 (amber). Chance is about 10% on all three. Note how PCA-RX's rank
changes between v1 and v2.*

## Table 4 - reconstruction and masks (the fine-tuning runs)

Per-sensor validation SAM in degrees (EnMAP / PRISMA / AVIRIS-NG, lower is better) and EnMAP test IoU for
water and cloud. Earlier models were validated differently (part 41), so their SAMs are not comparable
and are left out.

| run | SAM EnMAP | SAM PRISMA | SAM AVIRIS-NG | IoU water | IoU cloud |
|---|---|---|---|---|---|
| FT01 (head only, F001 frozen) | - | - | - | 0.82 | 0.80 |
| FT02b ep9 | 3.99 | 4.37 | 5.57 | - | - |
| FT02c ep10 | 3.93 | 4.21 | 5.41 | - | - |
| FT03 ep10 | 3.92 | 4.21 | 5.48 | 0.841 | 0.815 |
| FT05 ep5 | 3.94 | 4.20 | 5.33 | 0.838 | 0.818 |

FT05's epoch-5 figures meet its criterion 3 (part 58, section 6). On detection at epoch 5 it ties FT03 on
the real bench (criterion 1 not met: above on only one of its three measures) and is half a point ahead on
bench-v2, with the held-out `host` source not falling (criterion 2 met). Five epochs remain.

## What each step bought

Take FT03's caught-at-top-1% on bench-v2 apart, step by step:

```
v0.2.0 -> F001     22.1 -> 29.2   +7.1 points    pretraining (and see "unexplained" below)
F001   -> FT02c    29.2 -> 32.8   +3.6 points    three real sensors, mixed batches
FT02c  -> FT03     32.8 -> 32.3   -0.5 points    joint training: no change (a tie, below)
FT03   -> + cloud  32.3 -> 33.4   +1.1 points    a mask, not a model
FT03   -> FT05 ep5 32.3 -> 32.8   +0.5 points    implant-denoising, halfway (real bench: tie)
```

**Is -0.5 a real drop?** Treat each of the 2,913 objects as a coin with a one-in-three chance of being
caught. The standard error of a share `p` over `n` objects is `sqrt(p x (1 - p) / n)`:

```
p x (1 - p) = 0.33 x 0.67 = 0.2211
0.2211 / 2,913 = 0.0000759
sqrt(0.0000759) = 0.0087        about 0.9 points
```

A 0.5-point gap is about half a standard error: a tie. The +3.6 is about four. (Both models score the same
objects, so the true uncertainty of a difference is smaller than this; it is a rough ruler, not a test.)

## Three results nobody can explain yet

1. **EnMAP hosts do not move.** About 16% for every model and for PCA-RX, on bench-v1 and bench-v2 alike,
   while PRISMA hosts more than double. Cloud exclusion lifts them (15.7% -> 18.6%), so some EnMAP false
   alarms are cloud; the rest is open.
2. **v0.2.0 ties FT03 on real objects and trails it badly on implants** (44.2% vs 45.4% real; 22.1% vs
   32.3% bench-v2). One account says a reading artefact was ruled out. The only saved check
   (`C:/es/eval/resample_check/report.md`, 2026-10-17) says the opposite: v0.2.0 on the real bench through
   the generic linear resample that implant hosts get scored **15.8%** at the top 1% (pAUC 0.094) against
   44.2% through the product's band filter. If that check stands, v0.2.0's implant rows measure the
   resample, not the model. The other suggestion, that v0.2.0 trained on scenes from the bench sites, is
   unverified.
3. **bench-v2's per-tier foreignness.** Recomputed from the accepted bench's own `objects.json`, it sits
   1.19-1.40 times the real objects' (limit 1.25); the README's 0.90-1.15 comes from a calibration build
   no longer on disk (anomaly course, part 20).

## What would change the board

- **FT05's last epochs.** Criterion 1 is the real bench: not below FT03 on pAUC, TPR at 0.1% and caught top
  0.1%, above on two of the three. At epoch 5 it is above on one.
- **FT04**, a mask fine-tune, is proposed before cloud exclusion is wired into `backend/` - FT03 calls 11%
  of the desert scene set4 cloud.
- **Implants on water.** bench-v2 has none, though vessels are in scope; FT05 trains with them.

## Common confusions

**"FT02c is the best model."**
It leads bench-v2 by half a point, inside the noise, and ties FT03 on the real bench. FT03 is the stage-3
model because it also has the segmentation head.

**"Cloud exclusion is FT03's improvement."**
The mask is FT03's head, but the gain is the mask's: it lifts every model alike (part 57).

**"A model above chance is a good detector."**
Chance is the floor. PCA-RX is the bar: on the real bench every model clears it; on bench-v2 only the
fine-tuned ones do.

**"Table 4 shows FT05 is better than FT03."**
It shows reconstruction held. Detection - the point of FT05 - is in Tables 1 and 3: a tie on real objects at
epoch 5.

## Check yourself

1. FT03 catches 14.3% of 909 real objects at the top 0.1%, PCA-RX 8.1%. How many objects is that each, and
   how many times chance (1.1%) is each?
2. On bench-v2, how many more objects does FT03 catch at the top 1% than PCA-RX?
3. Using the standard-error ruler, is F001 -> FT02c (+3.6 points on 2,913 objects at p about 0.31) a real
   gain?
4. Which table would you use to decide whether FT05 passes, and which rows must it beat? Does its epoch-5
   row pass?
5. Why does Table 4 leave out v0.2.0's and F001's SAM?

<details>
<summary>Answers</summary>

1. FT03: `0.143 x 909 = 130`; PCA-RX: `0.081 x 909 = 73.6`, about 74. Times chance: `14.3 / 1.1 = 13.0`
   and `8.1 / 1.1 = 7.4`.
2. `(0.323 - 0.302) x 2,913 = 0.021 x 2,913 = 61.2`, about 61 objects.
3. `0.31 x 0.69 = 0.2139`; `/ 2,913 = 0.0000734`; `sqrt = 0.0086`, about 0.9 points. `3.6 / 0.9 = 4`
   standard errors: yes, well beyond noise.
4. Table 1, the real bench: FT03's row (pAUC 0.158, TPR at 0.1% 0.058, caught top 0.1% 14.3%) - not below
   on any, above on two. Table 3 is reported beside it and must show the `host` source not falling. At epoch
   5: pAUC tied (0.15801 against 0.15805), TPR above (0.059), caught top 0.1% equal (130 = 130) - above on one
   of three, so not yet.
5. They were validated with a different setup (different data and score; part 41), so the numbers are not
   on the same ruler as the fine-tuning runs' per-sensor validation.

</details>

---

That is the course so far. New results land first in the log of
`research/experiments/indradhanu-finetune/README.md`, the FT05 README, and
`research/experiments/implant-bench/README.md`.
