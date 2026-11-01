# N001b - spectral namer, level 1 by matching (closed set)

N001 with one change: `level1: "matching"` - natural library spectra are class C in the class scores and
P(man-made) = sigmoid(logsumexp_{c<C} S_c - S_natural + b) (model.level1_logit). Why: [N001](../N001/README.md)
(linear level-1 head at chance after epoch 1). Config: [config.json](config.json) (= `configs/namer/N001b.json`).
Bundle `allotrope-f4f3abd-namer3`; started 2026-10-31 11:59 UTC on the block-2 H100 (i-0ff678b9ba1eeab92); outputs
on S3 under experiments/indradhanu-finetune/spectral-namer/N001b/ (the GPU role's write prefixes). Bank (EnMAP /
PRISMA / AVIRIS-NG): 378/338/380 spectra, 159/149/160 man-made, 34/32/34 held out. 347,685 parameters.

## ACE baseline on the v1 classes

The separability check scored the 17 old families. Re-scored from its confusion tables onto the 7 classes (roofing
counted as concrete/asphalt - the tables do not split it by entry), level-2 class accuracy over planted man-made
spectra, and in brackets only where ACE also said man-made:

| sensor | fill 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| EnMAP | 0.24 (0.50) | 0.47 (0.70) | 0.63 (0.79) | 0.72 (0.82) |
| PRISMA | 0.41 (0.67) | 0.60 (0.78) | 0.69 (0.81) | 0.72 (0.83) |
| AVIRIS-NG | 0.38 (0.64) | 0.58 (0.76) | 0.67 (0.80) | 0.70 (0.81) |

The namer's level-2 accuracy chooses among the 7 man-made classes only, so the fair comparison lies between the two
ACE figures. ACE was scored leave-one-out over all spectra, the namer on the held-out 20% - not the same spectra.

## Results

Finished 2026-10-31 21:12 UTC, 40 epochs (~14 min each), EXIT 0. Checkpoints `ep1..ep40.pt`, `best.pt` (= epoch 15,
chosen on mean level-2 accuracy at fill 0.25, which was flat from epoch 5) and `last.pt` (epoch 40) on S3 under
experiments/indradhanu-finetune/spectral-namer/N001b/. **Use epoch 40**: level 1 kept improving after epoch 15 while
the selection metric did not move.

**Level 1: man-made recall at ACE's natural false-alarm rate** (epoch 40; ACE in brackets):

| sensor | fill 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| EnMAP | 0.65 (0.48) | 0.86 (0.66) | 0.90 (0.80) | 0.93 (0.87) |
| PRISMA | 0.85 (0.61) | 0.94 (0.77) | 0.98 (0.85) | 0.99 (0.87) |
| AVIRIS-NG | 0.71 (0.59) | 0.90 (0.76) | 0.94 (0.84) | 0.97 (0.87) |

**Level 2: class accuracy on held-out spectra** (epoch 40; ACE on the v1 classes, all man-made - conditional):

| sensor | fill 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| EnMAP | 0.59 (0.24 - 0.50) | 0.68 (0.47 - 0.70) | 0.70 (0.63 - 0.79) | 0.71 (0.72 - 0.82) |
| PRISMA | 0.68 (0.41 - 0.67) | 0.69 (0.60 - 0.78) | 0.69 (0.69 - 0.81) | 0.69 (0.72 - 0.83) |
| AVIRIS-NG | 0.65 (0.38 - 0.64) | 0.70 (0.58 - 0.76) | 0.72 (0.67 - 0.80) | 0.73 (0.70 - 0.81) |

**Against the pass criteria (fill 0.25, +0.10 over ACE):**
- **Level 1: passes on all three sensors**, by +0.20, +0.17 and +0.14.
- **Level 2: passes against ACE's unconditional figure on EnMAP (+0.21) and AVIRIS-NG (+0.12); fails on PRISMA
  (+0.09).** It is below ACE's conditional figure everywhere.
- The namer's biggest margin is at low fill: at fill 0.1 it beats ACE's unconditional class accuracy by 0.27-0.35.
- Its class accuracy hardly changes with fill (0.68-0.73 from 0.25 to 1.0) - the limit is generalising to unseen
  spectra, which [N003](../N003/README.md) addresses.
- No bootstrap interval: the trainer logs aggregate metrics only, not per-sample predictions; an interval needs a
  re-evaluation of epoch 40 that keeps them.

### Intervals (2026-11-01, `eval_ep40_v2`: 95% bootstrap resampling library SPECTRA, 500 reps)

Epoch 40, re-evaluated with `scripts/namer_eval.py` (same protocol; ~300k samples, 221-254 distinct spectra per
sensor, ~34 of them unseen man-made). ACE is a point value (its own uncertainty is not included).

| sensor, fill | level-1 recall at ACE's FA | ACE | level-2 class accuracy | ACE (all man-made) |
|---|---|---|---|---|
| EnMAP 0.1 | 0.65 [0.55, 0.72] | 0.48 | 0.59 [0.48, 0.70] | 0.24 |
| EnMAP 0.25 | 0.86 [0.78, 0.92] | 0.66 | 0.68 [0.55, 0.79] | 0.47 |
| PRISMA 0.1 | 0.85 [0.78, 0.92] | 0.61 | 0.69 [0.53, 0.81] | 0.41 |
| PRISMA 0.25 | 0.94 [0.87, 0.98] | 0.77 | 0.70 [0.54, 0.82] | 0.60 |
| AVIRIS-NG 0.1 | 0.71 [0.63, 0.79] | 0.59 | 0.65 [0.53, 0.76] | 0.38 |
| AVIRIS-NG 0.25 | 0.90 [0.83, 0.94] | 0.76 | 0.70 [0.57, 0.82] | 0.58 |

- **Level 1 is better than ACE on every sensor and fill, interval included.** At fill 0.25 the lower bound clears ACE
  + 0.10 on EnMAP (0.78 vs 0.76) and PRISMA (0.87 vs 0.87, just); AVIRIS-NG's point estimate does (+0.14) but its
  lower bound (0.83) does not reach 0.86.
- **Level 2 is decisively better than ACE at fill 0.1** (lower bounds 0.48-0.53 against ACE 0.24-0.41). At fill 0.25
  it is better on EnMAP (lower bound 0.55 vs 0.47); on PRISMA and AVIRIS-NG the interval overlaps ACE. With ~34 unseen
  man-made spectra, the class test cannot resolve differences much below ~0.12.
