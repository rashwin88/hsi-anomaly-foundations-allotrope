# 63 - The uncertainty head, run: it learned its error and lost detection

> **The one thing this part teaches:** FT06 trained part 62's uncertainty head on frozen FT04 and
> the head did learn to predict its own error (Spearman about 0.6 on every sensor), yet dividing the
> residual by that prediction found fewer real objects - because the head expects large errors exactly
> on targets, so predicting your own error well and detecting anomalies turn out to be different goals.

## Where we are

**Status: run 2026-10-20 as FT06, failed criterion 2 (detection), not adopted.** Part 62 taught the
design while it was still a proposal. This part teaches what happened when it ran. Indradhanu V2
(FT04, part 61) is unchanged and still the model to use.

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-finetune/FT06-uncertainty/README.md` | setup, pass criteria, results, observations - the record |
| `research/experiments/indradhanu-finetune/FT06-uncertainty/eval_report.md` | the full evaluation tables, by tier and sensor |
| `research/experiments/indradhanu-finetune/FT06-uncertainty/b_at_targets.py` | the diagnosis script: where the head puts its expected error |
| `docs/proposals/uncertainty-head.md` | the design; its status line now reads "failed detection, not adopted" |
| `app/foundation_models/trainers/finetune_trainer.py` | `_scale_loss` and `spearman` - mode `"scale"` |
| `app/evaluation/scorers.py` | `scaled_rm`, with `scale_power` and `scale_floor_pct` |

> **Analogy.** Part 62's forecaster judged a downpour by how far it missed their forecast, scaled by
> how big a miss they expected. Now suppose the forecaster is never told about your town, only about
> the towns around it - that is the hidden-pixel rule. A storm parked over your town still spills
> rain and wind into every neighbour. The forecaster reads the neighbours, says "something big is
> going on near there, give or take a lot", and is quite right. But the storm no longer looks
> surprising, because the forecaster expected a big miss in exactly that place.

## What was trained

FT06 is a fine-tuning run in a new trainer mode, `mode: "scale"` (`docs/lld/finetune.md`). Its
config (`FT06-uncertainty/config.json`) is FT04's - the same mixed batches of EnMAP, PRISMA and
AVIRIS-NG at shares 0.4 / 0.35 / 0.25 - with four changes:

| setting | value | what it does |
|---|---|---|
| `mode` | `"scale"` | trains the uncertainty head and nothing else |
| `scale_decoder_dim` | 64 | builds the head: `SegFormerDecoder(..., decoder_dim=64, out_channels=1)`, the 95,632 parameters you counted in part 62 |
| `init_from` | FT04's `best.pt` | starts from Indradhanu V2's weights |
| `num_epochs` | 3 | three passes, learning rate 5e-4 on a cosine schedule |

**Frozen means frozen.** In `FineTuneTrainer.__init__`, mode `"scale"` sets `requires_grad` to
`True` only for parameter names starting with `scale_head.`, and builds the optimiser over those
alone. `_train_pass_mode` then puts the whole model in evaluation mode and switches only
`scale_head` back to training mode, so FT04's BatchNorm statistics (part 10) are not updated either.

The loss is `_scale_loss`. Its docstring states it in two lines:

```
e(p) = mean over valid bands of |normalised residual|  (detached: the target, not something to shrink)
loss = mean over hidden p of  e(p) / b(p) + log b(p)    - lowest where b(p) = e(p)
```

The head outputs `log b`, which the loss clamps to the range -10 to 5 before using it, and the loss
is averaged only over pixels that were hidden from the encoder, valid, and eroded away from invalid
edges. Each validation batch also adds the head's NLL and `spearman(lb[hidden], e[hidden])` to a
per-sensor tally that lands in `finetune_metrics.csv` as `<sensor>_val_scale_spearman`.

**A definition you need now: Spearman correlation.** Rank both lists from smallest to largest, then
measure how well the two rank orders agree: 1 means the same order, 0 means unrelated, -1 means
reversed. The repo's `spearman` does exactly that: two `argsort` calls turn values into ranks, the
ranks are centred on their mean, and the result is their dot product divided by the product of their
lengths. It asks "does a larger `b` go with a larger error?", not "is `b` the right size?".

Tiny example, three hidden pixels. Do it on paper:

```
b = 0.10, 0.30, 0.20      ranks 0, 2, 1      centred -1, +1,  0
e = 0.05, 0.20, 0.40      ranks 0, 1, 2      centred -1,  0, +1

dot product      (-1)(-1) + (1)(0) + (0)(1) = 1 + 0 + 0 = 1
lengths          sqrt(1 + 1 + 0) = 1.414     sqrt(1 + 0 + 1) = 1.414
product          1.414 x 1.414 = 2.0
Spearman         1 / 2.0 = 0.5
```

At inference, `predict_full_scene_with_scale` (`app/foundation_models/inferencers/spectral_stem_inferencer.py`)
appends one zero channel to the scene as a placeholder, runs the ordinary tiled two-mask pass of
part 29 with the model's `with_scale` flag on, and splits the last output channel off as `log b`. So
each pixel's `b`, like its reconstruction, comes from the pass in which it was hidden - the same
condition as training.

## Result 1: the head learned its error

Criterion 1 asked for validation Spearman(b, e) of at least 0.5 on every sensor. From the README:

| epoch | EnMAP | PRISMA | AVIRIS-NG | val NLL |
|---|---|---|---|---|
| 1 | 0.584 | 0.620 | 0.607 | -1.331 |
| 2 | 0.600 | 0.631 | 0.625 | -1.338 |
| 3 | 0.604 | 0.636 | 0.640 | -1.341 (best) |

**Met, on every sensor, from the first epoch, and rising every epoch.** The NLL falls each epoch
too, and because mode `"scale"` ranks epochs by that NLL (`docs/lld/finetune.md`: "scale by its
NLL"), `best.pt` is epoch 3. Keep that in mind: "best" here means best at predicting its own error.

## Result 2: detection got worse

The evaluation (`configs/eval/ft06.json`) scored the real PRISMA bench with cloud excluded on FT04's
masks, against FT04 + residual Mahalanobis (RM, parts 45-46). The scorer `scaled_rm` in
`app/evaluation/scorers.py` does three things:

```
b = exp(log_b)
b = max(b, 5th percentile of b over valid pixels) ** scale_power    # scale_floor_pct = 5 by default
score = RM on (cube - recon) / b                                     # whitening refitted on the scaled residual
```

`scale_power = 1.0` is the design (residual / b); `scale_power = 0.5` is residual / sqrt(b).

| | pAUC <= 1% | TPR @ 0.1% | caught top 1% | caught top 0.1% | ROC-AUC |
|---|---|---|---|---|---|
| FT04 + RM (the bar) | **0.170** | 0.065 | **46.0%** | 15.8% | **0.852** |
| FT06 + RM | 0.170 | 0.065 | 46.0% | 15.8% | 0.852 |
| FT06 + scaled RM (residual / b, as designed) | 0.125 | 0.064 | 37.8% | 13.2% | 0.708 |
| FT06 + scaled RM, residual / sqrt(b) | 0.163 | **0.072** | 44.8% | **16.6%** | 0.805 |
| *epoch 1* + residual / sqrt(b) | 0.176 | 0.079 | 46.1% | 17.3% | 0.822 |

Read it row by row:

- **FT06 + RM equals FT04 + RM to every digit.** That is the frozen-backbone check working, not a
  bug: the reconstruction is FT04's, so plain RM on it is FT04's score.
- **As designed, residual / b loses a quarter of its pAUC** (0.170 to 0.125) and 8.2 points of
  objects caught at the top 1% (46.0% to 37.8%).
- **The sqrt variant lifts the very top and loses overall.** TPR at 0.1% goes 0.065 to 0.072 and
  caught top 0.1% 15.8% to 16.6%, but pAUC drops to 0.163 and caught top 1% to 44.8%.
- On bench-v2.1, caught top 1%: FT04 36.2%, scaled 33.9%, sqrt 36.2%.
- Part 62 expected most of any gain on medium and subtle objects. The opposite happened: the subtle
  tier's caught top 1% went 40.3% to 37.7% with sqrt (`eval_report.md`, "by tier").

Criterion 2 needed pAUC at least 0.180, TPR at 0.1% not lower, and caught top 1% and 0.1% not lower.
No row meets all of it. FT06 is not adopted.

## Why: the head expects large errors exactly where targets are

`b_at_targets.py` asks where the head puts its expected error, on sets 1, 2 and 4, for epoch 1 and
epoch 3. For each scene it takes the target pixels, and a ring around them - pixels within 8 dilation
steps of a target but beyond 3 (`binary_dilation(t, iterations=8) & ~binary_dilation(t, iterations=3)`),
which the README calls "3-8 px". It reports two numbers:

1. **the percentile of `b` at targets**, ranked against every valid pixel in the scene;
2. **the target / ring ratio**: `exp(median log b on target - median log b on ring)`, which is the
   median `b` on the target divided by the median `b` on its own surroundings.

The README's findings:

| | set1 | set2 | set4 |
|---|---|---|---|
| target / ring `b` ratio, epoch 1 | 1.05 | 1.12 | 1.04 |
| target / ring `b` ratio, epoch 3 | 1.07 | 1.19 | 1.13 |

and target pixels sit at the **87th-90th percentile** of `b` across the scene. Two separate facts,
and both hurt:

- **Across the scene,** targets sit on built-up surfaces, which are hard to reconstruct, so the head
  rightly expects large errors there. That is the same finding as the mixture-of-backgrounds scorer
  in `research/experiments/scoring-gains/` (anomaly course, part 24).
- **Locally,** `b` on the target is higher than on the target's own ring. That should not happen if
  the head only judged "how hard is this kind of place": the ring is the same kind of place.

**Why the hidden-pixel rule did not stop it.** Part 62's rule hides the pixel from the encoder. But
the encoder hides whole 4 x 4 tokens (part 53's caveat 1: stage 1 is non-overlapping, kernel 4,
stride 4), and a target is not confined to its token. Its PSF halo - the sensor's blur that spreads
some of a target's light into neighbouring pixels (anomaly course, part 20) - and any part of a
target larger than one token sit in the *neighbouring* tokens, which are visible. The head learns
"something odd is next to here, expect a big error". It is right about that. And dividing by `b`
then explains the anomaly away.

**The sentence to remember: a head trained to predict its own error will predict a large error on
an anomaly whenever it can see any trace of it, and then the anomaly is no longer surprising.**

### Dividing by `b`, by hand

Use set2's epoch-3 ratio, 1.19. Keep it to one whitened band and ignore that the whitening is
refitted (it rescales every pixel alike, so it does not change the order). RM's score is a squared
distance (`local_mean_score`: `||x - ring mean||^2`); take the ring mean as 0.

A target pixel with residual 3.0; a background pixel in its ring with residual 1.0; and a background
pixel elsewhere, on a flat field, with residual 2.6. Give the ring and the flat field `b = 1.00`, so
the target has `b = 1.19`.

```
plain RM
  target         3.0 ^ 2 = 9.00
  ring pixel     1.0 ^ 2 = 1.00
  flat field     2.6 ^ 2 = 6.76         target ranks first

residual / b
  target         3.0 / 1.19 = 2.521     2.521 ^ 2 = 6.36
  ring pixel     1.0 / 1.00 = 1.000     1.000 ^ 2 = 1.00
  flat field     2.6 / 1.00 = 2.600     2.600 ^ 2 = 6.76    the flat field now outranks the target
```

The target's score against its own ring fell from 9.00 to 6.36: divided by `1.19 x 1.19 = 1.416`,
about 29% lower. Worse, it fell below an ordinary bright pixel elsewhere, which is how objects drop
out of the top 1%. With sqrt the target becomes `3.0 / sqrt(1.19) = 3.0 / 1.091 = 2.750`, squared
7.56 - still above 6.76 here, which is why the sqrt variant hurts less. And set2, where the ratio is
largest, lost the most: pAUC 0.128 to 0.097 (README).

## Why epoch 1 looked better than epoch 3

The ratios grew with training: set1 1.05 to 1.07, set2 1.12 to 1.19, set4 1.04 to 1.13. As the head
got better at its own job (Spearman up every epoch), it also learned more of the "something odd next
door" signal, because that signal genuinely predicts large errors. More of it means more of each
target divided away. So epoch 1 with sqrt scored pAUC 0.176 and caught 17.3% at the top 0.1%, and
epoch 3, the `best.pt` by NLL, fell below FT04.

**This is the central lesson.** Predicting your own error well is not the same goal as detection.
The checkpoint chosen by the head's own loss was the worst for the product.

### The selection-risk caveat

Do not read the epoch-1 row as a gain to ship. The sqrt variant was **chosen after seeing epoch-1
results on the real bench**, and then judged on the same bench. When you try variants and keep the
one that looks best on the data that will also grade it, some of the improvement is luck in that
data, and you cannot tell how much. The README flags exactly this. And even taken at face value,
epoch 1's pAUC of 0.176 is short of the 0.180 bar.

## What might rescue it - not tried

The README lists two ideas, both untried:

1. **Hide a wider context when training the head** - for example 3 x 3 tokens around each predicted
   one, so the head never sees a target's halo or its other parts.
2. **Predict `b` from the reconstruction branch's features of a heavily masked pass only**, so there
   is little of the target left in view.

Both cost a retrain. The README recommends neither before a cheaper idea. One practical note: the
evaluation took about 2.3 hours rather than the expected 1, because two models plus the scale pass
filled the laptop GPU's 20 GB.

## Common confusions

**"Spearman met its bar, so the head works."**
It works at its own goal. Criterion 1 was a sanity check: a head that cannot predict its error is
useless. Passing it was necessary, not sufficient, and the detection criterion is the one that
counts.

**"The hidden-pixel rule should have prevented this."**
It prevents the head from seeing the pixel. It does not stop the head from seeing the neighbouring
tokens, which carry the target's halo and any part larger than one 4 x 4 token.

**"FT06 + RM matching FT04 to every digit means the evaluation is broken."**
It is the check that the backbone stayed frozen. Different numbers there would have been the bug.

**"The 1.19 is a percentile."**
No. 1.19 is a ratio of medians, target `b` over ring `b`. The 87th-90th percentile is a separate
measure, of target `b` against the whole scene.

**"`best.pt` is the best model."**
It is the epoch with the lowest scale NLL, which mode `"scale"` uses to rank epochs. Here that was
the worst epoch for detection.

**"So ship epoch 1 with sqrt."**
Selection risk, and it is still below the 0.180 pAUC bar.

## Check yourself

1. In the three-pixel Spearman example, swap the last two values of `e` to `0.40, 0.20`. What is
   Spearman now?
2. Redo the hand example with set1's epoch-3 ratio, 1.07. What is the target's score, and does it
   still outrank the flat-field pixel at 6.76?
3. With ratio 1.19, by what factor does residual / b shrink a target's squared score against its
   ring? And residual / sqrt(b)?
4. Why does FT06 + RM give exactly FT04's numbers?
5. Spearman rose every epoch while detection fell. Name the mechanism.
6. Why is the epoch-1 sqrt row not evidence of a gain?

<details>
<summary>Answers</summary>

1. `b` ranks 0, 2, 1, centred -1, +1, 0. `e = 0.05, 0.40, 0.20` ranks 0, 2, 1, centred -1, +1, 0.
   Dot product `1 + 1 + 0 = 2`. Lengths `1.414 x 1.414 = 2.0`. Spearman `2 / 2.0 = 1.0`: the same
   order.
2. `3.0 / 1.07 = 2.804`, and `2.804 ^ 2 = 7.86`. That is still above 6.76, so the target keeps its
   place in this example - consistent with set1 losing less than set2.
3. Residual / b: `1.19 x 1.19 = 1.416`, so the score is divided by 1.416 (about 29% lower).
   Residual / sqrt(b): the residual is divided by `sqrt(1.19) = 1.091`, so the squared score is
   divided by `1.091 x 1.091 = 1.19` (about 16% lower).
4. Mode `"scale"` trains only `scale_head.` parameters and keeps the rest in evaluation mode, so the
   reconstruction is FT04's and plain RM on it is FT04's score.
5. The target / ring ratio of `b` grew with training (for example set2 1.12 to 1.19): the head
   learned more of the "something odd next door" signal, which predicts errors well and divides
   targets away.
6. The sqrt variant was picked after seeing epoch-1 results on the same bench that grades it, so part
   of its lead may be luck in that bench; and its pAUC of 0.176 is below the 0.180 bar anyway.

</details>

---

That is the course so far. The uncertainty head is closed; V2 (part 61) stands, and the README's
two rescue ideas wait behind any cheaper one.
