# 62 - Next: an uncertainty head (a proposal, not yet run)

> **The one thing this part teaches:** scoring changes on FT04's fixed reconstruction all failed
> (2026-10-19), so the next idea moves into the model - a small third head that predicts, for each
> pixel, how large an error the model expects there, trained with a Laplace likelihood whose best answer
> is exactly that expected error; the score then divides the residual by it - and this is a design
> waiting for agreement, with no training run and no result yet.

**Update, 2026-10-20 - this proposal has been run.** It was trained as FT06 on frozen FT04 and **failed
detection; it is not adopted.** The head did learn its own error (Spearman about 0.6), but it expects large
errors on targets themselves, so dividing by `b` hides them. This part is kept as the design you study
first; part 63 teaches the run and its results.

## Where we are

**Status: proposed 2026-10-19, not built, not trained.** Everything below comes from
`docs/proposals/uncertainty-head.md` and describes intent. Nothing in this part is a measured result
except the parameter count, which was checked by building the head's decoder class.

Part 61 shipped FT04 as Indradhanu V2. Its detection ties v0.2.0, and the same day an experiment tried
to win detection back on the *score* side, with FT04's reconstruction held fixed: blurring the residual,
fusing residual Mahalanobis (RM, parts 45-46) with PCA-RX, and whitening each background type
separately. None beat plain RM on real objects caught; the anomaly-detection course's part 24 works
through why, and `research/experiments/scoring-gains/README.md` is the record. Its verdict: "the gains
question goes to the model side - B4, the uncertainty head".

| file | what it holds |
|---|---|
| `docs/proposals/uncertainty-head.md` | the design, pass criteria, risks, two decisions for you |
| `research/experiments/scoring-gains/README.md` | why the score-side ideas were dropped |
| `app/foundation_models/components/spectral_stem_seg_former_mae.py` | where the head would be added (beside `seg_head`) |

> **Analogy.** A weather forecaster who says "20 mm of rain, give or take 2" and one who says "20 mm,
> give or take 15" can both see 35 mm fall. For the first it is a shock; for the second it is a wet
> day. What makes an observation surprising is the miss divided by how big a miss the forecaster
> expected.

## The gap RM leaves

RM fits **one covariance per scene**. It learns which error *directions* the scene shares - haze,
striping, dark water - but treats every *place* alike. The model's errors are not alike: field edges,
texture, glint, thin cloud and shoreline are hard to reconstruct everywhere, and their large residuals
are RM's false alarms. RM has no way to know "the model always struggles here".

## The design in four lines

1. **Head.** A third `SegFormerDecoder` on the shared encoder features, beside reconstruction and
   segmentation, with `decoder_dim=64` and `out_channels=1`. It outputs `log b(p)`: one number per
   pixel, the log of the expected error size.
2. **Loss.** A Laplace negative log-likelihood (the probability model that matches the L1 loss of part
   23), on hidden pixels only, with the residual `r` detached (no gradient flows back into the
   reconstruction):

   ```
   L_scale = mean over masked p of [ mean_c |r_c(p)| / b(p) + log b(p) ]
   ```

3. **Training.** FT04 frozen, head only, 3 epochs on the FT04 mix, about 2.5 hours on the L40S. The
   reconstruction and segmentation are untouched, so V2's masks and parity stay exactly as shipped.
4. **Score.** `scaled_rm`: RM on `r(p) / b(p)`, whitening refitted on the scaled residual.

**Predict the head's size.** Part 47 counted the segmentation head at width 128 and asked you to redo
it at width 64. Do it with one output channel instead of five, so the last layer makes `1 x 16 = 16`
sub-pixel channels:

```
projections  32 x 64 + 64 =  2,112     64 x 64 + 64 =  4,160
            160 x 64 + 64 = 10,304    256 x 64 + 64 = 16,448
fuse        256 x 64 + 64 = 16,448
refine 1    64 x 64 x 9 + 64 = 36,864 + 64 = 36,928
refine 2    64 x 16 x 9 + 16 =  9,216 + 16 =  9,232
total       2,112 + 4,160 + 10,304 + 16,448 + 16,448 + 36,928 + 9,232 = 95,632
```

Building `SegFormerDecoder(embed_dims=[32, 64, 160, 256], decoder_dim=64, out_channels=1)` and counting
gives **95,632** - the proposal's "~0.1 M".

## Why the loss's best answer is the expected error

Take one pixel whose error is `|r| = 0.2` and try three values of `b`:

```
b = 0.1:  0.2 / 0.1 + ln 0.1 = 2.0 + (-2.303) = -0.303
b = 0.2:  0.2 / 0.2 + ln 0.2 = 1.0 + (-1.609) = -0.609
b = 0.4:  0.2 / 0.4 + ln 0.4 = 0.5 + (-0.916) = -0.416
```

The lowest loss is at `b = 0.2`, the error itself. Too small a `b` is punished by the first term, too
large by the second. Averaged over many pixels like this one, the optimum is `b(p) = E|r(p)|`: the
typical error size at that kind of place.

And the score: two pixels each with residual 0.3. At a field edge the head expects `b = 0.3`, so
`0.3 / 0.3 = 1.0` - ordinary. On a flat field it expects `b = 0.05`, so `0.3 / 0.05 = 6.0` - a
surprise. Plain RM would have treated them alike.

## The one rule that keeps it honest

`b` is learned only where the pixel was **hidden from the encoder** (part 17-19's masking). The head
must judge how hard a place is from its context, never from the pixel itself; otherwise it learns
"this pixel looks odd, so expect a big error" and divides the anomaly away. Inference already predicts
every pixel while it is hidden (two complementary masks, part 29), so inference matches training.

## Pass criteria, fixed before any run

Against FT04 + RM with cloud excluded on FT04's masks, on the real PRISMA bench: pAUC up to 1% false
alarms at least +0.010 (0.170 to 0.180), TPR at 0.1% not lower, **and real objects caught at the top
1% and 0.1% not lower** - the clause added because of the scoring-gains result. bench-v2.1 caught top
1% may fall by no more than 0.5 points. A sanity check comes first: on held-out tiles `b(p)` must rank
pixels by their actual error with Spearman correlation at least 0.5. If the head cannot predict its own
error, the score cannot use it.

The proposal lists three risks: real targets in the training shards (mitigated by the hidden-pixel
rule and rarity), `b` collapsing to a constant (caught by the Spearman check), and small `b` inflating
noise over deep water or flat desert (floor `b` at its 5th percentile per scene). Two decisions are
open: whether to start the GPU instance, and frozen backbone first (recommended) or joint training.

## Common confusions

**"The uncertainty head has shown a gain."**
No. It is a proposal. There is no checkpoint and no number.

**"This is the same as RM's whitening."**
RM divides by a per-scene spread along each error *direction*. `b(p)` is a per-*pixel* scale. The
proposal keeps the per-band shape with RM on purpose and rejects per-band scales.

**"Why not let the head see the pixel? It would predict the error better."**
It would predict anomalies' errors as large too, and dividing by that hides them.

**"Training the head will change V2's masks."**
Not in the recommended design: the backbone, reconstruction and segmentation heads are frozen.

## Check yourself

1. For `|r| = 0.5`, compute `L = |r| / b + ln b` at `b = 0.25, 0.5, 1.0`. Which is lowest?
2. What would the head's parameter count be with `out_channels = 2`? Change only the last layer.
3. A pixel has residual 0.4 and `b = 0.08`; another has residual 0.8 and `b = 0.4`. Which scores
   higher after scaling?
4. Why is the residual detached in the loss?
5. Which pass-criteria clause would A1+ from the scoring-gains run have failed?

<details>
<summary>Answers</summary>

1. `b = 0.25`: `2.0 + (-1.386) = 0.614`. `b = 0.5`: `1.0 + (-0.693) = 0.307`. `b = 1.0`:
   `0.5 + 0 = 0.5`. Lowest at `b = 0.5`, the error itself.
2. Refine 2 becomes `64 x 32 x 9 + 32 = 18,432 + 32 = 18,464` instead of 9,232, so
   `95,632 - 9,232 + 18,464 = 104,864`.
3. `0.4 / 0.08 = 5.0` against `0.8 / 0.4 = 2.0`: the first, despite the smaller residual.
4. So the likelihood trains only `b`. Otherwise the reconstruction could lower its loss by changing its
   errors to suit `b`, and the frozen-backbone guarantee would not hold.
5. "Real objects caught at the top 1% and 0.1% not lower": A1+ lost 2.7 and 2.1 points.

</details>

---

Part 63 teaches what happened when this design was trained: FT06, the uncertainty head, run.
