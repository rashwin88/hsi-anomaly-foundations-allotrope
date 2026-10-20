# An uncertainty head for Indradhanu (gains idea B4)

**Status:** run 2026-10-20 as FT06 - **failed detection, not adopted** (research/experiments/indradhanu-finetune/FT06-uncertainty/: the head predicts its error, Spearman ~0.6, but expects large errors on targets themselves). Proposed 2026-10-19. Training needs the GPU instance (i-0a3afdd4efbcafe92) - starting it
is your call. **Builds on:** FT04 / Indradhanu V2 (research/experiments/indradhanu-finetune/FT04-final-joint/).

## The problem
Residual Mahalanobis (RM) whitens the residual with **one covariance per scene**. It discounts error *directions*
the scene shares (haze, striping, dark water) but treats every *place* alike. The model's errors are not alike
across places: field edges, texture, glint, thin cloud and shoreline are hard to reconstruct everywhere, and their
large residuals are RM's false alarms. FT03 / FT04 reconstruct better than v0.2.0, yet detection ties - better
reconstruction lowers the residual on targets and background together.

What RM lacks is the model's own estimate of **how wrong it expects to be at this pixel**. A big residual where the
model expected to be wrong is ordinary; the same residual where it expected to be right is the anomaly.

## Design
**1. Head.** A third decoder on the shared encoder features, beside reconstruction and segmentation:
`scale_head = SegFormerDecoder(embed_dims, decoder_dim=64, out_channels=1)` -> `log b(p)`, one scalar per pixel
(spatial uncertainty; the per-band shape is RM's job and stays there). ~0.1 M parameters.

**2. Loss - Laplace negative log-likelihood**, matching the L1 reconstruction loss, on masked pixels only:

    L_scale = mean over masked p of [ mean_c |r_c(p)| / b(p) + log b(p) ],   r = normalised residual, detached

The optimum is b(p) = E|r(p)|: the expected size of the error there. Computed only where the pixel was **hidden from
the encoder** - the head predicts how hard a pixel is from its context, never from the pixel itself, so it cannot
learn "this pixel is odd, so expect a big error" (which would hide the anomaly). Full-scene inference already
reconstructs every pixel while it is hidden (two complementary masks), so inference matches training.

**3. Training - frozen FT04, head only** (recommended): reconstruction and segmentation are untouched, so V2's
masks, SAM and parity stay exactly as shipped; 3 epochs on the FT04 mix (EnMAP / PRISMA / AVIRIS-NG), ~2.5 h on the
L40S. Rejected for now: joint training (risks the reconstruction FT04 passed on, and the NLL can trade accuracy for
variance); per-band scales (doubles the decoder output, overlaps RM's whitening).

**4. Score.** `scaled_rm`: RM on r(p) / b(p) - whitening refitted on the scaled residual. Also measured: the plain
`mean_c |r_c| / b` (no whitening), and fusion with RM (the A2 machinery).

**5. Evaluation.** `EvalModel.reconstruct` gains an optional scale map; new scorer kind `scaled_rm`; the same
configs as FT04 (real PRISMA bench, bench-v2.1), with and without cloud exclusion.

## Pass criteria (fixed now)
Against FT04 + RM, cloud excluded on FT04's masks, real PRISMA bench: pAUC <= 1% at least **+0.010** (0.170 ->
0.180) and TPR @ 0.1% not lower, **and real objects caught at top 1% and 0.1% not lower** (scoring-gains
lesson: A1 raised pAUC while losing objects); bench-v2.1 caught top 1% not lower by more than 0.5 points. And a sanity check:
on held-out tiles, b(p) must rank pixels by their actual error (Spearman >= 0.5) - if the head cannot predict its
own error, the score cannot use it.

## Files
`app/foundation_models/components/spectral_stem_seg_former_mae.py` (head, `predict_scale`), the fine-tune trainer
(`mode: "scale"`, frozen backbone), `training_config.py` (`scale_head`, `mode`), `SpectralStemInferencer`
(full-scene scale with the same tiling), `app/evaluation` (`scaled_rm`), an experiment folder
`research/experiments/indradhanu-finetune/FT06-uncertainty/`. Product wiring only if it passes.

## Risks
- **Anomalies in training data.** Real targets are in the shards; the head learns a large b wherever the context
  looks like a target. Mitigated by the hidden-pixel rule and by rarity; measured by the bench.
- **b collapses to a constant** (the head learns nothing spatial): caught by the Spearman check.
- **Dividing by a small b** inflates noise in "easy" areas (deep water, flat desert). Floor b at its 5th percentile
  per scene; part of the scorer.

## Decisions for you
1. Go ahead (it needs the GPU instance started, ~2.5 h of training plus evaluation)?
2. Frozen backbone first (recommended), or joint from the start?
