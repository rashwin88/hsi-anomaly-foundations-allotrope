# Detection-supervised fine-tuning (FT05)

**Status:** proposed 2026-10-17, for agreement. **Starts from:** FT03 (joint reconstruction + segmentation).
**Runs in:** research/experiments/indradhanu-finetune/FT05-implant-denoise/.

## Why

Every fine-tuning stage so far trained reconstruction and hoped detection would follow. It did, a little
(bench-v2 caught top 1%: F001 29.2% -> FT02c 32.8% -> FT03 32.3%), and the scoring's whitening absorbs most of
what better reconstruction buys. The detection score is "how badly the model reconstructs this pixel from its
surroundings". It is therefore best when the model reconstructs the *background* under an anomaly and refuses to
reproduce the anomaly itself. Nothing in training asks for that: the targets are always the input, anomalies
included.

bench-v2 now gives implants that look like the real objects (contrast, foreignness, halo; research/experiments/
implant-bench/README.md), so they can be training signal.

## The idea: implant-denoising (recommended first)

Train with implants in the **input** and the **clean tile as the target**:

```
tile x (training split)  --implant k objects (v2 generator, per tile)-->  x'      targets: x (clean)
model(x', masks)  ->  x_hat ;  loss = MAE loss(x_hat, x) on hidden tokens
                                     + w_near x L1 on hidden pixels within 3 px of an implant
```

- Inference reconstructs every pixel while its 4 x 4 token is hidden, so an anomaly is seen only through its
  neighbours: objects larger than a token, the PSF halo, adjacency. The loss teaches the model not to carry that
  into its prediction - to reconstruct what the background would be. The residual at an anomaly grows; nothing
  about scoring changes (residual Mahalanobis, cloud exclusion).
- No new head and no new score: the product path is unchanged, so a win ships as a checkpoint.
- Implants are tiny (0.02% of pixels), so the near-implant term (w_near) keeps them from drowning in the MAE loss.

**Alternative, second if wanted - a detection head:** a third head on the encoder (as the segmentation head)
predicting per-pixel "implanted" logits with BCE, used alone or fused with the residual score. More direct, but a new
score and a higher risk of learning the implants' appearance rather than anomalies in general.

## Data and leakage

- Implants are generated **on the fly in the loader workers**, into **training-split** tiles of all three sensors
  (the shards already on the L40S). bench-v2 hosts are test-split scenes; the four real bench sites are excluded
  from every training split.
- Generator: app/evaluation/implant.py's v2 pieces, per 128 px tile (tile-level background subspace, PSF sigma 1.4,
  real_v2.json for size / contrast / foreignness pairs, library resampled per sensor band list at start).
- **Held-out source:** training uses the `signature` and `material` sources only; bench-v2's `host` source is
  never trained on, and is reported separately as an out-of-distribution check. A different seed from bench-v2.
- 50% of tiles get 1-4 implants; the rest train as plain MAE, so reconstruction is not unlearned.

## Run

FT05: FT03 epoch 10 continued, joint mode as FT03 (segmentation weight 0.1, so the head stays), mixed batches,
10 epochs, LR 1e-4 cosine, SAM weight 0.5, w_near 5 (fixed before launch). About 3.5 h on the L40S.

## Pass criteria (written before launch)

1. **Real PRISMA bench decides** (the only real labels): not below FT03 on pAUC <= 1%, TPR @ 0.1% and caught
   top 0.1%, and above it on at least two of the three.
2. bench-v2: above FT03 on caught top 1% and pAUC - reported, but optimistic (same generator family), so it cannot
   pass the run on its own; the held-out `host` source must not fall.
3. Reconstruction: per-sensor validation SAM within 0.1 deg of FT03 on clean tiles; segmentation IoU within 0.02.

## Cost and risks

- Code: an implant augmentation for the trainer (~200 lines + tests), reusing app/evaluation/implant.py. About a day.
- **Risk: the model learns the implants' signature** rather than anomalies. Guards: the real bench decides; the
  `host` source held out; a different seed.
- **Risk: reconstruction worsens** on anomaly-rich real scenes, since it learns to smooth over outliers. That is the
  intent near objects; criterion 3 checks it does not spread.
- The EnMAP hosts' flat ~16% on every bench is unexplained; FT05 may not move it.
