# Detection-supervised fine-tuning (FT05)

**Status:** proposed 2026-10-17; agreed and launched the same night (user: "Go for it"), relaunched at commit 558ff5e
after a speed fix. **Done 2026-10-18: did not pass.** best.pt = epoch 5 (lowest validation loss): real PRISMA bench
ties FT03 (pAUC 0.15801 vs 0.15805, TPR at 0.1% 0.0594 vs 0.0582, caught top 0.1% 130 vs 130 - criterion 1 not met:
above on one of three); bench-v2 above (caught top 1% 32.8% vs 32.3%, pAUC 0.167 vs 0.164; `host` source 29.6% vs
29.4% - criterion 2 met); criterion 3 held at epoch 5 (SAM 3.94 / 4.20 / 5.33 deg). Epoch 10 fails criterion 3 on
AVIRIS-NG (SAM 5.67 vs FT03's 5.48). On bench-v2.1 epoch 5 ties (35.4% vs FT03's 34.9%). Caveat: half the training
implants were `signature` implants garbled by the descending-wavelength np.interp bug (fixed in b94dbfd); the FT05
README calls a rerun with correct signatures the fair test, and nothing suggests it would flip the result. The
detection-head alternative below is untried.
**Starts from:** FT03 (joint reconstruction + segmentation).
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

**As built (2026-10-18).**
- The near term is a **weight on the loss mask**, not an added L1 term: `_mixed_loss` multiplies the hidden-pixel
  mask by `1 + (w_near - 1) x near`, and `_masked_l1_sam` divides by the mask's sum, so both L1 and SAM are
  weighted means with near pixels counted 5 times (`finetune_trainer.py`). Same intent, different arithmetic from
  the formula above.
- Implants are seeded by the tile key (`implant_sample`): a tile gets the same implants every epoch.
- Placement anywhere valid, water included (`implant_tile`; vessels in scope, user 2026-10-17). bench-v2's builder
  still places on land only.
- **Speed:** ~2 h 40 min per epoch, not ~21 min: the GPU sits at 0-7% while the loader workers do the implanting
  on the CPU (g6e.2xlarge, 8 vCPUs; `num_workers` 6 per sensor loader, three loaders in lockstep). The background
  subspace was moved from an SVD to a covariance eigendecomposition first (558ff5e, 0.25 -> ~0.02 s per tile, same
  subspace). Cause found 2026-10-18: 18 workers, each with its own BLAS thread pool, oversubscribed the 8 vCPUs
  about 8-fold (4.7 s per tile instead of 1.8 s). Fix: `single_thread_worker` (one thread per worker) and
  `loader_workers_per_source: 2` - 43 min per epoch from epoch 6 (f472aa8).

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
