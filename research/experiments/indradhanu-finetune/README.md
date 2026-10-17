# Indradhanu fine-tuning: EnMAP + PRISMA + AVIRIS-NG, with a segmentation head

**Question:** starting from F001 (the SpectralEarth-pretrained reduced-attention model), does fine-tuning on the
three real sensors, with a cloud / water / shadow / snow segmentation head, improve anomaly detection on the
labelled PRISMA scenes, and give masks good enough to replace the threshold rules in `scene_segmentation`?

**Design, curriculum and gates:** `docs/proposals/fine-tuning-and-segmentation.md`. Pass criteria for each run
are written in its README before launch and not moved afterwards.

## Runs

| id | stage | mode | what | instance | hours | cost | gate result | verdict |
|---|---|---|---|---|---|---|---|---|
| FT01 | 1 | probe | segmentation head on frozen F001 (epoch 109), EnMAP | g6e.2xlarge (kept) | 1.4 | ~$4 | water 0.82 >= 0.6, pass | water 0.82, cloud 0.80, shadow 0.20, thin cloud 0.09 (label-limited) |
| FT02 | 2 | adapt | reconstruction on all three sensors, one sensor per batch | g6e.2xlarge (kept) | 4.25 | ~$12 | bench 0.855 < 0.861, **not met** | reconstruction better on all three; combined 0.706 -> 0.824; per-sensor see-saw under investigation |
| FT02b | 2b | adapt, mixed batches | FT02 ep8 continued, every batch all three sensors | g6e.2xlarge (kept) | 2.2 | ~$6 | bench 0.856 < 0.861, not met | see-saw gone; SAM 3.99 / 4.37 / 5.57; bench measures tie v0.2.0 |
| FT02c | 2c | adapt, mixed + edge term | FT02b ep9 continued, gradient L1 weight 1.0 | g6e.2xlarge (kept) | 2.2 | ~$6 | edge FP share unchanged, not met | edge term moved 1%: decoder bottleneck; SAM 3.93 / 4.21 / 5.41 |
| FT03 | 3 | joint, mixed batches | FT02c ep10 + FT01 head, seg_weight 0.1 | g6e.2xlarge (kept) | 3.0 | ~$8 | IoU 0.841 / 0.815 >= FT01, SAM held, detection held: **passed** | stage-3 model; head ready for stage 5 |

## Stage 0 records

**Shards** (train / test tiles): EnMAP `patches/enmap/native-v1z` 14,542 / 950 (224 bands, labels); PRISMA
`patches/prisma/native-v1z` 14,802 / 1,570 (234 bands); AVIRIS-NG `patches/aviris_ng/native-v1` 22,377 / 2,270
(372 bands, 3-8 m). The `v1z` sets are the native-v1 shards re-encoded to `pixels.i16dz` (exact, ~2.1x;
`recompress_ft.sh`) so all three fit on the L40S. Band lists for alignment: `band_lists/enmap_224.json`,
`band_lists/aviris_ng_372.json`; PRISMA's 234 centres are fixed and need none.

**EnMAP label QA** (2026-10-16, all 104 shards / 15,492 tiles / 210 scenes; per-scene CSV and scripts kept on the
dev laptop at `C:/es/labelqa/`, per-tile tables at `patches/enmap/native-v1-analysis/`). No ground truth exists,
so two plausibility tests were used: *cloud plausible* = mean 450-650 nm reflectance > 0.2 and flat (min/max of
the 460-500 / 540-580 / 640-680 / 840-880 nm means >= 0.7); *cirrus surface-driven* = within a tile, the cirrus
pixels brighten 2100-2250 nm more than 450-650 nm relative to clear land (real thin cloud brightens the dark
visible more).

| scene water vapour (g/cm2) | scenes | cirrus level 1 surface-driven | cirrus >= 2 surface-driven |
|---|---|---|---|
| < 0.5 | 3 | 100% | 100% |
| 0.5-0.75 | 8 | 90% | 100% |
| 0.75-1.0 | 20 | 79% | 42% |
| 1.0-1.25 | 33 | 26% | 10% |
| 1.5-2.0 | 60 | 14% | 8% |
| 2.0-2.5 | 33 | 66% | 32% |
| >= 2.5 | 32 | 42% | 3% |

| policy (train) | scenes | tiles affected | suspect labels removed | plausible removed |
|---|---|---|---|---|
| wv < 0.50 | 3 | 1.5% | 44% | 0.2% |
| **wv < 0.75 (adopted)** | **11** | **5.6%** | **62.6%** | **13.6%** |
| wv < 1.0 | 30 | 15.2% | 66.6% | 15.7% |
| elevation > 1,500 m | 13 | 6.6% | 43.1% | 18.8% |
| elevation > 3,500 m (proposal's first draft) | 0 | 0 | 0 | 0 |

Findings that changed the draft: `label_classes == 0` is EnMAP's cloud class (79% of all cloud-labelled pixels),
not no-data; no scene is above 2,776 m, so dry air, not altitude, marks the false labels; 4 scenes appear twice
under two processor versions (`band_lists/enmap_drop_duplicate_scenes.json`); snow and cloud overlap on 122,815
pixels; 53% of shadow pixels have no cloud within 900 m in the same tile (an upper bound - the cloud may be in a
neighbouring tile). Class balance after the policy and the remaining caveats are in the proposal.

## Log

One entry per run, diagnosis or decision, dated, newest last. Numbers live in the run READMEs; this is the
thread that connects them.

- **2026-10-16 - Label QA and policy.** EnMAP label audit over all 15,492 tiles (Stage 0 records above): class 0
  is cloud, dry air (wv < 0.75 g/cm2) not altitude marks false cloud/cirrus labels, 4 duplicate scenes dropped.
- **2026-10-16 - F001 stopped at epoch 113 of 200** (decision). Bench (residual Mahalanobis, own bands) 0.861 at
  epoch 60, 0.862 at epoch 109: flat for 49 epochs. Fine-tuning starts from `F001_ep109.pt`
  ([F001 README](../indradhanu-se-pretrain/F001-final-pretrain/README.md)).
- **2026-10-16 - FT01 probe done** ([FT01-probe](FT01-probe/README.md)). Head only, frozen F001, 15 epochs,
  1.4 h. IoU water 0.82, cloud 0.80, shadow 0.20, thin cloud 0.09 (flat all run - the label ceiling the QA
  predicted), snow not measurable (1 test tile). Gate (water >= 0.6) passed.
- **2026-10-16 - FT02 adapt done, gate not met** ([FT02-adapt](FT02-adapt/README.md)). Reconstruction on all
  three sensors, one sensor per batch, 20 epochs, 4.25 h. SAM fell on every sensor; bench 0.855 (best.pt, epoch
  8) / 0.853 (epoch 20) vs >= 0.861; combined rose 0.706 -> 0.824. The trained-wavelength record grew 202 -> 733
  (AVIRIS-NG +337, PRISMA +177, EnMAP +17), so FT02 reads all 185 bench PRISMA bands where F001 read 165.
  Per-sensor validation see-sawed (epoch 17: PRISMA 5.04 deg, AVIRIS-NG 8.24 deg).
- **2026-10-16 - Edge diagnosis** (FT02 README, Observations). 49-68% of top-1% false positives sit on the
  top-10% spatial edges (chance 10%), but 29-79% of ground-truth pixels are strong-edge pixels too; an
  edge-strength correction of the score map collapses the bench 0.86 -> 0.67, and the earlier sharpeners
  (difference-of-Gaussians, ring z, one-pixel shift-min) all lost to residual Mahalanobis. Conclusion:
  post-hoc edge correction cannot tell a field boundary from a target.
- **2026-10-16 - BatchNorm diagnosis** (FT02 README, Observations). The model's one BatchNorm (the stem's)
  re-estimated per PRISMA scene moves 0.35-0.76 sd; residual Mahalanobis barely moves (0.853 -> 0.858),
  combined swings (set1 0.943 -> 0.755). Sensor-pure batches leave its stored statistics set by the last
  batches of an epoch - consistent with the see-saw.
- **2026-10-16 - Decisions (user).** FT02 is not invalidated: build on it. Mixed-sensor batches: yes (FT02b).
  Regularisation (EMA, SpectralEarth replay, lower LR, L2-SP): shelved. Teacher ("oracle") head - an EMA
  teacher on the full band list distilling into a student on a degraded list: deferred until after mixed
  batches. Per-sensor heads and a synthetic-target detection head: discussed, not chosen.
- **2026-10-16 10:15 UTC - FT02b launched** ([FT02b-mixed](FT02b-mixed/README.md), code 9ae3215). FT02 epoch 8
  continued with every micro-batch EnMAP 3 / PRISMA 3 / AVIRIS-NG 2, x4 accumulation, 10 epochs. Weights only
  are carried over: the optimiser, the cosine LR (back to 1e-4) and the SAM ramp restart (SAM weight 0 in its
  first epoch, 0.45 in its tenth). The configured 1-epoch warm-up does not exist in code (`warmup_epochs` is
  never read). Results pending.
- **2026-10-16 - FT02c planned, after FT02b.** An edge term: L1 between the spatial gradients of reconstruction
  and input on hidden pixels. Inference reconstructs each pixel while its token is hidden, so a boundary that
  continues across the hidden token is learnable sharply, while a target isolated inside it is not. Caveat: a
  target larger than one 4 x 4 px token is partly visible in its neighbours. Folder [FT02c-edges](FT02c-edges/README.md); code 9ff0484.
- **2026-10-16 - FT02b done: see-saw gone, gate not met.** Per-sensor SAM 3.99 / 4.37 / 5.57 deg, all three improving
  together; PRISMA bench residual Mahalanobis 0.856 (< 0.861). Every bench measure ties v0.2.0 - the 4-scene bench
  cannot separate models under the default score ([FT02b-mixed](FT02b-mixed/README.md)).
- **2026-10-16 - FT02c done: edge term had no effect.** The term itself fell 1% in 10 epochs; edge false-alarm share
  unchanged on all 4 scenes. Decoder H/4 bottleneck: sharpening would need a masked higher-resolution decoder path
  ([FT02c-edges](FT02c-edges/README.md)).
- **2026-10-17 - Detection measures and the implanted-target benchmark.** pAUC <= 1% FPR, TPR at 0.1%, objects
  caught at 0.1%; bench-v1: 3,336 implanted objects on 18 held-out hosts, tiered subtle / medium / easy on the real
  objects' contrast. It separates the stages: easy pAUC 0.257 (v0.2.0) -> 0.342 (F001) -> 0.371 (FT02c)
  ([implant-bench](../implant-bench/README.md), docs/lld/implant-bench.md).
- **2026-10-17 - FT03 passed.** Joint, mixed batches, FT02c init + FT01 head, seg_weight 0.1: water / cloud IoU 0.841 /
  0.815, reconstruction and detection held ([FT03-joint](FT03-joint/README.md)).
- **2026-10-17 - Class-aware scoring: not shipped.** Per-class whitening + same-class ring removed cloud false alarms
  (implant 8% -> 1%) but water's tight statistics made water noise dominate (set1 water share of false alarms 14% ->
  92%; PRISMA pAUC 0.158 -> 0.134). User: "we don't detect anything on water and cloud" - next, exclude them.
- **2026-10-17 - Evaluation became one subsystem** (app/evaluation, scripts/evaluate.py, docs/lld/evaluation.md);
  stage-3 run reproduced every earlier number. Water / cloud exclusion (FT03 masks, zeroed after scoring) lifts the
  top of the ranking on every model (FT03 PRISMA: caught top 0.1% 14.3% -> 16.5%, pAUC 0.158 -> 0.165). A model-free
  PCA-RX beats all models on the implant bench but loses on the real one: bench-v1 is biased, a v2 comes before any
  detection-supervised fine-tuning ([implant-bench](../implant-bench/README.md)).
- **2026-10-17 - bench-v2 accepted.** Implants matched to the real objects' contrast, foreignness and halo; on 2,913
  objects FT02c / FT03 beat PCA-RX (caught 32.8% / 32.3% vs 30.2%, pAUC 0.166 / 0.164 vs 0.154), consistent with the
  real bench. Detection-supervised fine-tuning is unblocked. 39 real objects lie inside FT03's water / cloud mask
  (13 offshore in set1, likely vessels; 4 on bright desert FT03 calls cloud) - exclusion waits on the user's scope
  call and a mask fine-tune (FT04).

