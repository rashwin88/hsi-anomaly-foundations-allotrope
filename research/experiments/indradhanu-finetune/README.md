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
- **2026-10-18 - FT05 (implant-denoising) did not pass.** Epoch 5 ties FT03 on the real bench (pAUC 0.158 / 0.158);
  epoch 10 costs AVIRIS-NG reconstruction (5.67 vs 5.48 deg). Loader fix: single-thread workers, 2 per source (2 h 41
  -> 43 min per epoch). Bug found: PRISMA lists bands descending and np.interp was used unsorted - v0.2.0's implant-
  bench numbers were garbage on PRISMA hosts (the real-bench generic reading now equals the product path) and every
  transplanted signature implant was garbled (bench-v1, v2, FT05). bench-v2.1 rebuilt with the fix.
- **2026-10-18 - Correction: set4 is hazy, not a desert mask error.** FT03's "11% cloud" on PRISMA set4 was read as
  bright desert called cloud; a true-colour view shows a bright cloud / smoke streak and a large hazy patch, which
  the mask follows. Set4's pixels are bluish (NIR / blue 0.54-0.65 for 75%), not red. A bright-and-flat cloud guard
  built on the desert reading rejected 69% of EnMAP's labelled clouds and was dropped. FT04's purpose is masks on
  all three sensors (pseudo-labels) and rare classes, not a desert fix.
- **2026-10-18 - bench-v2.1: no new model beats the shipped v0.2.0 on detection.** With the descending-wavelength fix,
  v0.2.0 scores 35.5% caught (pAUC 0.168) against FT02c / FT03 / FT05 35.1 / 34.9 / 35.4% (0.169 / 0.168 / 0.170) and
  F001 33.1% - consistent with the real bench tie. The earlier pretraining gain on the implant bench was the bug.
  Residual Mahalanobis absorbs what better reconstruction changes; detection gains need another lever (scoring,
  masks, detection-targeted training). The new models' measurable wins: masks for cloud exclusion (+~1 point, any
  model), native bands, reconstruction.
- **2026-10-18 - FT04 (final joint) passed - the release candidate.** Pseudo-labels for PRISMA / AVIRIS-NG plus rare-
  class weights: masks held, reconstruction better on all three sensors, detection at or above FT03; with cloud
  exclusion on its own masks the best real-bench result so far (pAUC 0.170, TPR @ 0.1% 0.065 vs shipped v0.2.0
  0.159 / 0.058). Next: product wiring (FT04 checkpoint + masks + cloud exclusion in anomaly_scoring).
- **2026-10-19 - FT04 shipped as Indradhanu V2 (opt-in); score-side gains tried and dropped.** Product path verified
  against the evaluator (pAUC identical on PRISMA set1 / set4). Spatial blur, fusion with PCA-RX and a mixture of
  backgrounds all fail to find more real objects than residual Mahalanobis (research/experiments/scoring-gains/).
- **2026-10-20 - FT06 (uncertainty head) failed.** A head on frozen FT04 learns its own expected error (Spearman 0.60-
  0.64) but expects large errors on the targets themselves (b on a target 1.07-1.19x its surroundings), so dividing
  by it hides anomalies: pAUC 0.170 -> 0.125 (residual / b) or 0.163 (residual / sqrt b). Folder [FT06-uncertainty](FT06-uncertainty/README.md).
- **2026-10-21 - FT07 (gated training loss) and FT08 (+ 8 px block masking) failed.** Leaving the top 1% worst-
  reconstructed pixels (+ 2 px halo) out of the loss costs a small, resolved pAUC loss (0.170 -> 0.165; FT08 0.164);
  objects caught and bench-v2.1 tie. Reconstruction and masks held. Folder [FT07-gated](FT07-gated/README.md).
  With bench-resolution (research/experiments/bench-resolution/): every change since FT04 except cloud exclusion moves
  detection by less than the real bench resolves, or costs a little.
- **2026-10-22 - FT09 (progressive decoder): the first resolved gain since cloud exclusion.** FT04 rendered pixels it
  could see barely better than hidden ones (research/experiments/decoder-capacity/), so its SegFormer decoder was
  replaced by x2 stages with a masked full-resolution skip. Two phases - FT09a decoder alone (3 epochs, FT04 frozen),
  FT09b joint (5) - after a joint-from-scratch first attempt lagged. FT09b, cloud excluded: real pAUC 0.176 vs 0.170,
  bootstrap [+0.0002, +0.0118]; caught top 1% +1.1 pts, top 0.1% -0.4 (set3). A candidate, not a release; FT09c
  continues it. Folder [FT09-decoder](FT09-decoder/README.md).

