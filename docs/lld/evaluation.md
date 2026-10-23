# Evaluation

**Purpose** - one way to measure anomaly-detection models: a JSON config names the models, datasets, scorers,
metrics and groupings; `scripts/evaluate.py` runs every combination and writes tables, a summary and a report.
It replaced (2026-10-17) `scripts/evaluate_prisma_bench.py`, `scripts/evaluate_implant_bench.py`,
`app/utils/evaluation/` and the laptop one-offs, which each re-implemented band reading, reconstruction, scoring
and counting slightly differently.

## Interfaces

```python
# app/evaluation/
EvaluationConfig.from_file(path)            # config.py - pydantic; ModelSpec, DatasetSpec, ScorerSpec, MetricSpec
Evaluation(config, datasets=None, models=None).run() -> summary dict      # runner.py; overrides for tests
PrismaBench(root, scenes=None), ImplantBench(path, hosts=None)            # datasets.py - yield EvalScene
resample(cube, wl, grid) -> (G, H, W)       # datasets.py - linear, wl in any order (sorted first since b94dbfd)
EvalScene(dataset, name, sensor, cube, wavelengths_nm, valid, objects, attrs, units, reader, fingerprint).read(mode)
EvalModel(spec, cache_dir).reconstruct(scene) / .segment(scene)          # models.py
score(spec, cube, recon, valid, classes) -> (H, W)                       # scorers.py
scene_metrics(score, valid, objects, attrs, metrics, group_by)           # metrics.py
exact_roc, auc_exact, pauc, tpr_at, three_d_roc, object_hits, chance_hit # metrics.py
# app/evaluation/implant.py - the implanted-target benchmark's construction core (docs/lld/implant-bench.md)
```

```bash
python scripts/evaluate.py configs/eval/stage3.json [--output <dir>]
```

Config (configs/eval/stage3.json is the reference):

| block | kinds / fields |
|---|---|
| models | name, checkpoint (local or s3://), reading auto / native / product165 / se202 |
| datasets | prisma_bench (path = stage1_datasets root), implant (path = bench dir); scenes subset |
| scorers | residual_mahalanobis (d, local_window), combined, L1, SAM, class_aware (segmenter), exclude (base, classes, segmenter, fit_on_kept), pca_rx (no model); since 2026-10-19 `spatial_sigma` (blur the residual before whitening), fuse (parts, rule rank_mean / rank_max), mixture (k background types by k-means on the cube) - research/experiments/scoring-gains/, none adopted; scaled_rm (scale_power, scale_floor_pct; needs a model with an uncertainty head - `needs_scale` scorers are skipped for other models; FT06, not adopted) |
| metrics | roc_auc, pauc (max_fpr), tpr_at (fpr), objects_caught (top_pct list), three_d_roc |
| group_by | object attributes: tier, source, sensor, size |

## Data flow

```
for each dataset scene (one in memory):
  for each model: reconstruct once (product inferencer, product defaults) -> every model-based scorer -> free
  model-free scorers (pca_rx) once on the native cube
  class maps (class_aware / exclude) once per scene per segmenter, cached
  each score map -> scene_metrics -> per-object rows (hits, chance) + per-scene measures (overall and per group)
<output>/ per_object.csv, per_scene.csv, summary.json, report.md, config.json, provenance.json, cache/{scores,classes,checkpoints}
```

## Invariants

1. **The product's numbers.** Reconstruction uses the product inferencers and Indradhanu defaults (128 px, stride
   32, two complementary token masks); `roc_auc` is the product's `compute_roc`; scorers call the product's
   `compute_score` / `residual_mahalanobis`.
2. **Each model reads a scene its own way** (reading `auto`): wavelength stems native bands within their
   trained-wavelength record; fixed stems se202; v0.2.0 product165. PRISMA scenes reload through the product's
   band_filter_apply for non-native modes; other datasets are linearly resampled (close, not identical).
   `resample` must sort the wavelengths first: PRISMA lists its bands descending, and before b94dbfd the unsorted
   `np.interp` turned every PRISMA implant host into copies of two bands for v0.2.0. Since the fix the generic
   path reproduces the product path on the real bench exactly (`C:/es/eval/resample_check_fixed`); keep that
   check as the test of this invariant (docs/lld/implant-bench.md, "Corrections").
3. **Real and implanted objects are tiered by the same contrast function** (`implant.contrast_z`, diamond ring,
   tier edges 0.502 / 0.934), so `group_by: tier` means the same on both benchmarks.
4. **Chance is analytic** per object: 1 - (1 - q)^m, m its dilated pixel count (4-neighbour, the 1-px tolerance).
5. **Datasets are never pooled** in the summary: their objects are different populations.
6. **Cache keys** are dataset kind + scene name + model file identity (path, size, mtime, reading) + the full scorer
   spec + the scene's `fingerprint` (PRISMA: the .he5 file's size and mtime; implant host: the sha256s of
   `cube.npy` and `gt.npy` from manifest.json; since 229d423); changing any of them recomputes. Class maps are
   keyed by scene name + segmenter identity only - no fingerprint (failure modes). Reconstructions are never cached (0.25-1 GB each - they
   filled the laptop once).
7. **Excluded pixels stay in the measures.** `exclude` scores its classes 0 but leaves them valid: they remain
   negatives (tied at 0, at the bottom of the ranking) and any labelled object inside them remains a positive that
   the scorer can no longer find - the product would not flag it either. ROC-AUC therefore falls under exclusion
   while the top of the ranking improves.
8. **Every model-based scorer runs for every model**; a `segmenter` only supplies the class map. `exclude` with
   `segmenter: FT03_ep10` masks v0.2.0's scores with FT03's water / cloud.

## Failure modes

| What | Seen as |
|---|---|
| a scene not in reflectance (e.g. a future HAD100 radiance adapter) | `ValueError` naming the scene |
| exclude / class_aware without a segmenter, or a segmenter that is not a configured model | config `ValueError` |
| segmenter checkpoint without a head | `ValueError` from `EvalModel.segment` |
| duplicate model names or scorer labels | config `ValueError` |
| s3:// checkpoint without credentials | boto3 error at start (set AWS_PROFILE) |
| a dataset rebuilt in place (same kind, same scene names) with the same `output` | score maps: recomputed since 229d423 (fingerprint in the key). Class maps (`cache/classes`): **silent**, the old map is reused. Give each bench version its own path and output dir, or delete `<output>/cache/classes` |
| `provenance.json` `dirty` | fixed in 229d423 (git now runs from the repo root). Runs before it - the stage-3 run, which records commit 060aabc - always say `false`; trust their `commit` only together with the file times |

## Reference run: stage 3 (2026-10-17)

`configs/eval/stage3.json`: v0.2.0, F001_ep109, FT02c_ep10, FT03_ep10 on the PRISMA bench (4 scenes, 909 objects)
and implant bench-v1 (18 hosts, 3,336 objects); scorers residual Mahalanobis, combined, water + cloud exclusion by
FT03's head (mask and fit variants) and `pca_rx`. Output on the dev laptop at `C:/es/eval/stage3/` (report.md,
summary.json, per_object.csv, per_scene.csv, provenance.json); summarised in
`research/experiments/implant-bench/README.md` ("Stage-3 evaluation through app/evaluation").

1. **Reproduced every earlier number** (PRISMA bench pAUC / ROC-AUC / caught of 909 at the top 1%: v0.2.0
   0.159 / 0.854 / 402, F001 0.148 / 0.861 / 400, FT03 0.158 / 0.853 / 413; implant v0.2.0 19.0% / 7.7% caught at
   the top 1% / 0.1%).
2. **Water / cloud exclusion lifts the top of the ranking on every model** (FT03, PRISMA bench, mask variant:
   caught top 0.1% 14.3% -> 16.5%, top 1% 45.4% -> 46.1%, pAUC 0.158 -> 0.165, TPR at 0.1% FPR 0.058 -> 0.065).
   ROC-AUC falls 0.853 -> 0.830, most of it on set1 (0.932 -> 0.851), where FT03 calls 55% of the
   scene water: those pixels tie at 0, and so do 125 of set1's 638 ground-truth pixels. 39 of the 909 objects
   (27 in set1) lie wholly inside FT03's water / cloud mask; plain residual Mahalanobis catches 13 of them at the
   top 1%, exclusion 4 (through the 1-px tolerance). Fitting on kept pixels only (`fit_on_kept: true`) is no better
   than masking after scoring (FT03: 15.7% vs 16.5% at the top 0.1%).
3. **A model-free PCA-RX beats every model on bench-v1 and loses on the real bench.** Implant: caught top 1%
   30.2% vs FT03's 26.9%, pAUC 0.209 vs 0.172. PRISMA bench: pAUC 0.128 vs 0.158, TPR at 0.1% FPR 0.029 vs 0.058,
   caught top 0.1% 8.1% vs 14.3%. bench-v1 is biased toward cube-level spectral oddity
   (docs/lld/implant-bench.md, "bench-v2").

## Later runs (2026-10-17 / 18)

| config | output (dev laptop) | what it showed |
|---|---|---|
| `configs/eval/bench_v2.json` | `C:/es/eval/v2` | bench-v2 acceptance: FT02c / FT03 above PCA-RX (caught top 1% 32.8% / 32.3% vs 30.2%; docs/lld/implant-bench.md, "bench-v2") |
| `configs/eval/cloud_only.json` | `C:/es/eval/cloud_only` | FT03, plain vs cloud-only vs water + cloud. PRISMA bench: pAUC 0.158 / 0.166 / 0.165, caught top 0.1% 14.3% / 15.8% / 16.5%, ROC-AUC 0.853 / 0.850 / 0.830. bench-v2 caught top 1%: 32.3% / 33.4% / 33.7%. Cloud-only leaves set1 unchanged; its gain is set3 and set4 |
| `C:/es/eval/resample_check.py` (one-off, not in the repo) | `C:/es/eval/resample_check` | v0.2.0 on the PRISMA bench read through the generic linear resample instead of band_filter_apply: caught top 1% 15.8%, pAUC 0.094, ROC-AUC 0.752 (vs 44.2% / 0.159 / 0.854). **Resolved 2026-10-18:** the descending-wavelength bug (b94dbfd) |
| the same script, after b94dbfd | `C:/es/eval/resample_check_fixed` | generic resample = product path: 44.2% / 14.1% / 0.159 / 0.854 / 0.058 |
| `configs/eval/ft05_ep5.json` | `C:/es/eval/ft05_ep5` | FT05 epoch 5 vs FT03, plain and cloud-only. PRISMA bench: pAUC 0.15801 vs 0.15805, TPR at 0.1% 0.0594 vs 0.0582, caught top 0.1% 130 vs 130 - a tie. bench-v2: caught top 1% 32.8% vs 32.3%, pAUC 0.167 vs 0.164, `host` source 29.6% vs 29.4% (signature implants garbled - before b94dbfd) |
| `configs/eval/bench_v2_1.json` | `C:/es/eval/v2.1` | bench-v2.1 (fixed signatures, foreign tol 1.2), all models, PCA-RX, cloud exclusion by FT03's masks. Caught top 1%: v0.2.0 35.5%, F001 33.1%, FT02c 35.1%, FT03 34.9%, FT05 ep5 35.4%, PCA-RX 33.2%; exclusion ~+1 point for every model. No new model beats v0.2.0 (docs/lld/implant-bench.md, "bench-v2.1") |
| `configs/eval/ft04.json` | `C:/es/eval/ft04` | FT04 vs FT03, real bench and bench-v2.1, cloud exclusion with each model's own masks and swapped. Real bench pAUC / TPR at 0.1%: FT04 0.162 / 0.061, FT04 + own masks 0.170 / 0.065 (the release candidate), FT03 + own masks 0.166 / 0.064; masks interchangeable (FT04 + FT03 masks 0.170, FT03 + FT04 masks 0.165) |
| `configs/eval/scoring_gains.json` | `C:/es/eval/scoring_gains` | Scoring gains A1-A3 on FT04 (2026-10-19). Spatial blur raises real pAUC 0.162 -> 0.177 but loses objects at every size; RM + PCA-RX fusion adds nothing; mixture of backgrounds +6 points on bench-v2.1 and 0.162 -> 0.111 on the real bench (real targets cluster in small built-up types). RM stays the score |
| `configs/eval/ft06.json` | `C:/es/eval/ft06` | FT06 uncertainty head vs FT04 (2026-10-20). Real bench, cloud excluded: RM 0.170 / 46.0% caught top 1%; residual / b 0.125 / 37.8%; residual / sqrt b 0.163 / 44.8% (top 0.1% 16.6% vs 15.8%). Not adopted. ~2.3 h: two models plus the scale pass filled the 20 GB laptop GPU |
| `configs/eval/scene_adapt.json` | `C:/es/eval/scene_adapt` | Gated per-scene adaptation of FT04 (`ModelSpec.adapt_steps / adapt_lr / adapt_gate_pct / adapt_halo`; app/foundation_models/scene_adaptation.py), 2026-10-20. Every N (50 / 200 / 1000) below FT04 (pAUC ~0.160 vs 0.170 cloud excluded); gated = ungated. Not adopted. Adapted variants carry their settings in the model id (cache key) |
| `configs/eval/ft07.json`, `ft08.json` | `C:/es/eval/ft07`, `ft08` | FT07 gated loss / FT08 + block masking vs FT04 (2026-10-21), cloud excluded: real pAUC 0.165 / 0.164 vs 0.170 (bootstrap intervals below zero), objects tie. Not adopted |
| `configs/eval/ft09b.json` | `C:/es/eval/ft09b` | FT09a (progressive decoder alone) / FT09b (+ joint) vs FT04 (2026-10-22), cloud excluded: FT09b real pAUC 0.176 vs 0.170, bootstrap +0.0060 [+0.0002, +0.0118] - the first resolved gain since cloud exclusion; caught top 1% 47.1% vs 46.0%, top 0.1% 15.4% vs 15.8% (unresolved). FT09a a tie |
| `configs/eval/ft09c.json` | `C:/es/eval/ft09b` | FT09c (FT09b + 5 epochs) 2026-10-23: real pAUC 0.172 - a tie with FT04 (+0.0020 [-0.0026, +0.0064]) and a resolved loss vs FT09b (-0.0040 [-0.0076, -0.0010]). The object bootstrap does not cover checkpoint-to-checkpoint variation, which here is as large as FT09b's gain |
| `scripts/compare_scorers.py` | - | paired object bootstrap between two (model, scorer) pairs of a finished run (`app/evaluation/significance.py`); research/experiments/bench-resolution/. Real-bench resolution ~+-0.005-0.01 pAUC, ~+-1-1.5 pts caught at top 0.1% |

## Decisions

- **Scene-major, then model-major** loops: one scene and one reconstruction in memory - the old scripts ran out of
  RAM holding several (2026-10-17).
- **Score maps cached, reconstructions not**: new metrics or groups re-run no model. Cached float16, except maps
  past its range (> 6e4 or non-finite), kept float32 since 2026-10-19 - blurred / mixture squared distances overflowed
  to inf. Adding a ScorerSpec field changes every key (the key is the spec's JSON), so old caches recompute.
- **Product's sampled AUC kept as `roc_auc`** for continuity with every earlier table; the exact AUC is available
  per group (`roc_auc_exact|...`).
- **Classical baseline as a scorer** (`pca_rx`: `residual_mahalanobis` with a zero reconstruction, so the
  whitened top-30 PCA distance from the 21/61 ring mean on the native cube itself - a local PCA-RX) - no model, run
  once per scene. It is the check that a benchmark rewards what a model adds over the cube's own statistics.
- **Cloud excluded by masking after scoring; water kept** (user, 2026-10-17). First "we don't detect anything on
  water and cloud"; then, after 13 of the 39 real objects inside FT03's water / cloud mask turned out to be single
  pixels > 3 px offshore (likely vessels), vessels on water were put in scope. FT03's head was the segmenter; FT04's
  is the release candidate's (FT04 + cloud exclusion on its own masks, 2026-10-18). FT04 was first proposed as a
  mask fix because FT03 called 11% of set4 "bright desert" cloud; corrected 2026-10-18, that 11% is haze and a
  cloud / smoke streak the mask follows, and FT04 ran for masks on all three sensors and the rare classes. Class-aware
  scoring (per-class whitening) was tested first and not shipped (log, research/experiments/indradhanu-finetune/
  README.md: set1 water share of false alarms 14% -> 92%, PRISMA pAUC 0.158 -> 0.134). Water + cloud exclusion is
  kept as a scorer variant for comparison. Not yet in `backend/`: there, water and cloud are left out only by an
  optional upstream `keep_mask` from the threshold-based `scene_segmentation` or `cloud_mask` actions.
- **Not done**: HAD100 / ABU adapters (radiance; ABU has no wavelengths - research/experiments/benchmarks/README.md);
  MNF-RX as a scorer; an HTML report.
