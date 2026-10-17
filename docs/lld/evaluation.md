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
EvalScene(dataset, name, sensor, cube, wavelengths_nm, valid, objects, attrs, units, reader).read(mode)
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
| scorers | residual_mahalanobis (d, local_window), combined, L1, SAM, class_aware (segmenter), exclude (base, classes, segmenter, fit_on_kept), pca_rx (no model) |
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
3. **Real and implanted objects are tiered by the same contrast function** (`implant.contrast_z`, diamond ring,
   tier edges 0.502 / 0.934), so `group_by: tier` means the same on both benchmarks.
4. **Chance is analytic** per object: 1 - (1 - q)^m, m its dilated pixel count (4-neighbour, the 1-px tolerance).
5. **Datasets are never pooled** in the summary: their objects are different populations.
6. **Cache keys** are dataset kind + scene name + model file identity (path, size, mtime, reading) + the full scorer
   spec; changing any of them recomputes. The dataset's path and content are **not** in the key (failure modes).
   Class maps are keyed by scene + segmenter identity. Reconstructions are never cached (0.25-1 GB each - they
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
| a dataset rebuilt in place (same kind, same scene names) with the same `output` | **silent**: cached score maps from the old data are measured against the new ground truth. Give each bench version its own path and output dir, or delete `<output>/cache/scores` |
| `provenance.json` `dirty` | always `false` today: `_write_provenance` runs `git status -- app scripts` with cwd `app/evaluation/`, so the pathspecs resolve to `app/evaluation/app` and `app/evaluation/scripts`. The stage-3 run records commit 060aabc, which does not contain `app/evaluation/` at all. Trust `commit` only together with the file times |

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

## Decisions

- **Scene-major, then model-major** loops: one scene and one reconstruction in memory - the old scripts ran out of
  RAM holding several (2026-10-17).
- **Score maps cached, reconstructions not**: new metrics or groups re-run no model.
- **Product's sampled AUC kept as `roc_auc`** for continuity with every earlier table; the exact AUC is available
  per group (`roc_auc_exact|...`).
- **Classical baseline as a scorer** (`pca_rx`: `residual_mahalanobis` with a zero reconstruction, so the
  whitened top-30 PCA distance from the 21/61 ring mean on the native cube itself - a local PCA-RX) - no model, run
  once per scene. It is the check that a benchmark rewards what a model adds over the cube's own statistics.
- **Water and cloud excluded by masking after scoring** (user, 2026-10-17: "we don't detect anything on water and
  cloud"), with FT03's head as the segmenter; class-aware scoring (per-class whitening) was tested first and not
  shipped (log, research/experiments/indradhanu-finetune/README.md: set1 water share of false alarms 14% -> 92%,
  PRISMA pAUC 0.158 -> 0.134). Not yet in `backend/`: there, water and cloud are left out only by an optional
  upstream `keep_mask` from the threshold-based `scene_segmentation` or `cloud_mask` actions.
- **Not done**: HAD100 / ABU adapters (radiance; ABU has no wavelengths - research/experiments/benchmarks/README.md);
  MNF-RX as a scorer; an HTML report.
