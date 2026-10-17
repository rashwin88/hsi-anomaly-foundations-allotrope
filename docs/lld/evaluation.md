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
6. **Cache keys** are scene + model file identity (path, size, mtime, reading) + the full scorer spec; changing any
   of them recomputes. Reconstructions are never cached (0.25-1 GB each - they filled the laptop once).

## Failure modes

| What | Seen as |
|---|---|
| a scene not in reflectance (e.g. a future HAD100 radiance adapter) | `ValueError` naming the scene |
| exclude / class_aware without a segmenter, or a segmenter that is not a configured model | config `ValueError` |
| segmenter checkpoint without a head | `ValueError` from `EvalModel.segment` |
| duplicate model names or scorer labels | config `ValueError` |
| s3:// checkpoint without credentials | boto3 error at start (set AWS_PROFILE) |

## Decisions

- **Scene-major, then model-major** loops: one scene and one reconstruction in memory - the old scripts ran out of
  RAM holding several (2026-10-17).
- **Score maps cached, reconstructions not**: new metrics or groups re-run no model.
- **Product's sampled AUC kept as `roc_auc`** for continuity with every earlier table; the exact AUC is available
  per group (`roc_auc_exact|...`).
- **Classical baseline as a scorer** (`pca_rx`: the same whitened-PCA distance on the cube) - no model, run once.
- **Not done**: HAD100 / ABU adapters (radiance; ABU has no wavelengths - research/experiments/benchmarks/README.md);
  MNF-RX as a scorer; an HTML report.
