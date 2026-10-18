# 9. Known issues

Found in a full-tree audit on **2026-08-24**. The repo carries no `TODO`/`FIXME` markers by
policy, so this file is the register. Fix an entry, delete the entry.

## No verification behind the backend

**1. `backend/` has no tests, and that has cost real outages.**
`scripts/run_tests.*` covers `app/` only — 67 tests. There are none for `backend/`, none for
`frontend/`, none for `app/foundation_models/`, and there is no CI.

This is not theoretical. Two total outages sat in `main` undetected until someone actually
started the stack on 2026-08-24:

- **`anomaly_scoring` raised `ImportError` on every job** — it imported a
  `PixelStatsOverride` that was never defined.
- **The worker could not start at all**, so scene onboarding, every action run and every
  export were dead. `envi_helper.py` and `hotsat_helper.py` imported `ENVIFileComponents`
  and `HotSatFileComponents`, neither of which existed anywhere in the repo.

Both were invisible to reading. The second is the instructive one: the api stayed perfectly
healthy — `/healthz/db` green — because the lazy-import rule keeps `scene_onboard` out of the
api's startup path. Only the worker imports it at module level, and nothing ever imported the
worker except the worker.

**The cheapest guard is a smoke test that imports every worker entry point.** A test doing
`import allotrope_worker.handlers` would have caught it in milliseconds. A second guard worth
having: `docker compose up` and assert all five containers reach `running`, since `exited` and
`restarting` both look like progress in the logs.

Until that exists, after any change under `app/` or `backend/`, check the worker is genuinely
up rather than crash-looping:

```bash
docker compose -f docker/docker-compose.yml ps -a     # worker must be 'running'
docker compose -f docker/docker-compose.yml logs worker | tail -5
```

A healthy worker logs `worker starting (id=… types=action_run,…)`. A crash-looping one shows
`restarting` and repeats a traceback.

## Correctness risks

**3. Train/inference regime mismatch (SegFormer).** Inference defaults to
`masking_strategy="checkerboard"`, but training and validation use random two-pass masking.
The inferencer's `MIN_VALID_FRACTION = 0.1` carries a comment claiming it "matches
training"; training actually uses `0.4`.

**4. Uncovered-pixel fallback diverges by model family.** SegFormer falls back to the
original scene (residual exactly zero — safe). The conv autoencoders fall back to zeros,
producing a large spurious residual in uncovered regions that reads as an anomaly.

**5. Normalization stats computed under the wrong mask.**
`app/background_stats/landsat_thermal_stats.py` computes mean/std under
`pure_validity * custom_quality_mask`, while the trainers train under
`pure_validity * predicted_cloud_mask`. Relatedly, three thermal trainers read
`custom_quality_mask.npy` and then never use it — a dead read that still forces the key to
exist in every shard.

**6. Saved checkpoints have an empty `optimizer_state_dict`**, so `resume_mode="resume"`
restores an empty optimizer state. (Noted in the old checkpoint inventory; still true.)

**7. `HotSatDatasetBuilder` silently drops `units`.** It passes `units=_DN_UNITS` to
`VendableThermalDataset`, which has no such field and no `extra="allow"` — Pydantic v2
discards it. The intended guard against treating raw DN as °C is therefore not in effect.

**8. Scene search strips `_` and `%` instead of escaping them.** `list_scenes` in
`backend/allotrope/api/scenes.py` removes both characters from `q` before the `ILIKE`, so
any search containing an underscore — which is every PRISMA/EnMAP product id
(`PRS_L2D_STD_…`) — becomes a different string and matches nothing. Fix: escape them and
pass `escape="\\"`, as `allotrope_inventory/api/datasets.py` does.

**9. SAM and `combined` rank dark water as anomalous.** On near-black pixels a tiny residual is a
large spectral angle, so on sea-heavy scenes those maps light up ordinary water (PRISMA set1: 75%
sea, SAM ROC-AUC 0.34-0.48). Indradhanu's default is now `residual_mahalanobis` (2026-10-15),
which does not have this problem (set1 0.94 for v0.2.0), but `SAM` and `combined` remain
selectable per model and still score with the plain angle: `compute_score`'s opt-in
`sam_dark_fade` (see [5. Detectors](05-detectors.md#sam-on-dark-water)) is not passed by
`_anomaly_scoring_run.py`. Whether to enable it for those methods, or retire it, is open.

**10. The splib07 slim bundle drops most man-made spectra.** `scripts/curate_splib07.py` keeps
ASD-instrument files only and skips Beckman (BECK) spectra, on the claim that their
200-3000 nm range does not overlap our sensors. It does. Most USGS man-made spectra (plastics,
fabrics, roofing, oil) were measured on the Beckman, so `data/splib07_slim` has 3 usable
man-made entries out of 519, and `spectral_library_match` can rarely name a man-made material.
Its "artificial" chapter is mostly paint pigments and lab chemicals. `data/family_library`
(see [`lld/spectral-family-match.md`](lld/spectral-family-match.md)) carries them for family
matching; the SAM library itself is unchanged.

**11. `gaussian_resample_to_target` silently drops bands for ~10 nm sources.** It returns NaN
for any target band with fewer than 3 source samples inside its window. That is fine for 1 nm
library spectra. But it dropped 42 of 165 bands when resampling classic AVIRIS (~10 nm)
imagery onto the common grid. Any image resampled with it from a ~10 nm sensor loses bands
without an error. Densify the source first (as `scripts/curate_family_library.py` and step 1
of the man-made library experiments do), or lower `min_lib_points_per_band` for images.

**12. The local deployment's AVIRIS-NG vendables cannot be read by this repo.**
- **What is on disk:** in the stack that runs from `/home/fitt/allotrope`,
  `vendable.pkl` stores `normalized_hyperspectral_cube` and `validity_cube` as
  `{"__memmap__": True, "filename": ..., "dtype", "shape", "offset"}` dicts. They point at
  `primary_cube_bsq.npy` and `validity_cube.npy` beside the pickle.
- **Why it breaks:** nothing in this repo or its history writes or reopens that form. With
  this repo's code, every action that `pickle.load`s such a vendable fails on its first
  array operation. That includes SAM in `spectral_library_match`.
- **Status:** found 2026-10-16 while testing family matching. The deployed images carry code
  this repo does not.

## Deliberate limitations in segmentation sharding

Added 2026-08-25 with the `enmap_seg` lane. All known and accepted, not defects. Design
rationale in `docs/lld/segmentation-sharding.md`.

- **A scene that fails mid-patching leaves nothing to inspect.** The partial `.tar` is
  removed in `finally` so a resume cannot mistake it for finished work; the trade is that
  debugging a failure means re-running that scene.
- **`S3SceneStorage.shard_exists` catches `ClientError` broadly**, so a permissions failure
  reads as "shard absent" and the work is redone. Errs toward doing work rather than
  skipping it, which is the safe direction, but it delays surfacing an auth problem.
- **`app/utils/files/enmap_scene_cover.py` has no test.** It parses a real METADATA.XML and
  those are gitignored payloads. A four-tag synthetic document is enough to exercise it —
  proven incidentally during development — so this is worth closing.
- **Storage is large.** At stride 64 a 165-band float16 patch is ~8.2 MB (5.4 pixels +
  2.7 per-band validity + 0.1 labels), so 212 scenes come to **~562 GB**. Per-band validity
  is retained deliberately even though the current trainers read only band 0, to keep
  per-band masking open. Dropping it to a 2D mask saves a further ~33%.
- **A float16 shard must be cast to float32 by its trainer.** The models are fp32.
  Segmentation shards store float16; the reconstruction lane still stores float32.

## Raw-dataset inventory — deliberate limitations

Added 2026-10-13 with `backend/allotrope_inventory/`. Design rationale in
[`lld/inventory.md`](lld/inventory.md).

- **No automated tests.** Every path was exercised by hand against live S3 and Supabase on
  2026-10-13/14 (upload, resume, checksum rejection, server-side import with provider MD5s,
  bulk registration, unpack, retire, the UI proxy), but nothing re-runs it. A harness was
  scoped — moto in server mode plus a throwaway PostGIS container — and dropped.
- **Registration fields are immutable.** Metadata, footprint, acquisition time and kind are
  set at `POST /datasets`; there is no `PATCH`. `ingest` / `import-s3` warn when a resume
  passes different values. This is why scenes imported from file names carry no footprint.
- **The whole-file SHA-256 of a multipart file is client-declared.** S3 cannot compute one
  for a multipart object; integrity rests on per-part checks. Bulk-registered (single-PUT)
  files don't have this gap.
- **Not built yet:** a Google Drive source, `rebuild-index` from the manifests, and per-patch
  metadata for SpectralEarth (the patches carry no georeference or date).
- **Presigned URLs die with the credentials that signed them.** On EC2 the inventory signs
  with the instance role's temporary credentials (rotated every few hours), so a URL can
  expire well before its 24 h `ExpiresIn`. Uploads that outlive it fail with 403; re-running
  `ingest` (resume) issues fresh URLs. Same for `/download`.
- **Re-running an interrupted `unpack-archive` re-reads the whole archive** — a `.tar.gz`
  cannot be seeked — though members already registered aren't uploaded again. About 3.9 h
  for SpectralEarth.
- **Orphaned multipart uploads** from a request that fails after `start_upload` but before
  commit are left to the bucket's 7-day abort rule.
- **Writes are CLI-only from outside the box.** Port 8020 binds to localhost; the public
  route `/inventory-api/` is GET-only behind the app login. Remote scripts need either SSM
  or a future API-key route through Caddy.

## SpectralEarth — known gaps

Added 2026-10-14 with `app/utils/spectralearth/`. Rationale in
[`lld/spectralearth.md`](lld/spectralearth.md).

- **No cloud screening yet.** SpectralEarth ships no quality layers, and DLR filtered
  clouds only per source scene (< ~10%, by eye), so some patches are partly or mostly
  cloud. The `ScreenChain` slot exists; a cloud screen is unbuilt and must first be shown
  to work without the 1380 nm band SpectralEarth dropped. Until then, cloud enters training
  as "normal" surface.
- **No georeference, date or source-scene id per patch** — patches can be found by site
  and patch id only. Searching by place or time needs a DLR metadata table we don't have.
- **The band table rests on a drop list from related literature** (HySpecNet), stated but
  not enumerated in the SpectralEarth paper. Three independent checks agree with it (see
  the LLD); a DLR-published band list would settle it.
- **`common_grid_pipeline.py` duplicates the EnMAP builder's step order** rather than the
  builder calling it. A change to either must be mirrored in the other.

## Spectral family matching — open gaps

Added 2026-10-15 with `app/spectral_match/family_match.py` and `data/family_library`.
Rationale and results in [`lld/spectral-family-match.md`](lld/spectral-family-match.md).

- **Not wired into `spectral_library_match` yet.** The module, library and tests are ready;
  the LLD gives the integration steps.
- **Radiometric scale unverified.** The method scores pixel minus local background, so pixels
  must be 0-1 surface reflectance like the library; SAM never needed this. Confirm that the
  onboarding vendable's `normalized_hyperspectral_cube` is reflectance for each sensor before
  wiring. A PRISMA band-filtered vendable sampled during this work had medians of 0.001-0.09.
  `match_families` logs a warning when the median is outside 0.005-1.5.
- **It names materials; it does not filter false alarms.** Ordinary pixels get a man-made
  name at most alpha (5 %) of the time, but false alarms are not ordinary. On the 4 PRISMA
  ground-truth scenes, 9-24 % of false-alarm pixels got a man-made name, and they outnumber
  real targets ~50 to 1. The UI must present the result as a likely material, not as
  evidence that a pixel is man-made.
- **Small evidence base on real targets.** Named-target rates were 38-47 % in 2 of the 4
  PRISMA scenes and 14-15 % in the other 2, plus 5 deployed sheets at Gudalur.
- **Fill is not reported.** No estimator recovered fill on real deployed materials without an
  in-scene or field spectrum of the material.

## Spectral-stem pretraining and fine-tuning — open gaps

Added 2026-10-14 with the `spectral_stem_mae` trainer and the EnMAP native shards. Rationale in
[`lld/spectral-stem.md`](lld/spectral-stem.md) and [`lld/enmap-shards.md`](lld/enmap-shards.md).

- **Mixed-sensor batches have no label-key policy.** `band_list_path` aligns bands, but EnMAP
  samples carry `label_*`/`meta.json` and SpectralEarth/PRISMA samples do not, so default
  collation fails on a mixed batch. Pre-training is single-sensor; this blocks only mixed
  fine-tuning.
- **The loss does not use the EnMAP cloud labels yet.** Fine-tuning on EnMAP as-is would teach
  cloud as normal surface; screening cloudy pixels out of the loss (except on the high, dry
  plateau scenes whose cirrus/cloud labels are false) is unbuilt.
- **No PRISMA shards.** PRISMA L2D (223 scenes, in the inventory) needs its own native-band
  reader before it can be sharded.

## Evaluation - known limits

Added 2026-10-17 with `app/evaluation/` and the stage-3 run. Rationale in [`lld/evaluation.md`](lld/evaluation.md)
and [`lld/implant-bench.md`](lld/implant-bench.md).

- **Implant bench-v1 is biased toward cube-level spectral oddity.** A model-free PCA-RX beats every model on it
  (objects caught top 1% 30.2% vs FT03's 26.9%) and loses clearly on the real PRISMA bench (pAUC 0.128 vs 0.158,
  TPR at 0.1% FPR 0.029 vs 0.058). The implants' difference from their ring lies outside the scene's main spectral
  directions far more than real objects' does (foreign fraction median 0.042 vs 0.006), and their edges are hard
  (halo / object contrast 0.60 vs 0.74). bench-v1 still orders the models consistently; it must not be a training
  target or the only judge. bench-v2 replaced it (2026-10-17), and bench-v2.1 replaced bench-v2 (2026-10-18).
- **`np.interp` on descending band lists (fixed 2026-10-18, b94dbfd).** PRISMA lists its bands descending; the
  evaluator's `resample` and the signature implants called `np.interp` unsorted, which NumPy does not check. v0.2.0's
  bench-v1 / bench-v2 numbers are void and every signature implant in bench-v1, bench-v2 and FT05's training was
  garbled. Any new interpolation over a sensor's band list must sort first (`app/evaluation/implant.interp_spectrum`);
  see [`lld/implant-bench.md`](lld/implant-bench.md), Invariant 5.
- **Cloud exclusion is not in the product.** Chosen 2026-10-17 (first water and cloud, then cloud only - vessels are
  in scope); the release candidate since 2026-10-18 is FT04 with cloud excluded on its own masks. Today it exists
  only as the `exclude` scorer in `app/evaluation/scorers.py`. In the product, water and
  cloud are left out only through an optional upstream `keep_mask` from `scene_segmentation` (NDWI / brightness
  thresholds) or `cloud_mask`. Under exclusion, 39 of the 909 real
  labelled objects (27 in PRISMA set1) cannot be found, because FT03 calls their every pixel water or cloud.
- **The class-map cache does not know the dataset's content.** Score maps are keyed by the scene's file
  fingerprint since 229d423; class maps (`cache/classes`) are not, so a bench rebuilt in place under the same scene
  names, evaluated into the same output directory, silently reuses old masks. Use a new bench path and output
  directory per version.
- **`provenance.json`'s `dirty` flag was always false before 229d423** (the git pathspecs resolved under
  `app/evaluation/`), so the stage-3 run records commit 060aabc although it ran on the then-uncommitted
  `app/evaluation/`.

## Config knobs that do nothing

Declared, set in real configs, never read by any code:

- `LRScheduleConfig.warmup_epochs` — no warmup path exists in `_build_scheduler`
- `CheckpointConfig.save_to_s3` / `s3_checkpoint_key` — no upload code exists (the
  `spectral_stem_mae` trainer uploads via its own `model_config.checkpoint_s3_uri` instead)
- `SpatialMaskedAutoEncoderConfig.masking_range` — the trainer hardcodes
  `uniform_(0.13, 0.25)`; only the *normalized* variant honours the config

## Dead code and registry gaps

- **`InferenceHarness`** (`app/utils/anomaly_detection/inference_harness.py`) is fully
  built, documented and covered by 22 tests — and unused. Production calls
  `get_detector()` → `fit()` → `detect()` directly, reaching into the detector's private
  `_spatial_mask` in between. Deliberately left alone for now; the options and the blocker
  are written up in
  [`docs/proposals/inference-harness-adoption.md`](proposals/inference-harness-adoption.md).
- `LRX` and `MNF_LRX` are in the detector registry but absent from the backend's
  capabilities table, so they are unreachable from the product. `StatisticalEnsembler` is in
  neither.
- `ADModel.CRD` and `FoundationModelName.SPECTRAL_COMPRESSOR` are enum members with no
  implementation.
- `inferencer_factory` has 5 entries; `trainer_factory` has 7. Two architectures can be
  trained but not run.

## Ops

- **Port mismatch.** Compose binds `3010:80` and `8010:8000`, but `docker/reset-stack.sh`,
  `scripts/remote_load.sh` and `scripts/remote_tunnel.sh` all still use `3000`/`8000`. The
  documented SSH tunnel forwards to dead ports.
- **Process-local caches break under more than one api process.** Preview masks, unpickled
  vendables and composite rasters are in-memory LRUs. Safe only because the CMD is a
  single-process uvicorn — adding `--workers` or a replica silently breaks preview.
- **`.dockerignore` has two no-op rules** — it excludes `data/aviris_samples/` and
  `model_break_down/`, but those live at `samples/…` and `research/model_break_down/`. The
  whole ~330 MB `research/` tree ships into the build context.
- **`bootstrap.sh` swallows a `seed-admin` failure** and continues, leaving a stack that
  boots but rejects every login (stderr warning only).
- **`models/job.py` declares `project_id` as a plain UUID** while a migration added a real
  FK — Alembic autogenerate would try to drop it.
- **Bare directory patterns in `.gitignore` have silently eaten source files twice.** A bare
  `lib/` matches at *any* depth, so `frontend/src/lib/` was uncommittable and two real
  modules (`elkLayout.ts`, `npy.ts`) were lost — the frontend simply would not build. A bare
  `.claude/` did the same to the project skills. Both are now anchored (`/lib/`,
  `.claude/*` + a negation). **Audit the rest of `.gitignore` for unanchored directory
  patterns** — this file was assembled from the standard Python template, which assumes a
  single-language repo, and this one has three.
- **`python -m allotrope.cli --help` likely crashes in the api image.** It runs typer
  0.13.1 against click 8.4.2; typer 0.13 calls `Parameter.make_metavar()` without the `ctx`
  click 8.2 made mandatory. The commands themselves run — only the help/options panel
  breaks. Pin `click<8.2` (the inventory CLI pins `click==8.1.8`) or bump typer.
- **A leaked Docker Desktop WSL forwarder can squat the published ports** (`wslrelay.exe`
  holding 3010/8010/5432 with no containers running), so `up` fails with "ports are not
  available". Restart Docker Desktop to clear it.
