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

## Deliberate limitations in segmentation sharding

Added 2026-08-25 with the `enmap_seg` lane. All known and accepted, not defects. Design
rationale in `docs/lld/segmentation-sharding.md`; the S3-coupling debt this sits on top of
is costed in `docs/tech-debt/s3-coupling-in-sharding.md`.

- **`EnmapSegmentationSharder` implements `s3_searcher` / `s3_downloader` and never touches
  S3.** The names come from the ABC, which predates the storage seam. Renaming them means
  editing three other sharders that produce Indradhanu's training data, so it waits for
  the migration described in the debt entry.
- **A scene that fails mid-patching leaves nothing to inspect.** The partial `.tar` is
  removed in `finally` so a resume cannot mistake it for finished work; the trade is that
  debugging a failure means re-running that scene.
- **`S3SceneStorage.shard_exists` catches `ClientError` broadly**, so a permissions failure
  reads as "shard absent" and the work is redone. Errs toward doing work rather than
  skipping it, which is the safe direction, but it delays surfacing an auth problem.
- **Stratifying under `S3SceneStorage` downloads every scene** — `fetch_scene` has no
  metadata-only mode, so splitting 212 scenes pulls ~64 GB. Cheap on a local/Drive mount
  (~3 ms/scene), which is the intended path.
- **`app/utils/files/enmap_scene_cover.py` has no test.** It parses a real METADATA.XML and
  those are gitignored payloads. A four-tag synthetic document is enough to exercise it —
  proven incidentally during development — so this is worth closing.
- **Storage is large.** At stride 64 a 165-band float16 patch is ~8.2 MB (5.4 pixels +
  2.7 per-band validity + 0.1 labels), so 212 scenes come to **~562 GB**. Per-band validity
  is retained deliberately even though the current trainers read only band 0, to keep
  per-band masking open. Dropping it to a 2D mask saves a further ~33%.
- **A float16 shard must be cast to float32 by its trainer.** The models are fp32.
  Segmentation shards store float16; the reconstruction lane still stores float32.

## Config knobs that do nothing

Declared, set in real configs, never read by any code:

- `LRScheduleConfig.warmup_epochs` — no warmup path exists in `_build_scheduler`
- `CheckpointConfig.save_to_s3` / `s3_checkpoint_key` — no upload code exists
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
- **A leaked Docker Desktop WSL forwarder can squat the published ports** (`wslrelay.exe`
  holding 3010/8010/5432 with no containers running), so `up` fails with "ports are not
  available". Restart Docker Desktop to clear it.
