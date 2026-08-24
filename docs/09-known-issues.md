# 9. Known issues

Found in a full-tree audit on **2026-08-24**. The repo carries no `TODO`/`FIXME` markers by
policy, so this file is the register. Fix an entry, delete the entry.

## Broken — blocks real use

**1. The frontend build fails — and `.gitignore` is the root cause.**
`frontend/src/pages/ModelDetailPage.tsx:33` imports `layoutWithElk` from `"../lib/elkLayout"`,
but `frontend/src/lib/` does not exist. `npm run build` (`tsc -b && vite build`) fails, which
also means **`docker compose up` cannot build the frontend image on a clean clone.**

The reason it's missing: `.gitignore:22` is `lib/` — a bare pattern inherited from the
standard *Python* gitignore template. With no leading slash, git matches it at **any** depth,
so `frontend/src/lib/` was silently never committable.

*Fix:* change `.gitignore:22` to `/lib/` (anchoring it to the repo root), then write
`frontend/src/lib/elkLayout.ts`. Recreating the file without fixing the ignore rule just
reproduces the bug for the next person.

Note `npm run dev` skips `tsc -b`, so it appears to start fine — the failure only surfaces
when you navigate to `/models/:architecture`.

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

## Config knobs that do nothing

Declared, set in real configs, never read by any code:

- `LRScheduleConfig.warmup_epochs` — no warmup path exists in `_build_scheduler`
- `CheckpointConfig.save_to_s3` / `s3_checkpoint_key` — no upload code exists
- `SpatialMaskedAutoEncoderConfig.masking_range` — the trainer hardcodes
  `uniform_(0.13, 0.25)`; only the *normalized* variant honours the config

## Dead code and registry gaps

- **`InferenceHarness`** (`app/utils/anomaly_detection/inference_harness.py`) is fully
  built, documented and covered by 22 tests — and unused. Production calls
  `get_detector()` → `fit()` → `detect()` directly. Either adopt it or delete it.
- **`app/models/intermediate_concepts/band_responses.py`** raises on import:
  `error_pixel_values = Optional[...] = Field(...)` is a chained assignment that attempts a
  subscript assignment on `typing.Optional`. Unnoticed because nothing imports the module.
  Delete it.
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

## Test coverage

118 test functions, all under `tests/`, mirroring `app/`. Coverage is `--cov=app` only.
**Zero tests for `backend/`, `frontend/`, or `app/foundation_models/`.** There is no
`.github/` directory — no CI at all.
