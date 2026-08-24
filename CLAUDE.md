# CLAUDE.md

Orientation for working in this repo. Written 2026-08-24 from a full-tree audit.

## What this is

**Allotrope** is a full-stack web product for satellite hyperspectral/thermal **anomaly
detection** — Postgres + JWT auth + React SPA + a job worker — not just an ML pipeline.

Sensors end-to-end: `prisma`, `enmap`, `aviris_ng` (hyperspectral); `landsat9`, `hotsat1`
(thermal).

Domain model: **Scene** (raw → vendable → annotations) → **Project** (bound to one Scene) →
**Action** (chains into a DAG) → ActionOutput / Visualization / Export / Note.

The analysis chain:

```
band_filter_apply    → filtered_vendable.pkl (+ nearest-valid pixel fill)
 → scene_segmentation | cloud_mask   → keep_mask.tif
   → anomaly_scoring                 → anomaly_score.{tif,png}, ROC/metrics
     → anomaly_detection_prep        → composite_score.tif + anomaly_mask.tif (human threshold)
       → spectral_library_match      → matches.parquet + match_map.tif   [hyperspectral]
         → POST /actions/{id}/export → submission bundle zip
```

## Skills

Two project skills in `.claude/skills/` load automatically — you should not need to be told
to use them:

- **`allotrope-orientation`** — domain and codebase briefing. Load before answering
  questions about this repo, tracing a feature, or planning a change.
- **`iterative-nano-chunking`** — the default coding workflow here. Propose a design and stop
  for agreement; break the work into chunks of ≤20 changed lines; execute **one chunk per
  message** and stop for feedback; finish by updating docs and LLDs.

The nano-chunking loop is not optional for non-trivial changes, and "these next few are
small so I'll batch them" is exactly the failure it exists to prevent.

## Docs

`docs/01-orientation.md` … `docs/09-known-issues.md` — nine files, read in order. They are
the current, verified documentation; treat them as authoritative.

Low-level designs live in `docs/lld/<subsystem>.md`, created when a change adds a subsystem
or logic whose rationale isn't obvious from reading it.

Also authoritative:
- **`final design/`** — the product/UX spec the frontend implements. Frontend source cites
  it by section. `ROADMAP.md` is the live work tracker.
- **`spectal_match_sample/WALKTHROUGH.md`** — the spectral-matching algorithmic spec, cited
  from `_spectral_library_match_run.py`.

The old `docs/tech/` tree, the legacy `docs/<topic>/` folders, and the root `CONTEXT.md` /
`TOC.md` / `TODO.md` / `TESTIN.md` were removed on 2026-08-24: `docs/tech/` chapter 1
contained fabricated classes and enum members, six files in chapter 5 documented a
`pixel_stats_override` mechanism that does not exist, and every source link in the tree was
broken (`../../app/` resolves to `docs/app/`). All of it is recoverable from git if needed.

There are **zero `TODO`/`FIXME` markers by policy**. Unfinished work is tracked as prose
"Step N" references in docstrings and in `final design/ROADMAP.md`. Grep `"Step "`.

## Layering rule — respect this

- **`app/`** — portable science. numpy/torch/rasterio only. **No DB, no FastAPI imports.**
- **`backend/allotrope/`** — shared by api and worker (routers, ORM, auth, action registry).
- **`backend/allotrope_worker/`** — worker only (claim loop, heartbeat, reaper, handlers).

Action-type modules must **not** import `app.*`, torch or rasterio at module top level — only
inside `run()`/`summarize()`/`preview()`. That's what the `_<kind>_run.py` files are for: they
hold the heavy implementation behind the lazy-import boundary so the api's import graph stays
light.

To trace a feature: the science is in `app/`, but the thing that *runs* it is a
`backend/allotrope/action_types/<kind>.py` + `_<kind>_run.py` pair, dispatched by
`backend/allotrope_worker/action_run.py`.

## Key facts that will surprise you

- **No Redis/Celery/RQ.** The queue is the `jobs` Postgres table, claimed with
  `FOR UPDATE SKIP LOCKED` + committed immediately; ownership is then expressed by heartbeat
  freshness. Stale jobs are reaped into `failed`.
- **All FastAPI handlers are sync `def`.** No async anywhere.
- **No `/api` prefix in FastAPI and no CORS** — nginx proxies `/api/*` → `api:8000` and strips
  it. Single origin by construction.
- **Wire ids are prefixed**, never bare UUIDs: `scene_<uuid>`, `action_<uuid>`, …
  (`backend/allotrope/api/wireformat.py`).
- **The frontend has no state library, no fetch library, no UI library, no env vars, no lint,
  no tests.** One 5.9k-line `index.css`. `API_BASE = "/api"` is hard-coded. Live updates are
  polling. Match this style.
- **Vendables carry no spatial reference** — every action-written GeoTIFF has an identity
  transform. CRS/affine is recovered at export time by re-reading the raw file
  (`app/georef/`).
- **The 165-band common grid** (10 nm, 460–2450 nm, atmospheric windows excluded) is what makes
  PRISMA/EnMAP/AVIRIS mutually comparable and lets one model train on mixed-sensor shards.
- **Foundation models carry Indic codenames** and the backend selects by codename:
  Pratibimba, Antardhana, Tirohita, Asanskrita, Drashta, **Chakshu** (thermal SegFormer MAE),
  **Indradhanu** (hyperspectral SegFormer MAE). Architectures live in
  `app/foundation_models/components/`, not `app/ml_models/`.
- **Plain RX on hyperspectral was deliberately dropped** (2026-05-11 — near-singular
  covariance, distances → 1e11). MNF-RX is the replacement. Don't reintroduce it.
- **No CI.** There is no `.github/`. `--cov=app` only; `backend/` and `frontend/` have zero
  tests.

## Known breakage — verify before relying on either

1. **The frontend build fails.** `frontend/src/pages/ModelDetailPage.tsx:33` imports
   `../lib/elkLayout`; `frontend/src/lib/` does not exist because root `.gitignore:22` is
   `lib/` — a bare pattern that matches at any depth. Fix the ignore rule to `/lib/` first,
   then write the file.

Others: `app/models/intermediate_concepts/band_responses.py` has a chained-assignment bug that
raises on import (module is unused); `InferenceHarness` is fully built and tested but unused in
production; deploy scripts still reference ports 3000/8000 while compose binds 3010/8010.

## Running things

```bash
docker compose -f docker/docker-compose.yml up -d      # full stack → http://localhost:3010
python -m pytest -m "not large_files and not large_benchmarks and not network_access"
python scripts/train_foundation_model.py configs/<exp>.json
```

`tests/test_payloads/` is gitignored — the `large_files` marker gates the tests that need it.
