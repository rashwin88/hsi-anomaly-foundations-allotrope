---
name: allotrope-orientation
description: Domain and codebase briefing for the Allotrope repo (hsi-anomaly-foundations-allotrope) - hyperspectral and thermal satellite anomaly detection. Load this BEFORE answering any question about this codebase, tracing a feature, planning a change, or writing code here. Covers the domain vocabulary (hyperspectral cube, band, spectrum, vendable, anomaly-as-reconstruction-error), the Scene/Project/Action model, the app-vs-backend layering rule, the conventions that silently break things if ignored, and what is currently broken. Triggers on - anomaly detection, hyperspectral, HSI, PRISMA, EnMAP, AVIRIS, Landsat, HotSat, vendable, band filter, spectral, RX detector, SegFormer, Indradhanu, Chakshu, action type, scene onboard, splib07, or any work inside this repository.
---

# Allotrope: orientation

Read this before touching anything. It exists so you don't have to rediscover the domain
or trip the same wires everyone else does.

## What the product is

A **web application** for finding anomalies in satellite imagery. Not a script collection.

A user uploads a **Scene** (a raw satellite file), creates a **Project** on it, runs a chain
of **Actions** over it, picks a threshold by eye, and exports a georeferenced bundle naming
each candidate anomaly and its likely material.

```
Scene ──► Project ──► Action ──► Action ──► … ──► Export
```

The shipped chain:

```
band_filter_apply → scene_segmentation | cloud_mask → anomaly_scoring
  → anomaly_detection_prep → spectral_library_match → export
```

## Domain in 90 seconds

- A **hyperspectral** sensor records 200+ narrow wavelength bands per pixel. That per-pixel
  reflectance-vs-wavelength curve is a **spectrum**, and it acts as a material fingerprint.
  So an image is a 3D **cube**: height x width x bands.
- **Thermal** sensors have one band: surface temperature.
- **Anomaly = surprise.** There are no labels. We learn what *normal* looks like in this
  scene and score each pixel by how badly it fits. Two families do this:
  - **Classical detectors** - statistical distance from the scene's own background (RX).
    No training, no checkpoint.
  - **Foundation models** - networks trained to *reconstruct* imagery. The reconstruction
    error IS the anomaly score.
- A **vendable** (`VendableDataset`) is the normalised cube + validity masks that every
  downstream component consumes. It is the currency of the system.
- All hyperspectral sensors are resampled onto **one common 165-band grid**
  (10 nm, 460-2450 nm). That is what lets a single model serve PRISMA, EnMAP and AVIRIS-NG.

Sensors: `prisma`, `enmap`, `aviris_ng` (hyperspectral); `landsat9`, `hotsat1` (thermal).

## Codebase map

```
app/          portable science - numpy/torch/rasterio. NO database, NO FastAPI.
backend/
  allotrope/          shared by api AND worker - routers, ORM, auth, action registry
  allotrope_worker/   worker only - claim loop, heartbeat, reaper, handlers
frontend/     React 19 + Vite SPA
scripts/      patch generation, training, splib curation, deploy
research/     experiments, notebooks, per-model walkthroughs
final design/ the product/UX spec the frontend implements
docs/         01-orientation .. 09-known-issues
```

## The rules that bite

**1. The layering rule.** `app/` is portable science and must never import the database or
FastAPI. `backend/` orchestrates.

**2. The lazy-import rule.** Action modules must NOT import `app.*`, `torch` or `rasterio`
at module top level - only inside `run()`/`summarize()`/`preview()`. The api imports every
action module at startup; a top-level heavy import loads torch into the web process. This is
why you see paired files: `anomaly_scoring.py` (light) + `_anomaly_scoring_run.py` (heavy).

**3. Wire IDs are prefixed, never bare UUIDs.** `scene_<uuid>`, `action_<uuid>`,
`project_<uuid>`, `job_<uuid>`. Use `backend/allotrope/api/wireformat.py`. Returning a raw
UUID is a bug.

**4. No `/api` prefix in FastAPI, and no CORS.** nginx proxies `/api/*` and strips it. Same
origin by construction. Don't add CORS middleware; don't hardcode `/api` in a response body.

**5. Every API handler is sync `def`.** No async anywhere - the DB session is sync and one
`async` endpoint will block the loop.

**6. The queue is Postgres**, not Redis/Celery. `SELECT ... FOR UPDATE SKIP LOCKED`, then
ownership is expressed by heartbeat freshness. Handlers must NOT commit or rollback - the
runner owns the commit.

**7. The frontend is deliberately dependency-light.** No state library, no fetch library, no
UI kit, no Tailwind, no env vars, no tests. Adding one is a design change, not a convenience.

**8. Thresholds are percentile, never absolute.** Typical residuals differ by an order of
magnitude between scenes; a fixed cut flags everything in one and nothing in another.

**9. Vendables carry no spatial reference.** Action-written GeoTIFFs have identity
transforms; CRS/affine is recovered at export time from the raw file (`app/georef/`).

**10. There are zero TODO/FIXME markers by policy.** Unfinished work lives in
`docs/09-known-issues.md`, `final design/ROADMAP.md`, and prose "Step N" references in
docstrings. Grep `"Step "`.

## Adding things

- **A capability** = a new **Action type**, not a new route. Add
  `backend/allotrope/action_types/<kind>.py` exporting `KIND`, `META`, `validate_config`,
  `run`, `summarize`, `preview`; register it in `__init__.py`. `META` drives the UI picker,
  so you do not touch frontend code to add one.
- **A model** - architecture in `app/foundation_models/components/`, trainer, inferencer,
  register in **both** factories, then add a capabilities entry in
  `backend/allotrope/foundation_models/resolver.py`. Step 5 is the one people forget.

## Currently broken - check before assuming

1. **The frontend build fails.** `ModelDetailPage.tsx:33` imports `../lib/elkLayout`, which
   does not exist because root `.gitignore:22` is `lib/` - a bare pattern matching at any
   depth. Fix the ignore rule to `/lib/` first, then write the file.

Full register: `docs/09-known-issues.md`.

## Verify, don't trust prose

Documentation in this repo has historically drifted ahead of implementation. A previous doc
tree documented classes that did not exist, enum members that were invented, and a config
mechanism that had been reverted. **Before relying on any doc claim about a class name, enum
member, config field, or default value, check the source.** The `docs/01-09` set was written
against verified source in 2026-08, but the same rule applies to it as it ages.

## Where to read more

`docs/01-orientation.md` through `docs/09-known-issues.md`, in order. Also `final design/`
(product/UX spec, cited by section from frontend source) and
`spectal_match_sample/WALKTHROUGH.md` (spectral matching spec, cited from
`_spectral_library_match_run.py`).
