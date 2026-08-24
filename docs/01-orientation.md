# 1. Orientation

**Read time: 8 minutes.** Start here.

## The problem

A satellite photographs a patch of ground. Somewhere in it is something that doesn't
belong — a gas leak, an illegal dump, a hot spot, an unusual material. Nobody has labelled
it. Nobody knows what it looks like in advance.

Allotrope finds those pixels, and for hyperspectral imagery, tells you what material they
probably are.

## Why hyperspectral

An ordinary photo has 3 colour channels. A **hyperspectral** sensor records 200+ narrow
wavelength bands per pixel. That per-pixel curve of reflectance against wavelength is a
**spectrum** — effectively a material fingerprint, because different minerals, gases and
plants absorb light at different wavelengths.

So an image isn't a 2D picture but a 3D **cube**: height × width × bands.

**Thermal** sensors are the simple cousin — one band, surface temperature.

## The core idea: anomaly = surprise

We never train on labelled anomalies, because we have none. Instead:

1. Learn what **normal** looks like in this scene.
2. Score every pixel by **how badly it fits normal**.
3. The worst-fitting pixels are the candidates.

Two families of method do this, and both ship:

- **Classical detectors** — statistical distance from the scene's own background
  (the RX family). No training, no checkpoint. → [5](05-detectors.md)
- **Foundation models** — networks trained to *reconstruct* imagery. Show one a scene and
  it redraws it from context; wherever the redrawing is badly wrong, something unusual is
  there. The reconstruction error **is** the score. → [4](04-models.md)

## What the product does

A web application, not a script collection.

```
Scene ──► Project ──► Action ──► Action ──► … ──► Export
  ▲                      │
  │                      └── each Action's output feeds the next
upload
```

1. **Upload a Scene** — a raw satellite file. The system normalises it into a standard cube
   (a "vendable") and renders previews.
2. **Create a Project** on that Scene.
3. **Run Actions** — chained processing steps.
4. **Pick a threshold** interactively, looking at the score map.
5. **Export** a georeferenced bundle: GeoTIFF + Shapefile + CSV naming each candidate, its
   coordinates, and (hyperspectral) its matched material.

The shipped Action chain:

```
band_filter_apply → scene_segmentation | cloud_mask → anomaly_scoring
  → anomaly_detection_prep → spectral_library_match → export
```

## Sensors

| Sensor | Kind | Upload |
|---|---|---|
| PRISMA | hyperspectral, 239 bands | single `.he5` |
| EnMAP | hyperspectral, 224 bands | folder |
| AVIRIS-NG | hyperspectral | folder |
| Landsat 9 | thermal (B10) | single `.tif` |
| HotSat-1 | thermal | folder |

All hyperspectral sensors are resampled onto **one common 165-band grid**, which is what
lets a single model serve all of them. → [3](03-data-pipeline.md)

## Codebase map

```
app/          portable science — numpy/torch/rasterio. NO database, NO FastAPI.
  foundation_models/   architectures, trainers, inferencers
  detectors/           classical RX-family detectors
  utils/               band filtering, patching, scoring, spectral indices, STAC
  models/              Pydantic domain types (vendables, configs)
  spectral_match/      USGS splib07 material matching
  georef/              recover CRS/affine from the raw file at export time

backend/
  allotrope/           shared by api AND worker — routers, ORM, auth, action registry
  allotrope_worker/    worker only — claim loop, heartbeat, reaper, handlers

frontend/     React 19 + Vite SPA
scripts/      patch generation, training, splib curation, deploy
research/      experiments and notebooks (committed outputs; large)
final design/  the product/UX spec the frontend implements
```

### The one rule to internalise

**`app/` is portable science. `backend/` is orchestration.** `app/` never imports the
database or FastAPI. And action modules must not import `app.*`, torch or rasterio at module
top level — only inside `run()`. → [6](06-backend.md)

To trace a feature: the science lives in `app/`, but the thing that *runs* it is a
`backend/allotrope/action_types/<kind>.py` + `_<kind>_run.py` pair.

## Where to go next

| You want to… | Read |
|---|---|
| Run it locally | [2. Setup](02-setup.md) |
| Understand cubes, bands, vendables | [3. Data pipeline](03-data-pipeline.md) |
| Work on the neural models | [4. Models](04-models.md) |
| Work on RX / scoring / material matching | [5. Detectors](05-detectors.md) |
| Add an API endpoint or an Action | [6. Backend](06-backend.md) |
| Touch the UI | [7. Frontend](07-frontend.md) |
| Ship it somewhere | [8. Deploy](08-deploy.md) |
| Know what's broken | [9. Known issues](09-known-issues.md) |

## A note on the name

"Allotrope" — different structural forms of the same element. The foundation models carry
Sanskrit codenames (Indradhanu, Chakshu, …) and the backend refers to them that way.
