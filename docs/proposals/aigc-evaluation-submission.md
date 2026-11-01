# AIGC Stage-III evaluation submission - closing the gaps

**Status:** agreed 2026-10-31 (user: native bands with backward compatibility; points per pixel as now; the 3
suggested submissions; the namer in the report as preliminary). **C1-C6 built and tested the same day; C7 done on the
local stack** (images rebuilt, worker `running`, export rehearsed on a committed PRISMA prep: `hyper_*` bundle, uint8
{0,1} mask in the product's UTM zone). Measured on Set-1: the old EPSG:4326 fit put pixels a median 2.4 px (95% 4.1,
max 5.6) from the product grid; internal evaluation (pixel-space) was never affected. Still open: the thermal smoke
(no data yet), C6 on the real atmospheric-correction output (Q4), the namer column on a real N001 checkpoint. Plan only: the target is the **local stack**
(WSL `/home/fitt/allotrope`, http://localhost:3010). A prod / airgap-VM deploy is **not approved** and is listed as a
separate last step.

**Deadlines:** report (chapter "Findings on Evaluation Dataset: 02 Nov 2026") due 02 Nov 2026 midnight; presentation
04 Nov 2026 09:00.

**Short answer:** most of the pipeline is already built. The product already makes a georeferenced boolean 0/1 mask
GeoTIFF with a full shapefile set: it is the `anomaly_detection_prep` export. The gaps are:
1. **PRISMA georeferencing is approximate** (an EPSG:4326 bounding-box fit instead of the HE5's own UTM grid).
2. **Hyperspectral prep bundles are named `thermal_`.**
3. **Indradhanu V2 cannot be run from the UI**, because no UI control sets the band filter to native bands.
4. **There is no per-object SAM vs foundation-model table** for the report.
5. **No L1 to reflectance path exists in the product.** That deliverable has to come from the other session's
   atmospheric-correction module plus a writer script.

Branch note: this branch (`exp/airgap-vm-update`) is 54 commits ahead of `main`, and `main` lacks residual-Mahalanobis
and Indradhanu V2. The running local images do have both: checked inside `allotrope_worker` on 2026-10-31.

---

## 1. Requirement -> current state -> gap

| # | Requirement | Current state (evidence) | Gap |
|---|---|---|---|
| R1 | Reflectance product from L1 (atmospheric + radiometric correction), GeoTIFF or BSQ, all wavelengths, 0-1 or x10000, water bands may be dropped | Not in the product. Onboarding reads L2 only: PRISMA L2D (`app/utils/dataset_builder/prisma_dataset_builder.py:2,60`) and EnMAP L2A (`enmap_dataset_builder.py:2,41`). No ISOFIT / ACOLITE / 6S anywhere under `app/`, `backend/` or `scripts/`. The atmospheric-correction module lives on branch `vinoth/atm-isofit4` (worktree `C:/wt/atm-isofit4`), owned by another session (`research/experiments/atm-correction/README.md`). The repo has no ENVI or GeoTIFF reflectance writer. | **Large.** Out of UI scope. Needs the other session's module plus a ~80-line writer script (C6). |
| R2a | Boolean 0/1 anomaly map as **GeoTIFF**, real CRS | The prep commit writes uint8 0/1 with `nodata=None` (`backend/allotrope/api/action_threshold.py:439-455`). The export copies it in and applies the georef from the raw scene file (`app/anomaly_detection/export.py:84-111`, `backend/allotrope/api/action_export.py:206-239,282-299`). Works for **any** sensor, hyperspectral included. | (a) **PRISMA georef is approximate.** `_resolve_prisma` fits EPSG:4326 `from_bounds` to the min/max of the per-pixel lat/lon (`app/georef/from_scene_file.py:139-145`). The ground-truth rasters are UTM: set1 has transform `[30,0,545007.75,0,-30,2501437.5]` in EPSG:32642, and that origin is the HE5 `Product_ULcorner_easting/northing` minus / plus half a pixel (`research/experiments/gt-had-prisma/set1/metrics.json`). The 4326 fit is therefore not the evaluator's grid; it can be off by several pixels at the edges, and it treats pixel-centre lat/lon as edges. EnMAP and AVIRIS read the vendor GeoTIFF / ENVI directly, which is fine. (b) A **hyperspectral** prep bundle is named `thermal_*` (`export.py:150-164`). The 2026-05-14 rules require `hyper` / `thermal` in names (`final design/ROADMAP.md` 14.8). |
| R2b | Same map as **.shp** | geopandas 1.0.1 + fiona 1.10.1, sidecars `.shp/.shx/.dbf/.prj/.cpg` (`app/anomaly_detection/export.py:199-215`). Pinned in `backend/requirements.txt` and `requirements-worker.txt`, and import fine in the running `allotrope_api`. It writes **one Point per anomaly pixel** with a continuous `score` column (`export.py:128-139,202`). | The `score` attribute is heatmap-like, so drop it from the shapefile to be safe. Points or polygons is an open question (Q2). |
| R2c | No heatmap-style output | The prep bundle holds no score raster; it reads `composite_score.tif` only to fill the `score` column. **But** (i) the hyperspectral `spectral_library_match` bundle's raster `hyper_anomaly_materials.tif` is int16 **family codes** (-2..F), not 0/1 (`app/spectral_match/export.py:133,228-233`), and it contains no boolean mask at all. (ii) The project export (`POST /projects/{id}/exports`, ROADMAP 17) zips every output tree, `anomaly_score.tif` included. | **Submit only prep bundles.** Never submit a project export or the materials raster as the detection map. Optionally, add a boolean mask to the hyperspectral bundle (N3). |
| R2d | Up to 3 results per scene | One committed mask per prep action; a re-commit overwrites it (`action_threshold.py:291-293,453`). Several prep actions can hang off one scoring output, and several scoring actions can hang off one band filter. Each export zip carries the action's short id, so zips are already distinct. | **None for the backend.** It is a procedure: 3 committed prep actions, then 3 exports. Optional `?tag=` for readable names (N1). |
| R2e | Match detected pixels to USGS / open libraries with a **foundation model and classic SAM**; pros and cons in the report | SAM top-K against splib07 (`matches.parquet`) and calibrated Simplex-ACE family match (`family_matches.parquet`), both per pixel, in `spectral_library_match` (`backend/allotrope/action_types/spectral_library_match.py:1-35`). They are exported side by side per pixel (`app/spectral_match/export.py:180-206`). The family match is **classical**, not a foundation model. The namer (`app/spectral_match/namer/`) has no inference API and is not wired in. It is training as N001 on the H100 until 2026-11-02 11:30 UTC. The trainer writes `ep{n}.pt` every epoch (`namer/train.py:229-236`), so an intermediate checkpoint is usable. | **No per-object table, and no foundation-model column.** C4 and C5: a script, plus a small namer inference module. |
| R3 | Thermal: boolean GeoTIFF + .shp | The code path exists end to end: `cloud_mask` -> `anomaly_scoring` (default `Chakshu`, `anomaly_scoring.py:198-212`; or `Thermal-GRX`) -> prep -> `build_thermal_bundle`. The georef comes from the thermal GeoTIFF (`from_scene_file.py:237-275`). Chakshu `segformer_mae_v0.8.0_epoch495.pt` is present in the `/models` volume. **No thermal scene is in the local DB**, so the chain is unexercised on this stack. HotSat DN-units guard not in effect (`docs/09-known-issues.md` #7). | **Untested** (C7: one smoke run when the data arrives). |
| M1 | Indradhanu V2 (Heavy, FTH3 ep5, `v2.1.0`) usable | The manifest `/models/spectral_stem_finetune/current.json` serves `spectral_stem_finetune_v2.1.0_epoch5.pt`, codename "Indradhanu V2". The resolver applies `thin_cloud_is_cloud=False`, `exclude_classes=("cloud",)` and the residual-Mahalanobis default (`backend/allotrope/foundation_models/resolver.py:163-177`). It appears in the UI picker for hyperspectral scenes: manifest `sensor` is `hyperspectral`, filtered at `frontend/src/components/NewActionDialog.tsx:1013-1035`. **Not the default:** `_DEFAULT_PRISMA/ENMAP/AVIRIS_NG` still say `["Indradhanu"]` = v0.2.0 (`anomaly_scoring.py:190-205`). **Blocked in the UI:** V2 needs `use_common_wavelength_grid=false` (`_anomaly_scoring_run.py:93-106`), every band-filter default is `True` (`band_filter_apply.py:129,144,157`), and the dialog has no control for it (no hit for `use_common_wavelength_grid` in `frontend/src`). The local DB has 18 band filters, all `true`, and no V2 scoring action. | **Must fix** (C3). |
| M2 | Residual-Mahalanobis scoring | A **product option, not eval-only**. It is in `_VALID_METHODS` (`anomaly_scoring.py:66`) and is the default scoring method for both Indradhanu architectures (`resolver.py:153-158,169-170`). It is implemented in `app/utils/anomaly_detection/residual_mahalanobis.py` and called from `scoring.py:50`. | None. |

## 2. Change list (ordered)

### Must-have for 02 Nov (local stack)

| # | Change | Files (approx. changed lines) | Risk |
|---|---|---|---|
| C1 | **PRISMA georef from the HE5 grid.** If the root attrs carry `Epsg_Code` and `Product_{UL,UR,LL}corner_{easting,northing}`, build `Affine(dx,0,ULe-dx/2, 0,-dy,ULn+dy/2)` with `dx=(URe-ULe)/(W-1)` and `dy=(ULn-LLn)/(H-1)` (the corners are pixel centres). Check that `(H,W)` matches the lat/lon array, otherwise fall back to today's bbox fit. Add a unit test: set1's attrs (in `research/experiments/gt-had-prisma/set1/metrics.json`) must give the ground-truth transform exactly. | `app/georef/from_scene_file.py` `_resolve_prisma` (~20); `tests/test_utils/test_georef_prisma.py` (new, ~30) | Medium: changes every PRISMA export. The test pins it to the ground-truth grid. The api image needs a rebuild (h5py is already in it). |
| C2 | **Prep bundle prefix by modality, no score in the shapefile.** `prefix = "thermal" if sensor_type in ("landsat9","hotsat1") else "hyper"` for the folder and file names. Drop `score` from `keep_cols` in the shapefile (keep it in the CSV, which is not submitted). Update the docstring. | `app/anomaly_detection/export.py:1-20,150-164,202` (~10) | Low. The frontend fallback name (`ActionDetailPane.tsx:2550`) is only used when the header is missing, so it needs no change. |
| C3 | **Make Indradhanu V2 runnable from the UI.** Option A, recommended for the deadline: flip `use_common_wavelength_grid` to `False` in the 3 hyperspectral band-filter defaults (no frontend change). Then V2 and MNF-RX work. Old v0.2.0 "Indradhanu" fails with its clear "needs the 165-band grid" error. Option B: a "Keep native bands (Indradhanu V2)" checkbox in `NewActionDialog.tsx` for `band_filter_apply` (~15 lines, frontend). | A: `backend/allotrope/action_types/band_filter_apply.py:129,144,157` (3) + docs (`docs/09-known-issues.md` "Spectral-stem", `docs/training/.../23-using-indradhanu-v2.md`). B: `frontend/src/components/NewActionDialog.tsx` (~15) | A: low code risk, but it is a product decision (known-issues lists "V2 default" as Phase 2), so **Q1**. Leave the `anomaly_scoring` codename defaults alone: the UI makes the user pick models explicitly (`NewActionDialog.tsx:69,215`). |
| C4 | **Namer inference module** (offline-callable): load a `train.py` checkpoint (`model`, `config`, `classes`), encode the family library at the scene's bands, compute the 2-7 px ring mean (reuse `generator.ring_and_noise`), and return per pixel `p_manmade`, `cls`, `p_cls`, `fill`. Pure `app/`: torch only, no DB. | `app/spectral_match/namer/infer.py` (new, ~70); `tests/test_spectral_match/test_namer.py` (+~15, random-weight smoke) | Low (new file). It depends on N001 producing a usable `ep*.pt` or `best.pt` before the report. |
| C5 | **Per-object report table (script).** Input: a `spectral_library_match` artifact dir, plus its upstream prep `anomaly_mask.tif` and the scene. 8-connected components (`scipy.ndimage.label`). Per object: `obj_id, n_px, centroid lat/lon` (via `app.georef`), SAM majority top-1 + median angle + top-3 names, family majority + mean `fconf` + `named` share, and namer `p_manmade`, class and `p_cls` when `--namer-ckpt` is given. It writes `objects.csv` and `objects.geojson`, plus a short `findings.md` stub (agreement rate of SAM vs family vs namer, and per-object disagreements) as the starting point for the report chapter. | `scripts/aigc_object_table.py` (new, ~150) | None to the product. Runs in the worker container (it has torch and scipy) or in the `C:/av` venv. |
| C6 | **Reflectance writer (script)** for R1. Input: the atmospheric-correction module's output cube, wavelengths, fwhm and georef. It removes water bands (default 1340-1460 and 1790-1960 nm; configurable), scales x10000 to int16 with nodata -9999, and writes (a) a GeoTIFF with per-band descriptions `"<nm> nm"` and `wavelength` / `wavelength_units` metadata in the ENVI domain, and (b) ENVI **BSQ + .hdr** with `wavelength`, `fwhm`, `wavelength units = Nanometers`, `reflectance scale factor = 10000`, `data ignore value`, and `map info` / `coordinate system string`. rasterio only. | `scripts/write_reflectance_product.py` (new, ~90) | None to the product. It needs the other session's module output (Q4). |
| C7 | **Verify on the stack.** Rebuild `api` and `worker`, then `docker compose ps -a` (the worker must show `running`). Then run a dry rehearsal on a labelled PRISMA scene (set1): 3 chains, 3 commits, 3 exports. Open each zip in QGIS over the ground-truth GeoTIFF. Check that `gdalinfo` shows the GT's EPSG and transform and that the mask values are {0,1}. Run C5 on one chain. Thermal: one Landsat-9 scene through cloud_mask -> Chakshu -> prep -> export, once the data arrives. | none | The stack was restarting during this audit: `fakevm` is up and `allotrope_inventory` is in `Restarting (255)`, probably the airgap-update session. Coordinate before rebuilding. |

**Suggested 3 submissions per hyperspectral scene** (all residual-Mahalanobis unless noted; the user confirms in Q3):
1. Indradhanu V2 (Heavy).
2. Indradhanu V2 + MNF-RX combined at the prep weights.
3. MNF-RX alone, or V2 at a second threshold.

Each one is a separate committed prep action.

### Nice-to-have (after the deadline or if time allows)

| # | Change | Files (approx. lines) | Risk |
|---|---|---|---|
| N1 | `?tag=r1` query parameter on `POST /actions/{id}/export`, appended to the zip and folder names. A "Result 1/2/3" select next to the prep's Export button. | `action_export.py:153-168` (~6), both builders (~4 each), `ActionDetailPane.tsx:2514-2580` (~12) | Low |
| N2 | Polygon shapefile (`rasterio.features.shapes` on the mask, one polygon per object, `value=1`) instead of, or as well as, points. | `app/anomaly_detection/export.py` (~20) | Low. Depends on Q2. |
| N3 | Hyperspectral `spectral_library_match` bundle also ships `hyper_anomaly_mask.tif` (0/1, read from the upstream prep) and the C5 per-object CSV. | `app/spectral_match/export.py` (~25), `action_export.py` (~8) | Low |
| N4 | Product-wire the namer: optional `namer_checkpoint: str \| None` on `SpectralLibraryMatchConfig` (default None means skipped, as with family's failure-tolerant block). The run calls C4 on the masked pixels and writes `namer_matches.parquet`. The export joins `nm_mm, nm_cls, nm_p` into the DBF and CSV beside SAM and family. | `spectral_library_match.py` (~8), `_spectral_library_match_run.py` (~35), `app/spectral_match/export.py` (~15) | Medium: worker torch path, lazy-import rule. Only after N001 passes (the proposal's own gate). |
| N5 | Reflectance-export action (vendable -> GeoTIFF/BSQ). Only useful once an L1 onboarding path exists. Today the vendables are vendor L2, so it would export the vendor's own product. | new `action_types/reflectance_export.py` + `_run.py` | Not worth it before L1 onboarding exists. |
| N6 | Fix the scene search stripping `_` (`docs/09-known-issues.md` #8): PRISMA ids are unsearchable in the UI. | `backend/allotrope/api/scenes.py` (~3) | Low |
| D1 | **Deploy (separate, not approved):** the airgap VM / prod update per `docs/08-deploy.md` (another session owns `docker/airgap-update/`). Keep volumes and `.env`. | - | Needs explicit approval. |

## 3. Fallback outside the UI

If the UI path slips, everything can be produced by scripts against the artifact volume:
- **Detection map:** `build_thermal_bundle` / `resolve_scene_georef` are plain `app/` functions. A ~30-line driver can
  call them on any committed prep's artifact dir. The local `/artifacts` volume holds 22 committed preps (PRISMA,
  EnMAP, AVIRIS). Or threshold `composite_score.tif` directly and write the GeoTIFF and shapefile with the same code.
- **Scoring without the UI:** `scripts/evaluate.py configs/eval/<name>.json` (`docs/lld/evaluation.md`) already runs
  the V2 residual-Mahalanobis scorers on labelled scenes.
- **Report table:** C5, which is a script by design.
- **Reflectance:** C6 on the atmospheric-correction module's output, which is a script by design.

## 4. Open questions for the user

1. **C3:** flip the band-filter default to native bands now (V2 becomes the practical default, and old v0.2.0 needs
   an explicit 165-grid filter), or add the UI checkbox instead?
2. **Shapefile geometry:** keep one point per pixel, or one polygon per object (N2)? Is there any evaluator template
   for attribute names?
3. **The 3 results:** which models and thresholds? Without ground truth on the evaluation data the threshold is
   chosen by eye. Should a fixed percentile per model be decided now from the 4 labelled PRISMA scenes?
4. **R1 input:** which L1 level arrives (PRISMA L1, EnMAP L1B / L1C), and will the atmospheric-correction module
   (branch `vinoth/atm-isofit4`) be frozen and run by its owner in time? Should the reflectance stay in sensor
   geometry (PRISMA L1 is not map-projected) or be gridded? GeoTIFF, BSQ, or both?
5. **Evaluation L2 format:** will it arrive as standard PRISMA L2D `.he5` / EnMAP L2A folders (which onboard
   today), or as a plain GeoTIFF / ENVI cube (which does not onboard)?
6. **Foundation-model column:** if N001 has not passed by 02 Nov, report the namer as "experimental, epoch k"
   alongside SAM and family, or leave it out?
7. Is a **thermal** evaluation dataset confirmed? If so, which sensor? Chakshu is untested on this stack.
