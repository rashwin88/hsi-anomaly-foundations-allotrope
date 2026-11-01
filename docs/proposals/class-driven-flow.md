# Class-driven flow: leave classes out at scoring, export the classes picked at matching

**Status:** agreed 2026-11-01 (user: "load the scene - apply band filtering - do scene segmentation - use the output
of that with options to leave out various classes in anomaly scoring - then do the detection prep - then the spectral
library matches. Depending on the classes chosen in spectral library matches I want to get the full output pack. We
need to test this end to end."). To be built after the pre-freeze regression and before the update kit is built.
**Part A built 2026-11-01** on the local stack (not committed or deployed beyond it): gap 1 (`exclude_classes` at
scoring + dialog), gap 2 backend (the pack, `classes` / `confident_only` / `manmade_p`, `selection_counts`), addition
C (V2 input guard), addition E (worker GPU snapshot; not yet seen on a GPU), v1.0. The end-to-end test below passed on
PRISMA and Tanager (final design/ROADMAP.md OB-20).
**Match screen (Gap 2 UI, additions A, D): built 2026-11-01** on the local stack, in the Tufte style - two linked
maps (classified, anomaly points), All / Man-made / Natural plus material refinement, confident only, sensitivity
presets, the export named by the selection; not committed or deployed beyond the local stack
(docs/lld/spectral-namer-in-product.md §2 "Viewer").

## The flow

1. **Load scene** (any sensor). Exists.
2. **Band filter.** Exists.
3. **Scene segmentation, method `indradhanu_v2`.** Exists. It writes `class_map.tif` (land, water, cloud, shadow, snow)
   plus a thin-cloud display layer in `class_codes.bin`, and its own `keep_mask.tif` from its `exclude_classes`
   (default thick cloud).
4. **Anomaly scoring - NEW: choose the classes to leave out here.**
5. **Detection prep** (threshold + commit). Exists.
6. **Spectral match** (namer objects: man-made / natural, closest material class). Exists.
7. **Output pack - NEW: driven by the classes picked in the match screen.**

## Gap 1 - class exclusion in anomaly scoring

**Config:**
- `exclude_classes: list[str] | None` on `anomaly_scoring`.
- `None` keeps today's behaviour: use the upstream segmentation's keep_mask.
- A list (from: water, cloud, thin_cloud, shadow, snow) recomputes the keep mask from the upstream V2 segmentation's
  class map, `class_codes.bin` included for thin_cloud. There is no re-segmentation.
- It works for every model in the action:
  - V2: its excluded pixels are zeroed after scoring, as today;
  - classical detectors: the keep mask restricts their background statistics, as today.
- Validation: allowed only when the upstream segmentation is V2; otherwise a clear error.
- The choice is recorded in stats and summary.
- An upstream `indices` segmentation could map its classes_to_mask the same way. That is optional, only if trivial.

**UI** (NewActionDialog, anomaly_scoring):
- When the chosen keep-mask input is a V2 segmentation, show its classes as checkboxes, labelled with that action's
  coverage % (from its summary legend).
- Pre-ticked: the segmentation's own excluded classes.
- Plain labels: "Leave out: ☑ Cloud 2.9% ☐ Water 15.0% ...".

## Gap 2 - class-driven output pack from spectral match

**API:** `POST /actions/{id}/export` for spectral_library_match accepts:
- `classes=` (comma list from: natural, manmade, oil_fuel, plastic, textile_fabric, wood_paper, paint_coatings, metal,
  concrete_asphalt);
- optional `confident_only=true`.

`subset=manmade|natural|all` keeps working as a shorthand. Selection is by namer object:
- `natural` = objects called natural;
- `manmade` = all man-made objects;
- a material class = man-made objects whose closest class is that one.

**Pack contents**, all on the scene's real grid, named with the `hyper_` prefix:
- `hyper_anomaly_mask_<sel>.tif`: uint8 0/1, the selected objects' pixels only. This is the submission-format map.
- `hyper_anomaly_pixels_<sel>.shp` + sidecars: one point per selected pixel.
- `hyper_anomaly_objects_<sel>.shp` + sidecars and `hyper_anomaly_objects_<sel>.csv`: one point per selected object
  (centroid), with call, class, confident, p_manmade, p_class, n_px, lat, lon.
- `hyper_namer_level1.tif` and `hyper_namer_class.tif`: the full maps, for context.
- `hyper_quicklook_<sel>.png`: RGB plus the selected objects outlined.
- `MANIFEST.json`, recording the whole chain:
  - scene (id, sensor, file);
  - band filter (bands kept);
  - segmentation (method, classes left out, at both segmentation and scoring);
  - scoring (models, method);
  - prep (threshold mode and value, committed pixels);
  - match (namer checkpoint, confident_p);
  - selection (classes, confident_only, object and pixel counts);
  - software version and time.

There is no heatmap anywhere. The prep bundle is unchanged.

**UI** (spectral match screen):
- The class bars become multi-select checkboxes, with man-made / natural at the top.
- A "Confident only" toggle.
- The export button names the selection and count, e.g. "Export oil + plastic (64 objects)".
- The map and list follow the selection.

## End-to-end test

On PRISMA and Tanager, through the API and a headless browser:
1. load (existing scenes);
2. band filter;
3. V2 segmentation;
4. two scorings: (a) leave out cloud, (b) leave out cloud + water. Check that (b) has no scored pixels on water;
5. prep and commit;
6. match;
7. export three selections: man-made; oil + plastic; natural.

For each pack, check:
- grid and CRS;
- mask values {0,1};
- the mask pixel count equals the selection's pixel count;
- the objects CSV rows equal the selected objects;
- the shapefiles open;
- the manifest records the chain and the selection exactly.

## Additions agreed 2026-11-01 (same build)

**A. Man-made sensitivity on the match screen** (user: "solar panel type scenes being classified as natural - can we be
a bit more aggressive on man-made").
- Presets: Balanced 0.5 (today), Sensitive 0.35, Very sensitive 0.25, as the cut-off on an object's mean P(man-made).
  The screen recomputes calls client-side from namer_objects.parquet: map, list, counts and class bars, with no re-run.
- The export takes `manmade_p=<cut>`; the pack and manifest record it.
- Evidence: on recent scenes many objects sit just under 0.5 (EnMAP 330 of 1,434 objects with P in 0.3-0.5; PRISMA
  1,227 of 5,544).
- The material class of newly man-made objects is their top P(class | man-made), shown tentative unless it is at least
  namer_confident_p.

**B. Solar panel / glass spectra in the library, without retraining.**
- Level 1 is decided by matching against library spectra, and the library tower encodes any spectrum. So man-made PV
  spectra raise P(man-made) for PV-like pixels, and a new class `glass_solar` can be named, with no retraining.
- Source, time-boxed:
  - in-scene spectra from known solar parks in data we have (PRISMA / EnMAP inventory over Bhadla, Rajasthan ~27.53N
    71.92E, or Pavagada, Karnataka ~14.25N 77.45E, or any Tanager open-data solar farm), sampled from pixels well inside
    the park polygon (OSM `power=plant` + `plant:source=solar`, or a visible array), atmospherically corrected L2;
  - else literature supplements, if downloadable.
- Kept as a separate, clearly sourced library addition (`data/family_library` plus sources.md) with provenance per
  spectrum. Hold out some for a check: a held-out PV pixel's P(man-made) before and after.
- The UI and export show the new class "glass / solar panel".

**C. V2 input guard** (from the user's failed run on PRS_L2D_STD_20251127041712, a July 165-grid band filter).
- For scene_segmentation method indradhanu_v2, and for anomaly_scoring with V2, the input picker marks 165-grid band
  filters as incompatible (greyed out, with the reason) and points to adding a native one.
- If one still gets through, the action's failure message says it plainly ("this band filter is on the old 165-band
  grid; add a new Band filter (native bands) and use that"), not a traceback.

**D. An "anomaly points" panel on the match screen** (user 2026-11-01: "another panel which has just the anomaly
points also - easy to distinguish").
- A second map panel, side by side with the classified map. On a phone or narrow window it becomes a switch: Classified
  / Anomaly points.
- It shows **only the committed anomaly pixels** as high-contrast points on a dimmed, desaturated RGB:
  - one bright colour (e.g. yellow with a dark outline) that is not used by the class colours;
  - markers keep a minimum on-screen size, so single pixels stay visible when zoomed out.
- Its pan and zoom are **linked** to the classified panel: moving one moves the other.
- Hovering or clicking a point highlights the same object in both panels and the list.
- A toggle "Selected only" restricts the points to the current class selection and sensitivity, so you see exactly
  what the export will contain.
- The quick-look PNG in the output pack uses the same rendering.

**E. GPU usage in the monitoring pane** (user 2026-11-01: "the GPU usage is not shown there").
- **Cause:** `GET /metrics/host` (backend/allotrope/api/metrics.py `_gpu()`) runs `nvidia-smi` in the **api**
  container. The GPU compose override gives the GPU only to the **worker** (where the models run), so the api always
  reports `gpu.available=false`, even on the H200 VM.
- **Fix, with no compose change** (the VM's kit compose files stay as they are):
  - the **worker** samples `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu`
    (plus the torch device it actually uses) on its heartbeat, about every 5-10 s, and stores the latest snapshot with
    a timestamp in the DB (e.g. a small `worker_status` row or the existing heartbeat record - check the schema);
  - `/metrics/host` reports the api host's own nvidia-smi if present, else the freshest worker snapshot (marked "from
    worker <id>, Ns ago"), and stale (> 60 s) as unavailable;
  - the monitoring pane and top bar show GPU name, utilisation %, memory used / total and temperature, plus whether
    the models are actually running on GPU or CPU (the worker's torch device), so "GPU present but jobs on CPU" is
    visible.
- **Tests:** parse a captured nvidia-smi CSV; freshest-snapshot selection; staleness. The local stack has no GPU, so
  live verification is on the air-gapped VM: the operator checks the pane shows the H200 with non-zero utilisation
  while a V2 scoring runs.
- **Man-made vs natural on the points panel too** (user 2026-11-01: "split manmade and natural on the second pane
  too - add some small icon"). Each anomaly point is drawn with a **shape by call**, keeping the panel easy to read
  and colour-blind safe:
  - man-made: a small **square** (or a building/pin glyph at higher zoom);
  - natural: a small **circle** (or a leaf glyph at higher zoom).
  Both are in the bright points colour with the dark outline; there is no class colour on this panel.
- A small legend shows the two icons with counts. The sensitivity setting moves points between the two shapes live.
- Toggles "Man-made" / "Natural" show or hide each set.
