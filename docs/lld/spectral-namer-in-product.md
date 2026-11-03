# The spectral namer in the product

**Purpose.** Put the spectral namer (`app/spectral_match/namer/`,
[proposal](../proposals/spectral-namer.md)) in front of material matching in the
`spectral_library_match` action: call each anomaly **object** natural or man-made, give a
man-made one its **closest known class** with a confident / tentative flag, and send only
man-made objects' pixels on to SAM and family matching.

**Status.** Built 2026-11-01 on the local stack (design agreed with the user the same day);
not committed, not deployed to prod or the air-gapped VM. Checkpoint: run N001b, epoch 40.
The one-screen viewer and the man-made / natural subset export followed the same day (second
agreed design; §2 "Subset export", "Viewer"). The class-driven match screen (two linked maps, multi-select
classes, man-made sensitivity, anomaly-points panel; [class-driven flow](../proposals/class-driven-flow.md))
replaced that viewer the same evening; the class-driven export it calls is part A of that build.

| What | Where |
|---|---|
| Per-pixel inference | `app/spectral_match/namer/infer.py` `load_namer`, `name_pixels` |
| Objects and their maps | `app/spectral_match/namer/objects.py` |
| Worker stage, files, PNGs | `backend/allotrope/action_types/_spectral_namer_run.py` `run_namer` |
| Wiring into the action | `backend/allotrope/action_types/_spectral_library_match_run.py` `_namer_stage` |
| Config (`namer`, `namer_confident_p`) | `backend/allotrope/action_types/spectral_library_match.py` |
| Manifest lookup | `backend/allotrope/foundation_models/resolver.py` `resolve_spectral_namer` |
| Export (bundle schema `/3`, `subset`) | `app/spectral_match/export.py`; `backend/allotrope/api/action_export.py` `check_subset` |
| Object lat / lon for the viewer | `backend/allotrope/api/action_export.py` `spectral_namer_object_coords` |
| Viewer | `frontend/src/components/spectralMatch/` (`SpectralMatchModal`, `MatchMap`, `selection`, `points`, `panels`, `spectrum`, `data`); card in `ActionDetailPane.tsx` `SpectralLibraryMatchOutputViewer` |
| Tests | `tests/test_spectral_match/test_namer_objects.py`, `tests/test_backend/test_spectral_namer_outputs.py` |

---

## 1. Data flow

```
committed anomaly_mask.tif  ──┐
native onboarding vendable  ──┼─► name_pixels (every mask pixel, spatially valid)
/models/spectral_namer/       │      P(man-made), P(class | man-made) x 7, fill   per pixel
  current.json -> .pt  ───────┘
                                  ─► name_objects: 8-connected components
                                       mean P(man-made) > 0.5            -> man-made
                                       argmax of mean P(class | man-made) -> class
                                       that mean >= namer_confident_p    -> confident
                                  ─► namer_* files + summary["namer"]
man-made objects' pixels ─► SAM (matches.parquet, match_map.tif)
                         ─► family (family_matches.parquet, family_map.tif)
natural objects' pixels  ─► not matched: -3 in match_map.tif and family_map.tif
every anomaly pixel      ─► anomaly_pixel_spectra.npz (the viewer inspects natural ones too)
```

The library side is the family library baked into the worker (`/srv/family_library`,
`family_match.FAMILY_BUNDLE_DIR`), resampled per band list into
`<splib07_cache_dir>/namer/` by `bank.build_bank`. No GPU: the worker runs it on CPU.

## 2. Interfaces

**Config** (`SpectralLibraryMatchConfig`): `namer: bool = True`, `namer_confident_p: float =
0.55` (0-1). In the sensor defaults, so every new action from the dialog has the namer on.
`cfg.get("namer", False)` in the worker: a stored pre-namer config never turns it on.

**Manifest** `<models_dir>/spectral_namer/current.json`: `current.file` (the `.pt` beside it),
`version`, `epoch`, `codename.name`, `label`. Swapping to N003 is copying its `.pt` into the
directory and editing `current.file` / `version` / `epoch` - no code change. The `/models`
list skips this directory (`api/models.py` `_list_architectures`): it is not a
reconstruction model and has no val_loss or scoring methods.

**Files** (all in the action's output directory):

| file | content |
|---|---|
| `namer_pixels.parquet` | row, col, p_manmade, cls, p_cls, fill, obj_id - one row per named pixel, the pixel's own numbers |
| `namer_objects.parquet` | obj_id, n_px, row, col (centroid), row_min/max, col_min/max, p_manmade, call, cls, p_cls (NaN natural), confident, fill, and `prob_<class>` per class: the object's mean P(class \| man-made), natural objects included (since 2026-11-01; the match screen's sensitivity re-calls objects from it) |
| `namer_level1_map.tif` | int8: -1 outside the mask, 0 natural object, 1 man-made object |
| `namer_class_map.tif` | int8: -1 outside, 0 natural, 1..7 confident class, 11..17 the same class tentative |
| `namer_level1_map.png` / `_overlay.png` | orange `#E8743B` man-made, teal `#2A9D8F` natural; dark backing / transparent with a 1 px halo |
| `namer_class_overlay.png` | per-class colours on man-made objects, tentative at alpha 105 |
| `namer_legend.json` | codes of both maps -> label, colour, alpha; confident_p, codename, checkpoint |

All rasters carry the mask's identity transform, like every action-written GeoTIFF.

**summary.json**: a `namer` block - `status` (`ok` / `failed` / `skipped` / `disabled`),
codename, checkpoint, version, epoch, confident_p, `n_objects`, `n_objects_manmade`,
`n_objects_natural`, `n_objects_confident`, `n_pixels`, `n_pixels_manmade`,
`n_pixels_natural`, `class_counts`, `class_counts_confident` (objects per class), timing.
Top level gains `pixels_matched` (sent to SAM and family) and `pixels_natural_skipped`;
`n_pixels_matched` / `n_pixels_no_match` are now over `pixels_matched`. The family block
gains `n_namer_natural_skipped`, and its legend a `-3` code ("Natural (spectral namer), not
matched").

**Export** (`allotrope-export-hyper/3`, when `namer_pixels.parquet` exists): one point per
anomaly pixel, natural ones included. DBF / CSV gain, after row / col / lat / lon:
`nm_mm` (object P(man-made)), `nm_call`, `nm_cls`, `nm_p`, `nm_conf` (1 confident),
`nm_obj`. Natural pixels get family `namer_natural` and no SAM ranks. The zip adds
`hyper_namer_level1.tif` (int8, nodata -1, the scene's CRS and affine like the main raster)
and `hyper_namer_objects.csv` (one row per object, centroid lat / lon in EPSG:4326). No
namer PNGs. `MANIFEST.json` gains a `namer` block. The prep bundle is unchanged.

**Subset export**: `POST /actions/{id}/export?subset=manmade|natural` (default `all`, the
bundle above). Only the pixels of objects with that call; every coded raster is replaced by
one boolean map, so the bundle is the submission shape (boolean GeoTIFF + shapefile), no
heatmaps:

| member | content |
|---|---|
| `hyper_anomaly_mask_<subset>.tif` | uint8, 1 on the subset's pixels, 0 elsewhere, no nodata; the scene CRS and affine; tag `ANOMALY_MASK` |
| `hyper_anomaly_materials_<subset>.{shp,shx,dbf,prj,cpg}` | one point per subset pixel, the `/3` columns |
| `hyper_match_summary_<subset>.csv`, `hyper_namer_objects_<subset>.csv` | the `/3` tables, filtered |
| `MANIFEST.json` | schema `/3`, `subset`, `subset_note`; `namer` counts over the subset's objects |

Zip and folder `hyper_<scene>_<action>_<subset>`. The mask is cut from
`namer_level1_map.tif` (1 = man-made, 0 = natural), so it is exactly the objects' pixels and
a subset of the committed mask. A subset on an action without the namer files (pre-namer,
`namer=false`, failed) is a 422 that says so; `all` exports as before.

**Class-driven pack** (2026-11-01, [class-driven flow](../proposals/class-driven-flow.md) gap 2 and addition A):
`POST /actions/{id}/export?classes=<list>[&confident_only=true][&manmade_p=<0..1>]`
(`api/action_class_pack.py`, `app/spectral_match/class_selection.py`, `app/spectral_match/class_pack.py`).

- **Tokens**: `natural`, `manmade`, or a key of `namer/classes.py` `MANMADE_CLASSES`, read per request (a class
  added there, e.g. `glass_solar`, is a token at once). Unknown token: 422 naming the allowed ones.
- **Re-deciding at `manmade_p`** (default the action's 0.5, `objects.MANMADE_CALL_P`): at 0.5 the stored call, class
  and confident flag are used as they are. At another cut an object is man-made when its mean P(man-made) > cut;
  one still man-made keeps its class; a newly man-made one takes its top mean P(class | man-made) from the
  `prob_<class>` columns of `namer_objects.parquet` (written since 2026-11-01), else - older actions - a lower bound
  from `namer_pixels.parquet` (sum of `p_cls` over the object's pixels whose own top class it is, / n_px: exact for
  one-pixel objects, low otherwise), else `unknown`. It is confident only at `p_cls >= namer_confident_p`.
  The manifest records which source was used (`newly_manmade_class_source`).
- **Selection**: the union of the tokens (`natural` = objects called natural, `manmade` = all man-made, a class =
  man-made objects of it); `confident_only` then drops tentative man-made objects (natural objects stay).
- **Pack** `hyper_<scene8>_<action8>_<sel>.zip`, `<sel>` a slug such as `oil_fuel-plastic`, `manmade_p035`,
  `natural_conf` (shortened to `<n>classes` past 48 characters):

| member | content |
|---|---|
| `hyper_anomaly_mask_<sel>.tif` | uint8 0/1, the selected objects' pixels, scene CRS + affine, no nodata |
| `hyper_anomaly_pixels_<sel>.{shp,shx,dbf,prj,cpg}` | one point per selected pixel, fields below |
| `MANIFEST.json` | schema `allotrope-export-hyper-classes/1`, `software_version`, grid, `chain` (scene, band filter bands kept, segmentation method and classes left out at segmentation and at scoring, scoring models / methods, prep threshold and committed pixels, match namer checkpoint and confident_p), `selection` (classes, confident_only, manmade_p, counts, per_class), namer, `material_rule`, `georeference` (only when the scene has none), `files` |

Nothing else (v1.0.1, user 2026-11-01: minimal files). The objects shapefile / CSV, the two namer maps and the
quick-look PNG of v1.0 are gone; the Story GIF (`docs/lld/story-gif.md`) is separate and unchanged.

| DBF field | meaning |
|---|---|
| `row`, `col`, `lat`, `lon` | pixel; lat / lon EPSG:4326 (blank on the pixel grid) |
| `nm_obj`, `nm_call`, `nm_mm` | the pixel's object, its call at `manmade_p`, its mean P(man-made) |
| `nm_cls`, `nm_p`, `nm_conf` | closest namer class and its P(class \| man-made), confident flag; `nm_cls` is `natural` for a natural object |
| `family` | the pixel's family-match verdict (`family_matches.parquet`) |
| `mat1`, `scr1` | the pixel's own USGS splib07 SAM top-1 and its angle (degrees) |
| `material` | the family-library spectrum closest to the OBJECT's mean spectrum, within the object's class (the `natural` entries for a natural object); empty when the class has no entry (e.g. `unknown`), the action has no family library / pixel spectra, or its `family_legend.json` predates the per-entry `cls` (the screen then shows no reference either) |
| `mat_lib` | that entry's library, `USGS` or `ECOSTRESS` (id prefix; the family library is 250 USGS + 161 ECOSTRESS) |
| `mat_ang` | its spectral angle, degrees, 2 dp |

`material` is computed exactly as the match screen's object card ("closest material" plot): `data.ts`
`meanSpectrum` (per band, the mean over the object's pixels of valid finite values in `anomaly_pixel_spectra.npz`)
and `nearestInClass` (the largest cosine, on the bands both have finite, over the rows of `family_library_refl.npz`
whose `family_legend.json` `library[].cls` is the class; first on a tie), in `class_pack._object_materials`.

No heatmap, no coded material raster. Lat / lon columns are EPSG:4326, geometry in the scene CRS. **No
georeference** (v1.0.1): the pack still exports, on the pixel grid `app.georef.pixel_grid()` =
`Affine(1, 0, 0, 0, -1, 0)` with no CRS - point x = col + 0.5, y = -(row + 0.5), no `.prj`, lat / lon blank - and
the MANIFEST `georeference` says so; the response carries `X-Allotrope-Georef: none` and the screen shows a note.
`subset=` keeps working; giving both is a 422, as is `confident_only` / `manmade_p` without `classes`.

**Selection counts**: `GET /actions/{id}/spectral_library_match/selection_counts` with the same three params returns
`{objects, pixels, per_class, manmade_p, slug}` from the same `select()`, so the button's count is the zip's.

**Object coordinates**: `GET /actions/{id}/spectral_library_match/object_coords` returns
`{crs, coords: {obj_id: [lat, lon]}}` (EPSG:4326 centroids) through the export's georef path
(`_scene_georef`), because the browser cannot project and the action's rasters carry an
identity transform. 404 without `namer_objects.parquet`, 422 `crs_missing` without georef (it stays strict;
the viewer then shows row / col).

**Viewer** (`spectralMatch/`; class-driven redesign 2026-11-01,
[proposal](../proposals/class-driven-flow.md) Gap 2 UI and additions A, D; Tufte style). One
screen, no boxes or filled buttons: controls are text toggles (ink and underlined = on, light
grey = off), labels sit next to the data, numbers are tabular figures, and colour is only
man-made (`--accent-manmade` #c0392b), natural (`--accent-natural` #2e7d6b) and the anomaly
points / selection (`--accent-select` #d4a017). The tokens are tufte.css's; the canvas painters
repeat their values in `data.ts`.

| piece | file | what it does |
|---|---|---|
| selection, sensitivity, export query | `selection.ts` | `callAt`, `picked`, `tokens`, `exportQuery`, `fetchSelectionCounts` |
| two maps, linked | `MatchMap.tsx` | one panzoom per map, `PanLink` shares the transform |
| points panel drawing | `points.tsx` | `paintPoints` (canvas), `pointMarkers` (svg) |
| class rows, list, cards | `panels.tsx` | `ClassTree`, `ObjectList`, `ObjectCard`, `PixelCard` |
| spectrum chart | `spectrum.tsx` | `SpectrumPlot`: no box or grid, lines named at their right ends |

- **Header and controls:** a one-line count ("3,518 anomaly objects: 292 man-made, 3,226
  natural"), **Call man-made when P is above: Balanced 0.5 · Sensitive 0.35 · Very sensitive
  0.25**, **Colour: man-made / natural · material**, and the export, which names the selection
  and its size: "Export oil / fuel + plastic — 64 objects".
- **Sensitivity** (addition A): `callAt` recomputes every object's call from
  `namer_objects.parquet` at the chosen cut-off, live: maps, points, list, counts, class rows
  and the export label all follow. An object above the cut-off that the worker called natural
  takes its top `prob_<class>` as its class, tentative below `namer_confident_p`; its row says
  "at this sensitivity". Runs from before the `prob_<class>` columns use the pack's lower bound
  from `namer_pixels.parquet` (per class, the sum of `p_cls` over the object's pixels whose own
  top class it is, over `n_px`; `app/spectral_match/class_selection.py` `_top_classes`), so
  screen and pack agree; it leans tentative. Beside the three presets (pressed when the cut
  equals theirs) a slider sets any cut from 0.05 to 0.95 in steps of 0.01, with a live
  readout (v1.0.1, user 2026-11-01); the maps repaint from a deferred copy of the cut so the
  thumb stays responsive. Above 0.5 the worker's weaker man-made objects turn natural - intended.
  Off 0.5 the pack slug gains `_p<digits>` (0.62 -> `_p062`, 0.6 -> `_p06`).
- **Show: All · Man-made · Natural** in the controls line is the primary filter (user,
  2026-11-01); the class rows only refine man-made. Picking a material implies Man-made; All
  or Natural clears the picks (`selection.ts` `coarseSel`, `pickMaterial`). It also drives the
  anomaly points map (v1.0.1): All shows both kinds, Man-made only squares, Natural only
  circles; the caption toggles refine that until the switch moves again.
- **Class rows** (right column), the refinement and the legend at once: **man-made**, its
  material classes nested under it, then **natural**. Each row reads "oil / fuel, a thin bar,
  62": the bar's light line is all objects of the row, its heavier part the confident ones;
  material bars share the man-made row's scale, so they add up to it. "man-made" and
  "natural" act like the switch; the first material click narrows to that material, later
  ones toggle, and clearing the last pick returns to all man-made (italic "man-made" = some). **confident only**
  keeps man-made objects whose class is at least `namer_confident_p`; natural objects have no
  class and stay. Default: every man-made object.
- **Two maps, side by side** (small multiples: same grid cell, same scale, one pan / zoom):
  - **Classified**: RGB plus the overlay painted from `namer_pixels.parquet`; selected objects
    in full colour, the rest at alpha 45; colour by call or by closest material (tentative
    faded, natural grey).
  - **Anomaly points** (addition D): the RGB dimmed and desaturated (CSS filter), only the
    committed anomaly pixels with a dark one-pixel outline (a canvas at image size), and one
    marker per object at its centroid that keeps 7 CSS px on screen: a red **square** for
    man-made, a green **circle** for natural (`--accent-manmade`; `--accent-natural-point`
    #2e7d32, a clearer leaf green than `--accent-natural` at that size - user, 2026-11-01).
    Objects in the selection get an ochre halo, unless the selection is All. Past 6 screen px
    per image px the markers become glyphs, a red pin holding a building or a green leaf. The caption is the
    legend and the switches (ink, dotted underline off, solid on): "■ man-made 292 · ● natural 3,226 · selected only"; *selected
    only* limits the points to the selection at the current sensitivity, i.e. what the export
    will hold. Shape and colour both carry the call (shape for colour-blind readers); the panel
    has no material colours.
  - Hover anywhere (either map or the list) rings the object in both maps and highlights its
    row; a click selects it (ochre ring with a dark edge, the list scrolls to it, the card
    opens); a list click also zooms both maps to it. Hit-testing takes an anomaly pixel within
    4 px, else the nearest marker within its on-screen reach.
  - Below 900 px wide the two maps become a switch, **classified · anomaly points**.
- **List**: the selected objects, surest call first, virtualised (42 px rows). A row: the
  call's shape mark, closest material (or "natural"), "tentative", size, a hairline confidence
  bar, lat / lon.
- **Card**: "Object 412: man-made", closest material, size, lat / lon, and the spectrum chart -
  the object's mean shape against the nearest family-library spectrum *within the object's
  class* (smallest spectral angle; both L2-normalised), each line named at its right end.
  **Details** holds P(man-made) against the cut-off, P(class), centroid, fill, SAM top-3 and
  the family verdicts.
- **Export**: `POST /actions/{id}/export?classes=<tokens>&confident_only=<bool>&manmade_p=<cut>`,
  tokens `manmade` when every man-made row is on, else the material keys, plus `natural`. The
  label's count comes from `GET .../spectral_library_match/selection_counts` with the same
  query (debounced 250 ms), the screen's own count until it answers. While the api lacks that
  endpoint (a plain 404 "Not Found") the screen counts for itself and exports with the older
  `subset=manmade|natural` when the selection is exactly that at Balanced; any other selection
  says the server cannot export it yet. A 422 from an api that validates the new params gets
  the same fallback.
- **Older actions** (no `summary.namer` ok): one map with the worker's family / SAM overlay
  PNG, a family-count or top-materials list with the same thin bars, a pixel card with the
  same chart, and Export = the full bundle. A missing file loads as null and the panel that
  needs it stays empty.

Why the layers sit inside one panzoom'd wrapper: the earlier viewer panzoomed the RGB img
and moved the overlay, dots and ring after it on `transform` events from a listener bound
once and never re-bound, so after panzoom was recreated the points stayed still while the
image zoomed (user report, 2026-11-01). Inside one element they cannot drift, and pointer to
pixel goes through the wrapper's `getBoundingClientRect()`, which includes the transform.
Markers that keep a fixed on-screen size are drawn in image units times 1 / zoom, still inside
the wrapper. panzoom runs **without bounds**: with bounds on, a `zoomAbs` whose result has to
be pulled back inside them keeps the translation but drops the scale change, which broke
zoom-to-object and reset.

Why the link compares transforms instead of using a re-entry flag: panzoom applies a
transform in the next animation frame and only then emits `transform`, so a follower's echo
arrives after any flag set around `moveTo` has been cleared. Each map re-emits only a
transform that differs from the one it was last told to apply. Comparing with the link's
latest value instead bounced a stale echo back to the leader mid-drag.

## 3. Invariants

- **One call per object.** Every pixel of an object gets the object's level-1 call and class
  in the maps, the export and the matching decision. Per-pixel numbers are only in
  `namer_pixels.parquet` and the inspector. Do not "fix" the map to per-pixel calls without
  changing what is matched.
- **The namer sees every spatially valid mask pixel** (the same `pixel_mask` SAM used before),
  so `n_pixels = pixels_matched + pixels_natural_skipped` and the object counts sum to the
  mask.
- **Family's ring background still excludes the whole mask.** `match_families` gets the
  man-made (rows, cols) but the full `anomaly_mask`: natural anomaly pixels stay out of
  every man-made pixel's ring and of the calibration sample, as before.
- **A failed namer changes nothing downstream**: matching runs on every pixel and the family
  legend gets no `-3` code.
- **The pixel snapshot covers all anomaly pixels**, so `anomaly_pixel_spectra.npz` rows no
  longer equal the SAM rows. The viewer keys both by (row, col), never by position.

## 4. Failure modes

| What | Result |
|---|---|
| No manifest / checkpoint missing | `namer.status = "failed"`, error names the path; matching on all pixels |
| Checkpoint mismatch (wrong architecture) | `load_namer` raises `ValueError`; as above |
| `mode = "all_kept"` | `namer.status = "skipped"` (too many pixels; family skips it too) |
| `namer = false` / pre-namer config | `namer.status = "disabled"`; the card shows nothing namer |
| Every object natural | SAM writes empty tables, family `skipped` ("the namer called every anomaly object natural"); export still lists every pixel |
| Older action opened in the new viewer | no `namer` block: the same map with the family / SAM overlay, no filter, full export |
| `subset` asked of an action without namer files | 422 with a message giving the reason; `all` still exports |

`infer.name_pixels` no longer copies the cube: it gathers the requested pixels' spectra and
reads validity and finiteness band by band. The old version built two full boolean cubes and
a `nan_to_num` copy - about 5 GB extra on a Tanager scene.

## 5. Decisions

**Why objects.** The analyst points at things, not pixels, and at 30 m an object's edge pixels
are mostly background: their P(man-made) is lower than the core's, so a per-pixel map
speckles an object into man-made and natural halves. The object mean is also what the report
script already used (`scripts/aigc_object_table.py`). Objects are 8-connected, as there.

**Why match man-made pixels only.** A natural false alarm matched against libraries that are
mostly man-made comes back with a man-made name more often than ordinary ground does (the
family LLD says so of its own output). The namer's level 1 is the better natural / man-made
call: at fill 0.25 it beats classical ACE's man-made recall at the same natural false-alarm
rate by +0.20 / +0.17 / +0.14 on EnMAP / PRISMA / AVIRIS-NG
(research/experiments/spectral-namer/N001b/README.md). It also cuts matching time.

**Why "closest known class" and a tentative flag.** Run N002 held textile out of training:
unseen textile was called man-made only 46-47 % of the time and, when it was, was given a
confident known class 45-46 % of the time; abstaining on the best cosine or on the top
P(class) sent only 1-4 % to "unknown" (research/experiments/spectral-namer/N002/README.md).
The model cannot say "a new material", so the product never says "is": it says *closest
known class*. On the known classes the top P(class) does work as a confidence flag - at 90 %
coverage the kept classes are 0.80 / 0.84 / 0.81 accurate (N002) - so the flag stays, at
`namer_confident_p = 0.55`, inside the 0.50-0.61 range that 90 % coverage threshold took
across sensors at fill 0.25 (N002 evaluation, as given in the agreed design).

**Why N001b epoch 40.** Level 1 passes the pre-registered criteria on all three sensors and
kept improving after the `best.pt` epoch (15); N003 replaces it by a manifest edit.

**Why the call threshold is 0.5 on the object mean.** It is the model's own boundary (the
sigmoid of the level-1 logit). It is not recalibrated per scene; the proposal's "per-scene
thresholds" were not built.

## 6. Verified (2026-11-01, local stack, worker on CPU with 16 threads)

New `spectral_library_match` actions with the catalog defaults on two committed preps:

| | Tanager 20250205_175845_00_4001 | PRISMA (prep f0b5046e) |
|---|---|---|
| anomaly pixels / objects | 3,303 / 304 | 74,344 / 3,518 |
| man-made objects (pixels) | 82 (481) | 292 (1,622) |
| natural objects (pixels) | 222 (2,822) | 3,226 (72,722) |
| confident classes | 62 | 116 |
| namer stage | 7.6 s | 9.2 s |
| SAM, namer off -> on | 0.135 -> 0.029 s | 1.34 -> 0.036 s |
| family, namer off -> on | 8.8 -> 5.8 s | 19.8 -> 7.3 s |
| action total, namer off -> on | 10.7 -> 15.3 s | 24.4 -> 18.9 s |

Checked on both: object and pixel counts sum to the mask; `matches.parquet` and
`family_matches.parquet` cover exactly the level-1 man-made pixels; `-3` sits exactly on
the natural pixels in both match rasters; the export is schema `/3` with
`hyper_namer_level1.tif` in the scene CRS (Tanager EPSG:32613,
`(30, 0, 589050, 0, -30, 3653460)`; PRISMA EPSG:32644) and one CSV / DBF row per anomaly
pixel. On the PRISMA scene the man-made objects sit on the town; shoreline and terrain
false alarms are natural.

Subset export and viewer, same stack, after the redesign (images tagged `:pre-match-ui2`
before the rebuild):

| | Tanager (action b4ebf7a3) | PRISMA (action 318164f2) |
|---|---|---|
| `subset=manmade` mask pixels = `n_pixels_manmade` | 481 = 481 | 1,622 = 1,622 |
| `subset=natural` mask pixels = `n_pixels_natural` | 2,822 = 2,822 | 72,722 = 72,722 |
| mask CRS / affine | EPSG:32613, `(30, 0, 589050, 0, -30, 3653460)` | EPSG:32644, `(30, 0, 375472.78, 0, -30, 3721282.25)` |
| objects CSV rows (man-made / natural) | 82 / 222 | 292 / 3,226 |

Values {0, 1}; the mask equals the level-1 map's objects of that call; no pixel outside the
committed mask; zips `hyper_5ccb1db3_b4ebf7a3_manmade.zip` and so on. A pre-namer action and
a `namer=false` action return the 422. On PRISMA `subset=manmade` exports in 0.36 s and
`object_coords` answers in 0.08 s. Headless Chromium (Playwright 1.48) against
http://localhost:3010: overlay and RGB rects equal after a wheel zoom; double-click restores
the fit; a row click centres the object (ring centre within 0.2 px of the stage centre) at
zoom 19.8; a map click at an object's centroid selects it; a class bar filters the list
(Tanager plastic, 51); the browser export with the Natural filter downloads
`..._natural.zip`; about 20 list rows stay in the DOM after scrolling to the end of PRISMA's
3,226 natural objects.

**Class-driven match screen** (same stack, 2026-11-01; headless Chromium 1.48 at 1680 x 1000). Targets: Tanager
9824dc98 and its re-run 46db6036 (which has the `prob_<class>` columns), PRISMA 56b746f0 and AVIRIS-NG fae2bccf.

| | Tanager | PRISMA | AVIRIS-NG |
|---|---|---|---|
| objects, man-made at 0.5 | 1,103, 272 | 5,544, 405 | 2,375, 1,290 |
| man-made at 0.25 | 585 | 2,263 | 1,716 |
| "oil / fuel + plastic" at 0.5 | 247 | 141 | 936 |

- **Counts.** `selection_counts` returns the screen's own counts, per class too. That holds for the pixel-lower-bound
  classes of the older runs (9824dc98 at 0.25: oil 298, plastic 224) and for the `prob_<class>` classes of the
  re-run (oil 296, plastic 227).
- **Maps and layers.** After a wheel zoom and after a drag on the points map:
  - both maps' wrappers keep the same offset and size;
  - every canvas and svg layer equals its RGB rect.
- **Selection.** A list click rings the object in both maps. A hover gives a thin ring in both. A click on a points
  marker selects the object in the list and opens its card.
- **Markers.** Zoomed to an object, the markers become glyphs: red pins and green leaves. Square and circle counts
  follow the sensitivity.
- **Narrow window.** At 820 px wide the maps become a switch.
- **Older actions.** A pre-namer AVIRIS-NG action (77c0f8f3) opens with one map and "Export".
- **Downloads.** The browser export downloaded `hyper_5ccb1db3_9824dc98_manmade.zip`, the class pack. Story GIF
  downloaded a `GIF89a` file.

Class-driven flow, end to end (2026-11-01, local stack, API + headless Chromium; band filter native -> V2
segmentation -> V2 scoring leaving out cloud -> prep p99.5 -> match -> class packs):

| | PRISMA PRS_L2D_STD_20250214052558 | Tanager 20250205_175845_00_4001 |
|---|---|---|
| committed pixels | 7,435 | 4,565 |
| `classes=manmade` objects / mask px | 110 / 865 | 272 / 1,186 |
| `classes=oil_fuel,plastic` | 26 / 598 | 247 / 1,124 |
| `classes=natural` | 852 / 6,570 | 831 / 3,379 |
| `classes=manmade&manmade_p=0.35` | 240 / 1,672 (class source `prob_columns`) | 424 / 2,073 |
| grid | EPSG:32644 `(30, 0, 375472.78, 0, -30, 3721282.25)` | EPSG:32613 `(30, 0, 589050, 0, -30, 3653460)` |

Every pack: mask values {0, 1}, mask pixels = `selection_counts.pixels`, objects CSV rows = `selection_counts.objects`,
both shapefiles open in geopandas with that many features and the scene CRS, the quick-look present, MANIFEST
selection and chain exact, `software_version` `allotrope/1.0.0`, export 2.1-2.6 s. `classes=gold` is a 422. In the
browser the export button reads "Export man-made — 110 objects" (240 at Sensitive 0.35; Tanager 272 / 424) and
downloads `hyper_c7648a5f_2aa76402_manmade.zip` / `..._manmade_p035.zip`.

## Gotchas

1. **ck["epoch"] is 0-based.** `ep40.pt` stores `epoch = 39`; the manifest says 40 (the file).
2. **The namer never names glass, camouflage or anything outside the seven classes** - it
   gives the nearest of the seven. That is the tentative flag's job, and it is weak (N002).
3. **rgb.png must be the scene's full resolution** (it is: the upstream actions write it so).
   The viewer maps image pixel = scene (row, col); a downscaled RGB would misplace objects.
4. **No codename yet.** The manifest's `codename.name` is "Spectral Namer"; nothing routes on it.
5. **The 0.5 call is not scene-calibrated.** On a scene unlike the training sensors (Tanager is
   not one of them) treat level 1 as uncalibrated.
