# Part 28 - Producing a submission

> **The one thing this part teaches:** an AIGC Stage-III result is a committed `anomaly_detection_prep`
> exported as a boolean 0/1 GeoTIFF plus a point shapefile, with no score anywhere a judge could read
> as a heatmap, on the product's own map grid - which for PRISMA meant replacing a latitude / longitude
> bounding box with the HE5's UTM corner attributes, read as pixel *centres* (pitch = span / (n - 1),
> origin = upper-left centre minus half a pixel) - and up to three such results per scene, with a
> per-object table that puts SAM, the family match and the namer side by side for the report.

## Where we are

Parts 13 and 14 showed the export: geography is recovered from the raw file at the last moment, and a
bundle leaves the system. Part 23 put Indradhanu V2 in a chain; part 27 built the namer. On 2026-10-31
an evaluation circular arrived for Stage III, with a report due on 2 November 2026 (midnight) and a
presentation on 4 November 2026, 09:00. This part is how the product meets it.

The plan and its status live in `docs/proposals/aigc-evaluation-submission.md`. Its changes C1-C6 were
built and tested on 2026-10-31 and C7 (the rehearsal) was done on the **local** stack (WSL
`/home/fitt/allotrope`, http://localhost:3010): "images rebuilt, worker `running`, export rehearsed on a
committed PRISMA prep: `hyper_*` bundle, uint8 {0,1} mask in the product's UTM zone". A prod or
airgap-VM deploy is **not approved** and is a separate last step.

| file | what it holds |
|---|---|
| `docs/proposals/aigc-evaluation-submission.md` | requirements R1-R3, the change list C1-C7, the three suggested results, open questions |
| `app/georef/from_scene_file.py` | `prisma_utm_grid` (new) and `_resolve_prisma`, which now prefers it |
| `tests/test_utils/test_georef_prisma.py` | Set-1's attributes must give the ground-truth grid exactly |
| `app/anomaly_detection/export.py` | `build_thermal_bundle`: the prep bundle, now `hyper_` or `thermal_`, no `score` in the shapefile |
| `backend/allotrope/action_types/band_filter_apply.py` | the sensor defaults, now native bands |
| `backend/allotrope/action_types/_anomaly_scoring_run.py` | `_check_band_grid`, `_to_common_grid` |
| `scripts/aigc_object_table.py` | per-object SAM / family / namer table for the report |
| `scripts/write_reflectance_product.py` | the Level-1 reflectance deliverable's writer (not the correction itself) |
| `research/textbook/tools/compute_namer_submission.py` (+ `_output.txt`) | the worked numbers below |

> **Analogy.** A land registry accepts boundary maps only as surveyed lines on its own grid - not a
> sketch, not a shaded "likely boundary" map. You may file up to three claims per plot. Before this
> week the office was drawing PRISMA plots by eye from the corners of a road atlas; now it reads the
> surveyor's grid references off the deed.

## Step 1 - The rules, and which bundle meets them

From the circular, as recorded in the proposal's requirement table:

| # | rule | how the product meets it |
|---|---|---|
| R2a | boolean 0/1 anomaly map as a **GeoTIFF**, real CRS | the prep commit writes uint8 0/1; the export applies the raw scene's georef |
| R2b | the same map as a **.shp** | one Point per anomaly pixel, full sidecar set (.shp .shx .dbf .prj .cpg) |
| R2c | **no heatmap-style output** - it disqualifies | no score raster in the bundle, and (new) no `score` attribute in the shapefile |
| R2d | up to **3 results per scene** | three committed prep actions, three exports; each zip carries the action's short id |
| R2e | match detected pixels to libraries with a **foundation model and classic SAM**, pros and cons in the report | the per-object table (Step 5) |
| R1 | a reflectance product from L1 | not in the product; a writer script for another session's output (Step 6) |

**Submit only prep bundles** - the export of a committed `anomaly_detection_prep`
(`POST /actions/{id}/export`, built by `build_thermal_bundle`). Two other exports would break R2c:
the project export (`POST /projects/{id}/exports`) zips every output tree, `anomaly_score.tif`
included; and the hyperspectral `spectral_library_match` bundle's raster `hyper_anomaly_materials.tif`
holds int16 family codes (-2 .. F), not 0/1.

What changed in the prep bundle on 2026-10-31 (`export.py`, C2):

```python
prefix = "thermal" if spec.sensor_type in THERMAL_SENSORS else "hyper"   # THERMAL_SENSORS = ("landsat9", "hotsat1")
folder = f"{prefix}_{_short(spec.scene_id)}_{_short(spec.action_id)}"
...
keep_cols = ["row", "col", "latitude", "longitude", "geometry"]           # "score" removed
```

Before, a PRISMA bundle was named `thermal_*`, against the naming rule that the name says `hyper` or
`thermal`. The code comment on the dropped column: "No score column: the submission map is boolean, and a
continuous score reads as a heatmap (AIGC rules, 2026-10-31). The CSV keeps it." **The CSV is inside the
same zip** (`<prefix>_anomaly_summary.csv`, with the `score` column); the proposal calls it "not
submitted", so leave it out of what you upload. Two leftovers you will see: the function is still called
`build_thermal_bundle` and the manifest's schema is still `allotrope-export-thermal/1` for a `hyper_`
bundle.

The mask itself is untouched by export: it is the uint8 file the commit wrote, with the georef from the
raw scene file stamped on (the mask has no CRS of its own - part 14's identity transform).

## Step 2 - The PRISMA grid, fixed

**The problem.** A GeoTIFF's grid is six numbers `(a, b, c, d, e, f)`: `X = a col + b row + c`,
`Y = d col + e row + f`, where (col, row) is a position in pixels counted from the top-left *corner* of
the image, so the centre of the top-left pixel is (0.5, 0.5). Until 2026-10-31 a PRISMA export took the
minimum and maximum of the HE5's per-pixel latitude and longitude and fitted an EPSG:4326 box to them
(`from_bounds`). The ground-truth rasters are in UTM. On Set-1 the box put pixels a **median 2.4 px (95%
4.1, max 5.6) from the product grid** - 72 m, up to 168 m at 30 m pixels (proposal, Status). The
internal evaluation (`app/evaluation`) never noticed, because it compares masks in pixel space.

**The fix** (`prisma_utm_grid`, C1). A PRISMA L2D product is already gridded in UTM, and the HE5's root
attributes say how: `Epsg_Code`, and the easting / northing of the upper-left, upper-right and
lower-left corner pixels. The docstring: "the `Product_{UL,UR,LL}corner_{easting,northing}` attributes
are the corner pixels' *centres*, so the pitch is the corner span over (n - 1) and the origin is the UL
centre minus half a pixel."

Why `n - 1`? Five fence posts 30 m apart span `4 x 30 = 120` m from the first post's centre to the last's.
Centres of `n` pixels are `n - 1` pitches apart.

**Do Set-1 by hand.** The test file holds Set-1's attributes (stored as float32, like the file):
`Epsg_Code = 32642`, UL easting 545022.75, UL northing 2501422.5, UR easting 580872.5, LL northing
2465872.5; the image is H = 1186 rows by W = 1196 columns.

```
east span    = 580,872.5  - 545,022.75 = 35,849.75 m
pitch x      = 35,849.75 / (1196 - 1) = 35,849.75 / 1195 = 29.99979 m   -> round(., 2) = 30.0
north span   = 2,501,422.5 - 2,465,872.5 = 35,550 m
pitch y      = 35,550 / (1186 - 1) = 35,550 / 1185 = 30.0 m
origin x  c  = 545,022.75 - 30 / 2 = 545,007.75
origin y  f  = 2,501,422.5 + 30 / 2 = 2,501,437.5       (plus: north is up, rows go down)
transform    = (30, 0, 545,007.75, 0, -30, 2,501,437.5),  EPSG:32642 (UTM zone 42 N)
```

That is exactly the ground-truth raster's grid,
`[30, 0, 545007.75, 0, -30, 2501437.5]` (`test_set1_attributes_give_the_ground_truth_grid`). Check the
top-left centre: `X = 30 x 0.5 + 545,007.75 = 545,022.75`, the UL attribute. Read the attributes as edges
instead and the pitch would be `35,849.75 / 1196 = 29.9747` m - a quarter of a millimetre per metre,
which over 1196 columns is 30 m: one whole pixel at the far edge.

**The rounding** (`round(..., 2)`, comment "float32 attrs: 29.9998 -> 30.0"): the attributes are 32-bit
floats, so the span is not an exact multiple of 30. Rounding to centimetres snaps the pitch to the
product's 30 m. The price: the bottom-right centre comes out at 580,872.75 instead of the attribute's
580,872.5, a quarter metre - under 1% of a pixel.

**The fallback.** `_resolve_prisma` uses the product grid only when the attributes are present and sane
(`epsg > 0`, at least 2 x 2, spans positive) **and** the vendable's shape equals the latitude array's
shape. Otherwise it falls back to the old bounding-box fit, now described in the module docstring as
"approximate, can be pixels off" (`test_missing_attributes_fall_back`). The same function serves
`write_reflectance_product.py --georef-from <file>.he5` (Step 6).

## Step 3 - Native bands by default

Part 23 said V2 needs one manual setting on the band filter. Since 2026-10-31 it does not: every
hyperspectral sensor default in `band_filter_apply.py` reads

```python
"use_common_wavelength_grid": False,   # native bands (2026-10-31): V2 reads them; 165-grid models resample at scoring
```

The reason was C3: "Indradhanu V2 cannot be run from the UI, because no UI control sets the band filter to
native bands." The dialog prefills its form from these per-sensor defaults (`default_config_per_sensor`),
so the default now *is* the V2 setting. The LLD's standing decision is the same direction: "One inference
method: native bands" (`docs/lld/spectral-stem.md`, decided 2026-10-15).

And the old 165-band model keeps working, because `anomaly_scoring` now resamples for it
(`_to_common_grid`, "the same resampler band_filter_apply uses"):

```python
elif m.band_grid == "common165" and not is_thermal:
    model_cube, model_wl = _to_common_grid(cube_np, validity_np, wavelengths)   # (165, H, W), the grid
```

`_check_band_grid` lost its second error ("needs the 165-band grid"); its docstring now reads "Refuse a
native-band model (Indradhanu V2) on a 165-grid vendable ... A 165-grid model (v0.2.0) on native bands is
resampled by _to_common_grid instead." The one refusal left: V2 on a 165-grid band filter.

| band filter | Indradhanu (v0.2.0, `common165`) | Indradhanu V2 (`native`) | MNF-RX |
|---|---|---|---|
| native (the default since 2026-10-31) | runs, resampled inside scoring | runs | runs |
| 165 grid (`use_common_wavelength_grid: true`) | runs | **refused** | runs |

**So one scoring action can now run both Indradhanus**, on a native band filter - part 23's "build two
branches" is no longer needed (`docs/09-known-issues.md` says the same).

One trap, found by reading the code rather than the comments:

- The **pydantic field default is still `True`** (`BandFilterApplyConfig.use_common_wavelength_grid`,
  `Field(default=True, ...)`), and its description still says the grid is "Required by Indradhanu
  (v0.2.0)". The UI sends the sensor defaults, so the UI gets native bands; **an API call that omits the
  field gets the 165 grid**, and V2 then fails. Known-issues records this as deliberate ("an API call
  that omits the flag behaves as before").
- (Two other comments were stale when this part was first written - `_check_band_grid`'s docstring and
  the resolver's `band_grid` comment both still described the 165-grid default and a refusal both ways.
  Both were corrected on 2026-10-31; the resolver now says anomaly_scoring "refuses a native model on the
  165 grid and resamples native bands for a common165 model".)

The scoring *codename* defaults are unchanged: still `["Indradhanu"]` (v0.2.0), but the dialog makes you
pick models.

## Step 4 - Three results per scene

Each result is one committed prep action, so three results are a procedure, not code: three prep actions
(they may hang off one scoring output), three commits, three exports. A re-commit overwrites a prep's mask,
so do not reuse one prep for two results. The proposal's suggestion, for each hyperspectral scene, all
residual Mahalanobis (the user confirms in its Q3):

1. Indradhanu V2 (Heavy).
2. Indradhanu V2 + MNF-RX, combined at the prep weights.
3. MNF-RX alone, or V2 at a second threshold.

The thresholds are a person's call, as always (part 14); with no ground truth on the evaluation scenes,
the proposal asks whether to fix a percentile per model now from the four labelled PRISMA scenes (Q3).

## Step 5 - The per-object table

The report must compare a foundation model with classic SAM. Pixels are not objects, so
`scripts/aigc_object_table.py` groups the committed mask into **8-connected components**
(`scipy.ndimage.label` with a 3 x 3 structure of ones): two flagged pixels belong to one object if they
touch at an edge *or a corner*.

```
mask              8-connected (2 objects)    4-connected (3 objects)
1 1 0 0 0         1 1 0 0 0                  1 1 0 0 0
0 1 0 0 1         0 1 0 0 2                  0 1 0 0 2
0 0 1 0 1         0 0 1 0 2                  0 0 3 0 2
0 0 0 0 0         0 0 0 0 0                  0 0 0 0 0
```

The pixel at row 2, column 2 touches row 1, column 1 only at a corner: one object under 8-connectivity,
a separate one under 4.

Per object the script writes size, centroid (latitude / longitude through the scene's own georef, so
PRISMA uses Step 2's grid), and three opinions:

| column group | from | what |
|---|---|---|
| `sam_*` | `matches.parquet`, rank 0 | majority top-1 name and its share, its splib07 chapter, median angle, the three commonest names |
| `family*` | `family_matches.parquet`, answered pixels | majority family, its group, mean confidence, answered share |
| `namer_*` | `--namer-ckpt` (optional) | mean P(man-made), majority class and share, mean P(class \| man-made), mean fill |

It writes `objects.csv`, `objects.geojson` and a `findings.md` stub (agreement counts and the largest
objects), the starting point for the report chapter "Findings on Evaluation Dataset". Its own docstring
labels the family match honestly: "calibrated Simplex ACE (classical)". The namer column is the only
foundation-model column, and it reads an N001 checkpoint - `ep<n>.pt` exists after every epoch, so a
preliminary one is usable before the block ends; the stub then says "preliminary".

## Step 6 - The reflectance product writer

R1 asks for surface reflectance from L1 data. The atmospheric correction is another session's module
(branch `vinoth/atm-isofit4`); `scripts/write_reflectance_product.py` only writes its result in the
requested format: GeoTIFF and / or ENVI BSQ + `.hdr`, every wavelength recorded, water-absorption bands
removed (default `1340-1460,1790-1960` nm, both ends included), reflectance x 10000 as int16 with
nodata -9999.

```python
data = np.where(np.isfinite(cube), np.clip(np.round(cube * SCALE), -9998, 32767), NODATA).astype(np.int16)
```

On paper:

| reflectance | x 10000 | written | why |
|---|---|---|---|
| 0.2 | 2000 | 2000 | |
| 0.12345 | 1234.5 | 1234 | `np.round` rounds a half to the even neighbour |
| 0.12355 | 1235.5 | 1236 | the even neighbour is above this time |
| NaN | NaN | -9999 | nodata |
| 4.0 | 40000 | 32767 | clipped to int16's top |

The lower clip is -9998, not -32768, so no real value can collide with the nodata value. Bands at 1330,
1345, 1460, 1470 nm with the default drop: 1345 and 1460 go (1460 is an end of the range), 1330 and 1470
stay. `--georef-from` takes a raster with a CRS or a PRISMA `.he5`, the latter through Step 2's
`prisma_utm_grid`.

---

## Common confusions

**"Any export from the product will do."** Only the prep bundle. The project export carries
`anomaly_score.tif` (a heatmap); the material bundle's raster is family codes, not 0/1.

**"No score in the shapefile means no score in the zip."** The CSV in the same zip still has it; do not
upload the CSV.

**"PRISMA's corner attributes are the image's corners."** They are the corner *pixels' centres*. Read
them as edges and the far edge drifts by a whole pixel.

**"The georef error moved our benchmark numbers."** No: the evaluation compares masks in pixel space. Only
deliverables that a judge overlays on a map were affected.

**"V2 still needs a band-filter setting."** Not from the UI since 2026-10-31. Only an API call that omits
`use_common_wavelength_grid` still gets the 165 grid.

**"One scoring action cannot run v0.2.0 and V2 together."** It can now, on a native band filter: v0.2.0's
cube is resampled inside scoring.

**"The reflectance script does atmospheric correction."** It writes another module's output. Its docstring:
"The atmospheric correction itself is not here".

## Check yourself

1. A PRISMA product has W = 1000 columns; its UL and UR corner eastings are 400,015 and 429,985. Compute
   the pitch and the transform's `c`. What pitch would you get by (wrongly) dividing by W?
2. With Set-1's transform, where is the centre of the pixel in row 10, column 20?
3. A scene's band filter was created through the API without `use_common_wavelength_grid`. An
   `anomaly_scoring` action picks `["Indradhanu V2", "Indradhanu"]`. What happens?
4. Label this mask with 8- and with 4-connectivity and count the objects:
   `[[1, 0, 1], [0, 1, 0], [1, 0, 1]]`.
5. Write 0.03, -1.2 and 3.5 as the reflectance writer would.
6. Why is a re-commit on one prep action not a way to make result 2?

<details>
<summary>Answers</summary>

1. Span `429,985 - 400,015 = 29,970` m; pitch `29,970 / (1000 - 1) = 29,970 / 999 = 30.0` m;
   `c = 400,015 - 15 = 400,000`. Dividing by W: `29,970 / 1000 = 29.97` m.
2. `X = 30 x 20.5 + 545,007.75 = 615 + 545,007.75 = 545,622.75`;
   `Y = -30 x 10.5 + 2,501,437.5 = -315 + 2,501,437.5 = 2,501,122.5` (EPSG:32642).
3. The omitted field takes the pydantic default `True`, so the vendable is on the 165 grid. v0.2.0 would
   run, but `_check_band_grid` refuses V2 ("reads native bands: run band_filter_apply with
   use_common_wavelength_grid=false"), and the action fails. Fix: a band filter with the field `false` (or
   created from the UI).
4. 8-connected: the centre touches all four corners, so **1** object. 4-connected: no two pixels share an
   edge, so **5** objects.
5. `0.03 x 10000 = 300` -> 300; `-1.2 x 10000 = -12,000` -> clipped up to -9998 (so it cannot be mistaken
   for nodata, -9999); `3.5 x 10000 = 35,000` -> clipped down to 32,767, int16's largest value.
6. Commit overwrites the prep's `anomaly_mask.tif`, so result 1 would be lost; and the export carries the
   action's short id, so three distinct results need three prep actions.

</details>

---

## You have finished

You can now read the source. Where to go next:

- **`docs/01-orientation.md`** through **`docs/10-code-style.md`** - the reference docs
- **`docs/lld/`** - low-level designs for individual subsystems; for the last thirteen parts,
  `docs/lld/evaluation.md`, `docs/lld/implant-bench.md`, `docs/lld/spectral-stem.md` ("Serving", "One
  inference method: native bands") and `docs/lld/spectral-family-match.md`
- **`docs/proposals/spectral-namer.md`** and **`docs/proposals/aigc-evaluation-submission.md`** - the two
  plans behind parts 27 and 28, each with what was actually built
- **`docs/09-known-issues.md`** - what is currently broken, and what to check before trusting
  a green test run
- **`research/math/spectral-matching/spectral_matching_step_by_step.pdf`** - the mathematics of naming,
  slowly, ending with the namer as built

One habit worth keeping from this course: **when a comment and the code disagree, the code
wins**, and when a summary and the data disagree, the data wins. You met several such disagreements -
the regularisation default, the `MIN_VALID_FRACTION` comment and the discarded `units` field in parts 1
to 15; `compute_roc`'s threshold docstring in part 16; the LLD's old `strength_for_z` and "10 px apart" in
part 18; the cache key and `dirty` flag in part 19, both fixed since; bench-v2's per-tier foreignness in
part 20; "no real object sits on water" and "set4 is bright desert" in part 21; a "close to, not
identical" docstring that was hiding a factor of nearly three (15.8% against 44.2%), and a cross-check
whose result was misread, in part 22; a band filter description that still said the common grid was
"Required for cross-sensor models like Indradhanu" - true of v0.2.0, wrong for V2 - in part 23; a scorer
that met every pass criterion while finding fewer real objects in part 24; two expectations written down
in advance and both wrong, and "barely see" meaning small weight rather than few visits, in part 25;
family names that look like materials but were decided by words in file names in part 26; a model a
third to a ninth the size its proposal predicted, a "per class" split that was not per class and a duplicate-name
leak (both fixed the same day), and a level 1 trained unlike its design, in part 27; and, in this part, a
field default and its description that still describe the 165-band world. There will be more.

Next: [part 29](29-what-hypersl-is.md) - the external foundation model we evaluated and did not adopt, HyperSL, and the data on why.
