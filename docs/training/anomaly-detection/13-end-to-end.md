# Part 13 - End to end

> **The one thing this part teaches:** the components stop being separate - here is one
> PRISMA scene from upload to shortlist, with every step naming the part that explained it.

This part has no new concepts. If something here is unfamiliar, the part named beside it is
the one to re-read.

## The scene

A PRISMA acquisition over an industrial area. One `.he5` file, about 2 GB, 239 bands,
1,210 x 1,219 pixels. An analyst suspects an unreported flare.

## Step 1 - Upload and onboarding

The analyst picks `prisma`, selects the file, uploads. The api streams it to
`/data/staging/<job_id>/` in 1 MiB chunks and inserts a `scene_onboard` job. **The api does
no analysis** (part 3).

The worker claims the job and:

1. Moves the file to `/data/scenes/<scene_id>/raw/`.
2. Builds a `PrismaDatasetBuilder`, which reads through `HE5Helper` using the PRISMA template
   (part 4).
3. Converts DN to reflectance with per-band scale factors from the file's own metadata
   (part 2).
4. Runs the eight-stage band pipeline (part 7).
5. Writes the vendable and renders preview images.

What comes out:

```
cube        (165, 1210, 1219)  float32 reflectance, common grid
validity    (165, 1210, 1219)  int8, binary per pixel
wavelengths (165,)             460.0 ... 2450.0 nm
```

Note what is absent: **no latitude, no longitude, no projection** (part 5). That is recovered
at step 7.

Say 6% of pixels were lost to cloud and swath edges:

```
total pixels = 1210 * 1219 = 1,474,990
invalid  6%  =                  88,499
valid        =               1,386,491
```

## Step 2 - Create a Project

The analyst creates a Project bound to this Scene (part 3). Nothing computes.

## Step 3 - band_filter_apply

The first Action. Re-vends the scene with explicit band settings and writes
`filtered_vendable.pkl`.

Two things happen here worth remembering. The pickle is why class names in
`vendables.py` are a wire format (part 5). And `nearest_valid_fill` runs, replacing each
invalid pixel with its nearest valid neighbour's real measurement so the patch embedding
never sees a cliff (part 7).

## Step 4 - scene_segmentation

Computes NDVI, NDWI and brightness, thresholds them into water, cloud, shadow and vegetation
masks, and produces a **keep_mask** - the region worth analysing.

NDVI on one pixel, red 0.04 and near-infrared 0.55:

```
NDVI = (NIR - RED) / (NIR + RED)
     = (0.55 - 0.04) / (0.55 + 0.04)
     = 0.51 / 0.59
     = 0.864
```

Well above the 0.4 vegetation threshold - healthy vegetation, and masked out if vegetation is
among the classes to exclude.

Say masking removes another 31% of the valid pixels:

```
valid after onboarding = 1,386,491
kept (69%)             =   956,679
```

## Step 5 - anomaly_scoring

The expensive one. The analyst selects two models by codename (part 10): **Indradhanu**, the
hyperspectral transformer, and **MNF-RX**, the classical detector (part 8).

**Indradhanu.** The scene is tiled into overlapping patches. Each is reconstructed twice
under complementary masks so no pixel contributes to its own reconstruction (part 11), and
results are overlap-averaged. The residual is collapsed with `combined` scoring - half L1,
half SAM (part 12).

**MNF-RX.** Noise covariance from neighbouring-pixel differences, whiten, keep the top 10
components, run RX there (part 8). Before `detect()`, the worker narrows the detector's
internal spatial mask with the keep_mask, so clouds and water never enter the covariance
estimate.

Two score maps, both `(1210, 1219)`, both `NaN` at invalid pixels.

**A note on how this runs.** All of this happens in the worker, and the heavy imports live
*inside* `run()`. That is why the api can serve requests without ever loading torch - the
lazy-import rule from part 3. It is also why, when this module once imported a name that did
not exist, the api stayed perfectly healthy while every job failed.

## Step 6 - anomaly_detection_prep, and the pause

Rescales each model's map to [0, 1], combines them, computes percentiles - and **stops** at
`needs_threshold` (part 3).

The analyst looks at the composite and picks the top 0.5%:

```
kept pixels = 956,679
top 0.5%    = 956,679 * 0.005
            = 4,783 pixels
```

They commit. `anomaly_mask.tif` is written and the Action completes.

## Step 7 - spectral_library_match

Hyperspectral only (part 14 has the detail).

Each flagged pixel's spectrum is compared against 519 USGS laboratory spectra by spectral
angle. **It reads the native onboarding vendable, not the band-filtered one**, so narrow
absorption features survive - the filtering that made sensors comparable would blunt the very
evidence used to identify a material.

## Step 8 - Export

Now georeferencing appears. `app/georef/` re-opens the original `.he5`, reads the per-pixel
latitude and longitude arrays PRISMA stores as swath data, and fits an affine transform
(part 5).

Out comes a zip: a GeoTIFF, a Shapefile with one point per flagged pixel carrying its matched
materials, a CSV summary, and a manifest.

If the georeferencing cannot be resolved, this step returns **422 `crs_missing`** rather than
shipping a bundle with no projection.

## The whole funnel

```
1,474,990   pixels in the scene
1,386,491   valid after onboarding          (-6%   cloud, swath edges)
  956,679   kept after segmentation         (-31%  water, vegetation, shadow)
    4,783   flagged at the 0.5% threshold
```

**From 1.47 million to 4,783 - a reduction of about 308 times.** Then a human reviews the
shortlist. That is the system's actual job: not deciding, but narrowing.

## What could go wrong at each step

| step | failure | symptom |
|---|---|---|
| 1 | wrong sensor selected | `ValueError` in the worker, job marked failed |
| 3 | class renamed in `vendables.py` | stored pickles unreadable |
| 4 | thresholds wrong for the terrain | keep_mask removes the target |
| 5 | worker crash-looping | jobs sit in `queued`, api reports healthy |
| 5 | normalisation stats wrong for the sensor | every pixel scores high, nothing useful |
| 6 | threshold too tight | real anomaly ranked 5,001st |
| 8 | georeferencing unresolvable | 422 `crs_missing` |

The fifth row is the one that has actually happened, twice.

## Check yourself

<details>
<summary>1. Where does georeferencing come from, and why is it not present earlier?</summary>

From re-reading the original raw file at export time. Vendables carry no spatial reference
because analysis works in pixel grids; only the deliverable needs map coordinates.
</details>

<details>
<summary>2. A scene is 900 x 1,100. 9% invalid after onboarding, segmentation keeps 62%, threshold 0.4%. How many flagged pixels?</summary>

```
total   = 900 * 1100        = 990,000
valid   = 990,000 * 0.91    = 900,900
kept    = 900,900 * 0.62    = 558,558
flagged = 558,558 * 0.004   = 2,234  (2,234.2, truncated)
```
</details>

<details>
<summary>3. Why does spectral matching use the native vendable rather than the filtered one?</summary>

Band filtering and resampling onto the common grid blunt narrow absorption features. Those
features are the diagnostic evidence for identifying a material, so matching works on the
unfiltered spectra.
</details>

<details>
<summary>4. Jobs are queued, the api is healthy, and nothing progresses. What happened, and why did nothing catch it?</summary>

The worker is crash-looping - most likely an import error. Nothing caught it because the
api's health check queries the database, and the lazy-import rule keeps worker-only modules
out of the api's startup path, so the api never touches the broken code.
</details>

<details>
<summary>5. The analyst reports every pixel scoring high on a HotSat scene. Most likely cause?</summary>

Normalisation statistics wrong for the sensor. HotSat ships raw DN while the checkpoint's
baked figures are Celsius, so input arrives hundreds of standard deviations out and the model
reconstructs noise. The fix is a per-scene `PixelStatsOverride` (part 10).
</details>

---

Next: [part 14](14-material-and-export.md) - naming the material and getting it out.
