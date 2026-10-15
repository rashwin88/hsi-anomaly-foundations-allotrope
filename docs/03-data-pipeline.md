# 3. Data pipeline — raw file to comparable cube

Every sensor produces something different. This layer turns all of them into one shape.

## The output: a "vendable"

A **`VendableDataset`** is the currency of the whole system — a normalised cube plus
validity masks. Everything downstream consumes one.

| Type | Sensors | Cube |
|---|---|---|
| `VendableHyperspectralDataset` | PRISMA, AVIRIS-NG | reflectance |
| `VendableEnmapHyperspectralDataset` | EnMAP | reflectance + 5 quality masks |
| `VendableThermalDataset` | Landsat 9, HotSat-1 | °C |

**Vendables carry no spatial reference.** Every GeoTIFF an Action writes has an identity
transform. CRS and affine are recovered later, at export time, by re-opening the raw file
(`app/georef/`). PRISMA is the awkward one — it's *swath* data with per-pixel lat/lon
arrays, which get fitted to an axis-aligned bounding-box affine, discarding swath curvature.

## Cube layouts — the thing that bites newcomers

The same cube can be laid out three ways, and sensors disagree:

| Layout | Axes | Used by |
|---|---|---|
| **BIL** band-interleaved-by-line | `H × C × W` | PRISMA native |
| **BSQ** band-sequential | `C × H × W` | EnMAP, Landsat native; **the ML standard** |
| **BIP** band-interleaved-by-pixel | `H × W × C` | visualisation |

`ImageCubeOperations.convert_cube()` converts between them. If a cube looks like noise,
check you haven't mixed up an axis order.

## How a raw file is read

Two layers, so one code path serves wildly different containers:

- **`FileHelper`** — wraps the physical file (`HE5Helper`, `TIFHelper`, `EnmapHelper`,
  `ENVIHelper`, `HotSatHelper`).
- **Templates** (`app/templates/`) — a dict mapping *logical component* → *where it
  physically lives*, via three reference kinds: `FILE_REFERENCE` (an internal path, e.g. an
  HDF-EOS dataset), `ROOT_METADATA_FIELD` (a root attribute), and
  `DIRECT_PROPERTY_DEFINITION` (a property on the opened dataset, e.g. rasterio `crs`).

Templates are **injected, not looked up**, which is what keeps the helpers sensor-agnostic.

> Only PRISMA, Landsat and EnMAP have templates. AVIRIS-NG and HotSat bypass the system and
> parse their own sidecars.

**Two known traps:** rasterio band indices are **1-based**; and PRISMA's cube contains a
slice for every wavelength *including invalid ones*, which carry `FWHM = 0.0`.

## Getting to physical units

| Sensor | Conversion |
|---|---|
| PRISMA | DN → reflectance via per-band scale factors from L2D metadata |
| EnMAP | `SR = DN × 0.0001` (uniform gain) |
| Landsat 9 | `ST(K) = 0.00341802 × DN + 149.0`, then K → °C |

Landsat additionally gets cloud masking: `B10AdaptiveCloudMasker` (a 5-component GMM — see
[05](05-detectors.md)) plus the provider's `QA_PIXEL` bitwise masks.

## The 8-stage band pipeline (hyperspectral)

Driven entirely by **`BandFilterConfig`**. This is the part that makes different sensors
comparable.

```
239 bands (PRISMA) / 224 (EnMAP)
  1. drop bands flagged bad by the sensor
  2. drop atmospheric absorption windows
  3. trim 3 detector-edge bands per end
  4. drop bands with <20% valid pixels
        ↓ ~188 bands survive
  5. apply quality masks (EnMAP: cloud, shadow, haze)
  6. invalidate pixels with >40% invalid voxels
        ↓ validity is now BINARY: a pixel is wholly valid or wholly invalid
  7. PCHIP-interpolate remaining spectral gaps
  8. resample onto the common wavelength grid
        ↓ 165 bands, 10 nm, 460–2450 nm
```

**Why step 6 matters.** Forcing binary validity means downstream code never has to reason
about "this pixel is valid in 140 of 165 bands". A pixel either has a full spectrum or it
has nothing.

**Why PCHIP, not cubic spline.** PCHIP is shape-preserving — it will not introduce
overshoot ringing around absorption features. A spline can invent a dip that looks exactly
like the diagnostic feature you're trying to detect.

**Why the common grid is the whole point.** After step 8, PRISMA, EnMAP and AVIRIS-NG all
produce identical cube shapes with an identical `wavelengths.npy`. That is what lets a
single hyperspectral model train on a shard stream mixing all three sensors.

Defaults worth knowing: exclusion ranges `(0,450) (912,978) (1131,1152) (1350,1450)
(1800,1950)`; `edge_bands_to_trim=3`; `min_valid_pixel_pct=20`;
`max_invalid_voxel_fraction=0.4`.

## One more fix: nearest-valid pixel fill

`app/utils/pixel_fill/` exists for a specific, real bug. The SegFormer's patch embedding
uses a 7×7 kernel that straddles valid/invalid boundaries. Invalid pixels sit at 0, which
normalises to about −2.3 — a cliff against valid pixels in the −0.7…+1.7 range. Those
boundary artefacts then dominate the anomaly ranking and crowd out genuine detections.

The fix uses a distance transform to replace each invalid pixel with its **nearest valid
neighbour's real measurement** — never an interpolation. Masks are not modified, so the loss
still excludes those pixels. On by default for PRISMA / EnMAP / AVIRIS-NG.

## Patches and shards (training only)

Models train on patches, not whole scenes. Two stages, both writing **webdataset** `.tar`
shards to S3:

1. **Intermediate** — per sensor: download scene → build vendable → cut patches → drop
   patches below the validity threshold → write shards.
2. **Final** — shuffle *across* scenes, so a batch isn't 32 patches of the same field. For
   hyperspectral this also **mixes PRISMA and EnMAP into unified shards**.

```
s3://allotrope-raw-data-india/patches/{sensor}/{split}/{stage}/w{W}_h{H}_s{S}/
```

**Where scenes come from is injectable.** All four intermediate sharders take a
`SceneStorage` (`app/utils/patch_generation/scene_storage.py`) — `S3SceneStorage` by
default, `LocalSceneStorage` to run against a mounted disk such as Drive on Colab. The
bucket and region live in one place, `app/utils/general_utils/s3_config.py`, overridable
with `ALLOTROPE_S3_BUCKET` / `ALLOTROPE_S3_REGION`.

Two properties worth knowing: **constructing a sharder does no I/O** (scene discovery is
lazy, so these classes are testable offline), and **a sharder never deletes a scene folder
itself** — cleanup goes through `storage.release_scene`, which is a no-op locally, because
under a local backend that path is the user's own data rather than a temp download.

Stride is conventionally `size // 2`. Sample contents:

| Hyperspectral | Thermal |
|---|---|
| `pixels.npy` (165,H,W) | `pixels.npy` (1,H,W) °C |
| `validity_cube.npy` | `validity_cube.npy` |
| `wavelengths.npy` (165,) | `predicted_cloud_mask.npy`, `pure_validity_mask.npy`, QA masks |
| `meta.json` | `meta.json` |

Run them with `scripts/generate_hyperspectral_patches.py` and
`scripts/generate_landsat_patches.py` (both support `--skip-intermediate` / `--skip-final`).

### Segmentation shards — a third, separate lane

`scripts/generate_segmentation_patches.py` writes EnMAP patches carrying the **provider
quality layers as training labels**, under its own `enmap_seg` sensor prefix. Indradhanu's
`enmap` and `hyperspectral` shards are untouched by it.

Six extra keys, uint8 `(1,H,W)`, EnMAP only:

| key | values |
|---|---|
| `label_cloud.npy` · `label_haze.npy` · `label_cloud_shadow.npy` · `label_snow.npy` | 0/1 |
| `label_cirrus.npy` | **0-3** by thickness, not binary |
| `label_classes.npy` | 0 = no-data (error), 1 = land, 2 = water, 3 = no-data (off-swath) |

Pixels in this lane are **float16**, halving the dominant term to ~5.4 MB per patch;
reflectance sits in ~0-1 where float16 errs by ~1e-5, well below sensor noise. **A trainer
reading these must cast to float32** — the models are fp32. The reconstruction lane is
unchanged at float32.

Values are recorded exactly as the provider wrote them — no thresholding, no remapping.
Interpretation is the trainer's job, so a change of mind costs no re-shard. Note that in
`label_classes.npy` **both 0 and 3 are no-data**; off-swath padding alone is ~25% of every
raster, outnumbering cloud pixels roughly 35:1, so treating it as a class would swamp
training.

Three deliberate differences from the reconstruction lane:

- **Quality masks are not applied** during vending. Normally cloud/shadow/haze pixels have
  their validity zeroed, which consumes the labels before patching and then makes the
  validity filter discard any patch more than half cloud — precisely the training data.
- **The train/test split is stratified on scene cloud cover**, not random. Cloud appears in
  only 37 of 212 scenes screened on 2026-08-25, so a random split can leave a test set with
  no cloud in it.
- **One shard per scene, resumable.** Sharding is intended to run on Colab against mounted
  Drive; a killed session costs the scene in flight, not the run. Scene folders are read
  through a `SceneStorage` backend rather than boto3 — see
  `docs/lld/segmentation-sharding.md`.

This lane has its own final stage, `LocalFinalShuffler`, because `FinalShuffler` reads over
S3 and there is no S3 on the Colab path. It interleaves several per-scene shards at once,
window-shuffles, and rolls ~1 GB output shards — one complete pass, every patch exactly
once, and inputs left untouched so the output can be verified before anything is deleted.
Its `group_size` is the mixing parameter, not a speed setting.

This path is **only for training data**. Live analysis in the product runs on whole scenes
through Actions.

## SpectralEarth — patches that arrive without their spectral meaning

[SpectralEarth](https://geoservice.dlr.de/web/datasets/enmap_spectralearth) is DLR's
pretraining set: **539,755 patches of 128 × 128 pixels from 11,636 EnMAP L2A scenes, over
415,981 sites** (17.6% imaged on more than one date). We hold it in the raw-dataset
inventory as the collection `SPECTRAL_EARTH_PATCHES`, one S3 object per patch
(`docs/lld/inventory.md`). It is the intended pretraining corpus for the next Indradhanu.

Each patch is a **bare 202-band int16 GeoTIFF** — no wavelengths, no georeference, no
date, no source-scene id, and none of EnMAP's quality layers. What the files do carry:

| | |
|---|---|
| Bands | EnMAP L2A's 224 **minus 1-based bands 127–141 and 161–167** (the two water-absorption regions, 1295–1507 nm and 1728–1968 nm). 91 VNIR + 111 SWIR, 418–2445 nm |
| Values | reflectance × 10,000 (EnMAP gain 0.0001, offset 0); nodata −32768 — in practice none: 0 of 500 sampled patches have a missing voxel |
| Layout | `enmap/<site id>/<patch id>.tif`; a site's patches are its dates |

The band table — EnMAP band, centre, FWHM and detector for each of the 202 planes — is
`app/constants/spectralearth_bands.json`, generated by
`scripts/build_spectralearth_band_table.py` from an EnMAP `METADATA.XML` with the same
parser the EnMAP builder uses. **Nothing should hard-code a SpectralEarth wavelength**;
read the table.

The data layer, all in `app/` (no backend imports):

```
app/models/dataset/spectralearth.py     SpectralEarthPatch: decode, reflectance(), validity(), to_vendable()
app/utils/spectralearth/catalog.py      every patch, from the inventory's files.jsonl.gz; split by site
app/utils/spectralearth/source.py       fetch + sha256-verify (+ optional disk cache)
app/utils/spectralearth/screens.py      per-pixel screens (NodataScreen today; a cloud screen next)
app/utils/spectralearth/torch_dataset.py  Dataset: native (202 bands) or common_grid (165)
app/utils/data_transformations/common_grid_pipeline.py  the EnMAP 8-stage path as one call
```

```python
cat = SpectralEarthCatalog.from_files_manifest("s3://allotrope-raw-inventory-india/enmap/SpectralEarth/ds_0b1abe70-8c20-4248-a3da-f0905027e8fa/")
train, test = cat.split(test_fraction=0.05, seed="se-v1")       # 512,849 / 26,906, no site on both sides
ds = SpectralEarthTorchDataset(train, SpectralEarthSource(cache_dir="/nvme/se"), mode="native")
```

`common_grid` mode runs the same steps EnMAP scenes take (band filter → screen mask where
EnMAP applies its quality masks → 40 %-invalid rule → gap interpolation → resample), so
its spectra are comparable with the existing PRISMA/EnMAP shards. At ~60 ms per patch per
core it is too slow to run live in a training loader; precompute if a model needs the
165-band grid. `native` mode costs nothing beyond decoding.

**Reading straight from S3 does not feed a GPU**: 18 samples/s through an 8-worker
DataLoader, 37 patches/s with 32 fetch threads (EC2, same region). Put the collection on
local NVMe through the source's cache, or shard it. A native batch of 32 is ~424 MB, so
loader containers need a large `--shm-size`.

**Shards.** Training reads WebDataset shards of 150 patches, not single objects:
`patches/spectralearth/se-v1/{train,test}/` (3,419 + 180 shards, raw int16) and a compressed copy
of the first 1,000 + 20 (`se-v1z`, band-delta + zstd, 1.9x smaller, exact). `stats.json` beside
them is the wavelength-normalisation curve. See `docs/lld/spectralearth.md`.

## EnMAP at native bands - fine-tuning and cloud/water labels

`scripts/shard_enmap.py` turns the 210 EnMAP L2A scenes in the inventory into shards at **native
bands** (all 224, each scene's own wavelengths - no resampling, no 165-band grid) with the six
provider quality layers as `label_*.npy` and per-tile facts (`meta.json`: centre lat/lon, elevation,
water vapour, sun angles, cover %). Split by datatake, stratified on cloud. Output
`patches/enmap/native-v1/`. Bands are matched by wavelength because 5 scenes number them
differently. See `docs/lld/enmap-shards.md`.

## PRISMA at native bands - fine-tuning

`scripts/shard_prisma.py` does the same for the 223 PRISMA L2D scenes: all 234 bands PRISMA flags
valid (one fixed band list across every scene), the product's DN-to-reflectance scale and validity
rules but none of its band pipeline, and the EnMAP sample layout so one trainer reads both. No label
keys - L2D has no cloud or water mask. Split by site after tiling (the inventory has no footprints),
with the anomaly-benchmark sites held out of both sides. Output `patches/prisma/native-v1/`. See
`docs/lld/prisma-shards.md`.

## AVIRIS-NG at native bands and pixel size - fine-tuning

`scripts/shard_aviris.py` shards the 79 AVIRIS-NG L2 flight lines of the India campaign (copied from
Google Drive into the inventory on 2026-10-15): the 372 good bands (`bbl` = 1), 3-8 m native pixels,
compressed pixels (`pixels.i16dz`), the same split rules as PRISMA. Output
`patches/aviris_ng/native-v1/`. See `docs/lld/aviris-shards.md`.

Health check after changes, or before a training run, on the EC2 box:
`scripts/spectralearth_sanity.py <collection uri> --n 500` — value ranges, nodata, sites,
loader speed, a percentile spectrum (the red edge must sit at 700–750 nm) and RGB previews.

---

**Next:** [4. Models](04-models.md) · [5. Detectors](05-detectors.md)
