# Planet Tanager as a native sensor

**Files:** `app/utils/files/tanager_helper.py`, `app/utils/files/tanager_envi.py` (ENVI input),
`app/models/file_processing/tanager_metadata.py`,
`app/utils/dataset_builder/tanager_dataset_builder.py`, `app/georef/from_scene_file.py`
(`_resolve_tanager`), plus the `sensor_type="tanager"` entries listed in
[6. Backend](../06-backend.md#adding-a-sensor).
**Tests:** `tests/test_utils/test_tanager.py` (synthetic HDF-EOS5 files written with h5py; one
`large_files` check against the real file when it is on the machine) and
`tests/test_utils/test_tanager_envi.py` (hand-written ENVI headers + raw bytes in BSQ, BIL and BIP).

## Purpose

Onboard Planet Tanager ortho surface reflectance and run the whole shipped chain on it:
band filter, segmentation, Indradhanu V2 / MNF-RX scoring, threshold, splib07 matching and a
georeferenced `hyper_*` export. Added 2026-11-01; the ENVI input (below) the same day.

## The product

Verified on item `20250205_175845_00_4001` from Planet's open STAC (CC BY 4.0), the
`ortho_sr_hdf5` asset, 1.3 GB:

| Where | What |
|---|---|
| `HDFEOS/GRIDS/HYP` attrs | `epsg_code` (int64 32613 in this file), `strip_id` = `<item id>_strip` |
| `HDFEOS INFORMATION/StructMetadata.0` | ODL text: `XDim=895`, `YDim=1020`, `UpperLeftPointMtrs=(589050.00,3653460.00)`, `LowerRightMtrs=(615900.00,3622860.00)`, `HE5_GCTP_UTM`, `ZoneCode=13`, `GridOrigin=HE5_HDFE_GD_UL`, `PixelRegistration=HE5_HDFE_CORNER` |
| `Data Fields/surface_reflectance` | (426, 1020, 895) float32, 0-1, `_FillValue` -9999, chunks (14, 64, 56) gzip; attrs `wavelengths` 376.44-2499.0 nm, `fwhm` 5.2-6.81 nm, `good_wavelengths` (368 good; the 58 bad span 1342-1437 and 1782-1967 nm) |
| `nodata_pixels` | uint8, 1 = no data (27.6 % of this scene, the area outside the swath) |
| `beta_cloud_mask`, `beta_cirrus_mask` | uint8, 1 = flagged, 0 = clear, 255 = fill (461 cloud pixels, 0 cirrus here) |
| also present, unused | `surface_reflectance_uncertainty` (426, H, W), AOD, water vapour, sun/sensor angles, path length, time |

GDAL's HDF5 driver reads the same grid: EPSG:32613, `(30, 0, 589050, 0, -30, 3653460)`.

## Interfaces

- `read_tanager_metadata(path) -> TanagerMetadata` — band table (sorted), grid, scene id,
  acquisition time. No cube read. Used by the helper, the georef resolver and onboarding's
  duplicate pre-check, so all three agree on id and grid.
- `pick_tanager_file(paths)` — the file to read: an `.h5`/`.he5`/`.hdf5` if there is one (an
  `sr`-named one first), else an ENVI `.hdr` whose data file exists. Used by
  `backend/allotrope/sensors/source_path.py`, onboarding's `_filename_for_id` and the georef
  resolver.
- `TanagerHelper.extract_specific_bands(bands, mode)` — (C, H, W) float32 in sorted band order;
  `extract_layer(name)` — one (H, W) layer.
- `TanagerDatasetBuilder.vend_dataset(band_filter_config=None) -> VendableHyperspectralDataset`;
  `cloud_cover_pct()` for the Scene row; `stac_item` (id, WGS-84 bbox, datetime, `proj:epsg`).
- `resolve_scene_georef(sensor_type="tanager", ...)` -> `(Affine, CRS)`.

## Data flow

```
upload ortho_sr.h5 ─▶ scene_onboard: pick_tanager_file ─▶ read_tanager_metadata (id = strip_id − "_strip")
   ─▶ TanagerDatasetBuilder.vend_dataset()           426 native bands, Planet's bad bands invalid
   ─▶ splib07 cache for that grid · visualizations · Scene row (EPSG, cloud %, bbox)

band_filter_apply (Tanager default) ─▶ vend_dataset(BandFilterConfig)
   flags + exclusion + edge trim ─▶ cloud/cirrus invalidation ─▶ spatial mask ─▶ PCHIP
   ─▶ [no common-grid resample] ─▶ nearest-valid fill    → 325 native bands on the test scene

export ─▶ _resolve_tanager: StructMetadata grid + epsg_code → GeoTIFF CRS/transform
```

## Invariants

- **Ascending wavelength everywhere.** The file's array is not guaranteed ascending (an
  earlier scene had band 3 at 366.38 nm where 386.38 nm belonged). The helper argsorts once
  and keeps `file_band_index`; reads go through it, so the cube and its wavelengths move
  together. Duplicate or non-finite wavelengths raise.
- **Voxel validity** = value ≠ `_FillValue` ∧ finite ∧ `nodata_pixels == 0` ∧ band good. The
  finite term is not decoration: the AVIRIS-NG builder counted NaN as valid, which broke
  spectral matching on GHOSt. Invalid voxels are written as 0.0, as EnMAP's transformer does.
- **Every FWHM > 0.** splib07 resampling silently returns nothing for FWHM ≤ 0. A band without
  a usable width is flagged bad and given its band spacing as width.
- **Reflectance is not rescaled.** It is already 0-1.
- **Family split** VNIR < 1000 nm ≤ SWIR, as AVIRIS-NG; `band_index` restarts per family.
- **Quality masks** count only value 1; 255 (fill) is not cloud. They apply only through
  `quality_masks_to_apply` (`cloud`, `cirrus`); the Tanager template applies both. Unknown
  names are logged and ignored, matching EnMAP's behaviour.
- **Grid**: outer-corner registration, so affine = `(dx, 0, ULx, 0, −dy, ULy)` with no
  half-pixel shift (PRISMA's corner attributes are pixel *centres*; Tanager's are not).
  EPSG from `epsg_code`; otherwise WGS-84 UTM from `ZoneCode`, negative = south (GCTP). The
  UL northing is not a hemisphere test — southern UTM northings are positive too.

## ENVI input (`tanager_envi.py`)

The same sensor also accepts an ENVI upload: a `.hdr` plus its data file (`.img`, `.bsq`,
`.bil`, `.bip`, `.dat`, or the header's stem with no extension; `x.img.hdr` beside `x.img`
works too). `read_tanager_metadata` and `TanagerHelper` choose the reader by suffix, so
onboarding, band filtering and export run unchanged on either format.

| Header field | Becomes |
|---|---|
| `wavelength`, `wavelength units` | band centres in nm; micrometres ×1000; no unit and every value < 100 reads as µm (logged) |
| `fwhm` | widths, same unit conversion. Missing: the band spacing (`np.gradient` of the sorted centres), `fwhm_derived = True`, logged |
| `bbl` | `good_band`; missing = all good. The band-filter exclusion ranges still apply |
| `data ignore value` | `fill_value`; missing = NaN, so only the finite test removes voxels |
| `reflectance scale factor` | data ÷ factor = 0-1. Missing: detected from a decimated sample (8 good bands, ≤ 256 × 256): p99 > 10 → ÷ 10000. `reflectance_scale_source` records `header` / `detected_0_1` / `detected_0_10000` and goes into the Scene's STAC properties |
| `map info`, `coordinate system string` | read by GDAL, not parsed here: CRS + affine for export, `TanagerGrid` for bbox and `native_projection` |
| `band names` | `band_names`, sorted with the bands |
| `description` | scene id fallback (below) |

- **GDAL reads the data.** One `rasterio` read of every band at once, for BSQ, BIL and BIP
  alike (a per-band read of a BIP file would scan it once per band). The AVIRIS-NG reader's
  hand-built memmaps misread BSQ; this path has no memmap.
- **Validity** = value ≠ fill ∧ finite ∧ band good. ENVI has no `nodata_pixels`; an
  outside-swath pixel is fill in every band, so it drops out anyway. There are no cloud or
  cirrus masks: each `quality_masks_to_apply` name is skipped with a warning and listed in
  `band_filter_apply`'s summary as `quality_masks_skipped`. `cloud_cover_pct` is NULL.
- **Scene id**: a Planet item id (`YYYYMMDD_HHMMSS_xx_yyyy`) in the file name, else in the
  header `description`, else the file stem. Acquisition time comes from the id, else an ISO
  `acquisition time` header field, else NULL.
- **Duplicates**: the id rule means an ENVI copy of a scene already onboarded from HDF5 *is*
  that scene, and onboarding refuses it (`scene already exists: … sensor_scene_id=…`), as for
  any repeat upload. The refused upload stays in `/data/staging/<job_id>/`, as every failed
  onboarding does.
- **HDF5 wins** when an upload carries both formats: it has Planet's masks.
- **Georef**: `_resolve_tanager_envi` opens the data file with rasterio. No CRS (no map
  info) raises `GeorefUnavailable` "has no map info"; since v1.0.1 the export then ships in
  pixel coordinates (`app.georef.pixel_grid()`, no CRS, no `.prj`, MANIFEST `georeference`,
  `X-Allotrope-Georef: none`) instead of 422 `crs_missing`. A rotated map info, or a CRS with no EPSG
  code, still exports, but leaves the Scene without a bbox or `native_projection`.
- **Frontend**: the Tanager file input accepts several files (`multiple`) so the pair goes in
  one upload; a data file with no extension needs the picker's "All files" filter.

## Verified end to end (2026-11-01, local stack, 16 worker threads, CPU)

Through the API on the file above: onboarding 25 s (426 bands, EPSG:32613, cloud 0.07 %,
splib07 cache 518 entries at 426 bands) · `band_filter_apply` 20 s (426 -> 325 bands: 58
flagged, 34 wavelength, 9 edge; 461 cloud pixels; 72.4 % spatially valid) · `anomaly_scoring`
340 s (Indradhanu V2 read 325 bands, 280 s; MNF-RX 5 s) · `anomaly_detection_prep` committed
at p99.5 -> 3,303 pixels · `spectral_library_match` 20 s (SAM matched 3,303 of 3,303; family
status ok, 438 named man-made) · both exports `hyper_*.zip`, masks EPSG:32613
`(30, 0, 589050, 0, -30, 3653460)`, prep mask values {0, 1}.

**ENVI copy of the same scene (same day, same stack).** A BSQ float32 `.hdr` + `.img`
written from `ortho_sr.h5` (file band order, `bbl` = `good_wavelengths`, nodata pixels set to
the `data ignore value` -9999, `map info` UTM 13N UL 589050 / 3653460, 30 m). Under its
item-id name the upload was refused as a duplicate of the HDF5 scene; renamed
`tanager_envi_e2e` (no id in name or description, so no acquisition time) it ran: onboarding
20 s (426 bands, EPSG:32613, bbox identical to the HDF5 scene, cloud NULL, scale
`detected_0_1`) · `band_filter_apply` 10 s, 325 bands, `quality_masks_skipped` cloud + cirrus,
661,031 valid pixels = the HDF5 run's 660,570 + its 461 cloud pixels · filtered cube
bit-identical to the HDF5 run's on every pixel valid in both · `anomaly_scoring` Indradhanu
V2 265 s, MNF-RX 6 s · prep p99.5 -> 3,306 pixels · masks EPSG:32613
`(30, 0, 589050, 0, -30, 3653460)`, values {0, 1}. Scores vs the HDF5 run on the 660,570 common
pixels: MNF-RX r = 0.9986; Indradhanu V2 Pearson 0.53, Spearman 0.90, and Pearson 0.993-0.996
on pixels 16 px or more from a cloud pixel (r 0.01 within 2 px, 0.16 at 2-16 px). The whole
difference is the cloud pixels the ENVI path cannot mask: they enter the 128 px inference
tiles and the residual covariance. An ENVI scene with cloud therefore scores worse beside
the cloud than its HDF5 original, and the format cannot fix that. The wavelengths were
written to 4 decimals (up to 5e-5 nm off the float32 HDF5 values), so the splib07 cache key
differs and a second cache file was built.

## Failure modes

| Symptom | Cause |
|---|---|
| Onboarding `ValueError: StructMetadata grid … != cube …` | file is not an ortho grid product, or truncated |
| Onboarding `Unrecognized File Extension` | an extension `FileSourceConfig` does not map (it accepts `.h5`, `.he5`, `.hdf5`, `.hdr`) |
| Onboarding `no .h5 … and no ENVI .hdr with its data file` | an ENVI header uploaded without its data file, or the data file named differently from the header's stem |
| Onboarding `GDAL reads N bands, header says M` | the data file does not belong to the header, or the header was edited |
| ENVI export in pixel coordinates (`X-Allotrope-Georef: none`) | the header has no `map info` (the resolver says so in the log) |
| Export in pixel coordinates (`X-Allotrope-Georef: none`) | `_resolve_tanager` raised `GeorefUnavailable`: no `.h5` under `raw/`, unreadable grid, or the mask's shape ≠ (YDim, XDim). It never rescales: band filtering does not crop a Tanager cube, so a mismatch means the wrong file |
| Scene id is `ortho_sr` | the file had no `strip_id` attribute; a second such upload collides on `(sensor_type, sensor_scene_id)`, and `stac_item` has no datetime |
| Memory | the cube is read whole: 1.55 GB float32 + 0.39 GB validity for 426 × 1020 × 895, ~3 GB peak during band filtering — EnMAP's order of magnitude |

## Decisions

- **Native sensor, not the AVIRIS-NG costume.** GHOSt went in dressed as AVIRIS-NG
  (`ghost_to_envi.py`). For Tanager that would have meant converting 1.3 GB to ENVI and
  throwing away what the file already carries: Planet's bad-band list (an AVIRIS bbl could
  hold it), the nodata layer, the cloud and cirrus masks (ENVI has no slot), and the exact
  UTM grid (re-expressed as ENVI map info). Scenes would also be labelled AVIRIS-NG in the
  Library and the splib07 cache key. Native costs one helper, one builder and a sensor slug.
- **`VendableHyperspectralDataset`, not the EnMAP variant.** The masks fold into validity at
  band-filter time; nothing downstream reads per-mask arrays off a vendable, and a new
  pickled class would be a wire-format change (see `vendables.py`).
- **In RAM, not memmap.** The scene is EnMAP-sized, not AVIRIS-NG-sized (6.4 GB).
- **No template.** One fixed GRID layout; `FileHelper.template` is `{}`.
- **Band-filter defaults copy AVIRIS-NG** (same ~5 nm sampling), native bands, nearest-valid
  fill on, `quality_masks_to_apply: ["cloud", "cirrus"]`. Cloud is excluded and water is not,
  per the product rule. Planet's 1342-1443 / 1783-1967 nm flags are wider than the default
  exclusion windows; they drop through `band_validity_by_position` like a bbl.
- **Scoring default `Indradhanu V2`**, the native-band model; it reads the bands it was
  trained on (`trained_band_mask`) out of the 325.
- **`cloud_mask` stays Landsat-only.** Tanager's cloud mask arrives with the file.
- **ENVI is a second format of the same sensor, not a new slug and not the AVIRIS-NG
  costume** (agreed 2026-11-01). One slug keeps one band-filter default, one scoring default,
  one splib07 cache key and one Library label for the same instrument; the costume would
  have read BSQ wrongly and labelled the scene JPL's.
- **GDAL for the ENVI cube and georef**, not a memmap and not a hand parser for `map info`:
  interleave, byte order, header offset and the map-info rotation term are GDAL's problem,
  and it is the same reader `_read_georef_from_geotiff` already trusts.
- **Skipped masks are a warning, not an error.** The Tanager band-filter default asks for
  `cloud` and `cirrus`; failing every ENVI scene on that would make the default unusable.
