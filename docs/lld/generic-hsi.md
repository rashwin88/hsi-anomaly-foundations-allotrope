# Any hyperspectral file: the `generic_hsi` sensor

**Files:** `app/utils/files/generic_hsi_helper.py` (picker, subdataset choice, reads),
`generic_hsi_bands.py` (where the band table is found), `generic_hsi_scale.py` (band order, FWHM,
no-data, scale), `generic_hsi_georef.py` (file CRS, `.igm`, refusal), `generic_hsi_preflight.py`
(the report), `app/models/file_processing/generic_hsi_metadata.py`,
`app/utils/dataset_builder/generic_hsi_dataset_builder.py`, the `generic_hsi` branch of
`app/georef/from_scene_file.py`, plus the `sensor_type="generic_hsi"` entries of the
[sensor checklist](../06-backend.md#adding-a-sensor). Frontend: `components/PreflightReport.tsx`,
the "Other hyperspectral (any format)" entry of `IngestPanel.tsx`, the pre-flight panel on
`SceneDetailPage.tsx`.
**Tests:** `tests/test_utils/test_generic_hsi.py` (the synthetic "surprise" suite; writers in
`tests/test_utils/generic_hsi_files.py`; one `large_files` check reads Planet's Tanager `.h5` as an
unknown HDF5), `tests/test_backend/test_generic_hsi_pipeline.py` (band_filter_apply.run ->
anomaly_scoring run with MNF-RX, in-process).

## Purpose

Once the AIGC evaluation starts the code is frozen, and the data may be GHOSt GeoTIFFs, OSK's native
ENVI (`.hsi` + `.hdr` + `.igm`) or something nobody has shown us. Native readers stay for PRISMA,
EnMAP, AVIRIS-NG and Tanager, because they use the provider's quality masks; `generic_hsi` is the
safety net for everything else ([proposal](../proposals/robust-hsi-ingestion.md), agreed and built
2026-11-01). GHOSt is its first user; there is no `ghost` slug, and the offline AVIRIS-NG costume
(`ghost_to_envi.py`) is no longer needed.

## Data flow

```
upload (any files) ─▶ pick_generic_file: the largest GeoTIFF / ENVI .hdr (by its data file) / HDF5-NetCDF
  └─ HDF: choose_subdataset - the 3-D dataset with the most bands, uncertainty/mask/angle names skipped
read_generic_hsi_metadata (no cube read; one decimated sample)
  band table   1 per-band tags · 2 ENVI header · 3 HDF attrs (GDAL root tags, then h5py) · 4 descriptions
               · 5 wavelength CSV/JSON in the upload            -> sorted, de-duplicated, nm
  FWHM         first exact list · else 9 nm if the metadata names GHOSt/OSK · else 1.95 x spacing
  no-data      ENVI `data ignore value` · GDAL nodata · always -9999, NaN/inf, all-zero spectra
  scale        operator > header factor > radiance units > heuristic           -> factor, reflectance
  georef       file CRS+affine · .igm lat/lon (EPSG:4326 bbox fit) · none
  └─ preflight report (JSON)  -> Scene.extra_metadata["preflight"], job.payload["preflight"]
GenericHsiDatasetBuilder.vend_dataset()  one GDAL read, scaled, invalid voxels 0.0, units
  └─ band filter -> spatial mask -> PCHIP -> (common grid)         [the Tanager / EnMAP pipeline]
export ─▶ resolve_generic_hsi_georef ─▶ (Affine, CRS) or GeorefUnavailable -> 422 crs_missing
```

## The pre-flight report

Written by `generic_hsi_preflight.assemble`, schema `allotrope.generic_hsi.preflight/1`. Keys:
`file`, `data_file`, `subdataset`, `format` (GDAL driver), `interleave`, `dtype`, `shape`
(`bands_in_file`, `bands` kept, `height`, `width`), `wavelengths` (`source`, `units_read`, `min_nm`,
`max_nm`, `mean_spacing_nm`), `fwhm` (`source`, `derived`, range), `bad_bands`, `scale` (`decision`,
`factor`, `offset`, `reflectance`, `source`, `reason`, `sample` p50/p99/p99.9/max), `nodata`,
`georef` (`source`, `exportable`, `crs`, `epsg`, `transform`, `bbox_wgs84`, or `reason`),
`options` (the operator's `allotrope_ingest.json`), `assumptions` (every guess, in words) and
`warnings` (not reflectance, no map, a header factor that leaves values above 1.5).

## Decisions and why

| Decision | Rule | Why |
|---|---|---|
| Reader | GDAL (rasterio) for every cube, one `read()` of all bands | Interleave, byte order, header offset and map-info rotation are GDAL's problem; the AVIRIS-NG reader's hand-built memmap misreads BSQ silently |
| Band order | sort by wavelength, keep `file_band_index`; drop duplicates (first kept) and wavelengths <= 0 | GHOSt dataset 3: position 218 is labelled `Band 224 (1396 nm)`. The band *label* is never an index |
| Units | um -> nm; no unit and every value < 100 read as um, noted | GHOSt writes `wavelenght_units=nm` (misspelt) - matched by "starts with `wavel`, contains `unit`" |
| Descriptions | the number in parentheses | `Band 472 (2507.2)` (dataset 2, no unit) must still parse |
| IMAGERY domain | used, but its FWHM marked inexact | GDAL prints `FWHM_UM` to 0.001 um: OSK's 8.7 nm reads as 9. The ENVI header wins |
| FWHM fallback | 1.95 x band spacing (`np.gradient`), or 9 nm when the metadata says GHOSt / Orbital Sidekick / OSK; always `derived` | The costume's 1.0 x spacing (4.48 nm) was half the true width and made library spectra too sharp. 1.95 x 4.48 = 8.7 nm, OSK's own figure |
| Scale heuristic | decimated sample (8 good bands, <= 256 x 256, zeros and no-data out): p99.9 <= 1.5 -> 0-1; median 30-6000 and p99 <= 12000 -> x10000; else not reflectance | 0-1 reflectance stays under 1.5 even with glint; x10000 land medians sit in the hundreds to low thousands. A DN cube (median ~20000+) fits neither |
| Header factor | ENVI `reflectance scale factor` (divides), GDAL scale/offset, `scale_factor`/`add_offset` tags | A stated factor beats a guess; a result still above 1.5 is a warning, not an error |
| Radiance | units text (`data units`, band `units`, the subdataset name) matching radiance / W m-2 / sr | Radiance read as reflectance gives confident nonsense |
| Override | `allotrope_ingest.json` `{"scale": "auto" | "x1" | "x0.0001" | "x<k>" | "radiance"}`; the upload form's `scale_override` writes it | Stored in `raw/`, so every re-vend (band_filter_apply) applies it; the scene id gets `@<override>` so a corrected re-upload is not refused as a duplicate |
| Not reflectance | vended unscaled, `units="not_reflectance"`; anomaly_scoring skips foundation models (`models_skipped_not_reflectance`), refuses if only foundation models were asked for; spectral match adds a `warning` | MNF-RX is unit-free. The default stays `["Indradhanu V2", "MNF-RX"]`: the data decides, not a per-scene default the catalog cannot express |
| No-data | declared value, plus always -9999, non-finite, and a pixel 0 in every band; a lone 0 stays valid unless 0 is declared | GHOSt declares 0; Tanager uses -9999; NaN broke spectral matching on the costume. A single true 0 (deep absorption) is real data |
| Georef | file CRS + non-identity affine (EPSG:4326 allowed); else `.igm`; else refuse | An unplaceable map must never be submitted |
| IGM | own ENVI parser + numpy, band roles from `band names`, else by range / gradient; EPSG:4326 affine fitted to the lat/lon bounding box | GDAL opens `x.igm` with `x.hdr` - the cube's header - before `x.igm.hdr`. The bbox fit is PRISMA's fallback, approximate for a rotated swath (noted in `assumptions`) |
| Segmentation | V2 refuses a not-reflectance scene ("V2 segmentation needs reflectance; ... use method 'indices' or fix the scale at upload"), checked from the band filter's `summary.json` `scene_units` before any load; the generic_hsi template carries `method_if_not_reflectance: "indices"`, so the default falls back to the index method | The head was trained on reflectance; the static catalog cannot hold a per-scene default, so the fallback is a config field only this template sets |
| Scoring summary | `model_codenames` lists the models that RAN | anomaly_detection_prep reads each listed model's raster; a skipped V2 has none (found in the E2E) |
| Masks | `quality_masks_to_apply` names skipped with a warning (`quality_masks_skipped`) | No provider masks exist; failing the default would make it unusable |
| Defaults | band filter native bands + usual windows, no masks; segmentation V2; scoring V2 + MNF-RX; prep and spectral match as every HSI sensor | Same recipe as Tanager |

## Invariants

- Ascending `band_cw_order`, every FWHM > 0, NaN never valid, invalid voxels 0.0.
- `pick_generic_file` is the one picker: `source_path`, `_filename_for_id`, the builder and the
  georef resolver all go through it (via `locate_data`), so they read the same file.
- Sidecars (wavelength file, `allotrope_ingest.json`, `.igm`) are searched under the scene's `raw/`
  folder (`upload_root`), wherever in the upload they sit.
- The export detail stays `crs_missing` (the UI matches on it); the reason - "this file has no map
  coordinates ..." - is in the `X-Allotrope-Reason` header and on the scene page.

## Failure modes

| Symptom | Cause |
|---|---|
| Onboarding `no hyperspectral cube found ... got extensions [...]` | nothing GDAL-readable: e.g. an `.hsi` without its `.hdr` |
| Onboarding `no wavelengths found ... Upload a wavelength file` | no band table in any of the four places; add a CSV / JSON with one value per band |
| Onboarding `GDAL reads N bands, the header says M` | the data file is not the header's |
| Scoring `this scene is not reflectance ...` | only foundation models picked on a DN / radiance scene; add MNF-RX or re-upload with an override |
| Export 422 `crs_missing`, header `this file has no map coordinates` | no CRS, no map info, no `.igm` with its own header |
| `IGM holds projected coordinates` | an `.igm` in UTM metres; only degrees are supported |
| Memory | the cube is read whole, as Tanager: float32 C x H x W plus a bool cube of the same size |

## Verified end to end (2026-11-01, local stack, through the API)

Inputs built from Planet's Tanager scene `20250205_175845_00_4001` (rows 64-384, cols 64-384: 426 bands x
320 x 320, 21 % no-data), script `/root/generic-hsi-e2e/e2e_generic.py` in WSL. Each ran onboard ->
band_filter_apply (default) -> anomaly_scoring (default `["Indradhanu V2", "MNF-RX"]`) ->
anomaly_detection_prep -> commit p99.5 -> export -> spectral_library_match -> export.

| Case | Pre-flight | Chain | Export |
|---|---|---|---|
| (a) GHOSt-like COG: pixel interleave, float32 0-1, `wavelength` + `wavelenght_units=nm`, nodata 0, EPSG:32613 | band_metadata, FWHM derived 1.95 x spacing (9.66 nm), reflectance_0_1, nodata 0, georef file | 426 -> 335 bands; V2 + MNF-RX; 404 px at p99.5; splib07 cache 518 entries | 200, mask EPSG:32613 `(30, 0, 590970, 0, -30, 3651540)`, values {0, 1}; match bundle 200 on the same grid |
| (b) same cube uint16 x10000, no georef | reflectance_x10000 (median 3730, p99 4887), georef none, warning "export will refuse" | 426 -> 331; V2 + MNF-RX; 404 px | 422 `crs_missing`, `X-Allotrope-Reason: this file has no map coordinates ...` (prep and match) |
| (c) OSK-style ENVI `.hsi` + `.hdr`, BSQ, `wavelength` + `fwhm` + map info, ignore -9999 | FWHM from envi_header (5.2-6.8 nm), reflectance_0_1, nodata -9999, georef file | 426 -> 335; V2 + MNF-RX; 404 px | 200, mask on the grid of (a), {0, 1}; match 200 |
| (d) (b) with override `radiance` | not_reflectance (operator), id `tanager_u16_nogeo@radiance` | scoring ran MNF-RX only (`models_skipped_not_reflectance: ["Indradhanu V2"]`); prep, commit ok; segmentation default fell back to indices, explicit V2 refused | 422, as (b) |
