# Segmentation with Indradhanu V2's head (`scene_segmentation`, method `indradhanu_v2`)

Added 2026-11-01. Code: `backend/allotrope/action_types/scene_segmentation.py` (config, dispatch),
`backend/allotrope/action_types/_scene_segmentation_run.py` (the run), `_anomaly_scoring_run.py`
(`upstream_v2_segmentation` and the per-model branch), `frontend/src/components/SegmentationViewer.tsx`.
Tests: `tests/test_backend/test_scene_segmentation_v2.py`.

## Why

Until 2026-11-01 Indradhanu V2's segmentation head ran only inside `anomaly_scoring`, to zero thick cloud after
scoring. Its `class_map.tif` / `mask_cloud.tif` were written under `models/<codename>/` and shown nowhere, and the user
could not choose what was excluded. Making it a step of its own gives one place to see the classes, one place to
choose the excluded ones, and a `keep_mask.tif` with the same contract as the index method's.

## Config

| field | default | used by |
|---|---|---|
| `method` | `"indices"` | both: `"indices"` = NDVI / NDWI / brightness thresholds (unchanged), `"indradhanu_v2"` = the head |
| `exclude_classes` | `["cloud"]` | V2: a subset of `water, cloud, shadow, snow`; water stays in because vessels on water are in scope (user, 2026-10-17) |
| `thin_cloud_is_cloud` | `false` | V2: `segment_full_scene(..., thin_cloud=...)`; false = thick cloud only, as the resolver sets for scoring |
| `model_codename` | `"Indradhanu V2"` | V2: must be a catalog model of architecture `spectral_stem_finetune` (checked in `validate_config`) |

- **Back-compat.** A stored config without `method` is `"indices"` (pydantic default, and `run` reads
  `cfg.get("method", "indices")`). The index fields stay in the schema and are ignored by V2.
- **Templates.** `default_config_per_sensor` for prisma, enmap, aviris_ng and tanager is the V2 block only
  (`_DEFAULT_V2`). The index defaults (`_DEFAULT_PRISMA` ...) are kept in the module for method `"indices"`.
- V2 needs a native-band band filter (`use_common_wavelength_grid=false`, the sensor default since 2026-10-31); the
  run calls scoring's `_check_band_grid` and refuses the 165 grid.

## Run

Same calls, in the same order, as `anomaly_scoring` makes for V2: load `filtered_vendable.pkl` ->
`list_catalog` -> `get_inferencer(InferenceConfig(...))` -> `read_bands(cube, wavelengths)` ->
`segment_full_scene(model_cube, valid, thin_cloud=thin_cloud_is_cloud)`, `valid = validity_cube.sum(0) > 0`.

Files under the action's output dir:

| file | content |
|---|---|
| `class_map.tif` | uint8, `CLASS_NAMES` codes 0 land, 1 water, 2 cloud, 3 shadow, 4 snow; **255 outside valid** (`segment_full_scene` returns 0 there, which would read as land) |
| `keep_mask.tif` | uint8, 1 = valid and class not in `exclude_classes` - the contract `anomaly_scoring` already reads |
| `class_codes.bin` | uint8 raw, row-major, `summary.scene_shape`: the class code, `| 0x10` (`haze_bit`) where the head's thin-cloud output is > 0.5 and the pixel is not cloud (only when `thin_cloud_is_cloud=false`), 255 outside valid. The viewer paints its overlay from this |
| `class_map.png`, `class_map_overlay.png`, `preview.png`, `rgb.png` | fixed colours per class (`CLASS_COLORS`), transparent outside valid; RGB with the classes blended at 0.5 (land untinted); a 768 px preview; the scene RGB the viewer draws on |
| `summary.json` = `diagnostics.json` | `method`, `action_id`, model codename / architecture / checkpoint, `exclude_classes`, `thin_cloud_is_cloud`, `pixel_counts`, `class_pct_of_valid`, `excluded_pct_of_valid`, `kept_pct_of_valid`, `legend` (key, label, colour, code, pixels, % of valid, excluded), `scene_shape`, `bands_read`, `device`, `load_seconds`, `segment_seconds`, `runtime_seconds` |

Thin cloud is **not a class code**. It is a display layer, its pixels keep their class by priority (cloud > snow >
shadow > water > land) and stay in `keep_mask`. Percentages are of valid pixels; the five classes add up to 100%.

## anomaly_scoring with an upstream V2 segmentation

`upstream_v2_segmentation(ss_dir)` returns the upstream summary plus its class map (255 -> 0) when the attached
`scene_segmentation` output has `method == "indradhanu_v2"` and a `class_map.tif`; otherwise None and nothing changes.

For a model with `exclude_classes` (V2) and a foundation family, given such an upstream:

- the head is **not** run (no `segment_full_scene` call);
- it is scored **exactly as without an upstream**: `compute_score` over spatial validity (eroded if
  `keep_mask_erosion_kernel_size` is set), then the excluded pixels - valid and `keep_mask == 0` upstream - are
  zeroed. Percentiles, ROC and the PNG use the same mask as before. So V2's score equals the chain where it ran its own
  head (same inputs, same `exclude_classes`, same thin-cloud rule);
- `stats.json` gets `segmentation: "from upstream scene_segmentation action_<uuid>"` (`"internal head"` otherwise,
  `null` for models without exclusion), `excluded_classes` / `cloud_rule` from the upstream config; the summary gets
  `upstream_segmentation`. `class_map.tif` / `mask_cloud.tif` are still written under `models/<codename>/`.

Everything else keeps the existing keep_mask handling: classical detectors narrow `_spatial_mask` to the upstream
keep_mask (so MNF-RX's covariance leaves out the excluded classes), other foundation models score on it, and V2 with
an **index** segmentation scores on its keep_mask and runs its own head as before.

**Invariant.** V2 must not score on the upstream keep_mask: residual Mahalanobis fits its whitening on the scoring
mask, so scoring on the cloud-free mask would change every pixel's score, not only the cloud's (the order the
evaluator measured is "score, then zero"; `docs/lld/spectral-stem.md`).

### Classes left out at scoring (`exclude_classes`, 2026-11-01)

`anomaly_scoring` config `exclude_classes: list[str] | None`, from `water, cloud, thin_cloud, shadow, snow`
(`SCORING_EXCLUDABLE_CLASSES`). Design: [class-driven flow](../proposals/class-driven-flow.md), gap 1.

- **None** (the default, and every action stored before): the segmentation's own `keep_mask.tif`, exactly as above.
- **A list**: `keep_mask_for_classes(ss_dir, classes)` rebuilds the keep mask from the upstream `class_codes.bin` -
  1 = valid and none of the classes; `thin_cloud` is the `HAZE_BIT` layer, so it can be left out without thick
  cloud. No re-segmentation. From there the semantics are the ones above: V2 scores over spatial validity and the
  excluded pixels (valid, keep 0) are zeroed after scoring; classical detectors narrow `_spatial_mask` to it.
  `cloud_rule` is `cloud_or_thin` when `thin_cloud` is in the list.
- On a segmentation with `thin_cloud_is_cloud = true` there is no haze layer (thin cloud is already cloud), so
  `thin_cloud` alone leaves nothing out.
- **Validation.** `validate_config`: known classes, and a `input_scene_segmentation_output_id` must be given. The
  worker: the upstream must be method `indradhanu_v2` with a `class_codes.bin`, else a `PlainActionError` ("Leaving
  classes out at scoring needs a Scene segmentation made with method Indradhanu V2 ...").
- **Recorded.** `summary.exclude_classes` (null or the list), `keep_source` names the classes; every model's
  `stats.json` has `exclude_classes_at_scoring`, and V2's `excluded_classes` / `excluded_percent` are the list's.
- **Dialog.** With a V2 segmentation picked as the keep-mask input, `NewActionDialog` `ScoringClassPicker` shows its
  legend classes (land left out) as "Leave out" checkboxes labelled with the legend's % of valid, pre-ticked with
  the segmentation's own `exclude_classes`. Untouched sends nothing (None). An index segmentation gets one line
  saying its own settings decide.

### The V2 input guard (2026-11-01)

V2 reads native bands; a band filter with `use_common_wavelength_grid = true` (every band filter before
2026-10-31, e.g. the July one on PRS_L2D_STD_20251127041712) cannot feed it.

- **Dialog**: for `scene_segmentation` with method `indradhanu_v2`, and for `anomaly_scoring` with a model whose
  `/models` `band_grid` is `native`, the band-filter picker disables outputs whose summary has
  `common_wavelength_grid_applied: true`, labels them "old 165-band grid, not for V2", skips them when it picks
  the default, and says to add a new Band filter (native bands by default).
- **Run time**: `_check_band_grid` raises `PlainActionError`; the first line of `failure_reason` is "This band
  filter is on the old 165-band grid; add a new Band filter (native bands) and use that." (the worker drops the
  `ValueError:` prefix for this class; the card shows the first line).

**Mismatch.** If the segmentation's band filter is not the scoring's, the worker logs a warning and still uses the
segmentation; a pixel valid for scoring but invalid upstream is then zeroed.

## Viewer

`SegmentationViewer` (props `actionId`, `summary`) renders a method `indradhanu_v2` output: `MatchMap` (one panzoom
wrapper carrying RGB and overlay, so the overlay moves with zoom) with the overlay painted client-side from
`class_codes.bin`, per-class toggles with swatch and % of valid, excluded classes tagged, an opacity slider, hover
readout. `isV2Segmentation(summary)` picks it; the index method keeps `SceneSegmentationOutputViewer`.
`rgb.png` is capped at 4096 px on its longer side and `class_codes.bin` is not; the hover maps between the two.

## Verified (2026-11-01, local stack, CPU, 16 threads)

| scene | segmentation | classes (% of valid) | V2 score vs the pre-change chain | MNF-RX |
|---|---|---|---|---|
| PRISMA, 1201 x 1206, 186 bands read | 49 s (segment 46 s) | land 82.1, water 15.0, cloud 2.9 (excluded), snow 0.04, shadow 0; thin cloud 2.1 kept | bitwise identical; class map and `mask_cloud.tif` identical; no `segment` step in scoring | changed, corr 0.993 on kept pixels (covariance now without cloud) |
| Tanager, 1020 x 895, 325 bands read | 33 s | land 99.998, water 0.002, no cloud | bitwise identical | identical (nothing excluded) |

The pre-change chain is V2 + MNF-RX with no segmentation on the same band filter output, run on the images from just
before this change. Both chains went on through prep (commit p99.5) and export (200). Seen in the PRISMA overlay:
terrain shadow is labelled water (shadow 0%), and bright salt flats are labelled cloud - the head's behaviour, the
same mask scoring used before.

Classes left out at scoring, end to end (2026-11-01, local stack): two V2 scorings off one V2 segmentation, (a)
`exclude_classes: ["cloud"]`, (b) `["cloud", "water"]`. PRISMA PRS_L2D_STD_20250214052558: (b) is 0 on all 226,218
water pixels, (a) scores every one of them, and (a) = (b) everywhere else (381 s / 481 s on CPU). Tanager: 15 water
pixels, same result. The V2 guard on PRS_L2D_STD_20251127041712 with a 165-grid band filter: the option is disabled
in the dialog with the note, and the run fails with the plain first line.

## Not done

- `ActionDetailPane` does not render `SegmentationViewer` yet, and `NewActionDialog` shows `method` as a free-text
  field (both files were being edited by another session; the hook lines are in the 2026-11-01 handover).
- The V2 method has no per-class mean spectra or index histograms (the index method's diagnostics charts).
