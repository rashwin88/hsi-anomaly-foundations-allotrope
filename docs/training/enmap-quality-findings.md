# EnMAP data quality findings (2026-10-13)

What an automated check of every EnMAP scene we hold turned up, and what each finding means for
the anomaly-detection pipeline. The numbers below were measured; where an impact is inferred from
reading the code rather than measured on pipeline output, it says so.

## What was checked

| Set | Location (Google Drive) | Scenes |
|---|---|---|
| L2A, original | `MyDrive/PS_11/EnMAP_datasets/stage_3/` | 100 |
| L2A, added 2026-10-13 (Odisha/WB coast, North East, Rajasthan/Punjab border) | same folder | 55 |
| L1C, picked across haze / season / sun angle / surface / latitude | `MyDrive/PS_11/EnMAP_datasets/enmap_L1C/` | 60 |

Every scene was checked against its own product layers, with no external data:
`QL_QUALITY_CLASSES` (footprint, land/water), the cloud / cloud-shadow / cirrus / haze / snow
masks, `QL_PIXELMASK`, and the spectral cube read at its 4× overview. Result tiers:

| Set | Flag | Check | Pass |
|---|---|---|---|
| L2A (155) | 16 | 65 | 74 |
| L1C (60) | 2 | 26 | 32 |

14 of the 16 L2A flags are simply cloudy scenes in the original 100 (27–65% cloud by the
product's own mask). The findings below are the ones that matter beyond "this scene is cloudy".

Scripts, per-scene results and download manifests are in `MyDrive/PS_11/EnMAP_datasets/_downloads/`:
`scene_qa.py` / `scene_qa.json` (the checks), `viz6.py` (the comparison panels),
`l1c_features.json` (per-scene AOT, water vapour, sun angle, surface fractions), and the batch
manifests. See `dlr-geoservice-downloads.md` for how the data was downloaded.

## Finding 1: L2A reflectance goes negative, mostly in the blue

**What we measured.** 30 of 155 L2A scenes have reflectance below zero on more than 0.5% of clear
land. In the two worst scenes, read at full resolution (no nodata involved):

| Share of clear pixels < 0 | 424 nm | 445 nm | 550 nm | 660 nm | 864 nm | Typical value | 1st percentile |
|---|---|---|---|---|---|---|---|
| Coastal scene `DT0000219322_…_006`, 2026-10-09, AOT 0.63 | 34.6% | 29.4% | 0.0% | 0.3% | 0.1% | −0.006 | −0.021 |
| High-altitude scene `DT0000178884_…_010`, 2026-02-18, 2,689 m | 29.1% | 23.6% | 10.3% | 6.4% | 1.4% | −0.014 | −0.045 |

In the coastal scene the negatives sit on dark vegetated land, not the sea; in the high-altitude
scene, 44% of lake pixels are negative at 445 nm.

**Why it happens.** Atmospheric correction subtracts an estimate of the light scattered by air and
aerosol. Over surfaces that reflect almost nothing in a band (water, shadow, dense dark vegetation
in the blue), that estimate can exceed the signal, and the remainder goes negative. A
too-high aerosol estimate makes it worse; the coastal scene has one of the highest in the dataset.
This is a known artefact of surface-reflectance products, not file corruption.

**Pipeline impact (from the code).**
- `EnmapL2aDnToSurfaceReflectanceTransformer` applies the gain and zeroes only nodata. The vendable
  builder (`EnmapDatasetBuilder`) marks pixels invalid only for nodata and `QL_PIXELMASK`. Negative
  values therefore pass unchanged into vendables, models and normalisation.
- The 165-band common grid starts at 460 nm, which drops the worst bands (418–450 nm). 460–480 nm
  still carries negatives, and over water and shadow so do the green and red.
- `scene_segmentation` masks water (NDWI > 0.3) and vegetation (NDVI > 0.4) by default, which
  should remove many of the affected pixels before scoring. Not verified on these scenes.
- Risks if they reach scoring: physically impossible spectra are hard for a reconstruction model
  to reproduce, so they can score as false anomalies along shorelines, in shadow and under dark
  canopy; they also shift the per-band normalisation statistics (`anomaly_scoring` computes
  per-scene mean/std over the keep mask for uncalibrated sensors).

**Recommendation.** Measure before changing anything: run `anomaly_scoring` on the two scenes
above and compare scores on pixels with and without negative values. Then decide between clipping
at 0, masking pixels below a tolerance (e.g. −0.01), or leaving them.

## Finding 2: aerosol (AOT) stuck at ≈ 0.35 over bright desert

**What we measured.** 17 of 155 L2A scenes report `sceneAOT` between 0.345 and 0.355; 15 are in
the northwest desert, and their bright-soil cover averages 70% against 20% for the rest.

**Why it matters.** The processor estimates aerosol from dark reference surfaces, which bright
sand and salt lack. The clustering strongly suggests a fallback value rather than a measurement
(inferred; not confirmed against DLR documentation). Where real dust is heavier, haze remains in
the reflectance; where lighter, the blue is over-corrected.

**Pipeline impact.** These scenes carry a scene-wide spectral bias of unknown sign, which is
domain shift for a model trained on mixed scenes. `sceneAOT` cannot be used to label haze or dust
for these scenes: 8 of the 10 L1C scenes in the "dusty" group were picked on this value. All were kept
after visual review, but their dust label rests on the image, not on `sceneAOT`.

## Finding 3: the product's own masks are unreliable on this terrain

| Scene | What the mask says | What the image shows |
|---|---|---|
| Rann of Kutch `DT0000127112_…_004`, sea level | Snow 22% | Bright salt flats; no snow |
| Thar desert `DT0000149928_…_004` | Cloud 4.4%, cloud shadow **0%** | Cumulus mostly masked; every cloud's shadow unmasked |
| Plateau `DT0000181492_…_014`, 4,726 m | Cloud 28%, cirrus 43%, 658 small cloud patches | Large stretches of clear bare ground marked as cloud |

Also: the snow mask fires at low elevation in 8 scenes, and the metadata `cloudCover` simply
repeats the mask (so it inherits the same misses).

**Pipeline impact (from the code).**
- `BandFilterConfig.quality_masks_to_apply` (`app/models/dataset/vendables.py`) defaults to
  `cloud`, `cloud_shadow`, `haze`; `EnmapDatasetBuilder` invalidates flagged pixels at vendable
  build, and `band_filter_apply` then fills invalid pixels from their nearest valid neighbours
  (`apply_nearest_valid_fill`, on by default). So: false cloud on clear ground (the plateau) **removes good data**, and missed cloud
  shadows (Thar) **stay in as valid pixels**. Snow is not applied by default, so the salt-flat false
  snow does not remove data.
- `scene_segmentation` adds its own masks. Its shadow test (VNIR brightness < 0.02) is unlikely to
  catch cumulus shadows on bright sand, whose brightness stays far above that; its cloud test
  (brightness > 0.4) may remove bright salt flats as cloud. Both inferred from the thresholds, not
  verified on these scenes.
- Unmasked cumulus shadows are compact dark spots with unusual spectra, which is what an
  anomaly detector flags. Expect false positives next to cumulus in desert scenes.

## Finding 4: a few spectra rougher than their peers

The check compares band-to-band jitter in each scene's median spectrum (inside windows clear of
water-vapour absorption) with all scenes of the same level. Two L1C scenes stand far out
(`DT0000067061_…_016`, `DT0000100722_…_005`). The first looks normal in true colour; its median
spectrum is shaped differently rather than visibly noisy, and the cause is open (cirrus mask 0%;
16% cloud and 13% water in the scene). Treat this flag as a pointer to look closer, not a defect.

## What the automated checks cannot see

Visual review caught five Rajasthan/Punjab scenes (dust haze, salt, scattered cumulus) that the
checks placed only in **Check**, through indirect signs (AOT default, false snow). The checks find
measurable failures (heavy cloud, over-correction, broken spectra, data gaps); subtle dust and
mask misses still need a human look. After that review, all 215 scenes were kept.

## Open follow-ups

1. Quantify Finding 1 on real pipeline output before choosing clip / mask / keep.
2. Decide whether to keep `cloud_shadow` and `cloud` in `quality_masks_to_apply` for desert and
   high-plateau scenes, or replace them with our own masks.
3. Check `scene_segmentation` on the three Finding 3 scenes: does it mask the salt as cloud, and
   does it miss the cumulus shadows?
4. Don't use `sceneAOT` as a haze/dust label; use visual review or an image-based haze measure.

## Viewers

Two private claude.ai pages show the evidence (share them from each page's Share menu before
sending the links on):
- Scene review, all 215 scenes with check results and thumbnails:
  <https://claude.ai/artifact/QVJRaEtZM1biGh4W66p3VZ>
- Findings with side-by-side mask panels for six scenes:
  <https://claude.ai/artifact/Ugh5pf3GSqjwk3xqiSUXVT>
