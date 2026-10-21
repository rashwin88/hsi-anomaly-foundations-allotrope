# Validation against DLR PACO L2A

Each run compares this pipeline's L2A with DLR's PACO L2A for the same EnMAP
datatakes, using `src/validate_vs_paco.py`. Files per run:

- `validation_<date>.csv` - one row per scene and class (land / water, split by
  PACO's `QL_QUALITY_CLASSES`)
- `validation_<date>_bands.csv` - median PACO, median ours, median bias and
  Pearson r for 13 sample bands
- `validation_<date>.txt` - the console output of the run

Columns: `mae_*` is mean |ours − PACO| in reflectance units over all bands,
460–2450 nm (the Allotrope common grid), below 700 nm and above 800 nm;
`bias_460` is the median signed difference at 460 nm; `sam_deg` is the median
per-pixel spectral angle.

## 2026-09-24

Image `enmap-isofit:1.1.3`, analytical line, full resolution (`--step 1`).
DT0000187759 was run with the summer preset's `h2o_min` at 0.2 (commit
`b0693fc`); the other four with 0.5, which does not bind for them - their
presolve H2O sits well above both floors.

| Scene | Site, date | Land MAE | Land MAE 460–2450 | Land SAM |
|---|---|---|---|---|
| DT0000189895 | Eastern India, 22.8°N, Apr 2026 | 0.0097 | 0.0092 | 3.9° |
| DT0000194619 | Rajasthan, 26.6°N, May 2026 | 0.0113 | 0.0115 | 4.5° |
| DT0000122280 | Ganges plain, 25.1°N, Apr 2025 | 0.0134 | 0.0125 | 5.1° |
| DT0000131004 | Hyderabad, 18.0°N, May 2025 | 0.0163 | 0.0163 | 4.8° |
| DT0000187759 | Himalaya, 31.1°N, 1.9–5.9 km, snow, Apr 2026 | 0.0345 | 0.0321 | 4.2° |

What the numbers say:

- **Plains land agrees to 0.009–0.016 mean absolute reflectance.** The blue
  excess at 460 nm varies by scene (+0.006 to +0.035), so it tracks each
  scene's haze rather than a fixed offset.
- **Water is +0.03–0.06 too bright in the NIR on every scene** - adjacency
  from bright surrounding land, which PACO corrects and ISOFIT does not.
- **Band 224 (2445 nm) runs high**, up to +0.05, at the low-signal sensor edge.
- **The Himalayan scene's MAE is pixel scatter, not bias**: median band biases
  stay within about ±0.02, but MAE below 700 nm is 0.065. Bright snow on steep
  terrain points at terrain-illumination handling, which differs between the
  two processors.

## 2026-09-30 - ACOLITE as a second engine

Image `acolite:8e04b97` (ACOLITE at upstream commit 8e04b97, built from
`acolite/Dockerfile`), settings `acolite/settings/enmap_fixed.txt`: ACOLITE's
own default aerosol mode (`dsf_aot_estimate=fixed` - one AOT for the whole
scene, from the darkest pixels), DEM pressure, no ancillary data. ACOLITE
output is repacked with `src/acolite_to_paco.py`; run with
`acolite/run_enmap_set.sh` then `acolite/compare_set.sh`.

**Quinten Vanhellemont's recommended mode, `ancillary_fixed`, has not run
yet**: it downloads NASA MERRA-2 aerosol files, which need an EarthData
login. `settings/enmap_ancillary_fixed.txt` is ready for when one exists.

`validate_vs_paco.py` changed today: the spectral angle now uses only bands
both products fill. ISOFIT numbers are unchanged by this (re-scored,
identical to 2026-09-24); ACOLITE leaves 50 bands blank and would otherwise
score 20-27 deg.

| Scene | ISOFIT land | ACOLITE land | ISOFIT water | ACOLITE water | ACOLITE AOT550 |
|---|---|---|---|---|---|
| DT0000189895 E. India | **0.0097** | 0.0194 | 0.0136 | **0.0048** | 0.33 (maritime) |
| DT0000194619 Rajasthan | **0.0113** | 0.0341 | - | - | 0.67 (maritime) |
| DT0000122280 Ganges | **0.0134** | 0.0177 | 0.0182 | **0.0079** | 0.37 (maritime) |
| DT0000131004 Hyderabad | **0.0163** | 0.0168 | 0.0279 | **0.0138** | 0.51 (continental) |
| DT0000187759 Himalaya | **0.0345** | 0.0422 | **0.0178** | 0.0326 | 0.33 (continental) |

(mean |difference from PACO| in reflectance, all bands)

What the numbers say:

- **ACOLITE fixes the water adjacency problem**: water error roughly halves
  or better on the three plains scenes with real water. The Himalaya "water"
  class is bright (PACO 0.26-0.36 in the visible), so it is ice, snow edge
  or silty glacial water rather than open water, and ACOLITE loses there.
- **ISOFIT stays better on land, on every scene.** ACOLITE mostly runs
  dark (e.g. -0.01 to -0.04 across the spectrum on DT0000189895). Worst on Rajasthan (bright desert): the dark-pixel method finds
  no truly dark pixels, over-estimates aerosol (0.67) and over-corrects blue
  (-0.045 at 460 nm). This is exactly the case Quinten says `ancillary_fixed`
  handles better.
- **ACOLITE has no per-pixel water-vapour retrieval** (fixed 1.5 cm here), so
  the 940/1130 nm bands carry scene-wide vapour errors that ISOFIT removes.
- **ACOLITE leaves 50 of 224 bands blank**: 48 where gas transmittance is
  below `min_tgas_rho=0.70` (including 2346-2445 nm, inside Allotrope's
  460-2450 nm common grid) plus 2 lost to a name clash in the VNIR/SWIR
  overlap (ACOLITE names bands by rounded nm; 911.6/912.0 and 993.1/993.3
  collide).

## 2026-09-30 - PRISMA L1 through ACOLITE, against ASI L2D

First look at work package 1: engine only, no L2D-layout product yet.
`acolite/run_prisma.sh` then `acolite/compare_prisma.sh`
(`src/validate_prisma_vs_asi.py`). Numbers are against ASI's L2D, which is
what Allotrope's PRISMA training data is, not against ground truth.

Two things had to be solved before any number meant anything:

- **ACOLITE will not take a PRISMA L1 on its own.** It insists on ASI's L2C
  product beside it, and reads just three angle arrays from it. At deployment
  there is no L2C, so `src/prisma_l1_geometry.py` computes those angles from
  the L1 (satellite position track vs pixel lat/lon). On 2024-05-11 the view
  zenith matches ASI's to 0.1 deg (18.30-21.43 vs 18.28-21.35); relative
  azimuth is 9.5 deg vs ASI's 11.9 deg, cause not yet known, effect on the
  correction small at these angles.
- **PRISMA L1 positions are off from ASI's ground-corrected L2D** by up to
  ~200 m across track and, on 2021-03-05, ~600 m along track, and the offset
  varies across the swath. The comparison measures a shift per tile (6 x 6)
  and undoes it; after that, per-band correlation is 0.83-0.99.

Land, mean |ACOLITE - ASI| over all bands (water where the scene has any):

| Scene | Sun zen. | `prisma_fixed` (500-970 nm) | `prisma_fixed_swir` (400-2400 nm) | AOT550, 500-970 -> 400-2400 |
|---|---|---|---|---|
| 2021-03-05 | 38.8 | 0.0608 | **0.0453** | 1.32 -> 0.68 |
| 2023-06-20 | 18.0 | 0.0431 | **0.0392** | 0.77 -> 0.63 |
| 2024-05-11 | 19.8 | 0.0381 (water 0.0262) | **0.0327** (water 0.0188) | 0.88 -> 0.70 |
| 2024-09-19 | 27.0 | 0.0211 (water 0.0132) | **0.0127** (water 0.0075) | 0.70 -> 0.38 |
| 2026-03-05 | 42.6 | 0.0367 | 0.0374 | 0.29 -> 0.33 |

What the numbers say:

- **ACOLITE's PRISMA default hunts for dark pixels only in 500-970 nm**, where
  land is never dark, so it over-estimates aerosol (up to 1.32). Searching
  400-2400 nm, as ACOLITE's EnMAP default does, lowers the AOT and improves
  4 of 5 scenes. `prisma_fixed_swir.txt` is the better setting of the two.
- **2024-09-19 is close to the EnMAP-ISOFIT level** (land 0.013, water 0.008).
- **2023-06-20 is off by -0.11 in the NIR regardless of aerosol** (ASI 0.343
  vs ours 0.236 at 828 nm; blue agrees). Aerosol cannot do that. Not
  explained yet; next check is whether ASI's surface NIR sits above the L1
  top-of-atmosphere NIR, which would point at ASI rather than us.
- **ASI's L2D is bright in the blue**: water at 407-468 nm is 0.06-0.09 on
  2024-05-11, where open water is usually 0.02-0.05. Matching ASI therefore
  means matching that brightness too; ACOLITE does not.
- Like EnMAP, ~80 of 234 bands come out blank (`min_tgas_rho`, name clash).
- `ancillary_fixed` (Quinten's advice) has not run: same EarthData blocker.
