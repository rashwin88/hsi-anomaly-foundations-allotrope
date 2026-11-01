# PV / glass spectra in the namer's library, without retraining (addition B) — did not work

**Question** (docs/proposals/class-driven-flow.md, addition B): the namer's level 1 and its classes are decided by
matching against library embeddings, and the library tower encodes any spectrum. If we add in-scene spectra of solar
parks as a new man-made class `glass_solar`, do PV pixels gain P(man-made) and get named, with no retraining?

**Answer: no.** With the served namer (N001b, epoch 40), adding 50 in-scene PV spectra changed nothing. Held-out PV
pixels had the same P(man-made) and the same class before and after, to three decimals. So did the 50 pixels the
library spectra were taken from. The class was **not shipped**: `namer/classes.py` keeps its seven classes, and
`data/family_library` is unchanged. Time-boxed to about two hours, on 2026-11-01.

## Data

Spectra with provenance per pixel: `pv_spectra.json` here. Scene crops and the marked RGB images stayed in the session scratchpad; the
bulk scenes are in `E:\pv_scenes\`.

| Scene (sensor, product) | Park | PV | Background | Notes |
|---|---|---|---|---|
| PRS_L2D_STD_20210722055729 (PRISMA L2D) | unnamed park, about 27.18N 71.83E (Phalodi, Rajasthan): main block | 32 | 10 | 30 m pixels mix panels with the ground between rows; monsoon haze; R1250/R1000 is 1.22-1.28 against 1.10-1.14 for the background |
| same scene | three small blocks, about 27.32-27.36N 71.96-72.00E | 16 | 3 | The east-mid block has no visible rows and was called PV from the bandgap step alone |
| Tanager 20250412_111800_00_4001 (ortho_sr) | Groningen NL, about 53.255N 6.685E | 20 | 7 | The cleanest PV: visible about 0.045, 1250 nm about 0.39 |
| Tanager 20250729_185845_51_4001 (ortho_sr) | Caborca MX, about 30.565N 112.283W | 14 | 9 | No step at 1.1 µm (CdTe, or sparse panels?); doubtful |

- No PRISMA or EnMAP scene in the inventory or on the local stack covers Bhadla or Pavagada. The EnMAP footprints come
  from METADATA.XML; the PRISMA objects carry no footprint.
- No Tanager open-data item covers a large Indian or US park.
- OSM Overpass timed out, so no park polygon is OSM-confirmed. Every pixel was picked by eye on an RGB render, and the
  marked crops were checked.

## Method

1. `add_pv_spectra.py` builds the library:
   - Each spectrum is interpolated onto the library's 2,151-channel ASD axis, as the curated ECOSTRESS spectra are.
   - It is added as `manmade.glass` (source `scene_pv`, with provenance per entry), mapped to `glass_solar` in
     `classes.class_of`.
   - The library version becomes `family_v1+pv1`, so the per-sensor caches rebuild.
   - Hold-out is spatial: per block, the easternmost 20 %. Caborca and east-mid are doubtful, so they are held out
     whole and never added.
   - Result: 50 spectra in the library and 12 held out, plus 20 doubtful.
2. `measure_pv.py` names each pixel with `infer.name_pixels` on its scene crop, so the 3-7 px ring is the real
   neighbourhood. It runs twice:
   - **before**: the old library and the seven classes, with the window-glass entry excluded as in v1;
   - **after**: the new library and eight classes.

## Results (`measure.json`, `measure_inlib.json`)

| Set | n | mean P(man-made) before → after | share > 0.5 | share > 0.25 | class when man-made |
|---|---|---|---|---|---|
| PV held out | 12 | 0.248 → 0.248 | 0.33 → 0.33 | 0.33 → 0.33 | oil / fuel (4 Groningen pixels), both runs |
| PV in the library (its own pixels) | 50 | 0.247 → 0.247 | 0.30 → 0.30 | 0.32 → 0.32 | oil / fuel (15), both runs |
| PV doubtful (Caborca, east-mid) | 20 | 0.694 → 0.694 | 0.70 → 0.70 | 0.70 → 0.70 | plastic (14), both runs |
| Background | 29 | 0.135 → 0.136 | 0.03 → 0.03 | 0.14 → 0.14 | concrete / asphalt (1), both runs |

- One Groningen pixel that is itself in the library got P(glass_solar | man-made) = 0.001; oil / fuel got 0.994.
- By sensor:
  - The PRISMA Rajasthan PV pixels sit at P(man-made) of about 0.002, so no sensitivity preset reaches them.
  - The Groningen pixels are already man-made at 0.6-0.8, named oil / fuel (dark, flat in the visible).
- The existing matches' natural / man-made counts are unaffected, because nothing shipped.

## Why it fails

The pixel tower and the library tower are separate encoders. N001b trained them so that a planted mixture (target
at fill f inside its real ring) meets the *pure lab* spectrum's embedding. An in-scene pixel spectrum given to the
library tower is a mixture, which that tower never saw. Its embedding lands away from where the pixel tower puts the
very same pixel: even a pixel matched against its own spectrum scores below the oil entries. So the claim "the
library tower encodes any spectrum, no retraining" holds only for spectra like the training library, i.e. pure
material spectra.

## What would work

- **Pure PV module spectra.** Lab or field reflectance of crystalline-Si and CdTe modules (literature, or a field
  spectrometer) added as `manmade.glass`. They match the library tower's training distribution. Not found as
  downloadable files in the time box: USGS splib07 and ECOSTRESS have no PV.
- **Retraining** with glass / PV spectra as a planted class (run N00x). That is the reliable route and is outside
  the code freeze.
- **Pixel-side prototypes**: embed the in-scene PV pixels with the pixel tower and match against those. This is a code
  change to `class_scores`; not attempted.
