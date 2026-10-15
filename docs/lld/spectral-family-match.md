# Spectral family matching

**Purpose.** Name the likely material family (metal, plastic, roofing, paint, …) of an anomaly
pixel in 30 m hyperspectral imagery, and say how confident that name is. The current matcher
compares whole pixels with lab spectra. That rarely names a man-made material, because at 30 m
a man-made object usually fills only part of a pixel.

**Status.** The code and library are on branch `spectral_match_improv`, tested but **not yet
wired into the `spectral_library_match` action**. The wiring is a planned product change; see
[Integrating into the pipeline](#7-integrating-into-the-pipeline).

| What | Where |
|---|---|
| Matching | `app/spectral_match/family_match.py` |
| Library tags, pruning, band choice | `app/spectral_match/family_library.py` |
| Family library (411 spectra, USGS + ECOSTRESS) | `data/family_library/` (provenance in `sources.md`) |
| Library build script | `scripts/curate_family_library.py` |
| Tests | `tests/test_spectral_match/test_family_match.py` |
| Worker image | `backend/Dockerfile.worker` copies the library to `/srv/family_library` |

The experiments behind the numbers below were run on Google Colab, October 2026. Code, logs
and full tables are on the team Drive at `PS_11/claude_experiments/unmixing_next/`
(`README.md` is the run-by-run record). The earlier step-by-step experiments are on the local
branch `vinoth/hypersl-unmixing`, under `research/experiments/manmade-library-unmixing`.

---

## 1. The problem in one picture

A PRISMA or EnMAP pixel is 30 m × 30 m. A vehicle, a roof or a tarpaulin is usually smaller,
so the pixel's spectrum is mostly its surroundings:

```
   one 30 m pixel                      what the sensor records
  ┌───────────────────┐
  │  grass  grass     │       pixel = 0.8 x (grass + soil around it)
  │     ┌─────┐       │             + 0.2 x (roof)
  │ soil│roof │ grass │
  │     └─────┘       │       Whole-pixel matching asks "which library spectrum
  │  grass   soil     │       looks most like this pixel?" and answers "grass".
  └───────────────────┘
```

The current matcher (`app/spectral_match/sam.py`, the Spectral Angle Mapper) does exactly
that. On planted tests it named the right man-made family for **6 %** of pixels at 30 % fill.
The best match for real targets in the 4 PRISMA ground-truth scenes was usually a shrub or a
grass.

There was a second, simpler problem: **the product library has almost no man-made
spectra.** Of the 519 entries in `data/splib07_slim`, 3 are usable man-made materials. The
rest of its "artificial" chapter is paint pigments and lab chemicals. See
[Gotchas](#gotchas), item 2.

## 2. How the new matcher works

```
anomaly pixels (from anomaly_detection_prep)
      │
      ▼
┌──────────────────────────┐   ring = 13 x 13 window minus the centre 5 x 5,
│ local background         │   anomaly pixels left out of it
│ m = mean of the ring     │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐   W = inverse square root of the scene covariance:
│ whiten                   │   directions the scene varies in naturally
│ z = W (pixel - m)        │   (brightness, greenness) count for little
└────────────┬─────────────┘
             ▼
┌───────────────────────────────────────────────┐
│ score each family                             │
│  man-made family: Simplex ACE                 │  "can the pixel's difference from its
│    = how much of z a non-negative mix of the  │   surroundings be built from this
│      family's spectra explains (0..1)         │   family's spectra?"
│  natural family: best single-spectrum ACE     │
└────────────┬──────────────────────────────────┘
             ▼
┌───────────────────────────────────────────────┐   same scores for ~10,000 ordinary
│ calibrate                                     │   pixels of the same scene: the
│  percentile of the pixel's score among the    │   "null". 0.999 = higher than 99.9 %
│  scene's ordinary pixels, per family          │   of ordinary pixels
└────────────┬──────────────────────────────────┘
             ▼
┌───────────────────────────────────────────────┐
│ decide                                        │
│  family = highest percentile                  │
│  name it only if percentile >= p*             │   p* = 1 - alpha / F
│  else "man-made, type unconfirmed"            │   (F = number of man-made families)
└───────────────────────────────────────────────┘
```

### Key ideas, in plain words

- **Compare with the surroundings, not with the whole library.** The pixel minus its ring
  mean is mostly "what the object adds", so a 20 % roof shows up as a roof-shaped difference.
  This is the replacement idea behind sub-pixel target detection.
- **Whitening.** A difference along a direction the scene varies in all the time (shade,
  vegetation vigour) is common, so it should count for little. Multiplying by the inverse
  square root of the scene covariance does that. This plus a cosine score is the Adaptive
  Coherence Estimator (ACE), the standard sub-pixel known-target detector.
- **Score a family, not a spectrum.** One "roofing" spectrum never matches a real roof
  exactly. Simplex ACE (Ziemann & Theiler 2017) lets any non-negative mix of the family's
  spectra explain the pixel, solved by non-negative least squares. It is never used for
  natural families: dozens of minerals and soils can mix into almost anything.
- **Cap the library.** Each family keeps at most 40 spectrally spread entries. Uncapped,
  ~1,900 minerals out-vote the ~180 man-made spectra by sheer number of chances.
- **Calibrate against the scene.** Raw scores are not comparable between families or scenes.
  Each family score becomes a percentile among the scene's own ordinary pixels.

### What alpha and p\* mean

alpha is **the share of ordinary background pixels you accept being given a man-made name.**
The code turns it into a percentile bar:

```
p* = 1 - alpha / F          F = number of man-made families (10 with the shipped library)
```

An ordinary pixel passes one family's bar with probability 1 − p\*. With F families it gets
some man-made name with probability at most F × (1 − p\*) = alpha (the union bound). So the
bar adjusts itself when families are added, for example field spectra of a known target.

The default is **alpha = 5 %**, which gives p\* = 0.995. Measured effect of the choice (each
cell: share of targets answered / share of answers that were right / share of empty pixels
given a man-made name):

| alpha | p\* (F = 10) | Planted targets | Gudalur sheets, library only | Gudalur, library + field spectra | Oil spill |
|---|---|---|---|---|---|
| 2 % | 0.998 | 69 / 53 / 0.3 | 28 / 27 / 1.0 | 38 / 82 / 1.1 | 7 / 27 / 0.4 |
| **5 %** | **0.995** | **73 / 52 / 1.2** | **35 / 27 / 2.1** | **~47 / 78 / ~2.5** | **11 / 26 / 1.1** |
| 10 % | 0.99 | 77 / 50 / 3.1 | 45 / 28 / 3.7 | ~55 / 75 / ~4.5 | 15 / 25 / 1.8 |
| 20 % | 0.98 | 81 / 49 / 5.4 | 57 / 28 / 6.5 | ~62 / 70 / ~8 | 20 / 21 / 3.9 |

"~" means interpolated, because with field spectra F = 15. The bound is conservative:
measured false names run at a quarter to a half of alpha.

**The bound holds for ordinary pixels, not for false alarms.** False alarms are unusual by
definition, so they get man-made names more often (9–24 % on the PRISMA scenes, section 5).

## 3. What was tried

Everything was compared on the same data with the same scoring. In order of when it was
tried:

### Naming the material

| Technique | Idea | Result | Fate |
|---|---|---|---|
| Whole-pixel SAM (product today) | angle between pixel and each library spectrum | 6 % right at 30 % fill | replaced |
| Unmixing (FCLS / MESMA) with a ring background | pixel = ring mean + shade + one library spectrum | 22 %; 32 % with the 40-per-family cap | superseded |
| Ring as a subspace | background = best 3–5 typical ring spectra | 44 % | superseded |
| ACE with library spectra | whitened cosine to each spectrum minus the ring mean | 51 %; detection AUC 0.94 | baseline |
| ACE + absorption-feature check | accept a name only if the pixel shows that material's absorption features | planted: 62 % right, 0.6 % false names. **Real deployed sheets: 0 % right** | dropped |
| Feature-check hardening | 3σ noise threshold on feature depth; hydrocarbon (1.21 + 1.73 + 2.31 µm) and aromatic rules; talc / dolomite look-alikes | only the 3σ threshold helps (false names 2.4 → 0.1 % on Gudalur); the rest add nothing or hurt | moot once the check is dropped |
| Per-family calibration of ACE | percentile per family, then the highest | slightly worse (47 against 51 %) | not used alone |
| Finite-target matched filters (FTMF, EC-FTMF) | model the target replacing part of the background; closed-form fill estimate | detection ≈ ACE; many false names on the Gudalur scenes (34–49 %) | dropped |
| **Simplex ACE + calibration + p\*** | section 2 | best on every data set (section 4) | **adopted** |
| HyperSL deep "endmember-free" unmixing | pretrained network learns endmembers itself | worse than classical unmixing on exact-truth sets; smooths away sub-pixel targets; 5–13 min per scene on a 96 GB GPU | rejected |

### Estimating how much of the pixel is the material (fill)

The FTMF estimate is unbiased and at the theoretical limit on synthetic data, and beats
unmixing below 20 % fill on planted targets. On real deployed materials every method failed:
it collapsed towards 0 on the Gudalur sheets and underestimated oil by ~0.8. **Report fill only
when a spectrum of the actual material from the scene or the field is available.** Otherwise
report "man-made, fill unknown".

### Removing false alarms (related work, same experiments)

None of these is in this branch; listed so they are not re-tried blindly.

- **Kept as the best filter: "background atypicality" (k-nearest-neighbour distance to the
  scene's background), never dropping a pixel the feature check verified as man-made.** It
  removes 30.4 ± 0.8 % of false alarms at ≤ 5 % of target objects lost, against 16 % for just
  raising the anomaly threshold. On water its protect rule hurt; check before relying on it
  there.
- **Rejected:**
  - "natural explains it better than man-made" margins and physics indices: no information;
  - convex-hull test against the surroundings: worse than doing nothing;
  - matched-pair implant classifiers: far worse alone;
  - a learned fusion: +8 points on the 4 PRISMA scenes, but it failed on held-out oil and
    Gudalur scenes.

## 4. How it was tested

### Data

| Data set | What it is | Why |
|---|---|---|
| Planted targets | 1,600 real pixels from 4 PRISMA scenes with a man-made library spectrum blended in at 5–50 % fill. The planted spectrum and its near-copies are removed from the library | exact truth for family and fill; development set |
| PRISMA ground-truth scenes | the 4 PS-11 Stage II PRISMA scenes with hand-labelled anomaly masks (~395 objects); candidates = the anomaly model's top 2 % | the real use case |
| Gudalur sheets | AVIRIS-NG at 4 m over the Nilgiris (2018) with 5 deployed 16 × 16 m sheets (green, red, yellow, black nylon; white cotton) and their field spectra; simulated to 30 m (below) | real materials, exact fill, an Indian scene; held out until the end |
| Gulf of Mexico oil spill | 18 AVIRIS scenes (2010) with oil masks, aggregated to ~30 m | real man-made hydrocarbon on water; held out |
| DLR HySU | HySpex 0.7 m, 5 materials × 5 sizes on grass (VNIR only) | used in the HyperSL comparison |

**Simulating 30 m from airborne data** (Gudalur and oil):
- bands resampled onto the 165-band grid with the product's Gaussian resampler;
- a 30 m Gaussian point-spread function (Gudalur) or a box (oil);
- 64 grid offsets in 4 m steps, so each 16 m sheet lands at many different fills (up to ~0.2
  of a pixel);
- EnMAP noise at its stated signal-to-noise (354 VNIR, 260 SWIR).

The true fill of every coarse pixel comes from the same weights applied to the 4 m ROI.

### Rules followed

- **Thresholds and settings come from the development set or from theory**, never from the
  held-out sets.
- **Methods are compared at the same coverage** (the same share of pixels answered), or with
  coverage reported beside accuracy.
- **Scores that depend on random background samples are repeated over 10 seeds** (mean ±
  sd). Per-scene differences under ~8 points count as noise.
- **Every number was reproduced on a fresh machine** before it was used as a baseline.

### Unit tests (`tests/test_spectral_match/test_family_match.py`, 7 tests, all pass)

- p\* arithmetic, including no man-made families.
- The calibration percentile.
- Tag parsing; untagged entries are excluded with a warning.
- The 40-per-family cap and the band / entry selection.
- Simplex ACE equals ACE for a one-spectrum family and never scores below its members.
- **End to end, with the shipped family library on the 165-band grid:**
  - ordinary pixels of a synthetic natural scene get a man-made name at most alpha;
  - a roofing target **held out of the library** is still named roofing.

Run them with `python -m pytest tests/test_spectral_match -q` (about 16 s).

## 5. Results

### Against the current approach (ACE + feature check)

Each cell: share of targets answered / share of answers right / share of empty pixels given a
man-made name.

| Data set | ACE + feature check | Simplex ACE, alpha = 5 % |
|---|---|---|
| Planted targets | 26 / 57 / 0.6 | 73 / 52 / 1.2 |
| Gudalur sheets, library only | 47 / **0** / 2.0 | 35 / **27** / 2.1 |
| Gudalur sheets, library + field spectra | 46 / **0** / 2.0 | ~47 / **~78** / ~2.5 |
| Oil spill | 8 / 18 / 0.2 | 11 / 26 / 1.1 |

- **Planted targets.** Simplex answers far more pixels at about the same accuracy. In
  absolute terms it names about 2.5× more targets correctly. When both answer the same 26 %,
  Simplex is right 75 % of the time against 57 %.
- **Real deployed sheets.** The current approach names none of them right, even when the
  sheets' own field spectra are in the library.

### On the 4 PRISMA ground-truth scenes, exactly as the product would run it

The branch's `match_families` was run with the family library resampled to each scene's own
PRISMA bands, on the anomaly model's top 2 %. That is ~18,000 candidates per scene, ~45 s per
scene on CPU. Script and per-scene JSON: Drive
`unmixing_next/code/prisma_family_check.py`, `runs/branch_prisma_check/`.

| Scene | Candidates | Target pixels | Targets named man-made | Target objects with a man-made name | False alarms named man-made | Targets named natural |
|---|---|---|---|---|---|---|
| Set 1 | 17,979 | 331 | **47 %** | **51 / 99** | 11 % | 31 % |
| Set 2 | 19,317 | 252 | **38 %** | **29 / 85** | 9 % | 25 % |
| Set 3 | 18,617 | 136 | 15 % | 13 / 59 | 15 % | 34 % |
| Set 4 | 16,853 | 489 | 14 % | 32 / 152 | 24 % | 38 % |

Families named on target pixels:

| Scene | Man-made families on targets | Natural families on targets |
|---|---|---|
| Set 1 | metal 52, plastic 36, roofing 34, textile 28 | mineral, rock, soil |
| Set 2 | building 29, oil 25, metal 17, plastic 12 | rock, mineral, soil |
| Set 3 | paint 6, building 5, metal 4 | mineral, rock |
| Set 4 | plastic 32, metal 12, roofing 11, paint 10 | mineral, rock, soil |

Names on false alarms are most often metal, textile, paint and plastic.

**Reading it:**
- **Sets 1 and 2: the names carry information.** Real targets are named man-made about 4×
  as often as false alarms, and half of set 1's objects get a man-made name.
- **Sets 3 and 4: they don't.** Their targets are no more library-like than their false
  alarms. An earlier test found the same split: library-like signal on real targets in 2 of
  the 4 scenes.
- **This is a naming step, not a false-alarm filter.** False alarms outnumber targets about
  50 to 1, so most man-made names in a scene still land on false alarms (set 4: ~3,900
  false-alarm pixels against ~70 target pixels). Show the result as "likely material: X
  (confidence)" on pixels the analyst already kept, never as proof that a pixel is man-made.

## 6. Interfaces

```python
# app/spectral_match/family_match.py
DEFAULT_ALPHA = 0.05

def ensure_family_cache(vendable, sensor_id: str, cache_dir: str = "/splib07_cache",
                        bundle_dir: str = "/srv/family_library") -> list[LibraryEntry]
def p_star(alpha: float, n_manmade_families: int) -> float
def match_families(cube, validity, anomaly_mask, rows, cols, entries,
                   alpha: float = DEFAULT_ALPHA, seed: int = 0) -> FamilyMatch
#   cube (B, H, W) surface reflectance 0-1 on the vendable's bands; validity (B, H, W) 1/0;
#   anomaly_mask (H, W) bool; rows, cols: the pixels to name.

@dataclass(frozen=True)
class FamilyMatch:            # all (P,) unless noted
    family: np.ndarray        # best family, e.g. "roofing"
    group: np.ndarray         # "manmade" | "natural"
    confidence: np.ndarray    # calibrated percentile of that family, 0..1
    best_entry: np.ndarray    # closest library entry inside the family (a hint, not the answer)
    answered: np.ndarray      # confidence >= p_star
    p_star: float
    n_manmade_families: int
    bands_used: np.ndarray    # (B,) bool

# app/spectral_match/family_library.py
FAMILY_CAP = 40
def tag_library(entries) -> TaggedLibrary            # tag = entry.chapter "<group>.<family>"
def balanced_subset(lib, cap=FAMILY_CAP) -> np.ndarray
def common_band_mask(pixel_valid, lib, min_pixel_share=0.99, min_entry_share=0.95)
```

**Data flow.**
- `data/family_library` has the same layout as `data/splib07_slim`. The worker image copies
  it to `/srv/family_library`.
- `ensure_family_cache` calls the existing `library.build_cache_for_vendable`, which resamples
  every entry to the scene's native bands and FWHM and writes
  `/splib07_cache/splib07_<hash>.npz/.json`. The hash includes the bundle version
  `family_v1`, so this cache never collides with the splib07 one.
- `match_families` reads nothing else from disk.

**Invariants.**
- Library and pixels must be on the same reflectance scale (0–1).
- Every family-library entry's chapter is `<group>.<family>`. An entry without one is excluded
  with a warning, so a new untagged spectrum cannot change results silently.
- Simplex scoring is applied to man-made families only.
- `rows, cols` must index the same grid as `cube` and `anomaly_mask`.

**Failure modes.**
- `ensure_family_cache` raises `RuntimeError` if the bundle is missing or the vendable lacks
  FWHM.
- `match_families` raises `ValueError` if fewer than 2 families survive the band checks.
- A warning is logged if the median pixel value is not 0–1 reflectance; the result is then
  meaningless.

**Decisions.**
- **A separate family bundle, not changing `splib07_slim`.** Changing it would change the
  current matcher's cache and results in the same review.
- **Tags carried in the chapter field.** That lets them pass through the existing cache
  unchanged, with no new table.
- **p\* derived from alpha by the union bound,** not tuned on any data.
- **The absorption-feature check dropped:** 0 % right on real deployed sheets.

## 7. Integrating into the pipeline

In `backend/allotrope/action_types/_spectral_library_match_run.py`, after step 4 ("Pick
pixels"). Imports stay inside `run()` (the lazy-import rule):

1. `entries = ensure_family_cache(vendable, ctx.sensor_type,
   cache_dir=settings.splib07_cache_dir)`.
2. `fam = match_families(cube, validity, anomaly_mask.astype(bool), rows, cols, entries,
   alpha=cfg["family_alpha"])`. Pass the **unsmoothed** cube: the experiments did not smooth,
   and the Savitzky–Golay step is the SAM path's.
3. Add columns to `matches.parquet`:
   - `family`, `family_group`, `family_confidence`, `family_answered`, `family_best_entry`.

   Add `alpha`, `p_star` and `n_manmade_families` to `summary.json`.
4. Add `family_alpha: float = 0.05` (and an on/off switch, if wanted) to
   `SpectralLibraryMatchConfig` in `spectral_library_match.py`. `META` drives the UI picker,
   so no frontend code changes there.
5. In the viewer, show "likely material: <family> (confidence)" when `family_answered`.
   Otherwise show "man-made, type unconfirmed" or the natural family.
6. Keep SAM's outputs as they are until the family result has been checked on a few product
   scenes.
7. Verify that the worker starts: `docker compose ps -a` must show `running`.

## Gotchas

1. **Radiometric scale (check first).** SAM ignores brightness; this method does not.
   - Confirm that the onboarding vendable's `normalized_hyperspectral_cube` is surface
     reflectance 0–1 for each sensor before wiring.
   - A PRISMA band-filtered vendable sampled during this work had medians of 0.001–0.09,
     which does not look like reflectance.
   - The PRISMA check above used reflectance read directly from the HE5 file.
2. **The product library has 3 usable man-made spectra.** `scripts/curate_splib07.py` skips
   Beckman spectra on the claim that their 200–3000 nm range does not overlap our sensors. It
   does. Most USGS man-made spectra are Beckman measurements. The family library includes
   them; the current SAM library still lacks them.
3. **Per-pixel band validity.** Whitening needs one band set. `common_band_mask` keeps bands
   valid for ≥ 99 % of the scored pixels and logs what it drops. SAM handles per-pixel
   validity, so the two can use different bands.
4. **The calibration sample needs no ground truth.** It excludes the anomaly mask grown by 3
   px. In the experiments some samples also excluded known targets, so real-scene false-name
   rates may run slightly higher than in section 2.
5. **ECOSTRESS man-made spectra are coarse in the SWIR** (~20 nm). Fine hydrocarbon features
   at 1.73 / 2.31 µm are blunted there; USGS resolves them.
6. **The product's Gaussian resampler silently drops (NaN) a 10 nm target band with fewer than
   3 source samples.** Fine for 1 nm library spectra. It dropped 42 of 165 bands when
   resampling ~10 nm AVIRIS imagery, so check any image resampling of ~10 nm sensors.
7. **Runtime.** About 45 s per PRISMA scene for ~18,000 candidates, CPU only. Most of it is
   the per-pixel non-negative least squares, about 10 man-made families per pixel.
8. **Small evidence base on real targets.** Gudalur has 5 sheets; the PRISMA scenes have ~395
   objects in 4 scenes, with mixed results. Treat the gains as directional until more
   labelled 30 m targets are available.

## References

- Ziemann, A., Theiler, J., 2017, Simplex ACE: a constrained subspace detector. *Optical
  Engineering* 56(8).
- Kraut, S., Scharf, L.L., 1999 (ACE); Manolakis, D., Shaw, G., 2002, Detection algorithms for
  hyperspectral imaging applications. *IEEE Signal Processing Magazine*.
- Ziemann, A., Kucer, M., Theiler, J., 2018, A machine learning approach to hyperspectral
  detection of solid targets. *Proc. SPIE* 10644 (finite-target matched filters).
- Liu et al., 2025, Endmember-free hyperspectral unmixing. *IEEE JSTARS* 18 (HyperSL).
- Library citations: `data/family_library/sources.md`.
