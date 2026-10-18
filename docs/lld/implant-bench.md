# Implanted-target anomaly benchmark

**Purpose** - a labelled anomaly benchmark large enough to tell Indradhanu versions apart, on the satellite sensors
the product serves, at real reflectance. No public one exists: the standard HAD benchmarks (ABU, HAD100) are airborne
radiance (research/experiments/benchmarks/README.md), and a search on 2026-10-17 found no labelled PRISMA / EnMAP /
Gaofen-5 anomaly set (only a methane-plume set, which labels gas, not surface objects).
The 4 labelled PRISMA scenes (909 objects) cannot separate models under residual Mahalanobis: every measure ties
within scene-to-scene noise (research/experiments/indradhanu-finetune/FT02b-mixed/README.md).

So: real held-out EnMAP / PRISMA / AVIRIS-NG test scenes, with synthetic objects made to look like the real bench
objects - their size, their background, their contrast - in three difficulty tiers.

## Interfaces

```python
# app/evaluation/implant.py
TIER_EDGES = (0.502, 0.934);  TIERS = ("subtle", "medium", "easy");  tier_of(z) -> str
ring_mask(obj, valid, exclude=None) -> (H, W) bool         # 4-neighbour dilation, steps 2-3 (a diamond)
contrast_z(cube, band_valid, obj, ring) -> float           # mean |o - r| / mean sigma over usable bands
grow_blob(rng, size) -> (size, 2) offsets                  # compact 4-connected blob
implant_signature(cube, obj, d, a)                         # x' = max(x + a d, 0)
implant_material(cube, obj, t, f)                          # x' = (1 - f) x + f t
solve_strength(o0, r, sigma, step, z, s_max=None) -> (s, z) # realised z: o0 + s step; None if the spot is
                                                           # already above z (choose another)
# bench-v2 additions (accepted 2026-10-17)
background_subspace(cube, valid, k=10, seed=0) -> (k, B)   # top-k principal directions of up to 100k valid pixels;
                                                           # gaps mean-imputed; covariance eigh (558ff5e, ~0.02 s/tile)
foreign_fraction(d, P) -> float                            # share of |d|^2 outside span(P); NaN bands ignored
match_foreignness(d, P, target) -> d'                      # out-of-subspace part rescaled: foreign_fraction = target
psf_weights(obj, sigma) -> (H, W)                          # footprint blurred by a Gaussian PSF, mean 1 on obj
solve_realised(o0, r, sigma, d0, P, z, f, shrink, tol=1.5) -> (s, d) | None   # z and foreignness solved on the
                                                           # REALISED object (natural offset + s shrink d)
evaluation_bands(wl, finite, valid) -> (B,) bool           # the bands the evaluator reads (product exclusions out,
                                                           # >= 60% finite over valid pixels)
implant_tile(rng, refl, wl, real, library, n, psf_sigma=1.4, sources=...) -> (H, W) ids   # training-time (FT05):
                                                           # v2 implants into one tile, placed anywhere valid
interp_spectrum(wl, src_wl, values) -> (len(wl),)          # np.interp with src_wl sorted first (b94dbfd): the
                                                           # PRISMA bench lists 2477 -> 453 nm (see Invariant 5)
```

```bash
python scripts/build_implant_bench.py real  <stage1_datasets> <work>/real.json
python scripts/build_implant_bench.py build <test shards> <work>/real.json <bench> [--seed 0]
    [--version 2 --psf-sigma <px> --foreign-tol <x> --hosts "sensor=prefix,prefix;..."]
    # v2:   --version 2 --psf-sigma 1.4 --seed 0                    (foreign-tol default 1.5)
    # v2.1: --version 2 --psf-sigma 1.4 --foreign-tol 1.2 --seed 0  (after b94dbfd)
python scripts/evaluate.py configs/eval/<name>.json     # dataset {"kind": "implant"}; docs/lld/evaluation.md
```

## Data flow

```
real:  4 PRISMA bench scenes (native) -> per GT object: size, z, ring NDVI, signature d = o - r -> real.json
build: test shards (EnMAP / PRISMA native-v1z, AVIRIS-NG native-v1)
         -> reassemble 6 scenes per sensor from 128 px tiles (meta x0 / y0); AVIRIS-NG lines cut to 1152 rows
         -> placement: >= 90% bands valid, not water, not cloud (EnMAP: provider labels; others: index rules),
            eroded 4 px; weighted to the real NDVI mix; real density (objects per valid pixel)
         -> per object: tier round-robin, size from the real sizes (cap 20), compact blob, ~15 px apart
            (the object's box grown by 4, then blocked 10 further);
            source alternates: signature (a real object's d, interpolated to the host bands, none farther than
            15 nm from a bench band) or material (USGS splib07 slim, resampled to the host bands; chapter drawn
            uniformly, then a spectrum); strength solved for a z drawn from the tier's real values; realised z
            measured after implanting and recorded (the tier reported is the realised one)
         -> <bench>/<sensor>/<scene>/{cube.npy int16, wavelengths.npy, gt.npy int32 ids, objects.json}, manifest.json
evaluate: product exclusions; per-band nearest-valid fill; each model's band reading; residual Mahalanobis and
         combined; objects caught top 1% / 0.1% by tier, source, sensor; pAUC <= 1% per tier
```

## Invariants

1. **Contrast is measured the same way on real and implanted objects** (`contrast_z`, `ring_mask`), and the tier
   edges are the real objects' z terciles. Changing either breaks the "like the real ones" calibration.
2. **One seeded generator** draws everything, in a fixed host order: the same seed and inputs give the same bench.
3. **Hosts are held out**: test-split scenes only, which were never trained on (split by datatake / site / line).
4. **Objects never touch**: each blocks its box grown by 4 + 10 px (about 15 px between objects), so a ring
   (steps 2-3) never contains another object.
5. **Never pass a band list to `np.interp` unsorted.** NumPy does not check, and "if the sequence `xp` is
   non-increasing, interpolation results are meaningless" (its docstring). The PRISMA bench's evaluated bands run
   2477 -> 453 nm; a PRISMA host's `wavelengths.npy` is two descending runs (977 -> 407 nm, then 2497 -> 943 nm).
   Unsorted, every query above the last listed wavelength returns the last listed value. Use `interp_spectrum`
   (or sort as `app/evaluation/datasets.resample` and the product's `spectral_resampler.py` do);
   `tests/test_evaluation/test_implant.py::test_descending_band_lists` pins it.

## Failure modes

| What | Seen as |
|---|---|
| a scene id prefix matches no tile | `SystemExit` naming it |
| tiles of one scene carry different band lists (> 0.5 nm) | `SystemExit` |
| no free spot after 200 tries | the object is skipped (fewer objects than the density target) |
| a material too close to its background to reach the target z with f <= 1 | f capped at 1; realised z (and tier) lower |

## Decisions

- **Implanted, not real, objects** (2026-10-17, user: "brilliant ... make them somewhat similar to the ones we do
  have"): thousands of objects at exact ground truth on our own sensors, reportable by tier, source and sensor.
  Accepted weakness: no shadows, adjacency effects or object texture - the real PRISMA scenes stay as the check.
- **Tiers by contrast z** (user: "subtle, medium, easy"), cut at the real objects' terciles 0.502 / 0.934.
- **Two spectrum sources reported separately**: transplanted real signatures (closest to what we have) and library
  materials (beyond the bench's materials).
- **Unlabelled real anomalies in the hosts** count as false alarms, equally for every model; not removed.
- **Not done**: sub-pixel objects smaller than one pixel (fill fraction already covers sub-pixel abundance within
  each object pixel); shadows; water hosts. Real objects DO sit on water: 39 of the 909 lie wholly inside FT03's
  water / cloud mask (checked 2026-10-17): set1 27 single pixels on water-dark NIR (median 0.040), 13 of them > 3 px
  offshore - likely vessels; set2 / set3 8 on shorelines; set4 4 inside the 11% of that scene FT03 calls cloud.
  That 11% was first recorded as "bright desert - a mask error"; corrected 2026-10-18: a true-colour view shows a
  bright cloud / smoke streak and a large hazy patch that the mask follows, and the pixels are bluish (NIR / blue
  0.54-0.65 for 75%), so the 4 are objects under haze (research/experiments/indradhanu-finetune/README.md, log). The first real-object scan used VIS < 0.04 as "water", which misses turbid coastal
  water (VIS ~0.08).

## bench-v2 (accepted 2026-10-17)

**Why.** In the stage-3 evaluation (docs/lld/evaluation.md) a model-free PCA-RX on the cube beat every model on
bench-v1 (objects caught top 1% 30.2% vs FT03's 26.9%) and lost clearly on the real PRISMA bench (pAUC 0.128 vs
0.158; TPR at 0.1% FPR 0.029 vs 0.058). bench-v1 rewards cube-level spectral oddity that real objects do not have,
so it cannot be a training target - nor the sole judge of a model.

**Diagnosis** (`C:/es/implant/v2/diag.py`, `diag.json`, dev laptop; PRISMA only: the 909 real objects against the
1,318 bench-v1 implants on the 6 PRISMA hosts). For each object, with d = object mean - ring mean:

| median | real | bench-v1 | real, easy | bench-v1, easy |
|---|---|---|---|---|
| foreign fraction: share of \|d\|^2 outside the scene's top-10 principal directions | 0.006 | 0.042 | 0.002 | 0.029 |
| halo / object contrast: z of the 1-px halo over z of the object | 0.74 | 0.60 | 0.70 | 0.38 |
| cube PCA-RX rank of the object's best pixel | 0.954 | 0.983 | 0.981 | 0.998 |

Real objects differ from their surroundings almost wholly *inside* the scene's own main spectral directions (they
are unusual mixtures of the scene's materials), and their contrast bleeds into the neighbouring pixels (the sensor's
point-spread function and adjacency). bench-v1 implants carry a difference that points outside those directions -
exactly what a cube RX detector rewards - into an untouched hard-edged footprint. `diag.py` measures z against a
ring at steps 3-4 (not the bench's 2-3) and tiers by that z, so its per-tier counts (real subtle / medium / easy
246 / 275 / 388) are not the bench's thirds.

**Plan, as built** (`place_and_implant(..., version=2, psf_sigma=...)` in `scripts/build_implant_bench.py`):

1. **z and foreignness as a pair from one real object** of the tier (`measure_real` now also records each real
   object's `foreign` and `halo`; `real.json` must be rebuilt).
2. **Difference rescaled to that foreignness**: `match_foreignness(d0, P, f_target)` keeps the in-subspace part
   d_in and scales the rest, d' = d_in + beta d_out with beta^2 |d_out|^2 / (|d_in|^2 + beta^2 |d_out|^2) = target.
   P = `background_subspace(host)`.
3. **A third source, `host`**: d0 = a land pixel of the same host at least 100 px (Manhattan) away, minus the ring
   mean. Sources rotate signature / material / host in runs of three; material becomes additive (d0 = t - r).
4. **PSF-blurred footprint**: the window gets `s x psf_weights(obj, sigma) x d'` (mean weight 1 on the object, the
   blur's tail on its neighbours), clipped at 0; the margin around an object grows from 4 to 7 px. The strength is
   solved with step `(1 - mean weight on the ring) x d'`, since the ring receives part of the tail too; the realised
   z, foreignness and halo ratio are measured afterwards and recorded (`foreign_target`, `foreign`, `halo` in
   objects.json; `version`, `psf_sigma` in manifest.json). In v2 the subspace, the strength solve and these
   measurements use the bands the evaluator reads (product exclusions removed, bands >= 60% valid over valid
   pixels - `ImplantBench`'s rule), not every host band.
5. **Acceptance, written before the build**: foreignness and halo medians within 25% of the real objects' overall
   and in every tier; and models beat PCA-RX on v2, as they do on the real bench. Detection-supervised fine-tuning
   waits on this (docs/proposals/fine-tuning-and-segmentation.md).

Unit tests cover the new functions (`tests/test_evaluation/test_implant.py`: foreignness by hand, matching to
0 / 0.1 / 0.5, PSF weights).

**Three fixes found by the calibration builds** (research/experiments/implant-bench/README.md, "Acceptance 1"):
measure on the evaluator's bands (`evaluation_bands`; the absorption / edge bands swamped foreignness and z);
`background_subspace` mean-imputes gaps (a plain mean made any band with one gap all-NaN - 130-182 of 234 PRISMA
host bands); foreignness solved on the realised object (`solve_realised`: a spot's own pixel-vs-ring offset is
already foreign 0.009-0.045, so matching the implanted signal alone overshot ~7x). Final sigma 1.4.

**Result.** `C:/es/implant/bench-v2` (dev laptop): 18 hosts, 2,913 objects (EnMAP 942, PRISMA 1,197, AVIRIS-NG
774), sources signature 981 / material 986 / host 946. Real objects re-measured into `real_v2.json` (medians on
the evaluator's bands and the bench ring: foreign 0.0071, halo 0.643, z 0.687 - not comparable with `diag.py`'s
0.006 / 0.74 above, which used every band and a ring at steps 3-4).

1. *Looks like the real objects* - README, calibration build on 2 PRISMA hosts (`C:/es/implant/cal_v2`, since
   deleted): foreign 0.0071 / 0.0081, halo 0.643 / 0.642, z 0.69 / 0.74, every tier within 0.90-1.15x. **Caveat
   (2026-10-18):** recomputed from the accepted bench's own `objects.json` on the same two hosts, foreignness is
   1.19 / 1.40 / 1.27x real (subtle / medium / easy; overall 0.0090, 1.27x) - outside the 1.25 limit in two
   tiers - while halo is 0.95 / 0.94 / 1.09x. Different draw (the final build placed EnMAP hosts first) or a code
   change between the builds; not investigated.
2. *Models beat PCA-RX, as on the real bench* - `configs/eval/bench_v2.json`, `C:/es/eval/v2`: caught top 1%
   v0.2.0 22.1%, F001 29.2%, FT02c 32.8%, FT03 32.3%, PCA-RX 30.2%; pAUC 0.118 / 0.145 / 0.166 / 0.164 / 0.154.
   Met for the fine-tuned models; F001 alone is below PCA-RX. (v0.2.0's figures are void - reading bug, below;
   every `signature` implant was garbled.)

v2 replaces v1 (v1 reproducible with `--version 1 --seed 0`); v2.1 replaces v2 (below). Open at the time: EnMAP
hosts ~16% for every model and PCA-RX; v0.2.0 ties FT03 on the real bench but trails on v2. The second was
resolved on 2026-10-18 (Corrections, below).

## Corrections (2026-10-18)

**Bug: descending band lists fed to `np.interp`** (fixed in b94dbfd; Invariant 5). Two places:

- `app/evaluation/datasets.resample`, the generic resample fixed-grid models (v0.2.0) get on implant hosts. On a
  PRISMA host every grid band became a copy of the 977 nm or the 943 nm band (the first and last listed). On the
  real bench read the same way (`C:/es/eval/resample_check`), v0.2.0 scored 15.8% caught top 1% / pAUC 0.094 /
  ROC-AUC 0.752 against 44.2% / 0.159 / 0.854 through band_filter_apply; after the fix the generic path gives
  exactly the product-path numbers (`C:/es/eval/resample_check_fixed`). An earlier note that the two readings were
  identical was a misread. **v0.2.0's bench-v1 (19.0%) and bench-v2 (22.1%) numbers are void.** Native-band
  models were never resampled and are unaffected. EnMAP and AVIRIS-NG hosts list ascending and were read
  correctly.
- the `signature` source, in the builder's `place_and_implant` and in `implant_tile` (`np.interp(wl, rw, d)` ->
  `interp_spectrum`): the real objects' `d` is
  sampled at the bench's descending 185 bands, so every host band above 453 nm received `d(453 nm)` - a flat step.
  Garbled in bench-v1 (half the implants), bench-v2 (a third) and FT05's training (half). Contrast and
  foreignness were still matched, on the flat step.
- Also fixed: the library FWHM on descending host lists (`np.gradient` was negative, so the 4-15 nm clip always
  gave 4 nm) - now `np.abs(np.gradient(wl))`, in the builder and in `finetune_trainer._library`. Material spectra
  on PRISMA hosts had been resampled with a 4 nm width rather than the band spacing; their effect was not measured
  separately.

**What changes.** The "pretraining is the big step" reading (v0.2.0 19-22% -> F001 25-29%) was the bug. bench-v2's
marginal acceptance-1 miss (all 6 PRISMA hosts: foreignness 1.10x / 1.28x / 1.27x real against a 1.25x limit)
motivated the tighter `--foreign-tol` of v2.1.

## bench-v2.1 (2026-10-18) - the implant benchmark now

Built `--version 2 --psf-sigma 1.4 --foreign-tol 1.2 --seed 0` after b94dbfd: `C:/es/implant/bench-v2.1` (dev
laptop), 18 hosts, 2,817 objects (signature 985, material 927, host 905). Per-tier foreignness vs real (PRISMA
hosts) 1.05x / 1.15x / **1.60x** (subtle / medium / easy): the tighter tolerance skips low-foreignness easy
targets, so the survivors skew high - acceptance 1's 25% limit is missed in the easy tier by a selection effect,
recorded, not fixed. Halo 0.92-1.06x.

`configs/eval/bench_v2_1.json` (`C:/es/eval/v2.1`), residual Mahalanobis, caught top 1% (0.1%) / pAUC <= 1%:
v0.2.0 35.5% (12.6%) / 0.168; F001 33.1% (11.7%) / 0.157; FT02c 35.1% (12.8%) / 0.169; FT03 34.9% (12.7%) / 0.168;
FT05 ep5 35.4% (12.8%) / 0.170; FT04 35.1% (12.7%) / 0.169 (`C:/es/eval/ft04`); PCA-RX 33.2% (9.8%) / 0.154.
Cloud exclusion adds ~1 point to every model (v0.2.0 35.5 -> 36.7, FT03 34.9 -> 36.0, FT04 own masks 35.1 ->
36.2). EnMAP hosts ~21% for every model and PCA-RX (unexplained).

Acceptance 2 holds at the top 0.1% (every model above PCA-RX: 11.7-12.8% vs 9.8%; at the top 1% F001 ties it).
**No new model measurably beats the shipped v0.2.0** - as on the real bench (pAUC 0.159 vs FT03 0.158).
The new models' measurable wins are masks (cloud exclusion), native bands and reconstruction.

**Scope gap.** Vessels on water are in scope (user, 2026-10-17), but `place_and_implant` still removes water and
cloud from the eligible area, so v2 and v2.1 hold no water objects and cannot measure a water exclusion's cost.
The training-time `implant_tile` (FT05) does place on water.
