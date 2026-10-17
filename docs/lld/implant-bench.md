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
# bench-v2 additions (in progress, 2026-10-17)
background_subspace(cube, valid, k=10, seed=0) -> (k, B)   # top-k principal directions of up to 100k valid pixels
foreign_fraction(d, P) -> float                            # share of |d|^2 outside span(P); NaN bands ignored
match_foreignness(d, P, target) -> d'                      # out-of-subspace part rescaled: foreign_fraction = target
psf_weights(obj, sigma) -> (H, W)                          # footprint blurred by a Gaussian PSF, mean 1 on obj
```

```bash
python scripts/build_implant_bench.py real  <stage1_datasets> <work>/real.json
python scripts/build_implant_bench.py build <test shards> <work>/real.json <bench> [--seed 0]
    [--version 2 --psf-sigma <px> --hosts "sensor=prefix,prefix;..."]     # v2, in progress
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
  each object pixel); shadows; water hosts (no real object sits on water - but 39 of the 909 real objects, 27 of
  them in set1, lie wholly inside FT03's water / cloud mask: boats, shoreline objects or FT03 errors, not checked;
  docs/lld/evaluation.md, reference run).

## bench-v2 (in progress, 2026-10-17)

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

**Plan** (`place_and_implant(..., version=2, psf_sigma=...)` in `scripts/build_implant_bench.py`, being written):

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

No v2 bench has been built and no results exist yet. Unit tests cover the new functions
(`tests/test_evaluation/test_implant.py`: foreignness by hand, matching to 0 / 0.1 / 0.5, PSF weights).
