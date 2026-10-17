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
# app/utils/evaluation/implant.py
TIER_EDGES = (0.502, 0.934);  TIERS = ("subtle", "medium", "easy");  tier_of(z) -> str
ring_mask(obj, valid, exclude=None) -> (H, W) bool         # 4-neighbour dilation, steps 2-3 (a diamond)
contrast_z(cube, band_valid, obj, ring) -> float           # mean |o - r| / mean sigma over usable bands
grow_blob(rng, size) -> (size, 2) offsets                  # compact 4-connected blob
implant_signature(cube, obj, d, a)                         # x' = max(x + a d, 0)
implant_material(cube, obj, t, f)                          # x' = (1 - f) x + f t
solve_strength(o0, r, sigma, step, z, s_max=None) -> (s, z) # realised z: o0 + s step; None if the spot is
                                                           # already above z (choose another)
```

```bash
python scripts/build_implant_bench.py real  <stage1_datasets> <work>/real.json
python scripts/build_implant_bench.py build <test shards> <work>/real.json <bench> [--seed 0]
python scripts/evaluate_implant_bench.py <bench> NAME=ckpt [NAME=ckpt ...]
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
  each object pixel); shadows; water hosts (no real object sits on water).
