# Implanted-target anomaly benchmark - bench-v1

**Why:** the 4 labelled PRISMA scenes (909 objects) cannot separate Indradhanu versions under residual Mahalanobis
(FT02b README), and no public labelled anomaly benchmark exists at satellite reflectance
(research/experiments/benchmarks/README.md). User, 2026-10-17: implanted targets "somewhat similar to the ones we do
have", with "subtle, medium, easy" categories. Design: docs/lld/implant-bench.md.

## Build (2026-10-17)

```bash
python scripts/build_implant_bench.py real  C:/Users/FITT/Documents/stage1_datasets C:/es/implant/real.json
python scripts/build_implant_bench.py build C:/es/implant/shards C:/es/implant/real.json C:/es/implant/bench-v1 --seed 0
```

Hosts: 6 held-out test scenes per sensor (one datatake / date / line each), reassembled from the native-v1z /
native-v1 test tiles; AVIRIS-NG lines cut to 1152 rows. 11 GB on the dev laptop at `C:/es/implant/bench-v1/`
(manifest.json holds every file's sha256). Not in S3 yet.

| sensor | hosts | objects | subtle / medium / easy |
|---|---|---|---|
| EnMAP (224 bands) | 6 | 1,179 | 385 / 412 / 382 |
| PRISMA (234 bands) | 6 | 1,318 | 429 / 463 / 426 |
| AVIRIS-NG (372 bands, 3-8 m) | 6 | 839 | 273 / 295 / 271 |
| **total** | 18 | **3,336** | 1,087 / 1,170 / 1,079 |

## Does it look like the real bench?

| | real (909) | implanted (3,336) |
|---|---|---|
| size px, 10th / 50th / 90th percentile | 1 / 2 / 10 | 1 / 2 / 9 |
| single-pixel share | 42% | 41% |
| contrast z, subtle | 0.13 / 0.31 / 0.46 | 0.16 / 0.32 / 0.46 |
| contrast z, medium | 0.53 / 0.69 / 0.87 | 0.54 / 0.69 / 0.87 |
| contrast z, easy | 1.00 / 1.42 / 2.91 | 0.99 / 1.43 / 2.86 |
| ring NDVI, 10th / 50th / 90th | -0.02 / 0.37 / 0.60 | 0.26 / 0.49 / 0.64 |

- 97% of objects land in their targeted tier; 2% of material implants could not reach their target with full fill.
- Material implants are mostly sub-pixel: fill fraction median 0.05, 90th percentile 0.33.
- Library chapters evenly drawn: artificial 347, minerals 336, soils 324, organics 323, vegetation 319.
- **Known bias:** backgrounds are greener than the real bench's (no host has set4's bright bare ground); the NDVI
  weighting cannot add ground the hosts lack. A v2 could add arid hosts.

## Real bench, by tier (the same z tiers, top-1% objects caught, residual Mahalanobis)

| model | subtle | medium | easy |
|---|---|---|---|
| F001 ep109 | 40% | 37% | 55% |
| FT02b ep9 | 40% | 40% | 56% |
| FT02c ep10 | 40% | 39% | 57% |

Medium is no easier than subtle on real objects: local contrast is not the whole story - how unusual an object is
across the scene matters too.

## Results (2026-10-17, `scripts/evaluate_implant_bench.py`)

Objects caught in the top 1% (top 0.1%) of scores, 1-pixel tolerance; pAUC <= 1% FPR per tier, mean over hosts.
**Chance** (random scores, same rules): 10% (1%) in every tier.

**Residual Mahalanobis (default)**

| model | all | subtle | medium | easy | easy pAUC | medium pAUC | EnMAP | PRISMA | AVIRIS-NG |
|---|---|---|---|---|---|---|---|---|---|
| v0.2.0 (shipped) | 19% (8%) | 9% (1%) | 14% (4%) | 34% (18%) | 0.257 | 0.091 | 16% | 15% | 30% |
| F001 ep109 | 25% (10%) | 11% (1%) | 21% (5%) | 45% (22%) | 0.342 | 0.112 | 15% | 30% | 32% |
| FT02 ep8 | 26% (10%) | 10% (1%) | 20% (5%) | 49% (24%) | 0.356 | 0.115 | 15% | 32% | 32% |
| FT02b ep9 | 27% (10%) | 10% (1%) | 22% (5%) | 50% (25%) | 0.367 | 0.121 | 15% | 34% | 33% |
| FT02c ep10 | 27% (10%) | 10% (1%) | 21% (5%) | 50% (25%) | **0.371** | **0.124** | 15% | 33% | 33% |
| FT03 ep10 | 27% (10%) | 10% (1%) | 21% (4%) | 50% (25%) | 0.371 | 0.121 | 15% | 33% | 33% |

**Combined**: 10% (v0.2.0) -> 12% (F001) -> 15-16% (FT02 / FT02b / FT02c); easy 19% -> 30%.

## Observations

- **The benchmark separates the models, in the order of their training.** Easy-tier pAUC rises at every stage:
  0.257 -> 0.342 -> 0.356 -> 0.367 -> 0.371; medium 0.091 -> 0.124. The largest step is pretraining (v0.2.0 -> F001,
  easy caught 34% -> 45%, PRISMA hosts 15% -> 30%); fine-tuning adds 4-5 points on easy and 3-4 on PRISMA. The 4-scene
  real bench showed none of this.
- **Subtle is at chance for every model** (10% / 1%). The tier is calibrated on the real objects' z, yet real subtle
  objects are caught 40% of the time at the top 1%: real objects at a given local contrast are easier than implants at
  the same z. Likely reasons: real objects are rare materials scene-wide, and the human labels favour what is visible.
  So the tiers order difficulty but are harder in absolute terms than the real bench; compare models within a tier.
- **EnMAP hosts are flat at 15-16% for every model** while PRISMA doubles. Not yet explained - candidates: the EnMAP
  hosts' greener, more uniform backgrounds; their pixel-mask defects (2-4 missing bands per pixel, filled); or that
  pretraining on SpectralEarth (EnMAP) already gave everything there was.
- Signature and material implants score alike (25% vs 29% for FT02c), so transplanted real signatures are not what
  makes the bench easy or hard.

## Stage-3 evaluation through app/evaluation (2026-10-17, `configs/eval/stage3.json`, output C:/es/eval/stage3)

Reproduced every earlier number (PRISMA bench pAUC / ROC-AUC / caught: v0.2.0 0.159 / 0.854 / 402, F001 0.148 /
0.861 / 400, FT03 0.158 / 0.853 / 413; implant v0.2.0 19.0% / 7.7%). New:

| implant bench | caught top 1% | top 0.1% | pAUC <= 1% | easy pAUC | medium pAUC |
|---|---|---|---|---|---|
| FT03 + residual Mahalanobis | 26.9% | 9.9% | 0.172 | 0.371 | 0.121 |
| FT03 + RM, water / cloud excluded | 27.2% | 10.3% | 0.174 | 0.378 | 0.118 |
| **PCA-RX on the cube (no model)** | **30.2%** | **10.5%** | **0.209** | **0.423** | **0.168** |

| PRISMA real bench | caught top 1% | top 0.1% | pAUC <= 1% | TPR @ 0.1% |
|---|---|---|---|---|
| FT03 + residual Mahalanobis | 45.4% | 14.3% | 0.158 | 0.058 |
| FT03 + RM, water / cloud excluded | 46.1% | 16.5% | 0.165 | 0.065 |
| PCA-RX on the cube (no model) | 39.7% | 8.1% | 0.128 | 0.029 |

**bench-v1 is biased toward cube-level spectral oddity.** A model-free PCA-RX beats every model on the implants and
loses clearly on the real objects (half the detection rate at 0.1%). Implants are linear blends of a foreign spectrum
into otherwise untouched pixels - hard edges, no texture, no blur - which suits a cube detector; real objects are
spatially structured. bench-v1 still orders the models consistently (v0.2.0 < F001 < FT02c ~ FT03) but must not be a
training target. v2: spatially coherent implants (PSF-blurred footprint, within-object texture), calibrated on a
second statistic, accepted only if it reproduces the real bench's ordering (models above PCA-RX).

## bench-v2 (2026-10-17) - accepted

Built with `build_implant_bench.py build ... --version 2 --psf-sigma 1.4 --seed 0` from `real_v2.json` (real objects
re-measured with foreignness and halo); C:/es/implant/bench-v2, 18 hosts, 2,913 objects (EnMAP 942, PRISMA 1,197,
AVIRIS-NG 774); sources signature / material / host in thirds.

**Acceptance 1 - looks like the real objects** (2 PRISMA calibration hosts, medians real / v2): foreignness 0.0071 /
0.0081, halo 0.643 / 0.642, z 0.69 / 0.74; every tier within 0.90-1.15x (limit 25%). Fixes needed on the way: measure
on the evaluator's bands; NaN-safe background subspace; solve foreignness on the realised object (a spot's own
pixel-vs-ring difference is already foreign 0.009-0.045, so matching the implanted signal alone overshot 7x).

**Acceptance 2 - models beat PCA-RX, as on the real bench** (`configs/eval/bench_v2.json`, C:/es/eval/v2):

| residual Mahalanobis | caught top 1% (0.1%) | pAUC <= 1% | TPR @ 0.1% | easy | medium | subtle |
|---|---|---|---|---|---|---|
| v0.2.0 | 22.1% (8.6%) | 0.118 | 0.067 | 38.3% | 15.2% | 10.9% |
| F001 ep109 | 29.2% (10.2%) | 0.145 | 0.075 | 48.5% | 23.1% | 13.1% |
| FT02c ep10 | 32.8% (11.4%) | 0.166 | 0.086 | 52.9% | 27.6% | 14.7% |
| FT03 ep10 | 32.3% (11.4%) | 0.164 | 0.086 | 52.7% | 26.9% | 14.2% |
| PCA-RX (no model) | 30.2% (9.1%) | 0.154 | 0.072 | 47.7% | 24.9% | 15.2% |

Met: fine-tuned models above PCA-RX on every headline measure (subtle ~tie). Pretraining alone (F001) does not beat
PCA-RX here; fine-tuning on the real sensors does. By host sensor: PRISMA 16.5% (v0.2.0) -> 35.2% (F001) -> 40.2%
(FT02c); EnMAP flat ~16% for every model and PCA-RX (unexplained, as in v1). Water / cloud exclusion: +1-3 points,
most on EnMAP hosts (FT03 15.7% -> 18.7%).

**v2 replaces v1 as the implant benchmark** (v1 kept reproducible by `--version 1 --seed 0`).

**Corrections (2026-10-18):**
- **v0.2.0's implant-bench numbers (v1 19.0%, v2 22.1%) are a reading artefact - disregard them.** On the real bench,
  v0.2.0 through the generic linear resample that implant hosts get scores 15.8% caught / pAUC 0.094 / ROC-AUC 0.752,
  against 44.2% / 0.159 / 0.854 through the product's band_filter_apply (C:/es/eval/resample_check/report.md). An
  earlier note here that the two readings were identical was a misread. Fix: read implant hosts for fixed-grid models
  through the product path. Native-band models (F001, FT0x) are unaffected.
- **The final bench-v2 misses acceptance 1 marginally in two tiers.** On all 6 PRISMA hosts foreignness is 1.10x /
  1.28x / 1.27x real (subtle / medium / easy; 1.25x overall) against a 1.25x limit; halo 0.93-1.07x. The 2-host
  calibration (within 0.90-1.15x) drew different random choices. Acceptance 2 (models above PCA-RX) and the model
  ordering stand. Candidate fix: tighten solve_realised's tolerance from 1.5x to 1.2x and rebuild.

