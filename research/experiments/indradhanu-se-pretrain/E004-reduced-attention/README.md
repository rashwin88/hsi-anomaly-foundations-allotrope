# E004 - Reduced-attention stem (variant C'), SpectralEarth pretraining

**Date:** 2026-10-14 -
**Status:** running (queued first on the kept instance)

## Question

Pretrained on SpectralEarth with the shared setup, how well does the reduced-attention stem
reconstruct held-out sites, and where does its loss flatten? It replaces the per-band
cross-attention stem of E003 as variant C in the A/B/C comparison.

**Expected outcome, written before launch:** ~80 patches/s per training step (measured), so
~6-8 min per 30,000-patch epoch including validation; validation SAM below ~5 deg within 20-40
epochs; early stop between epoch 40 and 100. Reconstruction close to E003's trajectory over its
first epochs (val L1 0.055 -> 0.046, SAM 17.3 -> 13.5 deg over epochs 1-2): the per-band skip
path and decoder keep band-level detail, attention runs over 40 nm bins.

## Setup

| | |
|---|---|
| variant | C' - `stem: reduced_attention`, `bin_nm: 40`: per-band kernel inside 42 bins (attention tokens), per-band skip path to the 32 channels, per-band wavelength-conditioned decoder (`app/foundation_models/components/spectral_reduced.py`) |
| why not E003's stem | per-band cross-attention measured at 13.5 patches/s, 2.8 GiB per patch on the L40S; reduced attention 80.0 patches/s, 0.68 GiB (probe 2026-10-14, see ../S01-laptop-smoke/ and the course part 40) |
| config | [config.json](config.json) - shared with E001/E002 except `stem` / `bin_nm`: batch 8 x 4 = 32, 6 workers, validation scored at full SAM weight |
| data | se-v1 train shards 000000-000199 (30,000 patches), test 000000-000009 (1,500), already on the instance's NVMe |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S 48 GB), on-demand, kept - stop/terminate protection on |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E004-reduced-attention/` |
| wall time | - |
| cost | - ($2.69/h) |

## Results

| metric | value | vs E001 (A) |
|---|---|---|
| recon L1 (se-v1 test, hidden pixels) | | |
| recon SAM, degrees | | |
| PRISMA ROC-AUC, mean of 4 scenes | | |
| PRISMA TPR at 1% FPR, mean | | |
| throughput, patches/s | | |
| peak GPU memory, GB | | |
| stopped at epoch | | |

## Graphs

## Observations

**Epoch-5 checkpoint on the PRISMA bench (2026-10-14, `scripts/evaluate_prisma_bench.py`).**
Combined ROC-AUC set1-4 / mean: native bands 0.263 / 0.781 / 0.725 / 0.741 / 0.628, se202
0.442 / 0.720 / 0.699 / 0.602 / 0.616; v0.2.0 in the same run 0.712 / 0.686 / 0.730 / 0.521 / 0.662.
Already ahead on set2 and set4; inference ~48 s per scene vs ~5 s for v0.2.0 (laptop GPU).

**Set1 is below chance because E004 cannot reconstruct water.** 75% of set1's valid pixels are
water (NIR 860 nm < 0.03); 599 of its 638 anomaly pixels are on land. Mean L1 on water is 0.047
for E004 vs 0.0046 for v0.2.0 - 10x worse, worst in the NIR (0.105 error on reflectance below
0.03, i.e. it paints a land-like red edge over water). So 93% of E004's top-1% pixels are water
and the real anomalies rank below them. Not PRISMA as such: sets 2-4 are PRISMA too. Likely
cause: SpectralEarth is land-dominated, so water is out of distribution after pretraining.
Expected to be fixed by fine-tuning on the EnMAP native shards (which contain water, labelled in
`label_classes`); worth re-checking at a later pretraining epoch first.

**Epoch-10 checkpoint, same bench.** Native 0.217 / 0.824 / 0.728 / 0.703 / mean 0.618; se202
0.518 / 0.775 / 0.692 / 0.631 / mean 0.654 (v0.2.0 0.662). Five more epochs of SpectralEarth did not
fix water: native set1 fell further (0.263 -> 0.217, 1 of 153 objects caught). se202 improved on
every scene but set3 and is now within 0.01 of v0.2.0 on the mean; set2 native (0.824) is the best
single-scene score of any model so far.

**Epoch-18 checkpoint (last.pt), same bench.** se202 0.725 / 0.790 / 0.756 / 0.732 / **mean 0.751**
- ahead of v0.2.0 (0.662) on all four scenes, set1 included. Native 0.205 / 0.808 / 0.768 / 0.536 /
mean 0.579. So E004 handles set1's water once PRISMA is resampled to SpectralEarth's wavelengths,
and fails on PRISMA's native list: the water failure is a band-list effect, not missing water in
pretraining (SpectralEarth is 6-11% water by pixel, measured 2026-10-14). Prime suspect: the skip
path's 1/N mean over bands (spectral_reduced.py line 96) weights each region by its band count -
PRISMA native is 16% visible / 71% SWIR vs SpectralEarth's 26% / 50%.

**Band-list sensitivity, measured (epoch 18, inference-only ablations, laptop).** Mean per-pixel
L1, water (NDWI > 0, NIR < 0.05) / land:

| condition | water | land |
|---|---|---|
| SE test patches, all 202 bands | 0.0155 | 0.0157 |
| same pixels, thinned to PRISMA spacing (151 of the 202) | 0.0309 | 0.0226 |
| thinned + skip path reweighted to each bin's training share | 0.0203 | 0.0205 |
| set1 native (185 bands) | 0.0462 | 0.0282 |
| set1 native + skip reweighting | 0.0352 | 0.0200 |
| set1 native minus its 20 bands at wavelengths SE never had | 0.0327 | 0.0221 |
| set1 native minus those + skip reweighting | 0.0345 | 0.0190 |
| set1 resampled to SE's 202 wavelengths | 0.0138 | 0.0160 |

Thinning SpectralEarth's own bands - no new sensor, no new wavelength - doubles the water error, so
the sensitivity is to the band list itself. The skip path's 1/N mean explains about two thirds of
that (water) and a third (land); unseen wavelengths matter on set1 but overlap with it. Neither
alone nor both close the gap: the wavelength networks k, W, U were only ever evaluated at one band
list in pretraining, so "any band list" was architecturally possible but never trained.

## Next
