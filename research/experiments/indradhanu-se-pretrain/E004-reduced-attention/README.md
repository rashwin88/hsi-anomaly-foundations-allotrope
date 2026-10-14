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

## Next
