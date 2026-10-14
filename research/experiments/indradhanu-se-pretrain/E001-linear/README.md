# E001 - Linear stem (variant A), SpectralEarth pretraining

**Date:** not started
**Status:** planned - starts automatically on the E003 instance when E003 stops (`../run_next.sh`)

## Question

The baseline every other variant is judged against. How well does a fixed linear stem reconstruct held-out SpectralEarth sites, and where does its loss flatten?

**Expected outcome, written before launch:** Several times faster than C per epoch (the laptop probe: 391 vs 5 patches/s); here likely limited by data loading (3 workers, batch 4) rather than the GPU. Validation SAM a little worse than C's at matched epochs if per-pixel adaptivity matters; early stop between epoch 40 and 100.

## Setup

Identical to [E003](../E003-cross-attention/README.md) - same instance, shards, normalisation,
model, optimisation and stopping rule - except:

| | |
|---|---|
| variant | A - `stem: linear`: today's fixed 1x1 compressor (`Conv2d(202, 32, 1)` + BN) and decompressor |
| config | [config.json](config.json) |
| instance | the E003 instance `i-0a3afdd4efbcafe92` (g6e.2xlarge, kept; never released) |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E001-linear/` |
| wall time | - |
| cost | - ($2.69/h) |

## Results

| metric | value | vs E003 |
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
