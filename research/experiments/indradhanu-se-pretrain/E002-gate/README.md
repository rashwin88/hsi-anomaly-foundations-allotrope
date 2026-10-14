# E002 - Gate stem (variant B), SpectralEarth pretraining

**Date:** not started
**Status:** planned - starts automatically on the E003 instance when E003 stops (`../run_next.sh`)

## Question

Does per-pixel adaptivity alone - without wavelength tokens or exact band masking - move reconstruction and detection relative to A?

**Expected outcome, written before launch:** Throughput between A and C, close to A. If B lands within seed noise of C on detection, B is preferred on cost (decision rule).

## Setup

Identical to [E003](../E003-cross-attention/README.md) - same instance, shards, normalisation,
model, optimisation and stopping rule - except:

| | |
|---|---|
| variant | B - `stem: gate`: per-pixel gate `sigmoid(MLP(x))` (hidden 32) rescaling each band, then A's compressor |
| config | [config.json](config.json) |
| instance | the E003 instance `i-0a3afdd4efbcafe92` (g6e.2xlarge, kept; never released) |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E002-gate/` |
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
