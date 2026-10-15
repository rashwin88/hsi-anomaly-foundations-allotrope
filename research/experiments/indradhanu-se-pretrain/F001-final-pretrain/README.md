# F001 - Final SpectralEarth pretraining of the next Indradhanu

**Date:** 2026-10-15 -
**Status:** running (started 09:39 UTC on the kept instance)

## Question

Pretrained for 200 epochs on 300,000 different SpectralEarth patches, how far do E005's
configuration's reconstruction and PRISMA detection go beyond what 20 epochs on 30,000 patches gave?
This is the checkpoint EnMAP/PRISMA fine-tuning (with the segmentation head) starts from.

**Expected outcome, written before launch:** validation keeps falling past E005's epoch 20
(0.0214 / 5.16 deg), with a small bump at each rotation that recovers within a few epochs; PRISMA
mean ROC-AUC on PRISMA's own bands above E005's 0.752 by the end. Roughly 30 hours, ~$82.

## Setup

| | |
|---|---|
| configuration | E005's exactly: `reduced_attention`, `bin_nm` 40, `band_weighting: width`, `band_augment_prob: 0.8`, `band_jitter_nm: 3.0`; batch 8 x 4, bf16 - [config.json](config.json) |
| schedule | one cosine schedule over 200 epochs (5 warm-up), not restarted at rotations |
| data | se-v1 train, rotated every 20 epochs through 10 blocks of 200 shards: block k = shards k*200 .. k*200+199 (30,000 patches); 2,000 shards, ~300,000 patches, 58% of the split. Validation: test shards 000000-000009 (1,500 patches) throughout, as E001-E005 |
| early stopping | off (fresh data every 20 epochs makes a plateau before a rotation likely); best.pt still tracks the lowest validation score |
| driver | [rotate_f001.sh](rotate_f001.sh) - prefetches the next block while the current trains, stops after epoch 20k (checkpoint already in S3), swaps the folder, resumes via auto_resume; restart-safe |
| code | commit `d89e0a1` (bundle `deploy/allotrope-d89e0a1.tar.gz`) |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S 48 GB), kept |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/F001-final-pretrain/` (checkpoints/, metrics.csv, train.log, rotation.log) |
| wall time | - |
| cost | - ($2.69/h) |

## Results

PRISMA is scored on its own bands only: the one standard inference for wavelength stems from
2026-10-15 (`scripts/evaluate_prisma_bench.py` defaults to it). Scenes set1-4.

| epoch | val L1 | val SAM | PRISMA combined ROC-AUC, own bands: set1 / set2 / set3 / set4 / mean |
|---|---|---|---|
| 7 | 0.0241 (ep 6) | 6.40 (ep 6) | 0.474 / 0.736 / 0.742 / 0.706 / **0.664** |
| 14 | 0.0219 (ep 13) | 5.53 (ep 13) | 0.430 / 0.765 / 0.720 / 0.724 / **0.660** |
| 20 | **0.0202** | **4.90** | 0.702 / 0.764 / 0.729 / 0.740 / **0.734** (first F001 checkpoint above v0.2.0's 0.662; rotation to block 1 at 12:41 UTC, clean) |
| 38 | **0.0192** | **4.47** | 0.763 / 0.762 / 0.688 / 0.695 / **0.727**; with the SAM dark-pixel fade 0.904 / 0.763 / 0.693 / 0.695 / **0.764** |
| 60 | | | |
| 100 | | | |
| 200 | | | |

## Graphs

## Observations

## Next
