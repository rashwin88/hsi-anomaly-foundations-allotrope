# E003 - Cross-attention stem (variant C), SpectralEarth pretraining

**Date:** 2026-10-14 (launched 16:58 IST, training 17:26 - ~20:00 IST)
**Status:** superseded - stopped after epoch 3 by decision, replaced by E004 (reduced-attention stem)

## Question

Pretrained on SpectralEarth with the shared setup, how well does the cross-attention spectral
stem reconstruct held-out sites, and where does its loss flatten?

This run alone decides nothing; E001 (A) and E002 (B) run next under the identical rule, and
the A/B/C decision rule in `docs/proposals/spectral-attention-bottleneck.md` is applied once
all three exist and are scored on the labelled PRISMA scenes.

**Expected outcome, written before launch:** validation SAM falls below ~5 deg within 20-40
epochs (today's Indradhanu reached 2-5 deg by epoch 100 on PRISMA+EnMAP, config notes of
`hyperspectral_segformer_exp_2.json`); early stop between epoch 40 and 100. Throughput on one
L40S ~15-25 patches/s (~20-35 min per 30,000-patch epoch); A10G roughly half that, L4 a quarter. Slower than that means a data
or kernel bottleneck worth fixing before E001/E002.

## Setup

| | |
|---|---|
| variant | C - `stem: cross_attention`, per-band compressor and decoder, no binning |
| git commit | `c002970` (branch `feat/indradhanu-se-pretrain`), bundle `s3://allotrope-training-data-prod-india/deploy/allotrope-c002970.tar.gz` |
| config | [config.json](config.json) |
| data | se-v1 train shards 000000-000199 (30,000 patches), test 000000-000009 (1,500); 30,000 train + 1,500 val patches per epoch (200 shards so the 250 GB disk of g6e.xlarge fits) |
| normalisation | `patches/spectralearth/se-v1/stats.json` (whole train split, 512,849 patches) |
| model | Indradhanu SegFormer (embed 32/64/160/256, heads 2/2/5/8, 2 blocks per stage, decoder 256), D = 32 |
| optimisation | batch 4 x 8 accumulation = 32 (4 so a 24 GB A10G/L4 fits), lr 5e-4 cosine to 1e-6 over 200 epochs, bf16 autocast, mask 0.65, L1 + 0.5 SAM (ramp 10 epochs) |
| stopping | early stop: <1% relative val improvement for 10 epochs, not before epoch 20, ceiling 200 |
| instance | first available of g6e.xlarge / g6e.2xlarge / g6e.4xlarge (L40S), g5.2xlarge (A10G), g6.2xlarge (L4); Prod, on-demand, ap-south-1a/b, AMI `ami-0eb5a4f0d81f671e3` (DLAMI PyTorch 2.7). No g6e capacity anywhere in the region on 2026-10-14 afternoon; spot placement score 1/10 for every GPU type |
| boot | [../bootstrap.sh](../bootstrap.sh) with `RUN=E003-cross-attention` |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E003-cross-attention/` (checkpoints/last.pt, best.pt, metrics.csv, train.log, bootstrap.log) |
| wall time | - |
| cost | - ($1.17-3.61/h depending on the instance) |

What differs from E001/E002: `model_config.stem` only.

## Results

| metric | value | vs baseline |
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

- **Epochs 1-2** (the only full rows): val L1 0.0550 -> 0.0462, val SAM 17.28 -> 13.49 deg; train
  12.1-12.4 patches/s over the whole epoch, 41 min per epoch, peak 11.25 GB at batch 4.
- **Validation-scoring flaw found here:** epochs were ranked by L1 + lambda(t) x SAM with lambda
  ramping 0 -> 0.5; epoch 1 (lambda 0) scored 0.055 and later epochs ~0.1 on that scale, so early
  stop would have fired at epoch 20 regardless of learning. Fixed in the trainer: validation now
  scored at the full SAM weight; early-stop state carries its scale and older state is ignored.
- **Too slow to be worth finishing:** the speed probe on this instance measured the per-band stem
  at 13.5 patches/s against 80.0 for the reduced-attention stem. Stopped after epoch 3; the
  instance was kept (never released). Checkpoints remain in S3 under this run's prefix.
- First boot crashed: the DLAMI's PyTorch activate script read an unset LD_LIBRARY_PATH under
  `set -u`; fixed in bootstrap.sh.

## Next
