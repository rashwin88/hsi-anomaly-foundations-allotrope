# F002-heavy - Indradhanu Heavy: 4x-wide pretraining on SpectralEarth

**Status:** running since 2026-10-28 14:05 UTC (first start 12:48 crashed, see below) on `i-06645b351d4dc9712` (p5.4xlarge, 1x H100 80 GB HBM3, 16 vCPU,
249 GB RAM, 3.5 TB NVMe), H100 capacity block `cr-08d2ff733d27a977c` (2026-10-28 11:30 -> 10-29 11:30 UTC); block 2
`cr-0dccba00b34dc29e1` (10-31 11:30 -> 11-02 11:30). Code bundle 7947bef. Launched on the user's "Go, but 4x wide";
named by the user ("This is indradhanu-heavy"); 80 epochs on the user's "we will lose steam after epoch 50-75".

## Question
Does capacity move detection? FT04 cannot render detail it can see (research/experiments/decoder-capacity/): each
4 x 4 token is squeezed to 32 numbers at stage 1 and the decoder had no full-resolution path. Every change since FT04
that kept the encoder (FT06-FT09) stayed within checkpoint noise. A wider encoder needs re-pretraining.

## Setup
| | |
|---|---|
| model | encoder embed [96, 192, 480, 768] (FT04: [32, 64, 160, 256]), heads [3, 6, 10, 12], spectral latent 64 (32), token_dim 32 (the reduced stem's wavelength encoding is 32 wide), progressive decoder dim 512, stages 256 / 128. **33.9 M parameters** (encoder 29.6 M, decoder 4.3 M) vs FT04 5.9 M |
| data | SpectralEarth se-v1, F001's normalisation stats (not recomputed); train shards rotated in blocks of 200 every 5 epochs (16 blocks, 3,200 of 3,419 shards); 30,000 patches per epoch |
| schedule | 80 epochs, lr 3e-4 cosine (fully decayed at 80), batch 32 x 1 (effective 32, F001's; was 16 x 2 at the first start), bf16, 12 loader workers, 16 test shards |
| driver | [heavy_run.sh](heavy_run.sh) (re-runnable: resumes from metrics.csv, BLOCK markers and S3 last.pt) |
| config | [config.json](config.json) |

## GPU usage (recorded on the user's request)
Sampled every 30 s by `nvidia-smi` into `gpu_log.csv` (mirrored to the run's S3 folder every 5 minutes, with `cpu_top.txt`).

| time (UTC) | GPU util | mem util | memory used | power | temp |
|---|---|---|---|---|---|
| 12:48:27 (start-up) | 0% | 0% | 9.1 GB | 114 W | 31 C |
| 12:48:57 | 23% | 17% | 15.9 GB | 158 W | 33 C |
| 12:50:27 | 70% | 26% | 15.9 GB | 370 W | 46 C |
| 12:51:27 | 85% | 49% | 15.9 GB | 377 W | 48 C |
| 12:51:57 | 75% | 45% | 15.9 GB | 381 W | 50 C |

Steady state so far: **63-85% utilisation, ~375 W of 700 W, 16 of 80 GB**. CPU: 12 loader workers at ~7% each, main
process 60% of one core, 86% of the 16 vCPU idle - the GPU, not the loaders, sets the pace; the headroom is batch size.

## Incidents
- **12:48 start crashed at the first validation, 3 times, then the driver gave up (13:10); GPU idle 13:10-14:05.**
  12 loader workers, 10 staged test shards: two workers got no shard and webdataset raised "No samples found ...
  fewer shards than workers". Fix: 16 test shards (`test_shards_per_size` 16). Restarted 14:05 with batch 32 x 1.
  My epoch watcher only looked for an epoch-1 row, so the failure went unseen for ~55 min; the watcher now alerts
  on any trainer exit or the driver stopping. With the late launch (12:38 for an 11:30 block), ~2.5 h of block 1
  were lost.

## GPU after the restart (batch 32 x 1)
14:05:57 85% / 26.5 GB / 391 W; 14:06:27 100% / 26.5 GB / 394 W; 14:12:57 100% / 26.5 GB / 404 W - against 63-85% at
16 x 2. Epoch 1: **369 s, 81.2 patches/s**, peak 18.4 GB (allocator); val L1 0.0285, SAM 8.84 deg (F001 epoch 1:
0.0298, 9.22 deg). 80 epochs ~8.6 h -> ~22:45 UTC.

## What follows in block 1
`research/experiments/indradhanu-finetune/h100_chain.sh` (running since 14:14): stages the fine-tune data now; after
the 80 epochs, FT04 pseudo-labels for PRISMA / AVIRIS-NG, then **FTH1** (Heavy fine-tuned in one joint run with FT04's
final recipe, 10 epochs) and **FTB1** (the control: F001 at FT04 size, same recipe).

## Results
_pending_
