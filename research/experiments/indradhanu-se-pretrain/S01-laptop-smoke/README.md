# S01 - Laptop smoke and GPU cost probe

**Date:** 2026-10-14
**Status:** done
**Kind:** wiring check, not an experiment - no result here says anything about which stem is better.

## Question

1. Does the whole training path run for all three stems: shards -> hot storage -> masking ->
   valid-band loss -> two-pass validation -> metrics.csv -> checkpoint -> reload?
2. At Indradhanu's real encoder size, what does each stem cost per training step in memory and
   time on one GPU - the question E000 was going to answer on AWS?

**Expected outcome, written before running:** all three run; cross-attention much slower
(course part 36 put its stem at 133x the arithmetic of the linear stem), memory-bound.

## Setup

| | |
|---|---|
| git commit | `c649dc0` + uncommitted step-1..3 work (sharder, spectral stem, trainer) |
| configs | [config_linear.json](config_linear.json), [config_gate.json](config_gate.json), [config_cross_attention.json](config_cross_attention.json) |
| data | the 20-patch smoke shards (one train, one test), stats.json from those 20 patches |
| smoke model | tiny encoder (embed 8/16/32/64, 1 block per stage), 2 epochs x 8 train / 4 val patches, batch 2, CPU |
| probe model | real Indradhanu encoder (embed 32/64/160/256, heads 2/2/5/8, 2 blocks per stage), random input, [gpu_probe.py](gpu_probe.py) |
| machine | laptop: NVIDIA RTX 4000 SFF Ada 20 GB, 24 CPUs, 63 GiB RAM; torch 2.5.1+cu124 |
| cost | none (local) |

## Results

### 1. Smoke (CPU) - all three run end to end

| stem | exit | epoch seconds | patches/s | val L1 (epoch 2) | val SAM deg (epoch 2) |
|---|---|---|---|---|---|
| linear | 0 | 0.7 | 12.2 | 0.095 | 14.5 |
| gate | 0 | 0.7 | 12.0 | 0.094 | 14.6 |
| cross_attention | 0 | 11.8 | 0.68 | 0.091 | 14.0 |

Checkpoints saved and reloaded into a freshly built model; the wavelength-stats curve is inside
the checkpoint. Logs: `log_<stem>.txt`; metrics: `metrics_<stem>.csv`.

### 2. GPU probe - one training step, real encoder, 128 x 128 patches

Peak memory (GiB) and throughput (patches/s):

| stem | precision | batch 4 | batch 8 | batch 16 | batch 32 | batch 64 |
|---|---|---|---|---|---|---|
| linear | bf16 | 0.4 GiB, 132 | 0.8, 248 | 1.4, 330 | 2.7, 371 | 5.3, **391** |
| gate | bf16 | 0.5, 114 | 1.0, 192 | 1.8, 221 | 3.6, 268 | 7.0, **282** |
| cross_attention | fp32 | 20.2, 2.7 | 32.1*, 0.9 | 56*, 0.3 | OOM | |
| cross_attention | bf16 | 15.2, **5.0** | 22.3*, 5.1 | 36*, 0.8 | OOM | |

\* above the card's 20 GB: Windows spills to shared system memory, which is why throughput
collapses. Batch 1 and 2 in bf16: 5.9 and 11.7 GiB, 5.6 and 5.4 patches/s.

### 3. Where C's cost sits - wavelength-bin probe (batch 4, bf16)

Prototype in [bin_probe.py](bin_probe.py): the compressor reads one token per fixed 40 nm
wavelength bin (42 tokens), each the masked mean of its bands' tokens - computed from bin means
because the value embedding is linear, so no (pixels, 202, 32) tensor is built.

| compressor | decoder | GiB per patch | patches/s |
|---|---|---|---|
| per band (C as built) | attention, per band | 3.81 | 4.8 |
| 40 nm bins | attention, per band | 3.14 | 9.2 |
| per band | linear 1x1 (not usable: fixed to 202) | 2.03 | 8.7 |
| 40 nm bins | linear 1x1 | 0.89 | 36.0 |

Bin width barely mattered (20 / 40 / 80 nm: 7.7 / 8.3 / 8.4 patches/s at batch 4). The cost is
split between the two halves: binning the compressor alone buys ~2x; the decoder's 202
per-band queries are the other half. **Decided 2026-10-14: no binning for now** - E003 runs C
as built, and the decoder is not changed. Binning stays a measured lever.

## Observations

- **The wiring works.** Every stage of the training path ran for all three stems, including
  checkpoint reload. Two bugs were found and fixed on the way:
  - `foundation_trainer._build_local_dataloader` handed webdataset bare Windows paths, which
    it reads as URLs with scheme `c`; Windows now gets `file:C:/...`, Linux is unchanged.
  - CUDA's fused attention cannot take more than 65,535 rows; with pixels as rows, 4 patches
    (65,536) failed with `invalid configuration argument`. Attention now runs in chunks of
    32,768 pixels, which is exact because pixels are independent.
- **Cross-attention costs ~3.7 GiB per patch in bf16** and tops out at **~5 patches/s** on this
  card - **~78x slower than the linear stem** (391) and 56x slower than the gate (282). The
  course's 133x arithmetic ratio was for the stem alone; end to end the shared SegFormer
  dilutes it, but not much.
- At 5 patches/s an epoch of today's 40,000 samples takes ~2.2 h here. The memory is
  activations kept for the backward pass - tokens, keys, values and their projections, each
  (pixels x 202 x 32) - not parameters.
- The gate costs 28% more time than linear, against the 3x of its stem arithmetic: the
  SegFormer dominates for both.
- Smoke losses are meaningless (8 patches, 2 epochs, lr 1e-3 to 1e-6); they only show the
  numbers are finite and in the expected range (L1 ~0.09, SAM ~14-21 deg at initialisation).

## Next

Decided 2026-10-14: run C as built (no binning, decoder unchanged) first, as E003 on one
g6e.2xlarge, stopping early once the losses flatten; then A (E001) and B (E002) under the same
rule. Binning (section 3) is the lever if C's cost becomes the limit.
