# Spectral attention bottleneck

**Status:** accepted design; built and running (2026-10-14, `docs/lld/spectral-stem.md`).
Decisions 1 and 3 were revised after measurement: C is now `reduced_attention` - attention over
40 nm bins with per-band skip and decoder paths - because per-band attention ran at 13.5
patches/s (E003, superseded) against 80 for the reduced stem. Validation is scored at full SAM
weight. Runs: E004 (C'), then E001 (A), E002 (B). Agreed 2026-10-14; supersedes the 2026-08-24
version of this file (recoverable from git). Delete this file once the work is built and its
decisions are written into the numbered docs and an LLD, per `docs/proposals/README.md`.
**Scope:** the spectral stem of the next Indradhanu, its SpectralEarth pretraining data, and
the experiment that decides whether it ships.
**Study material:** `indradhanu workings/31-*.md` to `39-*.md` teach every idea here from
first principles, with the arithmetic. This file states decisions; the course explains them.

---

## The problem

Today's `HyperspectralSegFormerMAE`
(`app/foundation_models/components/hyperspectral_seg_former_mae.py`) mixes bands exactly
twice, both times with a fixed 1x1 convolution:

```
(B, 165, H, W) -> PixelNormalize -> SpectralCompressor  Conv2d(165, D, 1) + BatchNorm2d
              -> SegFormerEncoder -> SegFormerDecoder   (attention over SPATIAL tokens only)
              -> SpectralDecompressor Conv2d(D, 165, 1) -> PixelDenormalize
```

One recipe for every pixel; inputs tied to 165 band *positions*; dead bands filled and mixed in
because a fixed linear map cannot skip an input. Course part 31 works through all three.

## Decisions

| # | decision | rejected alternative, and why |
|---|---|---|
| 1 | **Cross-attention compressor**: `D = 32` learned queries attend over a pixel's band tokens; output stays `(B, 32, H, W)` | self-attention over bands - quadratic, ~10 GiB of weights per patch (course part 36) |
| 2 | **Band tokens carry their wavelength** (sinusoidal, 16 periods from 10 to 10,000 nm, `d = 32`), not their index | index embeddings - re-tie the model to one band list |
| 3 | **Spectral cross-attention decoder**: each requested wavelength is a query against the pixel's 32 latent channels | keep `Conv2d(32, 165)` - cannot emit a band list other than 165 |
| 4 | **Pretrain on SpectralEarth's native 202 bands**; fine-tune and run on today's 165-band grid by passing the grid's wavelengths | pretrain on the grid - pays 60 ms/patch resampling and throws away SpectralEarth's resolution |
| 5 | **Normalise by wavelength**: one mean and std curve over wavelength, computed from the SpectralEarth train split; any sensor reads it off at its own wavelengths by linear interpolation | keep per-band-position z-score - no fixed positions exist any more; per-scene min-max - destroys absolute reflectance, which `spectral_library_match` needs |
| 6 | **Dead bands are masked with `-inf`** in the compressor and left out of the loss | fill-then-mix (today) - contaminates every channel |
| 7 | **Hide pixels, never bands** while scoring anomalies | band masking trains spectral self-completion - an anomaly can rebuild itself (course part 38, trap 1) |
| 8 | **Keep `BatchNorm2d` on the 32-channel compressor output** in every variant | drop it - would change the encoder's input distribution between variants and confound the comparison |
| 9 | **Spatial model unchanged**: SegFormer encoder, decoder, token masking (`mask_ratio 0.65`), as `configs/hyperspectral_segformer_exp_2.json` | co-designing it now - two changes at once, uninterpretable result |

## Shapes, one patch

```
pixels         (202, 128, 128) int16  -> x 0.0001 -> reflectance; -32768 -> band invalid
normalise      (202, 128, 128)        (r - mean(wl)) / std(wl)
as rows        (16384, 202)
band tokens    (16384, 202, 32)       Linear(1, 32)(value) + enc(wl)
compressor     (16384, 32)            32 queries, 4 heads, mask = band valid
image          (32, 128, 128)         BatchNorm2d(32)
---- unchanged SegFormer MAE ----     (32, 128, 128)
as rows        (16384, 32)
decoder        (16384, 202)           202 wavelength queries vs 32 channel tokens
denormalise    (202, 128, 128)        loss: L1 + lambda * SAM, raw reflectance,
                                      hidden pixels x valid bands only
```

Parameters (verified on the study prototype; re-verify on the real module): compressor
**5,345**, decoder **6,401**, against 5,376 + 5,445 for today's compressor and decompressor.

## The experiment that decides it

"More sophisticated" is not evidence. Three variants, same data, same spatial model, same
normalisation, same budget of samples seen:

| variant | spectral stem | input |
|---|---|---|
| **A** | today's `Conv2d(202, 32, 1)` + BN; `Conv2d(32, 202, 1)` out | 202 fixed positions |
| **B** | per-pixel gate `sigmoid(MLP(x))` (hidden 32) rescales each band, then A | 202 fixed positions |
| **C** | decisions 1-3 | 202 wavelength tokens |

A and B have fixed input positions, so on SpectralEarth they take the 202 bands as a fixed
list; only C can later move to another band list without retraining its input layer.

**Scores**, all reported per run:

- reconstruction on `se-v1` test sites: L1 and SAM (degrees), hidden pixels only;
- anomaly detection on the labelled PRISMA scenes (PS-11 stage-1 Set-1..4): ROC-AUC and TPR
  at 1% FPR of the SAM residual score. A and B read PRISMA resampled to SpectralEarth's 202
  wavelengths; C reads it twice - resampled, and on the 165-band grid - which tests
  decision 4 directly. Ground truth may sit ~1 px off; score with a 1 px tolerance.

**Decision rule, fixed before any run:** C proceeds to full pretraining only if it beats A on
detection in mean ROC-AUC across the four scenes. If B matches C within noise (seed spread of
A), B is preferred on cost. If neither beats A, the bottleneck is not the limit - stop and
rethink.

## Data: SpectralEarth shards

Reading single patches from S3 gives ~18 samples/s; training needs shards on local NVMe.

- **Format**: WebDataset tars, ~1 GB each, read by the existing `hot_storage` path
  (`app/abstract_classes/foundation_trainer.py`). Per sample: `pixels.npy` int16
  `(202, 128, 128)` and `wavelengths.npy`, keyed `se_<site>_<patch>`. No validity cube -
  derived from `-32768` on load. 150 patches per shard, fixed by a seed, so runs resume.
- **Shuffled across sites at write time**, so one scene's patches spread over many shards.
- **Split by `se-v1`** into `train/` and `test/` shard sets.
- **Output**: `s3://allotrope-training-data-prod-india/patches/spectralearth/se-v1/{train,test}/`
  plus `manifest.json` (shard -> keys, sha256, counts). Not registered in the inventory, which
  only tracks objects in its own bucket; the manifest is the record.
- **Normalisation curve** (decision 5): per-band count, sum and sum of squares accumulated
  over the train split while sharding; written beside the shards as `stats.json`.
- **Where**: the CPU prod box via SSM. No GPU.

## Compute and cost

- Comparison runs: `g6e.2xlarge` (1x L40S 48 GB, 8 vCPU) in Prod; quota allows one
  on-demand or four spot.
- Full pretrain: `p5.4xlarge` (1x H100); P quota is 0, a request for 16 vCPU is pending.
- **Memory**: the compressor's keys and values are 808 MiB per patch in fp32 (course
  part 36). Use bf16 and the fused `scaled_dot_product_attention` kernel, which does not keep
  the attention weights; expect a batch well below today's 128. A short throughput run (E000)
  sets the batch size before any comparison run. Fallback if it does not fit: band groups of
  4 (51 tokens).
- **Every GPU launch is approved individually** - instance, hours and cost stated first.

## Experiment tracking

Every run gets a folder under `research/experiments/indradhanu-se-pretrain/` (layout in that
folder's README): config, per-epoch `metrics.csv`, plots, and written observations.
Checkpoints go to S3, never to git.

## Build order

1. SpectralEarth sharder + normalisation stats.
2. Model components (wavelength encoding, compressor, decoder, gate, wavelength
   normalisation) with tests that reproduce the parameter counts.
3. Training wiring: a SpectralEarth data provider, band-valid masks through to the loss.
4. E000 throughput on one GPU, then E001-E003 (A, B, C).
5. Full pretrain on the winner; fine-tune on PRISMA/EnMAP shards.

## Risks

- **C may lose.** A harder-to-optimise stem can underperform a linear one; the decision rule
  above exists for that.
- **SpectralEarth has no cloud masks** (`docs/lld/spectralearth.md`). Cloudy patches train
  the model to treat cloud as normal. A cloud screen is a separate open item.
- **The PRISMA labelled data is small**: four scenes in total, one per set, and they exist
  only on the laptop (`stage1_datasets/Set-*`); they must be copied to S3 for the GPU box.
  Detection differences may sit inside seed noise; A is run with two seeds to measure it.
- **Resampling PRISMA to the 202 SpectralEarth wavelengths** must be checked against
  `BandFilterConfig`, whose default exclusion ranges remove some of them.
