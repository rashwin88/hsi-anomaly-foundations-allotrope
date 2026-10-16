# FT02b - FT02 continued with mixed-sensor batches

**Status:** launched 2026-10-16 on the user's go.

## Question
FT02 (one sensor per batch) improved reconstruction on all three sensors but see-sawed: some epochs PRISMA got
better while EnMAP and AVIRIS-NG got worse, then the reverse, and the PRISMA bench ended at 0.855 against a gate of
0.861. Does putting every sensor in every batch remove the see-saw and lift the bench?

Mixed batches change two things at once, deliberately: every gradient step balances the three sensors, and the stem's
BatchNorm sees mixed batches, so its stored statistics match what evaluation sees (FT02's swung with whichever sensor
ended the epoch - see FT02's README, "BatchNorm sensitivity").

**Expected, written before launch:** per-sensor validation moves together epoch to epoch (no epoch where one sensor
improves by > 0.3 deg SAM while another worsens by > 0.3 deg); bench residual Mahalanobis at or above 0.861.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT02's, plus `mixed_batches: true`; init FT02 epoch 8 (`FT02-adapt/checkpoints/best.pt`); 10 epochs (cosine, 1 warm-up, LR 1e-4) |
| batches | every micro-batch EnMAP 3 / PRISMA 3 / AVIRIS-NG 2 (share x 8, rounded), x4 accumulation = 32 per step, as FT02 |
| code | `SpectralStemSegFormerMAE.forward_mixed` (one stem pass per sensor, shared BatchNorm / encoder / decoder), `_AllSourcesLoader`, `FinetuneTrainer._mixed_loss` |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept; data already staged |

## Pass criteria (gate to FT03)
PRISMA bench, residual Mahalanobis, own bands: >= 0.861 (F001 epoch 109: 0.862). Reconstruction no worse than FT02's
epoch 8 on any sensor. Scored at epochs 3, 6 and 10.

## Results
## Observations
