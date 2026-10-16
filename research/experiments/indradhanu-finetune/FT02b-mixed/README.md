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

PRISMA bench, own bands (`C:/es/ep109/score.py`), ROC-AUC set1 / set2 / set3 / set4 / mean (objects caught of 909).
References: F001 ep109 residual Mahalanobis 0.862 (400); FT02 best (ep8) 0.855 (401), combined 0.824 (335).

| epoch | val SAM deg: EnMAP, PRISMA, AVIRIS-NG | residual Mahalanobis (gate >= 0.861) | combined |
|---|---|---|---|
| 4 | 4.10, 4.52, 5.62 | 0.935 / 0.852 / 0.759 / 0.881 / **0.857** (418) | 0.945 / 0.768 / 0.714 / 0.818 / **0.811** (324) |
| 9 (best.pt) | 3.99, 4.37, 5.57 | **0.856** (411) | **0.819** (320) |

Per-sensor validation, epochs 1-4 (L1 / SAM deg): EnMAP 0.0161 / 4.20 -> 0.0160 / 4.10; PRISMA 0.0166 / 4.70 ->
0.0150 / 4.52; AVIRIS-NG 0.0153 / 5.88 -> 0.0151 / 5.62 (epoch 2 AVIRIS-NG 6.19 the only step back). All three below
every FT02 epoch on PRISMA and AVIRIS-NG.

**Status:** done 2026-10-16 10:15-12:25 UTC (2.2 h), exit 0, best.pt = epoch 9. Gate (>= 0.861): **not met** (0.856).

**All bench measures** (2026-10-16, `C:/es/rescore/`, means of 4 scenes; `detection_measures` in
`app/utils/evaluation/prisma_bench.py`), residual Mahalanobis:

| model | ROC-AUC | pAUC <= 1% FPR | TPR @ 0.1% | TPR @ 1% | caught top 1% | caught top 0.1% |
|---|---|---|---|---|---|---|
| v0.2.0 (shipped) | 0.854 | 0.159 | 0.058 | 0.246 | 402 | 128 |
| F001 ep109 | 0.861 | 0.148 | 0.056 | 0.235 | 400 | 120 |
| FT02 ep8 | 0.855 | 0.150 | 0.054 | 0.237 | 401 | 127 |
| FT02b ep4 | 0.857 | 0.158 | 0.058 | 0.249 | 418 | 129 |
| FT02b ep9 | 0.856 | 0.158 | 0.059 | 0.248 | 411 | 128 |

Combined (old score) moves a lot with fine-tuning: pAUC 0.039 (F001) -> 0.093 (FT02b ep9), caught 201 -> 320; still
far below residual Mahalanobis. The 3D-ROC terms are degenerate for residual Mahalanobis: its heavy tail makes the
min-max-normalised background mean ~0.001 for every model (AUC_BS = ROC-AUC - 0.001), so they do not separate models.

## Observations

- **Under the default score the bench cannot tell these models apart.** Every measure, top-of-list included, spans
  less than the scene-to-scene noise, and the shipped v0.2.0 ties the fine-tuned models. Reconstruction improved a lot
  (PRISMA SAM 7.3 -> 4.4 deg) but residual Mahalanobis's whitening already absorbs most reconstruction error, so a
  better reconstruction buys little detection on 4 scenes. A larger labelled benchmark (HAD100 as reflectance, more
  PRISMA) is needed to see model differences.
