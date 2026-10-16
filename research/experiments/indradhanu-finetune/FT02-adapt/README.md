# FT02 - Reconstruction adaptation on EnMAP + PRISMA + AVIRIS-NG

**Status:** prepared 2026-10-16; runs after FT01, on the user's go.

## Question
Does continuing F001's masked reconstruction on the three real sensors, at their own bands and pixel sizes, keep
or improve anomaly detection while teaching the wavelengths SpectralEarth lacked?

**Expected, written before launch:** PRISMA bench under residual Mahalanobis at or slightly above F001's 0.861
(epoch 60); reconstruction on each sensor's test split falls over the run; AVIRIS-NG (3-8 m pixels, never seen)
starts worst and improves most; the trained-wavelength record grows to cover PRISMA's 1300-1500 and 1730-1960 nm
bands, so the bench reads all 185 PRISMA bands instead of 165.

## Setup
| | |
|---|---|
| config | [config.json](config.json): `mode: adapt`, init F001 epoch 109 (`F001_ep109.pt`; F001 stopped there 2026-10-16, detection flat since epoch 60), shares EnMAP 0.40 / PRISMA 0.35 / AVIRIS-NG 0.25, LR 1e-4 cosine (1 warm-up epoch), 20 epochs x 30,000 samples, band augmentation 0.8 + 3 nm jitter, width weighting |
| validation | each sensor's test split: L1 / SAM per sensor (`finetune_metrics.csv`); best.pt by the mean full-SAM score |
| bench | PRISMA four scenes, own bands, residual Mahalanobis (default) and combined, at epochs 5, 10, 20 (laptop) |

## Pass criteria (gate to FT03)
1. Bench mean ROC-AUC (residual Mahalanobis) >= 0.861 at the best checkpoint.
2. No scene below F001's epoch-60 score by more than 0.03.
3. Each sensor's test reconstruction lower at the end than after epoch 1.

## Results
## Observations
