# FT03 - Joint reconstruction + segmentation

**Status:** prepared 2026-10-16; runs after FT02 passes, on the user's go.

## Question
Trained together, does the segmentation head beat the frozen probe while detection holds?

**Expected, written before launch:** per-class IoU above FT01 (the encoder can now adapt to the labels); bench
within 0.01 of FT02 (reconstruction stays the dominant loss); the segmentation loss is taken on EnMAP batches
only, so PRISMA / AVIRIS-NG masks are judged by eye on cloud scenes.

## Setup
| | |
|---|---|
| config | [config.json](config.json): `mode: joint`, init FT02 best.pt (head fresh), seg_weight 1.0, same shares / LR / epochs / augmentation as FT02 |
| validation | per-sensor reconstruction + EnMAP per-class IoU |

## Pass criteria
1. Bench mean (residual Mahalanobis) not below FT02's by more than 0.01.
2. IoU above FT01 for water, cloud and thin cloud; shadow and snow reported only.

## Results
## Observations
