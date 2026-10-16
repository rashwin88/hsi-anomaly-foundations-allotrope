# FT01 - Segmentation probe on frozen F001

**Status:** done 2026-10-16, 03:40-05:03 UTC (1.4 h, ~$4), code 64f0106, exit 0. best.pt = epoch 14.

## Question
How much cloud, water, shadow and snow information does F001's encoder already hold? Only the segmentation head
(SegFormer-style, decoder width 128, ~372k parameters) trains; the backbone is frozen, BatchNorm statistics included.

**Expected, written before launch:** water is easy from the spectrum alone (IoU >= 0.80 on EnMAP test); cloud
>= 0.5; thin cloud lower (0.3-0.5 - its labels are the noisiest); shadow low (< 0.3, 0.09% of pixels, 21 test
tiles); snow not measurable on test (1 tile).

## Setup
| | |
|---|---|
| config | [config.json](config.json): `mode: probe`, EnMAP only, init F001 epoch 109 (`F001_ep109.pt`; F001 stopped there 2026-10-16, detection flat since epoch 60), LR 1e-3 cosine, 15 epochs x 14,542 samples, batch 8 x 4, no band augmentation |
| labels | the 2026-10-16 policy (`app/utils/segmentation/targets.py`): class 0 = cloud, class 3 = no-data, dry scenes (wv < 0.75 g/cm2) cloud/thin-cloud unknown, snow wins, cirrus 1 unknown; 4 duplicate scenes skipped |
| validation | EnMAP test split (6 datatakes, 950 tiles): per-class IoU each epoch (`finetune_metrics.csv`); best.pt by segmentation validation loss |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept |

## Pass criteria
None that block: FT01 measures. Its per-class IoU is the floor FT03 must beat. If water IoU is below 0.6 the
encoder lacks basic spectral separation and the plan is revisited before FT02.

## Results

Per-class IoU on EnMAP test (950 tiles), from `finetune_metrics.csv`; segmentation validation loss from train.log.

| epoch | water | cloud | thin cloud | shadow | snow | mean | val loss |
|---|---|---|---|---|---|---|---|
| 1 | 0.705 | 0.600 | 0.067 | 0.000 | 0.000 | 0.274 | |
| 3 | 0.777 | 0.733 | 0.076 | 0.000 | 0.000 | 0.317 | |
| 6 | 0.780 | 0.784 | 0.065 | 0.104 | 0.000 | 0.347 | |
| 9 | 0.815 | 0.766 | 0.061 | 0.185 | 0.000 | 0.365 | |
| 12 | 0.822 | 0.807 | 0.089 | 0.165 | 0.055 | 0.388 | 0.3724 |
| **14 (best.pt)** | **0.822** | **0.801** | **0.091** | **0.199** | 0.175 | **0.418** | **0.3696** |
| 15 | 0.823 | 0.800 | 0.089 | 0.193 | 0.169 | 0.415 | 0.3706 |

Against the expectation written before launch: water 0.82 (expected >= 0.80, met); cloud 0.80 (expected >= 0.5, well
above); thin cloud 0.09 (expected 0.3-0.5, far below); shadow 0.20 (expected < 0.3, as expected); snow 0.17 on a
single test tile, not a measurement. Gate: water >= 0.6, so the plan stands.

## Observations

- F001's frozen features already separate water and cloud: 0.82 / 0.80 IoU from a 372k-parameter head alone.
- Thin cloud never moved (0.06-0.09 for 15 epochs) while every other class improved: the label QA found cirrus
  labels surface-driven in many water-vapour bands, so this is most likely a label ceiling, not a feature gap.
  Do not tune FT03 on it; treat thin-cloud IoU as unreliable until a cleaner label exists.
- Shadow appeared from epoch 6 (0.09% of pixels) - rare-class weighting should lift it in FT03.
- Validation loss was still falling at the end (best at 14 of 15); a longer probe would gain a little, not change the picture.
- `metrics.csv` carries zeros for L1/SAM in probe mode (no reconstruction is run); the seg loss is only in train.log.
