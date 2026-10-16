# FT01 - Segmentation probe on frozen F001

**Status:** prepared 2026-10-16; runs after F001 and `ft_stage.sh`, on the user's go.

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
## Observations
