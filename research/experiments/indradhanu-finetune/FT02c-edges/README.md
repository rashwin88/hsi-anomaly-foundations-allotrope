# FT02c - FT02b continued with an edge term

**Status:** prepared 2026-10-16; launches after FT02b, on the user's go (agreed in principle 2026-10-16).

## Question
Half or more of the top-1% false alarms sit on strong spatial edges (FT02 README, "Edge diagnosis": 49-68%, chance
10%), and score-map corrections cannot remove them without removing targets, which are edges too. Does training the
reconstruction to be sharp at edges - on hidden pixels only - cut the edge false alarms without hiding targets?

**Why it should not hide targets:** inference reconstructs every pixel while its token is hidden (two complementary
masks, `app/foundation_models/inferencers/spectral_stem_inferencer.py`), so the model predicts each patch from its
surroundings. A field boundary, road or coastline continues across the hidden patch and can be predicted sharply; an
isolated target inside the patch cannot. **Caveat:** stage-1 patch embeddings overlap their neighbours by 1-2 px, so a
target larger than a patch is partly visible from next door; a sharper decoder may fill in some of it.

**Where the blur comes from:** the decoder fuses all encoder stages at H/4 and PixelShuffle(4) turns one feature vector
into a 4x4 pixel block (`app/foundation_models/components/seg_former_decoder.py`). An edge crossing a block must be
encoded in that one quarter-resolution vector. If the edge term cannot sharpen it, the next step is architectural - a
masked full- or half-resolution skip into the decoder.

**Expected, written before launch:** the share of top-1% false alarms on strong edges falls on at least 3 of 4 bench
scenes; bench residual Mahalanobis does not fall below FT02b's.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT02b's, plus `grad_loss_weight: 1.0`; init FT02b `best.pt`; 10 epochs |
| edge term | `gradient_l1` in `app/foundation_models/trainers/finetune_trainer.py`: mean over bands of \|d x_hat - d x\| for horizontal and vertical neighbour pairs, both pixels hidden and valid in that band; weight 1.0 because it is in the same reflectance units as L1 (laptop smoke: 0.014-0.016, the order of L1). Logged as `train_grad_l1` in `finetune_metrics.csv`, not in train L1 |
| measurement | PRISMA bench at epochs 3, 6, 10; the edge diagnosis (`C:/es/edges/edges.py`, same rules) on FT02b best and FT02c best |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept |

## Pass criteria
Edge false-alarm share down on >= 3 of 4 scenes, and bench residual Mahalanobis >= FT02b's best (and >= 0.861 to open
FT03).

## Results
## Observations
