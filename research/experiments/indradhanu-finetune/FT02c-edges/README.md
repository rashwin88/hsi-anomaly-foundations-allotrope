# FT02c - FT02b continued with an edge term

**Status:** done 2026-10-16 12:25-14:39 UTC (2.2 h), code 9ff0484, exit 0; best.pt = epoch 10.

## Question
Half or more of the top-1% false alarms sit on strong spatial edges (FT02 README, "Edge diagnosis": 49-68%, chance
10%), and score-map corrections cannot remove them without removing targets, which are edges too. Does training the
reconstruction to be sharp at edges - on hidden pixels only - cut the edge false alarms without hiding targets?

**Why it should not hide targets:** inference reconstructs every pixel while its token is hidden (two complementary
masks, `app/foundation_models/inferencers/spectral_stem_inferencer.py`), so the model predicts each patch from its
surroundings. A field boundary, road or coastline continues across the hidden patch and can be predicted sharply; an
isolated target inside the patch cannot. Stage-1 patch embeddings do not overlap (kernel 4, stride 4, padding 0,
`seg_former_encoder.py`), so nothing of a hidden 4x4 token leaks into its neighbours. **Caveat:** a target spanning more
than one token is partly visible in the neighbouring kept tokens; a sharper decoder may fill in some of it.

**Where the blur comes from:** the decoder fuses all encoder stages at H/4 and PixelShuffle(4) turns one feature vector
into a 4x4 pixel block (`app/foundation_models/components/seg_former_decoder.py`). An edge crossing a block must be
encoded in that one quarter-resolution vector. If the edge term cannot sharpen it, the next step is architectural - a
masked full- or half-resolution skip into the decoder.

**Expected, written before launch:** the share of top-1% false alarms on strong edges falls on at least 3 of 4 bench
scenes; bench residual Mahalanobis does not fall below FT02b's.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT02b's, plus `grad_loss_weight: 1.0` and `sam_ramp_epochs: 0` (SAM weight 0.5 throughout: `init_from` restarts the epoch counter, so FT02b's ramp restarted at 0 - not repeated here); init FT02b `best.pt`; 10 epochs |
| edge term | `gradient_l1` in `app/foundation_models/trainers/finetune_trainer.py`: mean over bands of \|d x_hat - d x\| for horizontal and vertical neighbour pairs, both pixels hidden and valid in that band; weight 1.0 because it is in the same reflectance units as L1 (laptop smoke: 0.014-0.016, the order of L1). Logged as `train_grad_l1` in `finetune_metrics.csv`, not in train L1 |
| measurement | PRISMA bench at epochs 3, 6, 10; the edge diagnosis (`C:/es/edges/edges.py`, same rules) on FT02b best and FT02c best |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept |

## Pass criteria
Edge false-alarm share down on >= 3 of 4 scenes, and bench residual Mahalanobis >= FT02b's best (and >= 0.861 to open
FT03).

## Results

Per-sensor validation SAM (deg), epoch 1 -> 10: EnMAP 4.02 -> 3.93, PRISMA 4.44 -> 4.21, AVIRIS-NG 5.63 -> 5.41 (FT02b
best: 3.99 / 4.37 / 5.57). The edge term itself barely moved: `train_grad_l1` 0.01163 -> 0.01153 (-1%) over 10 epochs.

**Bench, residual Mahalanobis** (means of 4 scenes, `C:/es/rescore/`): FT02c ep10 ROC-AUC 0.855, pAUC <= 1% 0.159,
TPR @ 0.1% 0.059, caught 413 (top 1%) / 131 (top 0.1%); FT02b ep9 0.856 / 0.158 / 0.059 / 411 / 128. Combined fell:
0.819 -> 0.806, caught 320 -> 296.

**Edge diagnosis** (`C:/es/edges/edges.py`, same rules), residual Mahalanobis, share of top-1% false positives on
strong edges, set1-4: FT02b ep9 69 / 52 / 52 / 60%; FT02c ep10 69 / 52 / 52 / 61%.

**Verdict: not met.** Edge false-alarm share unchanged on all 4 scenes (criterion: down on >= 3); bench unchanged
within noise (>= FT02b holds on caught, not on ROC-AUC 0.855 vs 0.856).

## Observations

- The edge term could not be reduced: 0.01163 -> 0.01153 in 10 epochs (-1%), against L1 and SAM that kept falling.
  The decoder predicts each 4x4 pixel block from one H/4 feature vector (PixelShuffle), so sub-block edge position
  has to be carried in that vector; with the loss on, it was not learned. The likely fix is architectural - a
  higher-resolution path into the decoder, masked so it cannot see hidden tokens - not a loss term.
- The per-sensor reconstruction gains over FT02b (SAM 3.99/4.37/5.57 -> 3.93/4.21/5.41) are more plausibly from the
  constant SAM weight 0.5 (FT02b ramped it from 0) than from the edge term.
