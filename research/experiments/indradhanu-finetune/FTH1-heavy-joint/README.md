# FTH1 / FTB1 - Indradhanu Heavy fine-tuned vs the same-recipe control

**Status:** done 2026-10-29 (FTH1 01:02-03:18 UTC, FTB1 03:18-05:29) on the H100 capacity block, `h100_chain.sh`.
**The first model-side detection gain since FT04 that holds across neighbouring checkpoints.** FTH2 / FTB2 (+8 epochs)
running after them.

## Setup
One 10-epoch joint run each, FT04's final recipe (mixed batches, EnMAP labels + PRISMA / AVIRIS-NG pseudo-labels from
FT04, rare-class weights [1,1,1,3,3], seg_weight 0.1), lr 1e-4 cosine:
- **FTH1** from F002-heavy (Indradhanu Heavy, 33.9 M; progressive decoder; seg head 256 wide) - [config](config.json)
- **FTB1** from F001_ep109 (FT04's size, 5.9 M; SegFormer decoder) - [config](../FTB1-base-joint/config.json)

## Validation (epoch 10)
| | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (four-stage curriculum) | 3.88 / **4.11** / 5.46 | **0.839 / 0.810** |
| FTB1 | 3.90 / 4.48 / **5.34** | 0.782 / 0.721 |
| FTH1 | 3.94 / 4.74 / 5.60 | 0.795 / 0.717 |

The one-run recipe falls short of FT04's curriculum on reconstruction and masks for both; shadow / snow IoU 0.

## Detection (`configs/eval/fth1_ftb1.json`, real PRISMA bench, [eval_report.md](eval_report.md))
**Size effect - same recipe, same (FT04) masks:**

| checkpoint | FTB1 pAUC | FTH1 pAUC | FTH1 - FTB1 (paired bootstrap 95%) | caught top 1% FTB1 -> FTH1 |
|---|---|---|---|---|
| epoch 8 | 0.162 | **0.175** | **+0.0132 [+0.0022, +0.0243]** | 45.9 -> 47.9% (+2.0 [+0.2, +3.9]) |
| epoch 9 | 0.162 | **0.171** | +0.0093 [-0.0013, +0.0195], P 0.96 | 46.4 -> 47.1% |
| epoch 10 | 0.161 | **0.175** | **+0.0145 [+0.0045, +0.0246]** | 46.3 -> 47.2% |

**Against shipped FT04 (its own masks):** FTH1 + FTB1's masks, epochs 8 / 9 / 10: pAUC **0.182 / 0.178 / 0.182** vs
0.170 (+0.0130 [+0.0009, +0.0242]; +0.0082 [-0.0027, +0.0181]; +0.0123 [+0.0011, +0.0224]); caught top 1% **48.5 /
47.7 / 48.1%** vs 46.0% (epoch 10: +2.1 [+0.1, +4.0]); caught top 0.1% 15.4 / 15.3 / 15.6% vs 15.8% (tie). The best
detection measured so far.

**FTH1's own cloud masks are broken for detection:** excluding cloud with them collapses every model (ROC-AUC ~0.74,
caught top 1% ~33%) - its head marks far too much as cloud on the bench scenes, though its EnMAP validation cloud IoU
(0.717) is like FTB1's (0.721). FTB1's masks are good (they even lift FT04: 0.170 -> 0.176).

## Mask diagnosis (2026-10-29)
Share of each bench scene the heads call cloud (class map = cloud or thin cloud > 0.5), and targets inside it:

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| FT04 | 0.0% / 0 | 0.0% / 0 | 0.8% / 0 | 11.8% / 2 |
| FTB1 | 0.0% / 0 | 0.0% / 0 | 1.7% / 0 | 0.3% / 0 |
| FTH1 | 0.4% / 17 | 0.0% / 0 | 2.1% / 0 | **75.4% / 209 of 230** |

**All of it is the thin-cloud output** - FTH1 on set4: thick cloud 0.4% (0 targets), thin cloud 75.4% (209 targets) =
the haze; FT04's 11.8% on set4 is thin cloud too. Excluding **thick cloud only** with FTH1's own head:
epoch 10 pAUC **0.181** (set1-4 0.253 / 0.162 / 0.086 / 0.222), caught top 1% 48.0%, top 0.1% 15.5%; epoch 8 0.181 /
48.5% / 15.4% - the same as with FTB1's masks. **Heavy stands alone at pAUC 0.181 vs shipped FT04's 0.170.**

Follow-up (not done): the product's exclusion (`SpectralStemInferencer.segment_full_scene`, cloud = cloud or thin
cloud) also removes haze for Indradhanu V2; measure thick-only on FT04 before changing shipped behaviour.

## Observations
- **Bigger pretrained model -> better detector**, consistently across three neighbouring checkpoints (two resolved),
  the rule FT09c taught. Heavy's lead over F001 seen in pretraining (pAUC +0.011) survives and grows in fine-tuning.
- Reconstruction is not the story: FTH1 reconstructs PRISMA worse than FTB1 and detects better.
- Heavy needs a working mask head before it can stand alone: diagnose what it calls cloud, then a head-only stage on
  the frozen backbone (FT01-style probe) or keep a separate segmenter.
- Caveats: 4 scenes; the bootstrap resamples objects, not scenes.
