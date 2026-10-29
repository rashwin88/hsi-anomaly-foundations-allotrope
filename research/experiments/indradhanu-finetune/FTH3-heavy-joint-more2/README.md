# FTH3 - Heavy +6 joint epochs: the plateau, and the release candidate

**Status:** done 2026-10-29 09:17-10:39 UTC, the last run of H100 block 1. FTH2 (18 joint epochs) + 6 at lr 3e-5
([config](config.json)). **Detection has plateaued; FTH3 epoch 5 is the release candidate.**

## Validation
| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FTH2 ep 8 | 3.88 / 4.48 / 5.39 | 0.818 / 0.736 |
| 4 | 3.87 / 4.47 / 5.40 | 0.827 / 0.731 |
| **5** | 3.89 / 4.48 / 5.41 | 0.825 / 0.749 |
| 6 | 3.85 / 4.44 / 5.47 | 0.820 / 0.747 |

## Detection (`configs/eval/fth3.json`, real PRISMA bench, own thick-cloud masks; [eval_report.md](eval_report.md), [bootstrap.txt](bootstrap.txt))
| | pAUC | caught top 1% | caught top 0.1% |
|---|---|---|---|
| FT04 shipped | 0.170 | 46.0% | 15.8% |
| FTH2 ep 8 | 0.186 | 49.0% | 15.7% |
| FTH3 ep 4 / **5** / 6 | 0.184 / **0.189** / 0.187 | 49.2 / **49.4** / 48.8% | 15.4 / **16.0** / 15.5% |

- vs FTH2 ep 8: -0.0021 [-0.0043, -0.0000] / +0.0028 [-0.0002, +0.0058] / +0.0011 [-0.0009, +0.0030] - a plateau.
- vs FT04 shipped: **+0.0148 [+0.0029, +0.0259] / +0.0196 [+0.0072, +0.0316] / +0.0179 [+0.0055, +0.0292]**; top 0.1%
  -0.4 / +0.1 / -0.3 pts (ties).
- Caught top 1% by tier, FT04 -> FTH3 ep 5: easy 58.2 -> 62.2%, medium 39.3 -> 43.9%, subtle 40.3 -> 42.0%.
- Heavy's thick-cloud masks are interchangeable across epochs 4-6 (within 0.001 pAUC).

## Verdict
**Release candidate: FTH3 epoch 5** (`ckpt_all/spectral_stem_finetune_v0.5.0_epoch5.pt`; neighbours 4 and 6 within
0.005) - pAUC +0.020 over shipped FT04, resolved, with the top 0.1% now a tie. Further joint epochs at this recipe do
not add detection. Product use needs the thick-cloud rule, a manifest / resolver path for the progressive-decoder
architecture, and a worker CPU-cost check (34 M parameters).

## Shipped (2026-10-29) as Indradhanu V2 `v2.1.0`
Same codename and architecture key as FT04 (`spectral_stem_finetune_v2.1.0_epoch5.pt` in the models volume; FT04
`v2.0.0` is the manifest's first alternative). The product excludes thick cloud only (resolver
`thin_cloud_is_cloud=False`).

**Product parity** ([product_parity.py](product_parity.py): resolver manifest -> get_inferencer -> read_bands ->
predict -> `compute_score(method="residual_mahalanobis")` -> `segment_full_scene(thin_cloud=False)` -> zero cloud),
against the evaluator's FTH3_ep5 "excl. FTH3_ep5 thick" (C:/es/eval/fth2_ftb2/per_scene.csv):

| scene | thick cloud excluded | pAUC product | pAUC evaluator | ROC-AUC product / evaluator |
|---|---|---|---|---|
| set1 | 0.00% | 0.2617 | 0.26165 | 0.9357 / 0.9354 |
| set2 | 0.00% | 0.1874 | 0.18742 | 0.8738 / 0.8735 |
| set3 | 1.49% | 0.0861 | 0.08609 | 0.7807 / 0.7807 |
| set4 | 0.47% | 0.2217 | 0.22167 | 0.8877 / 0.8876 |

Mean pAUC 0.1892, equal to the evaluator's.

**Worker CPU cost** ([cpu_timing.py](cpu_timing.py), inside `allotrope_worker`, 2 torch threads as the image pins,
PRISMA set1 1186 x 1196 x 185 bands): Heavy 1819 s (predict 1629, segment 185, score 4) against FT04 1058 s (predict
920, segment 133) - 1.7x. **With 16 threads** (compose `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`, 2026-10-29): Heavy
454 s (predict 402, segment 47, score 4) against FT04 274 s (predict 236, segment 32) - 4.0x and 3.9x faster, Heavy
still 1.7x FT04.
