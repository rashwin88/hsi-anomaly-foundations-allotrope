# HyperSL as an anomaly detector on the labelled PRISMA bench

Run 2026-10-31 for the textbook's Part X ("Why not HyperSL", `research/textbook/chapters/46-hsldetect.tex`) and the
course part 31. Question: **does HyperSL's masked reconstruction, used the way it was trained, find anomalies on our
909-object PRISMA bench, measured exactly as Indradhanu is measured?**

**Answer: no.** HyperSL's own residual is close to chance (its training loss as a score catches 63 objects, below the
~93 that random scores catch). Scored with the product's residual Mahalanobis, its reconstruction makes the score
*worse* than using no reconstruction at all (pAUC 0.083 against PCA-RX's 0.128, difference -0.045 [-0.059, -0.032]),
while Indradhanu FTH3 epoch 5's reconstruction adds +0.047 over the same PCA-RX. Not adopted.

## Config

| item | setting |
|---|---|
| data | the 4 labelled PRISMA L2D scenes in `C:/Users/FITT/Documents/stage1_datasets` (Set-1..4), 909 objects; validity and object maps exported from `app.evaluation.datasets.PrismaBench` (step 0) so every method is counted on the same pixels |
| HyperSL input | the 149-band PRISMA axis used by every earlier HyperSL experiment (`hypersl-splib-match/prisma_scene.py`, `drop_low_snr=True`, re-implemented in step 1); per-band min-max over the scene's valid pixels, as upstream `mutihsi.py` |
| model | upstream `model.py` unmodified, `ckpt.pt` = `10_base_mask95_checkpoint.pt` (sha256 `3d5f1914...a74a68c`), 256 / 8 / 4, 8 heads |
| masking | mask ratio 0.95 (`int(149 * 0.05) = 7` visible), 4 draws per pixel, `torch.manual_seed(1000 + draw)`, fp16 autocast |
| reconstruction | per band, the mean over the draws in which it was hidden (the last draw's output where it was never hidden: 2,866 of ~603 M (pixel, band) pairs; 2,936 expected) |
| HyperSL scores | `hsl_l2` mean squared residual in min-max units; `hsl_loss` MSE + (1 - cos) per pixel (its training objective); `hsl_rm` the product's `residual_mahalanobis` (d 30, window 21/61) on the residual undone to reflectance |
| baselines | FTH3 epoch 5 RM and "excl. FTH3_ep5 thick" from the cached score maps of `configs/eval/fth3.json` (`C:/es/eval/fth2_ftb2`, 2026-10-29), located by the runner's own cache key; PCA-RX = the evaluator's `pca_rx`; MNF-RX = `app.detectors.mnf_compression_detector.MNFCompressionDetector`, 10 components, product exclusions |
| cloud | every method also with FTH3 epoch 5's thick-cloud class map (cached, `FTH3_ep5-fefed77b7a`) scored 0 |
| measures | `app.evaluation.metrics.scene_metrics` (pAUC at FPR <= 0.01, ROC-AUC, TPR at 0.001, objects caught top 1% / 0.1%, tier groups); summary as `app.evaluation.report.summarise`; paired object bootstrap `app.evaluation.significance.paired_bootstrap`, 2,000 draws, seed 0 |
| hardware | step 1: RTX 4000 SFF Ada (20 GB) in WSL Ubuntu, torch 2.14.1+cu130 (`~/hypersl_eval/.venv`; Windows torch fails here), one CUDA job; steps 0 and 2: Windows product venv `C:/av` |

## Commands

```bash
# 0. masks from the evaluator (Windows)
C:/av/Scripts/python.exe step0_export_masks.py
# 1. HyperSL score maps (WSL, GPU)
wsl -d Ubuntu -- bash -lc 'cd "/mnt/c/.../research/experiments/hypersl-anomaly-bench" && ~/hypersl_eval/.venv/bin/python step1_hypersl_scores.py'
# 2. measures, baselines, bootstrap (Windows)
C:/av/Scripts/python.exe step2_evaluate.py
```

Score maps, masks and logs are outside the repo in `C:/Users/FITT/Documents/stage1_datasets/hypersl_anomaly_bench/`
(`set<n>_masks.npz`, `set<n>_scores.npz`, `step1_log.json`, `step1_run.log`, `step2_run.log`). `results.json` (everything,
per scene and per object group) and `results.md` (the two tables below) are here.

## Checks

- FTH3 epoch 5 reproduces its evaluation report exactly: RM 0.1749 / 46.4% / 13.9%; excl. 0.1892 / 49.4% / 16.0%
  (`research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/eval_report.md`).
- PCA-RX 0.1281 = the stage-3 run's 0.128 (`docs/lld/evaluation.md`).
- Never-hidden (pixel, band) pairs: set1 706 against 720.5 expected (`N x 149 x (7/149)^4`).
- `parameters()` = 9,155,969 = 9,027,841 trainable + 128,000 `pe` + 128 `div_term` (both declared as frozen parameters).

## Results

| method | caught top 1% | caught top 0.1% | pAUC (FPR<=0.01) | ROC-AUC | TPR at 0.001 |
|---|---|---|---|---|---|
| *chance* | 10.2% | | | | |
| FTH3_ep5 RM | 422 (46.4%) | 13.9% | 0.1749 | 0.866 | 0.058 |
| PCA-RX (no model) | 361 (39.7%) | 8.1% | 0.1281 | 0.839 | 0.029 |
| **HyperSL RM** | 350 (38.5%) | 5.4% | 0.0832 | 0.757 | 0.020 |
| MNF-RX | 249 (27.4%) | 3.4% | 0.0750 | 0.749 | 0.019 |
| **HyperSL L2** | 130 (14.3%) | 4.0% | 0.0175 | 0.624 | 0.007 |
| **HyperSL loss** | 63 (6.9%) | 0.9% | 0.0039 | 0.501 | 0.000 |
| FTH3_ep5 RM, cloud excl. | 449 (49.4%) | 16.0% | 0.1892 | 0.869 | 0.065 |
| PCA-RX, cloud excl. | 377 (41.5%) | 8.5% | 0.1390 | 0.842 | 0.031 |
| **HyperSL RM, cloud excl.** | 369 (40.6%) | 6.7% | 0.0947 | 0.761 | 0.024 |
| MNF-RX, cloud excl. | 280 (30.8%) | 5.2% | 0.0966 | 0.753 | 0.024 |
| **HyperSL L2, cloud excl.** | 137 (15.1%) | 4.3% | 0.0182 | 0.628 | 0.007 |
| **HyperSL loss, cloud excl.** | 63 (6.9%) | 0.9% | 0.0039 | 0.503 | 0.000 |

| difference (B - A) | pAUC diff [95% CI] | caught top 1% diff [95% CI] |
|---|---|---|
| **HyperSL RM - PCA-RX** | **-0.0450 [-0.0589, -0.0318]** | -1.2 [-4.0, +1.5] pts |
| HyperSL RM, cloud excl. - PCA-RX, cloud excl. | -0.0443 [-0.0577, -0.0320] | -0.9 [-3.5, +1.9] pts |
| PCA-RX - FTH3_ep5 RM | -0.0467 [-0.0604, -0.0327] | -6.7 [-8.8, -4.5] pts |
| HyperSL RM - FTH3_ep5 RM | -0.0917 [-0.1060, -0.0776] | -7.9 [-10.8, -5.2] pts |
| HyperSL RM, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.0944 [-0.1094, -0.0800] | -8.8 [-11.7, -6.2] pts |
| MNF-RX - FTH3_ep5 RM | -0.0999 [-0.1170, -0.0826] | -19.0 [-22.0, -16.3] pts |
| HyperSL L2 - FTH3_ep5 RM | -0.1574 [-0.1790, -0.1351] | -32.1 [-35.4, -28.9] pts |
| HyperSL loss - FTH3_ep5 RM | -0.1710 [-0.1936, -0.1479] | -39.5 [-42.9, -36.2] pts |

Per scene (pAUC, no exclusion): HyperSL RM 0.0875 / 0.0966 / 0.0286 / 0.1199 against PCA-RX 0.1734 / 0.1171 / 0.0822 /
0.1400 and FTH3 0.2616 / 0.1874 / 0.0543 / 0.1964 (set1-4) - below both on every scene. By tier (caught top 1%, easy /
medium / subtle): HyperSL RM 52.0 / 33.8 / 29.7%, PCA-RX 48.7 / 35.4 / 35.0%, FTH3 59.2 / 41.0 / 39.0%.

Runtime: 428-462 s GPU per scene, 1,773 s for the bench, 438 us per pixel at 4 draws.

## Observations

1. **Self-completion, measured.** HyperSL's training loss ranks anomalies *below* background (63 objects caught against
   ~93 by chance; ROC-AUC 0.501), matching the 2026-09-23 run where its reconstruction angle was lower on anomalies than on
   background on all four scenes (AUC 0.09-0.24, mask 0, per-curve scaling).
2. **Same scorer, opposite effect.** Against PCA-RX (residual Mahalanobis with zero reconstruction), Indradhanu's residual
   adds +0.047 pAUC and HyperSL's removes 0.045. Objects caught: HyperSL RM ties PCA-RX (-1.2 [-4.0, +1.5]).
3. **The subtle tier suffers most** (29.7% against PCA-RX's 35.0%), as expected if low-contrast targets are the most
   completely rebuilt from their own visible bands.

## Limitations

- 149-band axis (as the earlier HyperSL runs), not the evaluator's 186-band native axis, and no water bands (HyperSL
  pretrained with them). Mask ratio and draws not tuned. None of these was chosen on the bench.
- MNF-RX here is the product detector called directly (not via the action), with the bench's band filter; its ROC-AUC
  (0.749) agrees with the dual-window record's 0.752 on these scenes (`research/experiments/dual-window-mahalanobis/README.md`).
