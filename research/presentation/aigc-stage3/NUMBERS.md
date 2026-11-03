# Numbers on the slides and where they come from

Paths are relative to the repo root unless they start with `C:/`. The `.tex` files carry the same mapping in `%% src:` comments.
Short names: **FTH3** = `research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/` (README.md, eval_report.md, bootstrap.txt,
config.json, finetune_metrics.csv); **F002** = `research/experiments/indradhanu-se-pretrain/F002-heavy/` (README.md, config.json,
metrics.csv); **HSL** = `research/experiments/hypersl-anomaly-bench/README.md`; **NAMER** = `research/experiments/spectral-namer/`;
**ATM** = `C:/wt/atm-docker/research/experiments/atm-correction/EXPERIMENT_LOG.md`; **ATMREADME** = `C:/wt/atm-docker/isofit-atm-docker/README.md`;
**PROD** = `docs/lld/spectral-namer-in-product.md`; **TB** = `research/textbook/chapters/`; **COURSE** = `docs/training/anomaly-detection/`.

## Detection
| number | source |
|---|---|
| pAUC 0.189 (0.1892), 449 of 909 (49.4 %), top 0.1 % 16.0 %, ROC 0.869, TPR 0.065 | FTH3 eval_report.md; HSL (cloud excl. row) |
| FT04 0.170 / 46.0 % / 15.8 % / 0.852 / 0.065; FT04 RM 0.162 / 45.7 / 14.9 / 0.853 / 0.061 | FTH3 eval_report.md |
| 418 objects | computed: 46.0 % of 909 (FTH3 eval_report.md) |
| FTH2 ep 8 0.186 / 49.0 / 15.7 / 0.870 / 0.066; FTH3 RM 0.175 / 46.4 / 13.9 / 0.866 / 0.058 | FTH3 eval_report.md |
| +0.0196 [+0.0072, +0.0316], P 1.000; +0.0148 [+0.0029, +0.0259]; +0.0179 [+0.0055, +0.0292]; top 0.1 % +0.1 / -0.4 / -0.3 | FTH3 bootstrap.txt |
| tiers 58.2→62.2, 39.3→43.9, 40.3→42.0 % | FTH3 README |
| per scene 0.2617 / 0.1874 / 0.0861 / 0.2217; ROC 0.9357 / 0.8738 / 0.7807 / 0.8877; thick cloud 0.00 / 0.00 / 1.49 / 0.47 % | FTH3 README (product parity) |
| chance 10.2 % (= 93 objects), 1.1 % | FTH3 eval_report.md; HSL |
| PCA-RX 0.1281 / 361 / 8.1 % / 0.839 / 0.029; cloud excl. 0.1390 / 377 | HSL |
| MNF-RX 0.0750 / 249 / 3.4 % / 0.749 / 0.019; cloud excl. 0.0966 / 280 | HSL |
| bench resolution ±0.005–0.01 pAUC, ±1–1.5 pts | docs/lld/evaluation.md |
| 2,000 bootstrap draws | HSL |

## HyperSL
| number | source |
|---|---|
| RM 0.0832 / 350 / 5.4 % / 0.757 / 0.020; L2 0.0175 / 130 / 4.0 % / 0.624; loss 0.0039 / 63 / 0.9 % / 0.501; V2 RM 0.1749 / 422 | HSL results table |
| cloud excl. 0.0947 / 369 / 6.7 % | HSL |
| differences -0.0450 [-0.0589, -0.0318], -1.2 [-4.0, +1.5]; -0.0467 [-0.0604, -0.0327], -6.7 [-8.8, -4.5]; -0.0917 [-0.1060, -0.0776], -7.9 [-10.8, -5.2]; -0.0944 [-0.1094, -0.0800], -8.8 [-11.7, -6.2] | HSL difference table |
| +0.047 pAUC over PCA-RX | HSL observation 2 |
| min-max toy (0.20, 0.50)→(0.5, 1.0) / (0.4286, 0.75); (0.2, 0.333); 15.07°→36.87°; (3.0, -0.5); 41 %; 28 % | COURSE 30-min-max-and-the-library.md |
| cirrus 28.9 vs 67.9 %; cloud 87.1 vs 80.4; full recall/precision table; 82.75 / 84.70 % | COURSE 33-hypersl-cloud-mask.md |
| HySU AUC 0.97 / 1.00, 0.88 / 0.99, 0.68 / 0.77, blind 0.72 / 0.62 / 0.80 / 0.88 / 0.92 | COURSE 32-hypersl-unmixing.md |
| decision table | COURSE 34-hypersl-verdict.md |

## Anomaly score
| number | source |
|---|---|
| 0.570 / 0.662 / 0.702 / 0.752 / 0.831 / 0.839 / 0.854; 886 objects; Set-3 0.766 vs 0.749 | research/experiments/dual-window-mahalanobis/README.md |
| d = 30, ring 21/61, ≤ 200,000 pixels, 3–4 s per scene | docs/lld/residual-mahalanobis.md |
| 465 vs 17,205 entries; 3,280-pixel ring | TB 29-residualmaha.tex |

## Indradhanu architecture and training
| number | source |
|---|---|
| 35.3 M = 33,884,194 + 1,431,120 = 35,315,314; stem 17,825; encoder 29,580,288; decoder 4,279,616; output 6,337; FT04-size counts 14,721 / 3,331,072 / 2,164,736 / 4,257+64 | TB 41-heavy.tex |
| shapes (B,185,128,128), z (B,64,…), 665 of 1,024 tokens, (B,359,96), F1–F4, 53 bins, 64 queries, 4 heads, 185 bands kept, 733 trained wavelengths | TB 41-heavy.tex |
| widths [96,192,480,768], heads [3,6,10,12], reductions [8,4,2,1], decoder 512 / 256 / 128, mask 0.65, lr 3e-4, batch 32, 80 epochs, 30,000 patches, band augment 0.8, jitter 3 nm | F002 config.json |
| FTH3 lr 3e-5, batch 16×2, 6 epochs, sources 0.40 / 0.35 / 0.25, seg weight 0.1, class weights ×3 | FTH3 config.json |
| FTH2 = 18 joint epochs | FTH3 README |
| val SAM 3.89° (F001 4.06° at epoch 100), 8.84 at epoch 1, L1 0.0164; 369 s / epoch, 81.2 patches/s, 18.4 GB; 3,200 of 3,419 shards; best epoch 72 | F002 README, metrics.csv |
| pretrained only F001 0.155 / 45.1 / 14.0 %; F002 0.166 / 47.4 / 12.4 % | F002 README |
| PRISMA SAM 4.48°; IoU water 0.825, cloud 0.749, thin 0.090 | FTH3 README, finetune_metrics.csv |
| V1: 5.5 M params, 165-band grid, 1×1 compressor to 24–32 channels; FT04 5.9 M | docs/04-models.md |
| loss L1 + λ(t)·SAM, λ = 0.5·min(1, t/10) | TB 23-losses.tex; F002 config (sam_weight 0.5, ramp 10) |
| seg loss 0.1 × (BCE + Dice) | TB 27-finetune.tex |
| ~6× faster than per-band attention; band-list augmentation details; ROC-AUC 0.205 native vs 0.725 resampled | docs/lld/spectral-stem.md |
| haze 75.4 % of Set-4, 209 of 230 targets; thick-cloud rule | docs/lld/spectral-stem.md |
| v0.2.0 0.165 and FT04 0.170 (cloud excluded) | final design/ROADMAP.md (OB-14) |
| FT07 0.165, FT08 0.164, FT09b 0.176 (+0.0060 [+0.0002, +0.0118]), FT09c 0.172, FT06 0.125 / 0.163, scene adaptation ~0.160 | docs/lld/evaluation.md |
| worker cost 1819 / 454 / 274 s; 1629, 185, 4; 402, 47, 4; 236, 32 | FTH3 README |

## Spectral namer
| number | source |
|---|---|
| level 1 recall 0.65 [0.55, 0.72], 0.86 [0.78, 0.92], 0.85 [0.78, 0.92], 0.94 [0.87, 0.98], 0.71 [0.63, 0.79], 0.90 [0.83, 0.94]; ACE 0.48 / 0.66 / 0.61 / 0.77 / 0.59 / 0.76 | NAMER/N001b/README.md (Intervals) |
| level 2 0.59 / 0.68 / 0.69 / 0.70 / 0.65 / 0.70 with intervals; ACE 0.24 / 0.47 / 0.41 / 0.60 / 0.38 / 0.58 | NAMER/N001b/README.md |
| +0.20 / +0.17 / +0.14; 221–254 spectra, 500 reps; bank 378 / 338 / 380; 347,685 parameters | NAMER/N001b/README.md |
| N001 level 1 0.49–0.52, 0.50 at fill 1.0, ACE 0.87, BCE 0.684 | NAMER/N001/README.md |
| open set 0.46–0.47, 0.45–0.46, 1–4 %, 0.80 / 0.84 / 0.81; confident at 0.55 | NAMER/N002/README.md |
| PV 0.248 → 0.248, 50 spectra | NAMER/pv-library/README.md |
| ACE level-1 baseline table; class list | NAMER/README.md |
| stem 17,825 of 347,685; planting equation, kinds 0.5 / 0.25 / 0.25; schedule; 24,576 samples; ln 2 = 0.693, ln 7 = 1.946 | TB 42-namer.tex |
| a_min 0.05, gain 0.1, lr 3e-4, stem ×0.3, 40 epochs × 500 steps | NAMER/N001b/config.json |

## Product, sensors, deliverables
| number | source |
|---|---|
| 1,103 / 5,544 / 2,375 objects; 272 / 405 / 1,290; 585 / 2,263 / 1,716; 247 / 141 / 936 | PROD (class-driven match screen) |
| e2e: 7,435 / 4,565; 110 / 865; 272 / 1,186; 26 / 598; 247 / 1,124; 852 / 6,570; 831 / 3,379; 240 / 1,672; 424 / 2,073; export 2.1–2.6 s | PROD (class-driven flow e2e) |
| namer stage 9.2 / 7.6 s | PROD section 6 |
| EPSG:32644, (30, 0, 375472.78, 0, -30, 3721282.25); file names; attribute names | PROD |
| story frames: 1237×1202 px, 239 bands, 186 usable, 30.3 % no-data, 4.9 % cloud, top 8 % = 118,950 px | captions inside the story GIF key frames (scratchpad final/story/…_prisma_default/key1-7.png) |
| land 73.1, water 21.8, cloud 4.9, haze 3.7, snow 0.1 % | product screenshot final/seg-prisma_v2.png |
| PRISMA georef 2.4 px median, 95 % 4.1, max 5.6; affine | docs/proposals/aigc-evaluation-submission.md |
| Tanager 426 bands, 368 good, EPSG:32613, 30 m | docs/lld/tanager.md |
| EnMAP 224, AVIRIS-NG 372 bands | FTH3 config.json (band lists) |

## Atmospheric correction
| number | source |
|---|---|
| ISOFIT 4.1.5, sRTMnet, sensors, int16 ×10000, nodata -9999, sidecars | ATMREADME |
| T1–T6 times and core land MAE (33 / 36 / 44 / 26 / 33 / 36 min; 0.0124 / 0.0114 / 0.0089 / 0.0461 / 0.0157 / 0.0092) | ATM AC26-T11 |
| T7 41 min, MAE 0.0057, bias +0.0031, angle 2.27° | ATM AC26-T7B |
| PRISMA holdout 0.0244 → 0.0099; EnMAP 0.0098 | ATM AC26-HON |
| seg 50 / 200 / 200-empirical / 400-empirical: 40.8 / 17.8 / 11.3 / 7.5 min; 0.0056 / 0.0056 / 0.0056 / 0.0057; 2.15 / 2.17 / 2.17 / 2.19° | ATM AC26-FAST |
| 81 tests, 54 %, 22 GB, 8.98 GB | ATM AC26-BLD |

## Journey
| item | source |
|---|---|
| 26 Jan, 23 Mar, 10 Apr, 23 Jul, 14 Oct, 15 Oct, 19 Oct, 22 Oct, 28–29 Oct, 31 Oct, 1 Nov | `git log --date=short` (3625df1, 727e3e4, d2f8c3c, 0cc4335, c002970, 4b7da29, 20afa70, 8c14fe6, a0b6930/c4b3a63, 0c2efe8, 53ac8d7) |
| May: plain RX dropped, distances ~1e11 | CLAUDE.md |
| Aug: two outages reached main; fabricated docs removed 2026-08-24 | CLAUDE.md; commit c300ea7 |
| FT07 0.165, FT08 0.164 vs 0.170 | docs/lld/evaluation.md |
| haze: 209 of 230 targets | docs/lld/spectral-stem.md |
| HyperSL 0.083 vs 0.128 | HSL |
| N001 level 1 at chance (0.50) | NAMER/N001/README.md |
| PV no effect | NAMER/pv-library/README.md |

## Not sourced, so left out
- Trivy counts (app 42 critical / 670 high, 0 secrets; ATM 0 critical / 9 high): not in the repo or the ATM worktree (the ATM build log says Trivy was not installed). The slides describe the security scan without counts.
- N003 results: pending in its README.
