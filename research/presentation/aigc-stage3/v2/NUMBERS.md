# Numbers on the v2 slides and where they come from

**R p.N** = the submitted report, `PS11_stage3_report_allotrope21labs.pdf` (2 Nov 2026), page N. The report is the
primary source; repo files are cited only where the report is silent. `%% src:` comments in `main.tex` carry the same mapping.

| slide | numbers | source |
|---|---|---|
| 1 Title | 0.189 / 0.170; 449 / 909, 377; 86–94 % vs ACE 66–77 % | R p.1 |
| 2 Problem and data | 700+ scenes (Stage II); ≈3.3 TB, ≈11,636 scenes; 1,300+ curated, 2–40 % cloud; 60 EnMAP + 30 PRISMA pairs, 50+ Tanager; sensor table; 909 objects, tiers; measures and chance levels; ±0.005–0.01 | R p.1–2, Tables 1–2; R p.18 |
| 3 Pipeline | Figure 1 and its caption (1237×1202, 30.3 %, 4.9 %, 8 %, 118,950, 5,544, 405); 13 min; 454 / 274 s; 3–4 s; 8–9 s | R p.2, 3, 9, 15, 20 |
| 4 Spatial to spectral | V1 5.5 M, 165 grid, 1×1 compressor 165→24–32; V2 33.9 M + 1.43 M = 35.3 M; widths 96/192/480/768, heads 3/6/10/12; 185 / 224 / 372; 512,849; 0.165 → 0.170 → 0.189 | R p.3–5 (Figure 2, spec sheet, Table 3) |
| 5 Spectral stem | band ticks (EnMAP 224, AVIRIS-NG 372; 47 / 49 occupied bins computed from the lists); 40 nm bins from 400 nm, 53 bins, 64 queries, 4 heads, token width 32, width-weighted means, periods 10–10,000 nm; 80 %, 3 nm, ≥40 bands; 0.205 vs 0.725; 17,825; ≈6× | `research/experiments/indradhanu-finetune/band_lists/*.json`; `app/foundation_models/components/spectral_reduced.py`, `spectral_tokens.py`; F002 config; R p.3–4; `docs/lld/spectral-stem.md`; textbook ch. 41 |
| 6 Training | Table 4 (data, 33.9 / 35.3 M, 80 and 10+8+6 epochs, lr, batch, 9 h, 5.5 h, mix 40/35/25, 210/223/79 scenes, 14,542/14,802/22,377 tiles); 65 %; λ ramp to 0.5 over 10 epochs; 0.1 × (BCE + Dice), ×3; 150-patch archives, ≈1 GB, 3,419, 18 patches/s, 200 swapped every 5 epochs, 81 patches/s, 18.4 GB, 3,200; 30 km; 8.84° → 3.89°, 4.06°; epoch 72; epochs 4/6 0.184/0.187 | R p.4–5; curve plotted from `F002-heavy/metrics.csv` |
| 7 Detection results | Table 5 / Figure 4 bars; bootstrap +0.0196 [+0.0072, +0.0316], +0.0148 […], +0.0179 […]; −0.047 [−0.060, −0.033]; tiers; +0.1 pts; per scene 0.2617 / 0.1874 / 0.0861 / 0.2217 and ROC 0.936 / 0.874 / 0.781 / 0.888; 0.175 → 0.189; RM: 30 directions, 21/61 ring, 0.662 → 0.854 | R p.3, 6–7 |
| 8 Detection examples | Figure 6; class split 244/116/25/16/3/1; IoU 0.825 / 0.749; 75.4 %, 209 of 230; Table 8 | R p.3, 7, 11 |
| 9 Atmospheric correction | modules; radiometric rules; AOT rules (±0.15; 0.10–0.14; free); terrain/adjacency rules; bands kept 206 / 218 / 380 / 382; inversion windows; superpixels 50 / 400; r 0.947 → 0.980; NDSI 0.91–1.00; −0.2 / 2.5; adjacency formula, 1 km; ISOFIT 4.1.5 + sRTMnet | R p.17–19; container README (version) |
| 10 Correction accuracy | 0.0244 → 0.0099; 0.0098 → 0.0086; 0.0051; 0.0061–0.0144; 0.01–0.02; 15–27 / 5 scenes; +0.023 → +0.001; 62 / 94 min; 26–44 min; 13 / 13 / 16 / 30 / 35 / 40 min; 1.9×, 5×, 3.2×, ≤6e-8; 53 → 30 min; 40.8 → 7.5 min, 0.0056 → 0.0057; 373 → 41 min; 28/28 | R p.18–20 |
| 11 Material identification | 7 classes, 378 / 338 / 380; 347,685; ring 3–7 px; 5–100 % fill, 20 % held out; Figure 5; Table 6; 0.53 / 0.44 / 0.27–0.30 | R p.1, 9–10, 14 |
| 12 Field targets | HySU and Gudalur descriptions; Table 9 (L2 chart and L1 table); 5 of 5, 18 of 19, 16 of 22, 18 | R p.11–12 |
| 13 Against HyperSL | 9.2 M, 95 %; 63 vs 93; min-max example, 15.1° → 36.9°, 41 %; cirrus 28.9 vs 67.9 %; Figure 8; Table 10; Table 11; 0.4–0.7 | R p.12–14 |
| 14 Deliverables and limits | delivered attributes; hardware (<30 s, 16 min, CPU/RAM/GPU shares); Golmud: ≈8,000 → ≈3,000, ≈5,500 → ≈3,700; class shares and example matches; limits (46–47 %, 1–4 %, 0.248, 5 sheets, 22 targets, 454 / 274 s) | R p.15–16, 20–21 |
| 15 Journey | dates and failures | as in `../NUMBERS.md` (git log, CLAUDE.md, evaluation LLD, spectral-stem LLD, HyperSL bench, namer READMEs) |
| A1 | Table 3 / Figure 3 | R p.4 |
| A2 | N001 / N001b / N002 / PV results; Table 7; planting equation, loss, 347,685 / 17,825, AdamW 3e-4, 0.3×, 40×500; 500 resamples of 221–254; ≈34 spectra, 0.12 | namer READMEs, textbook ch. 42; R p.10, 15 |
| A3 | Table 12; ≈40 %, 0.6; min-max worked example; decision table; 149 bands; 29.7 vs 35.0 % | R p.12–15 |
| A4 | pAUC definition; Table 5 full | textbook ch. 31; R p.2, 6–7 |
| A5 | H200 140 GB, 16 cores, 64 GB; DEM 30 / 90 m; ISOFIT 4.1.5; 6S x86-64-v3/v4; 81 tests, 54 %, 22 / 8.98 GB; worker cost table; superpixel table; 75+ min, ≈6 h, 12–14 GB | R p.19–21; container README; ATM EXPERIMENT_LOG AC26-BLD, AC26-FAST; FTH3 README |

## Equations

- Spectral stem (slide 5): read from `spectral_reduced.py` / `spectral_tokens.py` (bin token with per-band weight vector
  `k(λ)`, width-weighted mean, bin-centre bias; learned queries attending over occupied bins; per-band skip `W(λ)`;
  per-wavelength decoder `U(λ)·z + c(λ)`).
- Atmospheric correction (slide 9): TOA conversion and adjacency formula from R p.18–19; the coupled radiance model and
  the optimal-estimation cost are the standard ISOFIT formulation (Thompson et al., 2018), not a number.
- Residual Mahalanobis, loss, namer scores, pAUC: as in the v1 deck (textbook chapters 23, 29, 31, 42).

## A6 HAD100 (added 2026-11-03)

| numbers | source |
|---|---|
| published mAUC rows (AETNet-KCSR, KCSR, CR, AETNet, LRX, GRX, AutoAD, LREN, RGAE at 50/100/200 bands) | Li et al., IEEE TGRS 2023, Table V, as transcribed in `research/experiments/dual-window-mahalanobis/README.md` 4b and EXPERIMENT_LOG.md |
| ours, official protocol: GRX, PCA30-RX at 50/100/200/276 b; Indradhanu V1 + RM 0.9740, local mean 0.9748, L1 0.8445; 5th of 11, 0.013 below best | `dual-window-mahalanobis/results/auc_summary.json` (had100_official_protocol); README 4b |
| without the 6 Kakinada crops (94), V1: V1 + RM 0.9724, local mean 0.9732, L1 0.8365 | computed from the per-crop step-6 output `C:/Users/FITT/Documents/public_had_datasets/dual_window_public/had100_protocol.json` (its all-100 means reproduce auc_summary.json exactly) |
| 94 raw tiles: 0.973 / 0.848 / 0.888 / 0.968, 90/94, 51/94, p = 0.06 | README 4a; auc_summary.json had100_94_tiles |
| 100 crops of 64x64, 94 tiles, 3,567 px, 460 objects, L1B radiance, 276 good bands, first 50/100/200; Kakinada 6/100, 1-2 km | `research/experiments/benchmarks/README.md` sections 2, 3, 5 |
| V1 model: Indradhanu v0.2.0 (epoch 200), radiance mapped affinely onto training stats | dual-window README sections 3-4, step3 docstring |
| V2 Heavy (FTH3 epoch 5, product inferencer + RM), input A (pooled affine onto training stats): RM 0.9714 / 0.9559 / 0.9675 / 0.9703 at 50/100/200/276 b; local 5/15 0.9715 / 0.9565 / 0.9688 / 0.9714; no Kakinada 0.9684 (RM), 0.9696 (local) at full; GRX / PCA30-RX no Kakinada at full 0.9611 / 0.9627 | research/experiments/had100-v2heavy/README.md, results/mauc_summary.json |
| V2 vs V1 per crop 47 / 52, Wilcoxon p = 0.5 (94 crops 45 / 49, p = 0.44); beats GRX and PCA30-RX at 200 and full, not at 50; 2-8 m, North America and Switzerland | had100-v2heavy README |
| 0.013 / 0.019 below the best at 200 b | 0.9869 - 0.9740; 0.9869 - 0.9675 |
