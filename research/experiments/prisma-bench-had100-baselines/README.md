# HAD100's published methods on our labelled PRISMA bench

Run 2026-11-03 for the AIGC Stage-III deck (slide A6 companion). Question: **how do the methods on HAD100's
leaderboard (Li et al., "You Only Train Once", IEEE TGRS 2023, Table V) do on our 909-object PRISMA bench, measured
exactly as Indradhanu V2 Heavy is measured?** Research only: no product code changed.

**Answer: none comes close to V2 Heavy.** The best HAD100 method here is CR (collaborative representation),
pAUC 0.127 with thick cloud excluded against V2 Heavy's 0.189 (difference -0.063 [-0.090, -0.035]); it ties the
no-model PCA-RX on objects in the top 1 % (376 vs 377, V2 Heavy 449) and roughly ties V2 Heavy at the very top
(top 0.1 %: 127 vs 145 cloud excluded, -2.0 [-4.7, +0.7] pts, unresolved). GRX, LRX and KCSR are below the product's
own classical MNF-RX on pAUC or objects. The HAD100 ranking (KCSR > CR > LRX > GRX) does not carry over to 30 m
PRISMA scenes: KCSR (approximated, below) and small-window LRX fall towards chance.

## Slide table

Mean over the 4 scenes for pixel measures; objects of 909. Cloud = FTH3 epoch 5's thick-cloud class map, scored 0
after scoring (as in the FTH3 report's Table 4). Chance: 93 objects at the top 1 %, 10 at the top 0.1 %.

| method | pAUC all / excl. | objects top 1 % all / excl. | top 0.1 % all / excl. | ROC-AUC all / excl. | d pAUC vs V2 Heavy, excl. [95 %] |
|---|---|---|---|---|---|
| **Indradhanu V2 Heavy + RM** | **0.175 / 0.189** | **422 / 449** | **126 / 145** | **0.866 / 0.869** | - |
| Indradhanu V2.0 (FT04) + RM | 0.162 / 0.174 | 415 / 427 | 135 / 152 | 0.853 / 0.857 | -0.015 [-0.027, -0.003] |
| PCA-RX (no model) | 0.128 / 0.139 | 361 / 377 | 74 / 77 | 0.839 / 0.842 | -0.050 [-0.064, -0.036] |
| CR 5/29, lambda 0.01 | 0.125 / 0.127 | 371 / 376 | 127 / 127 | 0.865 / 0.868 | -0.063 [-0.090, -0.035] |
| AutoAD (official code, tiled 2 x 2) | 0.103 / 0.108 | 336 / 344 | 60 / 66 | 0.818 / 0.820 | -0.082 [-0.107, -0.057] |
| MNF-RX (product classical) | 0.075 / 0.097 | 249 / 280 | 31 / 47 | 0.749 / 0.753 | -0.093 [-0.110, -0.075] |
| LRX 13/29 | 0.093 / 0.093 | 241 / 241 | 76 / 76 | 0.621 / 0.622 | -0.096 [-0.127, -0.065] |
| GRX, all bands | 0.079 / 0.081 | 273 / 282 | 65 / 67 | 0.753 / 0.756 | -0.108 [-0.130, -0.083] |
| GRX, first 100 bands | 0.065 / 0.067 | 228 / 233 | 67 / 69 | 0.732 / 0.735 | -0.122 [-0.146, -0.097] |
| GRX, first 50 bands | 0.049 / 0.051 | 176 / 178 | 55 / 57 | 0.693 / 0.696 | -0.138 [-0.164, -0.114] |
| KCSR 3/29 (approx.) | 0.042 / 0.042 | 195 / 195 | 76 / 76 | 0.561 / 0.562 | -0.147 [-0.173, -0.120] |
| LRX 5/15 | 0.029 / 0.029 | 71 / 71 | 39 / 39 | 0.362 / 0.363 | -0.160 [-0.188, -0.132] |

Full tables (every CR lambda, paired differences for objects at the top 1 % and 0.1 %, per scene): `results.md`;
everything incl. tiers and per-scene values: `results.json`, `per_scene.csv`.

Charts (slide A6 style, 80 x 47 mm, `make_chart.py`): `chart_prisma_had100.pdf/.png` (pAUC, cloud excluded, dot =
pooled value, line = 95 % object-bootstrap interval - not the 50-200 band span of the HAD100 chart) and
`chart_prisma_had100_objects.pdf/.png` (objects in the top 1 %, chance line 93). Copies under the same names in
`research/presentation/aigc-stage3/v2/figures/`; no existing chart was touched.

## Checks

- Reference numbers reproduced exactly from the same maps / code: V2 Heavy 0.1892 / 449 (cloud excl.) and
  0.1749 / 422; PCA-RX 0.1390 / 377; MNF-RX 0.0966 / 280 (`hypersl-anomaly-bench/README.md`, FTH3 `eval_report.md`).
- FT04 here uses FTH3's thick-cloud mask like every row (0.174); the deck's 0.170 is FT04 with its own masks.
- V1 (v0.2.0): its stage-3 cached maps could be matched to the stage-3 per-scene pAUC on set3 only (the bench's
  valid mask has changed since that run), so V1 is not in the table rather than shown on mismatched pixels.

## Input (identical for every method)

`app.evaluation.datasets.PrismaBench`, exactly what the evaluator's PCA-RX sees: the 4 labelled PRISMA L2D scenes in
`C:/Users/FITT/Documents/stage1_datasets` (Set-1..4), native bands after the product exclusions (185 / 186 / 186 /
186 bands), reflectance, nearest-valid filled, the evaluator's valid mask (~1.0 M valid pixels per scene); measures by
`app.evaluation.metrics.scene_metrics`, intervals by `app.evaluation.significance` (paired object bootstrap, 2,000
draws, seed 0). The chart's per-method interval is the same object bootstrap of that method alone.

## Methods and parameters

Windows are full widths (inner / outer); HAD100's best windows are its Table IV (tuned on its own test set).

| method | implementation | parameters | GPU time per scene |
|---|---|---|---|
| GRX | `methods.grx`: scene mean / covariance (float64), Cholesky | ridge 1e-6 x mean eigenvalue (tr S / C). Plain RX was dropped from the *product* for near-singular covariance; the ridge is what makes it computable. Also on the first 50 / 100 bands by wavelength (HAD100's "first N bands" of an ascending axis; 200 > 186 so "all" stands in for 200) | 1-3 s |
| LRX | `methods.lrx`: ring mean and ring covariance per pixel from float64 integral images of x and x x^T (all 17,205 band pairs) over row strips; Cholesky | 13/29 (HAD100 Table IV) and 5/15; ridge 1e-4 x tr(S_ring)/C. 5/15 has 200 ring samples for 185 bands, so its covariance is ridge-dominated | 150-167 s (13/29), 120-135 s (5/15) |
| CR (CRD) | `methods.crd`, Li & Du 2015 eq. 6 (as `func_CRD.m` in GT-HAD's copy): alpha = argmin ||y - X a||^2 + lambda ||Gamma_y a||^2, Gamma_y = diag(||y - x_i||), score ||y - X alpha||; computed exactly as (G + I)^-1 y with G = X D^-1 X^T (push-through identity) | 5/29 (HAD100 Table IV), lambda in {1e-3, 1e-2, 0.1, 1}; best on the bench is 1e-2 (pAUC) - **chosen on the bench, optimistic**, as HAD100 tuned on its test set. Spectra projected onto the top-30 eigenvectors of the uncentred second moment (99.97-99.99 % of energy kept), see caveats | 20 s for all 4 lambdas |
| KCSR | `methods.kcsr`, closest documented version of Ling et al. 2019 (the paper HAD100 cites as [17]): RBF-kernel representation of y by its ring under the simplex constraint (a >= 0, sum a = 1), projected gradient, score = feature-space residual k(y,y) - 2 k_y^T a + a^T K a | 3/29 (HAD100 Table IV); sigma^2 = mean squared distance from y to its ring; 100 iterations, step 1 / max row sum of K; ring subsampled on a stride-2 lattice (224 of 832 samples); same top-30 projection. **An approximation, labelled "approx."** | 110-280 s |
| PCA-RX, MNF-RX | the evaluator's `pca_rx` (d 30, ring 21/61) and the product's `MNFCompressionDetector` (10 components), as in hypersl-anomaly-bench | - | 4 s, 20 s |
| V2 Heavy, FT04 | cached RM maps of `configs/eval/fth3.json` (`C:/es/eval/fth2_ftb2`), found by the runner's own cache key | - | - |

## Deep methods

- **AutoAD** (official code, https://github.com/RSIDEA-WHU2020/Auto-AD, cloned into an isolated scratch folder, read
  before running; `models/skip.py` + `utils/common_utils.py` imported unmodified, `AD_main_loop.py`'s loop re-typed in
  `step2_autoad.py` with every setting unchanged). Input min-max scaled to [0, 1] over valid pixels (the sigmoid output
  needs it). **Tiled 2 x 2 (~600 x 640 px per tile, one Auto-AD run per tile)**: a full ~1200 x 1200 x 185 scene is too
  large for one run in the time available. Set1: 611 s for the 4 tiles (early stop after 401 / 401 / 701 / 951
  iterations), RTX 4000 SFF Ada alone.

  All 4 scenes: 611 / 889 / 936 / 675 s. Pooled, cloud excluded: pAUC 0.108 [0.086, 0.128], 344 objects in the top 1 %
  (V2 Heavy 449: -11.6 [-15.0, -8.3] pts), 66 in the top 0.1 %, ROC-AUC 0.820 - between CR and MNF-RX. Per scene
  (pAUC / top 1 %): 0.103 / 81, 0.090 / 94, 0.060 / 52, 0.177 / 117. `partial_AutoAD.json` is the earlier set1-only
  check (same numbers on set1).
- **RGAE**: skipped - no official Python release found (the paper's code is not on GitHub).
- **LREN**: skipped - the official code (github.com/xdjiangkai/LREN) is TensorFlow 1.5 / CUDA 10.2, and its LRR
  dictionary step is built over all pixels (infeasible at ~1 M pixels without re-engineering it).
- **AETNet, AETNet-KCSR**: skipped for time - AETNet needs training on anomaly-free PRISMA tiles from scenes outside
  the bench (and AETNet-KCSR then needs the exact KCSR, which is itself only approximated here).

## Caveats

1. **CR and KCSR run on a 30-dimensional projection**, not on the 185 bands (an exact per-pixel CR on 185 bands is
   ~2e13 flops per scene, ~45 min, and exact KCSR builds an 832 x 832 kernel per pixel). The projection keeps
   99.97-99.99 % of the spectral energy, so inner products and distances are nearly unchanged, but the discarded
   0.01-0.03 % may hold weak anomaly signal, and a 30-dimensional regression over 816 ring samples fits more easily
   than a 185-dimensional one. Both may be pessimistic for that reason.
2. **Parameters are HAD100's, not tuned for 30 m PRISMA** (except CR's lambda, chosen on the bench). HAD100's images
   are AVIRIS-NG crops at metre-scale pixels where targets are several pixels; our objects are mostly 1-3 px and
   clustered, so dual windows of 3-13 px inner width see neighbouring targets in their rings. LRX 5/15 and KCSR land
   near or below chance on ROC-AUC (0.36, 0.56): rings too small and sample-starved (LRX 5/15) or a ring representation
   that also reconstructs the target (KCSR on 30-dim spectra).
3. **Plain GRX needs a ridge here** (documented above); the scores are stable (no 1e11 distances).
4. **Four scenes.** The intervals are object bootstraps within the four scenes; they do not cover scene-to-scene
   variation. Per scene CR beats V2 Heavy on objects in set1 (115 vs 106) and loses on the other three
   (`results.md`).
5. ROC-AUC is the product's sampled `compute_roc` (as every earlier table).

## Reproduce

Score maps (float32 npz, per scene and method) and logs are outside the repo in
`C:/Users/FITT/Documents/stage1_datasets/had100_baselines/` (`<set>/<method>.npz`, `timing.json`,
`timing_autoad.json`, `step*_run*.log`). Run on the Windows product venv, one process at a time (two scene loads at
once exhausted the commit limit; C: had ~250 MB free during the run).

```bash
C:/av/Scripts/python.exe -X utf8 step1_classical.py                 # GRX, LRX, CR, KCSR  (~8 min / scene)
C:/av/Scripts/python.exe -X utf8 step2_autoad.py --repo <Auto-AD clone> --tile 640
C:/av/Scripts/python.exe -X utf8 step3_evaluate.py [--scenes set1 --tag _set1]   # results.json, per_scene.csv
C:/av/Scripts/python.exe -X utf8 summarise.py [--tag _set1]                     # results.md
PYTHONPATH=C:/pdfr C:/av/Scripts/python.exe -X utf8 make_chart.py               # the two charts
```
