# Indradhanu V2 Heavy on HAD100 (official protocol)

2026-11-03. Research only. No production code changed. For deck appendix slide A6.

**Result:** V2 Heavy does **not** beat V1 on HAD100. Primary row (V2 + RM, pooled-affine input, full
276 good bands): **0.9703**, against V1's **0.9740**. Per crop, 47 wins and 52 losses, Wilcoxon p = 0.5,
so this is a statistical tie leaning towards V1. The band subsets do not help either: V2 scores 0.9714
at 50 bands, 0.9559 at 100 and 0.9675 at 200.

## Protocol (same as V1's step 6)

The harness is [step6_had100_protocol.py](../dual-window-mahalanobis/step6_had100_protocol.py)
(README §4b), reused by import.

- 100 test crops of 64 × 64, at the offsets in HAD100 `main.py`'s `crop_dict`.
- The 276 good bands, `np.r_[15:109, 118:145, 158:187, 227:274, 328:407]`, then the first 50, 100
  and 200 of them.
- mAUC is the mean per-crop pixel ROC-AUC.
- Every pixel is scored. No mask is applied.
- Data: `C:/es/benchmarks/had100/HAD100` (AVIRIS-NG L1B radiance). Wavelengths are read per tile;
  across the 94 tiles they differ by at most 0.35 nm per band.

## Model

- `C:/es/models/FTH3_ep5.pt`, sha256 `e957e29f…77431423`. This is Indradhanu V2 `v2.1.0` (Indradhanu
  Heavy, FTH3 epoch 5). It is the same checkpoint as `spectral_stem_finetune_v2.1.0_epoch5.pt`,
  whose product parity is shown in `../indradhanu-finetune/FTH3-heavy-joint-more2/README.md`.
- The run uses the product code: `SpectralStemInferencer`, with patch 128, stride 32, batch 8 and two
  complementary token masks. `read_bands` keeps the bands the model was trained on; all 50, 100, 200
  and 276 requested bands were kept.
- Scorers come from `app/utils/anomaly_detection/residual_mahalanobis.py` and `compute_score`.
- V2 reads bands by wavelength, so each band setting is a separate V2 run with exactly those bands.
  V1 can only read its 165-band grid, so one full-spectrum V1 score fills every column.
- Crops are reflect-padded by 32 px to 128 and cropped back, as V1 did.
- Ran on the local RTX 4000 Ada with the `C:/av` venv. Runtime was 236 s for 100 crops × 4 band
  settings × 2 scalings, including the segmentation pass.

## Input scaling (HAD100 is radiance; V2 was trained on reflectance)

- **A, pooled affine (primary; V1's handling).** Each band is mapped affinely onto the checkpoint's
  own training mean and std at that wavelength (the `stats.*` buffers). There is one fit, pooled over
  all 94 full tiles. V1 used the same mapping onto `hyperspectral.json` on the 165-band grid.
- **B, blackbody TOA.** rho = pi·L / E_sun(lambda), where E_sun is a 5778 K Planck sun scaled to
  1361 W/m². No irradiance table is available offline. Because cos(SZA) is unknown, one scalar per tile
  matches the tile's mean to the training mean. This keeps the TOA spectral shape, atmosphere included.
  It scores lower than A at every setting.

## Results: mAUC

All 100 crops:

| method | 50 | 100 | 200 | full (276) |
|---|---|---|---|---|
| **V2 Heavy + RM global [A]** | 0.9714 | 0.9559 | 0.9675 | **0.9703** |
| V2 Heavy + RM local 5/15 [A] | 0.9715 | 0.9565 | 0.9688 | 0.9714 |
| V2 Heavy + RM product 21/61 [A] | 0.9712 | 0.9560 | 0.9676 | 0.9704 |
| V2 Heavy L1 [A] | 0.7130 | 0.6998 | 0.7783 | 0.8597 |
| V2 Heavy + RM global [B] | 0.9690 | 0.9371 | 0.9541 | 0.9612 |
| V2 Heavy + RM local 5/15 [B] | 0.9696 | 0.9382 | 0.9557 | 0.9626 |
| **V1 v0.2.0 + RM global** (165 grid, one score for all columns) | 0.9740 | 0.9740 | 0.9740 | **0.9740** |
| V1 v0.2.0 + RM local 5/15 | 0.9748 | 0.9748 | 0.9748 | 0.9748 |
| V1 v0.2.0 L1 | 0.8445 | 0.8445 | 0.8445 | 0.8445 |
| GRX (ours = paper) | 0.9799 | 0.9714 | 0.9649 | 0.9634 |
| PCA30-RX | 0.9797 | 0.9603 | 0.9636 | 0.9648 |

Without the 6 Kakinada crops (94):

| method | 50 | 100 | 200 | full (276) |
|---|---|---|---|---|
| **V2 Heavy + RM global [A]** | 0.9710 | 0.9533 | 0.9654 | **0.9684** |
| V2 Heavy + RM local 5/15 [A] | 0.9712 | 0.9539 | 0.9668 | 0.9696 |
| V2 Heavy + RM global [B] | 0.9687 | 0.9332 | 0.9512 | 0.9587 |
| **V1 v0.2.0 + RM global** | 0.9724 | 0.9724 | 0.9724 | **0.9724** |
| V1 v0.2.0 + RM local 5/15 | 0.9732 | 0.9732 | 0.9732 | 0.9732 |
| GRX | 0.9790 | 0.9695 | 0.9627 | 0.9611 |
| PCA30-RX | 0.9786 | 0.9579 | 0.9613 | 0.9627 |

The full table, including the B variants and L1, is in `results/summary.txt`. Paired comparison of
V2 [A] RM full against V1 RM: 47 wins / 52 losses on all 100 crops (Wilcoxon p = 0.5), and 45 / 49 on
the 94 non-Kakinada crops (p = 0.44).

## Observations and caveats

- At 200 and at full bands, V2 + RM beats GRX and PCA30-RX. At 50 bands it is below both. Its
  100-band dip (0.956) is shared with PCA30-RX (0.960). On HAD100, V2 is no better than V1 and about
  equal to the classical baselines.
- V2's reconstruction itself is much better than V1's on this data. L1 at full bands is 0.860 for V2
  against 0.845 for V1. Residual Mahalanobis absorbs that difference.
- The domain gap is large. The data is radiance, not reflectance, at 2–8 m GSD where the model
  trained at ~30 m, over North America and Switzerland, which the model never saw. Neither scaling
  removes the atmosphere.
- The Kakinada crops are near-domain for 6 of 100 and make little difference: removing them lowers
  every row by about 0.002.
- Segmentation head (not applied): on variant-A inputs, V2's thick-cloud class fires on 27 of 100
  crops (68k of 410k pixels), and on 11 crops under B. These are bright surfaces seen in radiance, not
  cloud. V1's protocol scored all pixels, so this run does too. A product run on these scenes would
  zero those pixels (`results/seg_head_classes_per_crop.json`).

## Files

- `run_had100_v2heavy.py`: the run, about 4 minutes on GPU.
- `summarise.py`: builds the tables and the Wilcoxon test.
- `results/had100_v2heavy_per_crop.json`: per-crop AUC for every variant, scorer and band setting.
- `results/v1_step6_per_crop.json`: a copy of V1's per-crop JSON.
- `results/mauc_summary.json`, `results/summary.txt`: the summary tables.
- `results/seg_head_classes_per_crop.json`, `results/runtime.json`: the segmentation report and the
  runtime.
- `run.log`: the log of the run.
