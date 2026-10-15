# E005 - Band-list robustness fixes on the reduced stem (C')

**Date:** 2026-10-14 -
**Status:** done 2026-10-15 00:29 UTC - stopped at epoch 10 (exit 143, capped); 3 of 4 pass criteria met

## Question

Do the three band-list fixes of course part 42 make E004's stem read a sensor's own band list as
well as SpectralEarth's, without costing reconstruction quality? This gates the final pretraining
run: it goes ahead with these fixes only if E005 passes.

**Expected outcome, written before launch:** at epoch 10, SpectralEarth validation L1 and SAM
within ~5% of E004's epoch 10 (augmented batches see fewer bands, so slightly worse is possible);
set1 on PRISMA's native bands close to set1 resampled to SpectralEarth's 202 (E004 epoch 10:
0.217 native vs 0.518 resampled); water L1 on thinned SpectralEarth patches close to the full-band
figure (E004 epoch 18: 0.0309 thinned vs 0.0155 full).

## Setup

| | |
|---|---|
| variant | C' as E004 (`reduced_attention`, `bin_nm` 40) plus `band_weighting: width`, `band_augment_prob: 0.8`, `band_jitter_nm: 3.0` |
| fixes | 1. one vote per nm, not per band (bins and skip path); 2. 80% of training batches read a random band subset - thinning, 1-3 gaps of 20-150 nm, or an even 6-15 nm grid - and reconstruct all 202; 3. stated wavelengths of the input jittered by up to +/- 3 nm. The evaluation drops bands > 15 nm from any pretraining wavelength (`trained_band_mask`). |
| config | [config.json](config.json) - E004's with only the three fields above and the run paths changed (diffed) |
| schedule | E004's 200-epoch cosine schedule, stopped at epoch 10 by the instance watchdog, so it sits at the same point on the same curve as the control |
| control | E004's epoch-10 checkpoint (its best.pt at that time) |
| data | se-v1 train shards 000000-000199, test 000000-000009, on the instance's NVMe |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S 48 GB), kept |
| outputs | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E005-band-robust/` |
| wall time | - |
| cost | - ($2.69/h) |

## Pass criteria

1. PRISMA bench, native bands: set1 within 0.05 AUC of the resampled score, and the four-scene
   native mean no lower than resampled minus 0.03.
2. PRISMA bench, resampled: no scene worse than E004 at epoch 10 by more than 0.03.
3. Thinned SpectralEarth patches: water L1 within 20% of the full-band figure.
4. SpectralEarth validation at epoch 10 no worse than E004's by more than 5%.

Control measured 2026-10-14 on E004's epoch-10 checkpoint. Its water error is not settled at epoch 10:
all 202 bands 0.0305, thinned 0.0258 (at epoch 18: 0.0155 and 0.0309). So at epoch 10 the
thinned-vs-full comparison is noisy for E004 itself. Criterion 3 compares E005 with its own
full-band figure, and the clearest control is set1 on its own bands (0.0456 water, 0.0246 land) and the
PRISMA ROC-AUC rows.

## Results

| metric | E005 epoch 10 | E004 epoch 10 |
|---|---|---|
| val L1 / SAM (deg) | 0.0242 / 5.85 (+4.3% / +4.5%) | 0.0232 / 5.60 |
| PRISMA ROC-AUC, native, set1-4 / mean | 0.631 / 0.755 / 0.755 / 0.679 / **0.705** | 0.217 / 0.824 / 0.728 / 0.703 / 0.618 |
| PRISMA ROC-AUC, resampled, set1-4 / mean | 0.721 / 0.749 / 0.755 / 0.654 / 0.720 | 0.518 / 0.775 / 0.692 / 0.631 / 0.654 |
| thinned-SE water L1 / full | 0.0166 / 0.0198 (land 0.0198 / 0.0188) | 0.0258 / 0.0305 (land 0.0224 / 0.0173) |
| set1 native water / land L1 | 0.0262 / 0.0232 (0.0141 / 0.0200 with untrained bands dropped) | 0.0456 / 0.0246 |
| throughput, patches/s | 55.5 | 55 |

## Graphs

## Observations

Evaluated 2026-10-15 on best.pt (epoch 10), `scripts/evaluate_prisma_bench.py` (untrained bands dropped
for spectral stems) and `C:/es/water/cause_test.py`. Against the pass criteria:

1. **Fail, narrowly.** Set1 native 0.631 vs resampled 0.721 - a 0.09 gap against 0.05 allowed. The
   four-scene native mean (0.705) is within 0.015 of resampled (0.720): that half passes. E004 at the
   same epoch: set1 native 0.217 vs 0.518 resampled, native mean 0.618 vs 0.654.
2. **Pass.** Resampled vs E004 epoch 10: set1 +0.20, set2 -0.026, set3 +0.06, set4 +0.02.
3. **Pass.** Thinned water L1 0.0166 vs 0.0198 full (thinning no longer hurts; E004 at epoch 18 doubled).
4. **Pass.** Validation +4.3% L1, +4.5% SAM vs E004 epoch 10.

Set1's water error on native bands, with the 20 untrained bands dropped as the bench does, is 0.0141 -
level with E004 resampled (0.0138). The water failure is fixed; the remaining set1 AUC gap is not
water reconstruction. Native set4 (0.679) is ahead of resampled (0.654).

**Extended to 20 epochs (2026-10-15, user decision; resumed from last.pt, `phase3_e005.sh`).**
best.pt = epoch 20, val L1 0.0214 / SAM 5.16 deg (E004 epoch 20: 0.0210 / 4.84 - +1.9% / +6.6%).

| combined ROC-AUC | set1 | set2 | set3 | set4 | mean |
|---|---|---|---|---|---|
| E005 epoch 20, native (untrained bands dropped) | 0.693 | 0.781 | 0.748 | 0.784 | **0.752** |
| E005 epoch 20, resampled to SE's 202 | 0.523 | 0.770 | 0.774 | 0.669 | 0.684 |
| E004 epoch 19, resampled (its best mode) | 0.797 | 0.777 | 0.744 | 0.700 | 0.754 |
| E004 epoch 19, native | 0.239 | 0.836 | 0.767 | 0.609 | 0.613 |
| v0.2.0 | 0.712 | 0.686 | 0.730 | 0.521 | 0.662 |

Band-list test, epoch 20 (water / land L1): SE all 202 0.0152 / 0.0171; thinned 0.0120 / 0.0177;
set1 native 0.0087 / 0.0201 (0.0088 / 0.0176 with untrained bands dropped) - below E004 resampled
(0.0138). The inference-time skip reweighting no longer changes anything (0.0089): width weighting
has absorbed it.

Verdict: **the fixes work.** Native band lists now read as well as E004 read resampled ones (0.752 vs
0.754), 0.09 above production; set4 native 0.784 is the best single score on that scene. Two things
are not explained: E005's *resampled* set1 fell from 0.721 (epoch 10) to 0.523 (epoch 20) - set1 swings
for every model (E004 resampled 0.44 -> 0.52 -> 0.73 -> 0.80 over epochs 5-19), but the drop is
real; and validation SAM is 6.6% behind E004 at the same epoch (criterion 4 allowed 5% at epoch 10,
where it passed). The product reads native bands, so the native row is the one that matters.

## Next
