# FT02 - Reconstruction adaptation on EnMAP + PRISMA + AVIRIS-NG

**Status:** done 2026-10-16 05:12-09:27 UTC (4.25 h, ~$12), exit 0; gate not met. Was: prepared 2026-10-16; runs after FT01, on the user's go.

## Question
Does continuing F001's masked reconstruction on the three real sensors, at their own bands and pixel sizes, keep
or improve anomaly detection while teaching the wavelengths SpectralEarth lacked?

**Expected, written before launch:** PRISMA bench under residual Mahalanobis at or slightly above F001's 0.861
(epoch 60); reconstruction on each sensor's test split falls over the run; AVIRIS-NG (3-8 m pixels, never seen)
starts worst and improves most; the trained-wavelength record grows to cover PRISMA's 1300-1500 and 1730-1960 nm
bands, so the bench reads all 185 PRISMA bands instead of 165.

## Setup
| | |
|---|---|
| config | [config.json](config.json): `mode: adapt`, init F001 epoch 109 (`F001_ep109.pt`; F001 stopped there 2026-10-16, detection flat since epoch 60), shares EnMAP 0.40 / PRISMA 0.35 / AVIRIS-NG 0.25, LR 1e-4 cosine (1 warm-up epoch), 20 epochs x 30,000 samples, band augmentation 0.8 + 3 nm jitter, width weighting |
| validation | each sensor's test split: L1 / SAM per sensor (`finetune_metrics.csv`); best.pt by the mean full-SAM score |
| bench | PRISMA four scenes, own bands, residual Mahalanobis (default) and combined, at epochs 5, 10, 20 (laptop) |

## Pass criteria (gate to FT03)
1. Bench mean ROC-AUC (residual Mahalanobis) >= 0.861 at the best checkpoint.
2. No scene below F001's epoch-60 score by more than 0.03.
3. Each sensor's test reconstruction lower at the end than after epoch 1.

## Results

PRISMA bench, own bands (`C:/es/ep109/score.py`, the product's `compute_score`), ROC-AUC set1 / set2 / set3 / set4 / mean
(objects caught of 909). Reference F001 epoch 109: residual Mahalanobis 0.862 (400), combined 0.706 (201).

| epoch | val L1 / SAM: EnMAP, PRISMA, AVIRIS-NG | residual Mahalanobis (gate >= 0.861) | combined |
|---|---|---|---|
| 1 | 0.0183 / 4.81, 0.0224 / 7.30, 0.0179 / 7.94 | 0.929 / 0.874 / 0.775 / 0.878 / **0.864** (413) | 0.827 / 0.764 / 0.795 / 0.653 / **0.760** (332) |
| 13 | 0.0165 / 4.16, 0.0193 / 6.04, 0.0164 / 6.49 | 0.930 / 0.846 / 0.759 / 0.885 / **0.855** (400) | 0.938 / 0.807 / 0.726 / 0.766 / **0.809** (330) |
| 8 (best.pt) | 0.0164 / 4.17, 0.0200 / 5.93, 0.0159 / 6.07 | 0.931 / 0.852 / 0.757 / 0.879 / **0.855** (401) | 0.943 / 0.781 / 0.736 / 0.834 / **0.824** (335) |
| 20 (last.pt) | 0.0162 / 4.07, 0.0199 / 6.31, 0.0156 / 6.19 | 0.934 / 0.848 / 0.748 / 0.882 / **0.853** (410) | 0.943 / 0.799 / 0.735 / 0.780 / **0.814** (320) |

**Gate: not met** - residual Mahalanobis 0.855 / 0.853 against >= 0.861 (F001 ep109 0.862). Reconstruction fell on
all three sensors (SAM epoch 1 -> 20: EnMAP 4.81 -> 4.07, PRISMA 7.30 -> 6.31, AVIRIS-NG 7.94 -> 6.19). Combined rose
0.706 -> 0.824 (best.pt), closing most of its gap to residual Mahalanobis.

Per-sensor validation see-saws: epochs 11-12 and 15-17 PRISMA improves while EnMAP and AVIRIS-NG worsen (epoch 17:
PRISMA SAM 5.04, AVIRIS-NG 8.24), then reverts. Validation tiles are fixed and unshuffled, so it is the weights.

Epoch 1 reads PRISMA through a record that now holds PRISMA's own wavelengths, so it reads all 185 of the bench scenes' bands where F001 read 165: the 20 F001 dropped
(SpectralEarth's gaps) are scored too; part of any change is those bands, not just better weights.
## Observations

- **BatchNorm sensitivity (2026-10-16, `C:/es/bn/adabn.py`, FT02 epoch 20).** The model has one BatchNorm layer (the
  stem's, in front of the encoder). Re-estimating its statistics on each PRISMA bench scene moves them 0.35-0.76 sd
  from the stored ones. Residual Mahalanobis barely changes (0.853 -> 0.858 mean, 410 -> 402 objects); combined swings
  hard (set1 0.943 -> 0.755, set4 0.780 -> 0.642). So the BatchNorm statistics move raw reconstruction error a lot -
  consistent with the per-sensor see-saw, since sensor-pure batches leave the stats set by the last batches of an
  epoch - but the whitening in residual Mahalanobis absorbs most of it, so they do not explain the gate miss.
  Fix for any rerun: keep BatchNorm in eval mode during adapt/joint (as probe does), or re-estimate it on a
  balanced sensor mix before each validation. Per-scene re-estimation is not a product option (combined falls).

- **Edge diagnosis (2026-10-16, F001 ep109 and FT02 ep13, `C:/es/edges/edges.py`, rules fixed before results).**
  Edge strength = max mean-band difference to a valid 4-neighbour; strong = scene top 10%. Of the top-1% residual
  Mahalanobis false positives, 49-68% sit on strong edges (chance 10%) - edges are the main false-alarm source.
  But 29-79% of ground-truth pixels are also strong-edge pixels (small targets are edges), and 39-91% of the hits.
  An edge-corrected score (rank minus the median rank of its edge-strength bin, 20 bins) collapses detection
  (0.86 -> 0.67 mean). Earlier score-map sharpening on F001 ep52 (`C:/es/postscore/`) agrees: difference-of-
  Gaussians F(C) catches 403 objects vs D's 383 but drops AUC 0.865 -> 0.718; ring z E(C) 0.790; one-pixel
  shift-min G-C 0.845. Conclusion: no score-map edge correction beats D; edge strength cannot tell a field
  boundary from a target. A "sharpening" gain would have to come from the reconstruction (blind-spot head), not
  from post-processing.
