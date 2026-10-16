# Residual Mahalanobis scoring

**Purpose** - turn a foundation model's reconstruction residual into an anomaly score that measures
how *unusual* each pixel's error is against the scene's normal errors, not how large it is. Default
score for Indradhanu (`hyperspectral_segformer_mae`) in the `anomaly_scoring` action since 2026-10-15.

## Interfaces

```python
# app/utils/anomaly_detection/residual_mahalanobis.py
residual_mahalanobis(original, reconstruction, keep, d=30, local_window=(21, 61)) -> (H, W) float32
whitened_pca(residual, keep, d=30, seed=0) -> (d, H, W) float32
local_mean_score(X, keep, window=(21, 61)) -> (H, W) float32
# app/utils/anomaly_detection/scoring.py
compute_score(original, reconstruction, validity, method="residual_mahalanobis", local_window=(21, 61) | None)
```

Backend: `resolver.py` lists `residual_mahalanobis` first in Indradhanu's `scoring_methods` and makes it
`default_scoring_method`; `anomaly_scoring.py` accepts it (`_VALID_METHODS`) and per-model overrides can
still pick `combined`, `L1` or `SAM`; `_anomaly_scoring_run.py` renders it with the sqrt stretch.

## Data flow

```
cube (C,H,W), reconstruction (C,H,W), keep (H,W)
  r = cube - reconstruction
  covariance of r over up to 200,000 valid pixels (seed 0) -> top-30 eigenvectors
  X = (r - mean) projected and divided by sqrt(eigenvalue)          whitened, (30,H,W)
  ring sums by integral image: count and sum of X between the 21x21 guard and the 61x61 window
  score = ||X - ring mean||^2, or ||X||^2 where the ring has < 60 valid pixels; 0 outside keep
```

## Invariants

1. **Same numbers as the research code.** The functions are a numpy port of `whitened_pca` and
   `local_mean_scores` in `research/experiments/dual-window-mahalanobis/dw_core.py`; checked equal on a
   synthetic scene (rtol 1e-5), and through `compute_score` the bench reproduces v0.2.0 0.854 / 0.840 and
   F001 (epoch 52) 0.865 / 0.849 (local / global) on the four labelled PRISMA scenes.
2. **Deterministic.** The covariance sample uses a fixed seed.
3. **Well conditioned.** 30 whitened components, not all bands - plain RX over 150+ bands was dropped in
   2026-05 for near-singular covariances.
4. **Other methods unchanged.** `local_window` is read only by this method.

## Failure modes

| What | Seen as |
|---|---|
| fewer than 30 valid bands | `d` falls to the band count (eigh returns fewer components) |
| a pixel whose ring is mostly nodata | global-mean score for that pixel, not an error |
| near-zero eigenvalues (flat residual) | floored at 1e-12 before whitening |

## Decisions

- **Local mean with a global covariance (21/61)** as the default, over the global mean: +0.011 to +0.016
  mean AUC, better on every F001 scene. The window was the best of four in the original sweep on the same
  four scenes, so the margin is slightly flattering; `local_window=None` remains one argument away.
- **Measured against ten alternatives (2026-10-15, F001 epoch 52 and v0.2.0, parameters fixed before the
  run):** rank-normalised `combined`, SAM dark-pixel fade, local z-score of the score map, a
  difference-of-Gaussians blob boost, and +/-1 px shift suppression, alone and chained. None beat this
  on AUC; the blob boost caught slightly more objects in the top 1% (403 vs 383 for F001) while losing AUC.
- **A separate name from the classical `mahalanobis`** (RX on the cube, MNF-RX / thermal GRX) so the two
  never get confused in configs or the UI.
- **CPU cost accepted:** ~3-4 s per scene against ~46 s of reconstruction for F001.
- **Rejected for now:** full-covariance local rings (dual-window RX) - slower, and the local *mean* gave
  most of the gain in the research sweep.
