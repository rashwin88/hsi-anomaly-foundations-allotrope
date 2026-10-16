# 5. Classical detectors, scoring, and material matching

The non-neural half of anomaly detection. These need no training and no checkpoint — they
estimate the scene's own statistics fresh, every time.

## The RX idea

Model the scene background as one multivariate Gaussian over the bands. Score each pixel by
its **Mahalanobis distance** from that background:

```
score(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)
```

`μ` and `Σ` come from the scene itself. A pixel far from the bulk of the distribution —
after accounting for how bands co-vary — is anomalous. That's it.

## The detectors

All live in `app/detectors/`, subclass `AnomalyDetector`, and return an `(H, W)` score map
with `NaN` at invalid pixels. `fit()` learns background stats, `detect()` scores.

| Detector | Background | Notes |
|---|---|---|
| `GlobalRXDetector` | whole scene | Two-stage band filter, then a coverage-based spatial mask |
| `ThermalGRXDetector` | whole scene | Single band, so RX collapses to a squared z-score — computed directly, since `spectral.rx` needs ≥2 dimensions |
| `LocalRXDetector` | a ring around each pixel | Outer window 25, inner guard 5. Batched Mahalanobis on GPU |
| `MNFCompressionDetector` | whole scene, in MNF space | **The one to use on hyperspectral** |
| `MNFCompressionLRXDetector` | ring, in MNF space | |
| `StatisticalEnsembler` | — | GRX + LRX, CDF rank-normalised, fused by product/max/mean. Not registered; notebook-only |

### Why MNF, and why plain RX is banned on hyperspectral

With 165 bands and a limited number of clean pixels, `Σ` becomes near-singular. Inverting it
amplifies noise, and distances explode to ~1e11 — the scores become meaningless.

**MNF** (Minimum Noise Fraction) fixes this: estimate the *noise* covariance from
neighbouring-pixel differences, whiten by it, then take the top ~10 components by
signal-to-noise. RX then runs in a well-conditioned 10-dimensional space.

Plain RX on hyperspectral was **deliberately removed on 2026-05-11** for exactly this
reason. Don't reintroduce it — use MNF-RX.

### One subtlety in the run path

The worker narrows the background region *after* `fit()` and *before* `detect()`, by ANDing
the upstream `keep_mask` into the detector's `_spatial_mask`. That keeps clouds, water and
shadow out of the covariance estimate. It reaches into a private attribute — ugly, but
deliberate and documented at the call site.

## Cloud masking

`B10AdaptiveCloudMasker` (`app/statistical_models/`) fits a 5-component Gaussian mixture to
the Landsat B10 temperature distribution and labels any cluster whose mean falls more than
12 °C below the scene median as cloud. Clouds are cold; that's the whole signal. It clips at
the 95th percentile first so genuinely hot anomalies don't drag the cluster means around.

## Scoring — turning residuals into a heatmap

`app/utils/anomaly_detection/scoring.py`. This is the shared step for **any** reconstruction
model: given the original cube, the reconstruction, and a validity mask, produce one
`(H, W)` score map.

| Method | Measures | Use when |
|---|---|---|
| `L1` | mean absolute residual | default, any band count |
| `MSE` | mean squared residual | matching an MSE-trained model |
| `SAM` | angle between spectra | multi-band only; catches *shape* change |
| `combined` | `w·L̃1 + (1−w)·S̃AM`, `w=0.5` | Indradhanu's default until 2026-10-15 |
| `residual_mahalanobis` | how *unusual* the residual's direction is (below) | **default for Indradhanu** |

**L1 and SAM carry independent information.** L1 is a *magnitude* signal, SAM a *shape*
signal. A bright cloud has large L1 but small SAM. A chemically odd material with normal
brightness but an unusual absorption profile has small L1 but large SAM. Reporting both lets
the operator choose which kind of anomaly matters.

Each is normalised by its maximum over **valid pixels only**, then invalid pixels are zeroed.

### Residual Mahalanobis - the default for Indradhanu

L1, SAM and `combined` measure the residual's *size*. A scene's normal residuals are large along a
few shared directions - overall brightness, haze, PRISMA's detector striping, dark water - and
tiny along the rest, so size alone rewards ordinary pixels that sit on a common error direction.
`residual_mahalanobis` (`app/utils/anomaly_detection/residual_mahalanobis.py`) projects the
residual onto its top 30 principal components, whitens each by its spread over the scene, and
scores each pixel by its squared distance from the mean of a ring between 21 and 61 pixels
around it (`local_window=None`: the scene mean). Common error directions are discounted; an error
in a direction the background never takes scores high even when it is small.

Measured 2026-10-15 on the four labelled PRISMA scenes, through `compute_score` on the bench's
reconstructions (mean ROC-AUC; objects caught in the top 1% of 909):

| | set1 | set2 | set3 | set4 | mean | caught |
|---|---|---|---|---|---|---|
| v0.2.0, `combined` | 0.712 | 0.686 | 0.730 | 0.521 | 0.662 | 194 |
| v0.2.0, `residual_mahalanobis` | 0.942 | 0.847 | 0.750 | 0.877 | **0.854** | 403 |
| F001 epoch 52, `combined` | 0.731 | 0.755 | 0.677 | 0.698 | 0.715 | 208 |
| F001 epoch 52, `residual_mahalanobis` | 0.929 | 0.859 | 0.787 | 0.885 | **0.865** | 383 |

The global-mean variant scores 0.840 / 0.849. The 21/61 window was the best of four in the
original sweep on these same scenes (`research/experiments/dual-window-mahalanobis/`), so the
local variant is slightly favoured here. About 3-4 s per scene on CPU. Scores are squared
distances with a heavy tail: the score PNG uses the sqrt stretch, and thresholds stay percentiles.
Spatial post-filters (local z-scores, blob filters, +/-1 px shift suppression) were tested on top
and did not help (comparison of 11 scoring methods, 2026-10-15; see docs/lld/residual-mahalanobis.md).

### SAM on dark water

SAM is scale-free, which is also its weakness: on a near-black spectrum (open sea, ~2%
reflectance) a residual of 0.01 is a large *fraction* of the signal, so the angle is large.
On PRISMA set1 (75% sea) the median angle is 0.34 rad on sea, 0.13 on land and 0.18 on the
labelled anomalies (F001) - SAM ranks ordinary water above the targets.

`compute_score(..., sam_dark_fade=t)` multiplies the angle by `clip(mean_reflectance / t, 0, 1)`
(mean over bands of the input spectrum), fading SAM linearly to 0 below `t` and leaving
pixels at or above `t` untouched. It applies to `SAM` and `combined`. The default is `None`
(plain angle, byte-identical to before), and the product action does not pass it yet - see
[known issue 9](09-known-issues.md).

Measured 2026-10-15 on the 4 labelled PRISMA scenes, bench inference
(`scripts/evaluate_prisma_bench.py`), `t = 0.05` chosen up front, not tuned:

| combined ROC-AUC | set1 | set2 | set3 | set4 | mean |
|---|---|---|---|---|---|
| v0.2.0, plain | 0.712 | 0.686 | 0.730 | 0.521 | 0.662 |
| v0.2.0, fade 0.05 | 0.891 | 0.686 | 0.733 | 0.521 | 0.708 |
| F001 epoch 14, plain | 0.430 | 0.765 | 0.720 | 0.724 | 0.660 |
| F001 epoch 14, fade 0.05 | 0.683 | 0.765 | 0.726 | 0.724 | 0.725 |

Only set1 has water, so the three land scenes show the fade is harmless, not that 0.05 is
right. The sensitivity is steep on set1 (v0.2.0 combined: 0.830 at 0.03, 0.908 at 0.08; F001:
0.524 / 0.810) - a reason to validate on more water scenes before picking a default.

### Thresholding is percentile, never absolute

A fixed cut like `score > 0.05` is indefensible across scenes — a calm lake has typical
residuals near 0.01, a fire-affected scene near 0.1. The same threshold flags everything in
one and nothing in the other.

So the convention is *"the top 1% of valid pixels in this scene"*. It adapts to the scene's
own distribution, which is what a human would do by eye. `compute_roc` likewise sweeps
**percentile-spaced** thresholds, because score distributions are heavy-tailed and linear
spacing wastes almost all its resolution.

Thresholding is not done here — it happens in the `anomaly_detection_prep` action, which
combines each model's map and then waits at status `needs_threshold` for a human to choose.

## Naming the material — spectral matching

Hyperspectral only, and the payoff for having 165 bands.

`app/spectral_match/` compares each anomalous pixel's spectrum against the **USGS splib07**
library — 519 curated lab spectra (minerals, organics, soils, vegetation, artificial) in
`data/splib07_slim/`, slimmed from the raw 6.7 GB distribution. The comparison metric is
**Spectral Angle Mapper**: the angle between the two spectra treated as vectors, which is
invariant to brightness, so illumination and slope don't matter.

Three things make it fast and correct:

- Library spectra are **Gaussian-resampled** onto the sensor's exact band grid using
  per-band FWHM, so lab and satellite spectra are compared like for like.
- The resampled library is cached, keyed by a **content hash** of
  `(sensor, wavelengths, fwhm, bad-band mask, chapters, min coverage, library version)` — so
  a settings change invalidates automatically and a re-run is a no-op.
- Pixels are bucketed by their pattern of valid bands, making each bucket a single dense
  matrix multiply.

It deliberately reads the **native onboarding vendable, not the band-filtered one**, so
narrow absorption features — the actual diagnostic evidence — survive.

Two limits matter at 30 m.
- **Whole-pixel matching mostly names the surroundings.** A man-made object usually fills
  part of a pixel, so the pixel's spectrum is mostly what is around it.
- **The library has only 3 usable man-made spectra** (known issue 10).

`app/spectral_match/family_match.py` with `data/family_library/` addresses both by naming
material *families* with calibrated Simplex ACE. Since 2026-10-16 the action runs it after
SAM, on the raw cube, and it is the answer the viewer and the export lead with: "likely
material: roofing" only above the bar p\* = 1 − `family_alpha` / F, otherwise "man-made, type
unconfirmed" or the natural family. SAM's outputs are unchanged and still shown. See
[`lld/spectral-family-match.md`](lld/spectral-family-match.md).

---

**Next:** [6. Backend](06-backend.md) · Deep reference: `docs/tech/06_classical_detectors/`
