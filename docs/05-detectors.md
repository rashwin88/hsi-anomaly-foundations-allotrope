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
| `combined` | `w·L̃1 + (1−w)·S̃AM`, `w=0.5` | default for Indradhanu |

**L1 and SAM carry independent information.** L1 is a *magnitude* signal, SAM a *shape*
signal. A bright cloud has large L1 but small SAM. A chemically odd material with normal
brightness but an unusual absorption profile has small L1 but large SAM. Reporting both lets
the operator choose which kind of anomaly matters.

Each is normalised by its maximum over **valid pixels only**, then invalid pixels are zeroed.

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

---

**Next:** [6. Backend](06-backend.md) · Deep reference: `docs/tech/06_classical_detectors/`
