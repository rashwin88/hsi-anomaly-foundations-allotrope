# Chapter 6 — Classical and Statistical Anomaly Detectors

This chapter is a textbook treatment of every non-deep-learning anomaly
detector shipped in Allotrope. It is split into thirteen sections, each
in its own file under
[`06_classical_detectors/`](./06_classical_detectors/). Read them in
order on a first pass — Section 1 introduces the RX statistic that
every later detector is a special case of, and Section 2 covers the
shared preprocessing that the first four detectors all rely on.

| # | Section | File |
| --- | --- | --- |
| 1 | The RX statistic — the engine inside every detector | [01_rx_statistic_intro.md](./06_classical_detectors/01_rx_statistic_intro.md) |
| 2 | Shared preprocessing — band filter and spatial mask | [02_shared_preprocessing.md](./06_classical_detectors/02_shared_preprocessing.md) |
| 3 | Global RX — `GlobalRXDetector` | [03_global_rx.md](./06_classical_detectors/03_global_rx.md) |
| 4 | Local RX — `LocalRXDetector` | [04_local_rx.md](./06_classical_detectors/04_local_rx.md) |
| 5 | MNF compression + GRX — `MNFCompressionDetector` | [05_mnf_compression_grx.md](./06_classical_detectors/05_mnf_compression_grx.md) |
| 6 | MNF compression + LRX — `MNFCompressionLRXDetector` | [06_mnf_compression_lrx.md](./06_classical_detectors/06_mnf_compression_lrx.md) |
| 7 | Thermal Global RX — `ThermalGRXDetector` | [07_thermal_grx.md](./06_classical_detectors/07_thermal_grx.md) |
| 8 | The Statistical Ensembler — `StatisticalEnsembler` | [08_statistical_ensembler.md](./06_classical_detectors/08_statistical_ensembler.md) |
| 9 | B10 Adaptive Cloud Masker — `B10AdaptiveCloudMasker` | [09_b10_cloud_masker.md](./06_classical_detectors/09_b10_cloud_masker.md) |
| 10 | Spectral Match — SAM against USGS splib07 | [10_spectral_match_sam.md](./06_classical_detectors/10_spectral_match_sam.md) |
| 11 | Background statistics — `landsat_thermal_stats.py` | [11_landsat_thermal_stats.md](./06_classical_detectors/11_landsat_thermal_stats.md) |
| 12 | Scoring residuals — `scoring.py` | [12_scoring_residuals.md](./06_classical_detectors/12_scoring_residuals.md) |
| 13 | Putting it together — when to reach for which | [13_when_to_use_which.md](./06_classical_detectors/13_when_to_use_which.md) |

## The unifying idea

Every detector in this chapter reduces to one core operation — the
squared Mahalanobis distance

$$
D(x) \;=\; (x - \mu)^{\top}\,\Sigma^{-1}\,(x - \mu)
$$

applied in *some* feature space (raw bands, MNF components, local
annulus, or single thermal band). The variations across detectors are
choices on three axes:

1. **Feature space.** Raw bands (GRX, LRX), MNF-compressed bands
   (MNF-GRX, MNF-LRX), single thermal band (Thermal GRX), or
   neural-network latent (foundation models with `scoring.py`).
2. **Background sample.** The whole scene (GRX), a local annulus
   (LRX), or a model-conditioned reconstruction (autoencoders).
3. **Conditioning of $\Sigma$.** Plain sample covariance (GRX),
   diagonal-loaded covariance (LRX), discard low-SNR components
   (MNF), or replace $\Sigma^{-1}$ with an L1 / SAM metric
   (residual scoring).

The supporting cast — `B10AdaptiveCloudMasker`, SAM,
`landsat_thermal_stats.py`, `scoring.py` — is built on the same
moments / inner-product machinery and is best read after the core RX
chapter (sections 1–8).
