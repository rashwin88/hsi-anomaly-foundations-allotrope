# Chapter 1 — The Data Handling Pipeline

This chapter is split into per-section files under [`01_data_pipeline/`](01_data_pipeline/). It walks the path raw sensor files take to become a `VendableDataset` — the canonical, sensor-agnostic in-memory (or memmap-backed) object every downstream component in Allotrope consumes.

The pipeline has five conceptual stages:

1. **File reading** — a `FileHelper` opens the vendor's container and lazily exposes pixel arrays and metadata.
2. **DN → physical quantity** — a `DataTransformer` converts Digital Numbers (DN) to surface reflectance (hyperspectral) or surface temperature (thermal).
3. **Stripe correction** — pushbroom artifacts are removed via moment-matching, frequency-domain notch filtering, or a composite of both.
4. **Band and pixel hygiene** — wavelength filtering, edge trimming, coverage pruning, spatial column masking, and spectral gap-fill via PCHIP / linear interpolation.
5. **Common-grid resampling** — every sensor's surviving spectrum is linearly resampled onto a single shared 10 nm grid so mixed-sensor training works with identical tensor shapes.

A `DatasetBuilder` orchestrates these stages for one scene and emits a `VendableHyperspectralDataset` or `VendableThermalDataset`.

---

## Sections

| # | File | Tagline |
|---|------|---------|
| 1 | [01_abstract_contracts.md](01_data_pipeline/01_abstract_contracts.md) | The three abstract base classes — `FileHelper`, `DatasetBuilder`, `DataTransformer` — that the entire pipeline rests on. |
| 2 | [02_dataset_builders.md](01_data_pipeline/02_dataset_builders.md) | Sensor-by-sensor walkthrough of every concrete `DatasetBuilder`. |
| 3 | [03_prisma_l2d_reflectance.md](01_data_pipeline/03_prisma_l2d_reflectance.md) | DN → surface reflectance for PRISMA L2D, with per-family scale and offset. |
| 4 | [04_enmap_l2a_reflectance.md](01_data_pipeline/04_enmap_l2a_reflectance.md) | DN → surface reflectance for EnMAP L2A, the simplest calibration in the codebase. |
| 5 | [05_landsat_l2sp_temperature.md](01_data_pipeline/05_landsat_l2sp_temperature.md) | DN → surface temperature for Landsat 9 L2SP — the only thermal calibrator. |
| 6 | [06_moment_matching_destripe.md](01_data_pipeline/06_moment_matching_destripe.md) | Broad-band per-detector statistics correction, with its stationarity-assumption failure mode. |
| 7 | [07_frequency_domain_destripe.md](01_data_pipeline/07_frequency_domain_destripe.md) | Narrow-band notch filter in the FFT that targets periodic stripe artifacts. |
| 8 | [08_composite_destripe.md](01_data_pipeline/08_composite_destripe.md) | FFT-then-moment-matching with a σ safety guard that rolls back regressions. |
| 9 | [09_spectral_band_filter.md](01_data_pipeline/09_spectral_band_filter.md) | Four-stage cascade dropping vendor-flagged, atmospheric, edge, and low-coverage bands. |
| 10 | [10_spectral_gap_interpolation.md](01_data_pipeline/10_spectral_gap_interpolation.md) | Shape-preserving PCHIP gap fill with pattern grouping for speed. |
| 11 | [11_spectral_resampling.md](01_data_pipeline/11_spectral_resampling.md) | Linear resampling onto the common 450–2400 nm @ 10 nm grid that every sensor exports. |
| 12 | [12_nearest_valid_pixel_fill.md](01_data_pipeline/12_nearest_valid_pixel_fill.md) | Spatial fill at SegFormer inference time that removes the "thermal cliff" boundary artifact. |
| 13 | [13_prisma_vend_dataset_walk.md](01_data_pipeline/13_prisma_vend_dataset_walk.md) | End-to-end choreography of one PRISMA scene through every pipeline stage. |
| A | [appendix_a_notation.md](01_data_pipeline/appendix_a_notation.md) | Cube layouts, symbols, abbreviations, file-path conventions. |
| B | [appendix_b_adding_sensors.md](01_data_pipeline/appendix_b_adding_sensors.md) | Runbook for onboarding a new sensor into the pipeline. |
