# Spectral Band Filtering & Interpolation Pipeline

## Overview

When `vend_dataset(band_filter_config=BandFilterConfig())` is called, the vendable dataset passes through an eight-stage post-processing pipeline before being returned. Each stage is configurable via `BandFilterConfig`. Passing `None` (the default) skips the entire pipeline — fully backward compatible.

```
Raw vendable (all bands, raw validity)
    |
    v
[1] Bad band flag removal
[2] Wavelength exclusion (atmospheric windows)
[3] Detector edge trimming
[4] Coverage-aware band pruning
    |                                    224 bands → ~188 bands
    v
[5] Quality mask invalidation (EnMAP only — cloud, shadow, haze)
    |
    v
[6] Spatial masking (pixel-column invalidation by voxel fraction)
    |
    v
[7] Spectral interpolation (PCHIP/linear gap-fill)
    |                                    Binary validity: fully valid OR fully invalid
    v
[8] Spectral resampling to common grid (optional)
    |                                    ~188 bands → 165 bands (10nm, 460-2450nm)
    v
Clean vendable (common grid, binary validity, ready for mixed-sensor training)
```

## Files

| File | Role |
|------|------|
| `app/models/dataset/vendables.py` | `BandFilterConfig` Pydantic model with all tuneable parameters |
| `app/utils/data_transformations/spectral_band_filter.py` | Stages 1-4: band-level filtering logic |
| `app/utils/data_transformations/spectral_interpolator.py` | Stage 7: PCHIP/linear spectral gap-filling |
| `app/utils/data_transformations/spectral_resampler.py` | Stage 8: PCHIP resampling to common wavelength grid |
| `app/utils/dataset_builder/prisma_dataset_builder.py` | Pipeline integration for PRISMA |
| `app/utils/dataset_builder/enmap_dataset_builder.py` | Pipeline integration for EnMAP |

## Stage Details

### Stages 1-4: Band Filtering

| Stage | What it does | Default |
|-------|-------------|---------|
| 1. Bad band flags | Drops bands where sensor validity flag != 1 | Always on |
| 2. Wavelength exclusion | Drops bands in atmospheric absorption windows | `(0,450), (912,978), (1131,1152), (1350,1450), (1800,1950)` nm |
| 3. Edge trimming | Drops first/last N bands of each detector (VNIR, SWIR) | `3` bands per end |
| 4. Coverage pruning | Drops bands with < threshold% valid pixels | `20.0%` |

### Stage 5: Quality Mask Invalidation (EnMAP only)

EnMAP scenes ship with per-pixel quality layers that flag atmospheric contamination. Before voxel-fraction spatial masking, pixels flagged by any of the configured quality masks have their entire validity column zeroed out. This catches pixels whose DN values look structurally valid but are atmospherically unreliable.

| Mask | What it flags | In default set? |
|------|--------------|----------------|
| `cloud` | Opaque cloud cover | Yes |
| `cloud_shadow` | Cloud shadow on ground | Yes |
| `haze` | Atmospheric haze | Yes |
| `cirrus` | Thin/thick cirrus (values 1/2) | No (thin cirrus may preserve usable signal) |
| `snow` | Snow/ice cover | No (valid land cover, not atmospheric contamination) |

Configurable via `quality_masks_to_apply` in `BandFilterConfig`. PRISMA has no equivalent quality layers and is unaffected.

### Stage 6: Spatial Masking

After band filtering, each pixel's validity profile is examined across the remaining bands. If more than `max_invalid_voxel_fraction` (default 40%) of a pixel's voxels are invalid, the **entire pixel column** in the validity cube is set to 0. This converts the validity landscape into three categories:

- **Fully valid** — all bands valid at this (h, w) location
- **Fully invalid** — all bands zeroed out (no usable data)
- **Partially valid** — some bands valid, some invalid (fed to stage 6)

### Stage 7: Spectral Interpolation

Partially-valid pixels have spectral gaps (missing bands interspersed with valid ones). These are filled using interpolation along the wavelength axis, then the corresponding voxels in the validity cube are flipped from 0 to 1.

**Method:** Hybrid PCHIP + linear interpolation.

- **Large pattern groups** (>=5000 pixels sharing the same invalid-band pattern): Vectorized [PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) (Piecewise Cubic Hermite Interpolating Polynomial). Shape-preserving, no overshoot between data points, handles sharp spectral features without ringing. One interpolator is built per group and applied to all pixels in that group via a single vectorized call.
- **Small groups / singletons**: `np.interp` (C-accelerated linear interpolation). Adequate for the typical ~3-band gaps and orders of magnitude faster than per-pixel PCHIP.

**Edge extrapolation:** Constant (clamped to nearest valid spectral neighbor). Avoids unphysical negative reflectance or values exceeding 1.0 that polynomial extrapolation can produce.

**Why PCHIP over alternatives:**

| Method | Pros | Cons |
|--------|------|------|
| Linear (`np.interp`) | Fastest, no overshoot | Creates kinks at knot points, distorts narrow absorption features |
| Cubic spline | Smoothest | Prone to Runge-type oscillations, can produce negative reflectance |
| PCHIP | Shape-preserving, C1-continuous, no overshoot | Slightly slower than linear |
| Akima | Similar to PCHIP, smoother | Less standard in HSI literature |

PCHIP is the standard choice in the HSI community for spectral gap-filling because it guarantees monotonicity preservation between data points — critical for maintaining realistic spectral shapes around absorption features and vegetation red edges.

**Rules:**
- Fully valid pixels: untouched
- Fully invalid pixels: untouched (no spatial borrowing)
- Partially valid pixels: gaps filled, validity flipped to 1

**After interpolation, the validity cube is strictly binary at the pixel level:** every pixel column is either all-1 (fully valid) or all-0 (fully invalid). Downstream code can derive spatial validity simply as `(validity_cube.sum(axis=0) > 0)`.

### Stage 8: Spectral Resampling to Common Grid

When `common_wavelength_grid` is set on `BandFilterConfig`, the cube is resampled from native sensor wavelengths onto a target grid using vectorized PCHIP interpolation. This is the final stage — it runs after all gaps are filled, so every valid pixel has a complete spectrum.

The default grid (`DEFAULT_COMMON_WAVELENGTH_GRID`) uses 10nm spacing from 460–2450nm but **excludes atmospheric absorption windows** to avoid fabricating spectral data in regions where bands were deliberately removed. This produces **165 bands across 5 clean spectral segments**:

| Segment | Range (nm) | Bands | Region |
|---------|-----------|-------|--------|
| 1 | 460–910 | 46 | Visible + NIR |
| 2 | 980–1130 | 16 | NIR/SWIR transition |
| 3 | 1160–1340 | 19 | SWIR-1 |
| 4 | 1460–1790 | 34 | SWIR-2 |
| 5 | 1960–2450 | 50 | SWIR-3 |

The 10nm spacing respects the coarsest sensor resolution (PRISMA ~12nm) to avoid manufacturing artificial spectral detail. Every resampled band is within ~5nm of a real native measurement.

After resampling, **all sensors produce identical (165, H, W) cubes with identical wavelength arrays** — enabling mixed-sensor batching in a single DataLoader.

The grid is built by `build_common_wavelength_grid()` which accepts custom parameters for start/end/spacing and exclusion ranges. Custom grids can be passed via `BandFilterConfig(common_wavelength_grid=my_grid)`.

## Configuration

All parameters live in `BandFilterConfig`:

```python
class BandFilterConfig(BaseModel):
    exclusion_ranges: List[Tuple[float, float]]  # wavelength exclusion windows (nm)
    edge_bands_to_trim: int                       # bands to trim per detector end
    min_valid_pixel_pct: float                    # coverage threshold for band pruning
    max_invalid_voxel_fraction: float             # spatial masking threshold
    quality_masks_to_apply: List[str]             # EnMAP quality layers to use
    common_wavelength_grid: Optional[np.ndarray]  # target resampling grid (None = skip)
```

## Performance

Spectral interpolation was the main performance challenge. Initial per-pixel PCHIP took ~300s per scene. The final hybrid approach:

| Sensor | Time |
|--------|------|
| PRISMA | ~5.5s |
| EnMAP | ~10s |

Key optimizations:
1. **Pattern grouping**: Pixels with identical invalid-band masks share one interpolator
2. **Vectorized PCHIP**: `PchipInterpolator(x, y_2d, axis=0)` interpolates all pixels in a group at once
3. **Hybrid threshold**: Only groups with >=5000 pixels use PCHIP; smaller groups use `np.interp` (C-accelerated) which is adequate for ~3-band gaps

## Results (Test Scenes)

### PRISMA

| Metric | Raw | After filtering+interp | After resampling |
|--------|-----|----------------------|-----------------|
| Bands | 239 | 186 | **165** |
| Wavelength range | 407.0 – 2497.1 nm | 453.4 – 2477.1 nm | 460 – 2450 nm |
| Fully valid pixels | 68.6% | 70.2% | 70.2% |
| Partially valid | — | **0%** | **0%** |

### EnMAP

| Metric | Raw | After filtering+interp | After resampling |
|--------|-----|----------------------|-----------------|
| Bands | 224 | 188 | **165** |
| Wavelength range | 418.4 – 2445.3 nm | 454.3 – 2422.8 nm | 460 – 2450 nm |
| Fully valid pixels | 9.8% | 70.9% | 70.9% |
| Partially valid | — | **0%** | **0%** |

Both sensors produce **identical (165, H, W) cubes** with **identical wavelength arrays** after resampling.

## Usage

```python
from app.models.dataset.vendables import BandFilterConfig, DEFAULT_COMMON_WAVELENGTH_GRID

# Full pipeline with common grid resampling (recommended for training)
ds = builder.vend_dataset(band_filter_config=BandFilterConfig(
    common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID
))

# Pipeline without resampling (native band count preserved)
ds = builder.vend_dataset(band_filter_config=BandFilterConfig())

# Custom thresholds
ds = builder.vend_dataset(band_filter_config=BandFilterConfig(
    min_valid_pixel_pct=30.0,
    edge_bands_to_trim=5,
    max_invalid_voxel_fraction=0.3,
    common_wavelength_grid=DEFAULT_COMMON_WAVELENGTH_GRID,
))

# No pipeline (backward compatible)
ds = builder.vend_dataset()
```
