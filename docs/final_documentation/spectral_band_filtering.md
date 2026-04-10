### Spectral Band Filtering

`SpectralBandFilter` reduces a full hyperspectral cube to only the bands that are physically usable for anomaly detection. It applies two independent filters in sequence — a per-band validity flag check and a wavelength-range exclusion — and returns the indices of surviving bands.

**Source:** `app/utils/data_transformations/spectral_band_filter.py`

---

#### Why band filtering is necessary

Hyperspectral sensors deliver hundreds of contiguous spectral bands, but not all of them carry useful signal. Two categories of bands must be removed before any downstream processing:

1. **Sensor-flagged invalid bands** — The sensor metadata includes a validity flag per band. A flag of `0` indicates the band failed calibration, has excessive noise, or is otherwise unreliable. These flags come directly from the instrument vendor (e.g., PRISMA HE5 metadata, EnMAP XML metadata) and are surfaced through the `VendableHyperspectralDataset.band_validity_by_position` field. Note that the sensor flagged bands are already incorporated into the validity mask that is part of the Vendable dataset.

2. **Atmospheric absorption bands** — Even when the sensor hardware is functioning correctly, certain wavelength regions are absorbed by atmospheric gases (primarily water vapor and CO2). Reflectance values in these regions are dominated by atmospheric effects rather than surface properties, making them useless — or actively harmful — for anomaly detection. These wavelength windows are defined as exclusion ranges.

Feeding noisy or absorption-contaminated bands into covariance-based detectors (GRX, LRX, MNF) inflates the condition number of the covariance matrix, degrades Mahalanobis distance estimates, and produces false positives.

---

#### Default exclusion ranges

The module ships with `DEFAULT_EXCLUSION_RANGES`, calibrated for PRISMA but broadly applicable to any VNIR-SWIR sensor:

| Range (nm)    | Reason                          |
|---------------|----------------------------------|
| 0 – 450       | Low SNR, detector noise (UV edge)|
| 912 – 978     | Water vapor absorption           |
| 1131 – 1152   | Water vapor absorption           |
| 1328 – 1492   | Deep water vapor absorption      |
| 1784 – 1967   | Water vapor + CO2 overlap        |

These ranges can be overridden at construction time. For example, EnMAP experiments may use slightly different boundaries depending on the atmospheric conditions of the scene.

---

#### Constructor

```python
SpectralBandFilter(
    band_wavelengths: List[float],                          # center wavelength of each band (nm)
    band_validity_flags: List[int],                         # 1 = valid, 0 = invalid
    exclusion_ranges: List[Tuple[float, float]] = None,     # defaults to DEFAULT_EXCLUSION_RANGES
    spectral_families: Optional[List[SpectralFamily]] = None,  # per-band VNIR/SWIR assignment
    edge_bands_to_trim: int = 0,                            # bands to trim from each detector end
    band_level_validity_scores: Optional[List[float]] = None,  # per-band valid pixel %
    min_valid_pixel_pct: float = 20.0,                      # coverage threshold for pruning
)
```

- `band_wavelengths` and `band_validity_flags` must be the same length and in band order.
- `exclusion_ranges` is a list of `(low_nm, high_nm)` tuples. A band is excluded if its center wavelength falls within **any** of these ranges (inclusive on both ends).
- `spectral_families` is required when `edge_bands_to_trim > 0` to identify detector boundaries.
- `band_level_validity_scores` is required for coverage-aware pruning (bands with < `min_valid_pixel_pct` valid pixels are dropped).

---

#### Methods

**`get_good_band_indices() -> List[int]`**

Returns the indices of bands that pass both filters. The result is cached — repeated calls return the same list without recomputation. On the first call, a summary is logged at `INFO` level.

Filter logic per band (applied in order):
1. If `band_validity_flags[i] != 1` → **dropped** (sensor flag).
2. If the band's center wavelength falls within any exclusion range → **dropped** (atmospheric).
3. If the band is within the first/last N bands of its detector → **dropped** (edge trim).
4. If `band_level_validity_scores[i] < min_valid_pixel_pct` → **dropped** (coverage).
5. Otherwise → **kept**.

The returned indices are in ascending order and can be used directly to slice the cube: `cube[good_indices]`.

**`get_good_band_wavelengths() -> List[float]`**

Returns the center wavelengths of the surviving bands. Convenience wrapper around `get_good_band_indices()`.

**`summary() -> dict`**

Returns a detailed breakdown of the filtering:

```python
{
    "total_bands": 224,
    "dropped_by_flags": {"count": 0, "indices": []},
    "dropped_by_wavelength": {"count": 19, "indices": [...], "ranges": {...}},
    "dropped_by_edge": {"count": 12, "indices": [...]},
    "dropped_by_coverage": {"count": 5, "indices": [...], "threshold_pct": 20.0},
    "surviving": 188
}
```

Each drop category only counts bands not already dropped by earlier stages. The `ranges` sub-dict maps each wavelength-excluded band index to the specific `(lo, hi)` exclusion range that caused its removal.

**Note:** When used via `BandFilterConfig` inside `vend_dataset()`, the filter is part of a larger 8-stage pipeline that also includes quality mask invalidation, spatial masking, spectral interpolation, and optional resampling to a common wavelength grid. See `docs/spectral_band_filtering_report.md` for the full pipeline documentation.

---

#### Usage in the pipeline

`SpectralBandFilter` is used at the entry point of every anomaly detector. The pattern is consistent across GRX, LRX, and MNF:

```python
band_filter = SpectralBandFilter(
    band_wavelengths=vendable.band_cw_order,
    band_validity_flags=vendable.band_validity_by_position,
    exclusion_ranges=exclusion_ranges,   # detector-specific or default
)
good_indices = band_filter.get_good_band_indices()

# Slice the cube to only good bands
sub_cube = cube[good_indices]           # (B', H, W) where B' < B
```

It is also used internally by the `FrequencyDomainDestriper` to select probe bands for stripe angle detection — only good bands are analyzed to avoid letting noise from dead bands influence the stripe correction.

---

#### Typical filtering results (with default BandFilterConfig)

For a 224-band EnMAP scene:
- 0 bands dropped by validity flags (EnMAP has no per-band invalidity)
- ~19 bands dropped by wavelength exclusion (atmospheric absorption)
- ~12 bands dropped by edge trim (3 per detector end × 2 detectors × 2 ends)
- ~5 bands dropped by coverage (<20% valid pixels)
- **~188 bands survive**

For a 239-band PRISMA scene:
- ~5 bands dropped by validity flags (0nm placeholder bands)
- ~36 bands dropped by wavelength exclusion
- ~12 bands dropped by edge trim
- 0 bands dropped by coverage
- **~186 bands survive**

When resampling to the common grid (`DEFAULT_COMMON_WAVELENGTH_GRID`), both sensors produce **165 bands** on a 10nm grid (460–2450nm, excluding atmospheric windows).
