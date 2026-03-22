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
    band_wavelengths: List[float],        # center wavelength of each band (nm), length B
    band_validity_flags: List[int],       # 1 = valid, 0 = invalid, length B
    exclusion_ranges: List[Tuple[float, float]] = None,  # defaults to DEFAULT_EXCLUSION_RANGES
)
```

- `band_wavelengths` and `band_validity_flags` must be the same length and in band order (i.e., index 0 corresponds to band 0 in the cube).
- `exclusion_ranges` is a list of `(low_nm, high_nm)` tuples. A band is excluded if its center wavelength falls within **any** of these ranges (inclusive on both ends).

---

#### Methods

**`get_good_band_indices() -> List[int]`**

Returns the indices of bands that pass both filters. The result is cached — repeated calls return the same list without recomputation. On the first call, a summary is logged at `INFO` level.

Filter logic per band:
1. If `band_validity_flags[i] != 1` → **dropped** (sensor flag).
2. If the band's center wavelength falls within any exclusion range → **dropped** (atmospheric).
3. Otherwise → **kept**.

The returned indices are in ascending order and can be used directly to slice the cube: `cube[good_indices]`.

**`get_good_band_wavelengths() -> List[float]`**

Returns the center wavelengths of the surviving bands. Convenience wrapper around `get_good_band_indices()`.

**`summary() -> dict`**

Returns a detailed breakdown of the filtering:

```python
{
    "total_bands": 224,
    "dropped_by_flags": {
        "count": 5,
        "indices": [0, 1, 2, 220, 223]
    },
    "dropped_by_wavelength": {
        "count": 30,
        "indices": [10, 11, ...],
        "ranges": {10: (912, 978), 11: (912, 978), ...}  # index → which range excluded it
    },
    "surviving": 189
}
```

The `ranges` sub-dict maps each wavelength-excluded band index to the specific `(lo, hi)` exclusion range that caused its removal. This is useful for auditing — you can verify that the expected atmospheric windows are being caught.

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

#### Typical filtering results

For a 224-band EnMAP scene:
- ~5–10 bands dropped by validity flags (sensor-reported dead/noisy bands)
- ~20–35 bands dropped by wavelength exclusion (atmospheric absorption)
- ~180–195 bands survive for anomaly detection

For a 239-band PRISMA scene:
- ~5–15 bands dropped by validity flags
- ~25–40 bands dropped by wavelength exclusion
- ~185–210 bands survive

The exact numbers vary per scene because validity flags are scene-specific (a band may pass calibration in one acquisition but fail in another).
