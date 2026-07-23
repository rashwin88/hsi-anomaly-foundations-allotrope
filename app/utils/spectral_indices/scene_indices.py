"""Per-pixel spectral indices and confounder-class masks.

Lifts the recipe from `benchmarking/hyperspectral/segformer_mae_benchmark.ipynb` § 9.

Inputs are a normalized hyperspectral cube on a known wavelength grid
(post-`vend_dataset(BandFilterConfig(...))`) and a per-pixel spatial
validity mask. Outputs are NDVI, NDWI, VNIR brightness, and four
threshold-driven binary masks (water / cloud / shadow / vegetation),
plus a final keep_mask that is the union of those (intersected with
validity) inverted — the canonical input for downstream anomaly Actions.

The Allotrope worker action `scene_segmentation` is a thin wrapper
around `compute_scene_segmentation` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def find_band(wavelengths: np.ndarray, target_nm: float) -> int:
    """Index of the band closest to `target_nm`."""
    return int(np.argmin(np.abs(wavelengths - target_nm)))


@dataclass
class IndexBands:
    """Wavelength targets for index math."""

    red_nm: float = 660.0
    green_nm: float = 560.0
    nir_nm: float = 860.0
    vnir_brightness_end_nm: float = 910.0


@dataclass
class ClassThresholds:
    """Threshold values for the four exclusion classes."""

    ndwi_water: float = 0.3
    brightness_cloud: float = 0.4
    brightness_shadow: float = 0.02
    ndvi_vegetation: float = 0.4


@dataclass
class SceneSegmentation:
    """Output bundle of `compute_scene_segmentation`."""

    ndvi: np.ndarray            # (H, W) float32, -1..1
    ndwi: np.ndarray            # (H, W) float32, -1..1
    brightness: np.ndarray      # (H, W) float32, 0..~1

    mask_water: np.ndarray      # (H, W) uint8 binary
    mask_cloud: np.ndarray
    mask_shadow: np.ndarray
    mask_vegetation: np.ndarray

    # spatial_validity & ¬(union of selected class masks)
    keep_mask: np.ndarray       # (H, W) uint8 binary

    diagnostics: dict


def per_class_mean_spectra(
    cube: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, list[float]]:
    """Mean spectrum per class over the masked pixels.

    Args:
        cube:  (C, H, W) reflectance cube.
        masks: {class_name: (H, W) uint8 binary mask}. Empty masks are
               handed back as zeros (and a per-class count of 0 in
               diagnostics — caller's job).

    Returns:
        {class_name: [C floats]} — mean reflectance per band per class.
    """
    flat = cube.reshape(cube.shape[0], -1)
    out: dict[str, list[float]] = {}
    for cls, m in masks.items():
        idx = m.astype(bool).ravel()
        if not idx.any():
            out[cls] = [0.0] * cube.shape[0]
            continue
        m_spec = flat[:, idx].mean(axis=1)
        out[cls] = [float(v) for v in m_spec]
    return out


def index_histogram(
    values: np.ndarray, validity: np.ndarray, *, bins: int = 50,
    value_min: float | None = None, value_max: float | None = None,
) -> dict:
    """Histogram a (H, W) index over its valid pixels.

    Returns: {"counts": list[int], "edges": list[float], "min": float,
              "max": float, "n_valid": int}.
    """
    vals = values[validity.astype(bool)]
    if vals.size == 0:
        return {
            "counts": [0] * bins,
            "edges": [0.0] * (bins + 1),
            "min": 0.0,
            "max": 0.0,
            "n_valid": 0,
        }
    lo = float(vals.min()) if value_min is None else value_min
    hi = float(vals.max()) if value_max is None else value_max
    if hi - lo < 1e-9:
        # Degenerate distribution — pad the range so np.histogram doesn't
        # collapse to a single bin and so the chart has visible width.
        hi = lo + 1e-6
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n_valid": int(vals.size),
    }


def compute_scene_segmentation(
    cube: np.ndarray,
    validity: np.ndarray,
    wavelengths: np.ndarray,
    *,
    bands: IndexBands | None = None,
    thresholds: ClassThresholds | None = None,
    classes_to_mask: Iterable[str] = ("water", "cloud", "shadow", "vegetation"),
) -> SceneSegmentation:
    """Compute spectral indices and class masks for a hyperspectral cube.

    Args:
        cube:        (C, H, W) float — normalized reflectance cube.
        validity:    (C, H, W) int  — per-band validity (1=valid, 0=invalid).
                     Per-pixel spatial validity is the OR across bands.
        wavelengths: (C,) float    — band center wavelengths in nm.
        bands:       Wavelength targets for the index bands.
        thresholds:  Threshold values for the four exclusion classes.
        classes_to_mask: Subset of {water, cloud, shadow, vegetation}
                     contributing to keep_mask. Empty = keep_mask is
                     just spatial validity.

    Returns:
        SceneSegmentation with all rasters and a diagnostics dict.

    Implementation mirrors `segformer_mae_benchmark.ipynb` § 9 — same
    band-targeting (NDVI=660+860, NDWI=560+860), same brightness
    definition (mean reflectance from the lower VNIR edge to a
    configurable upper bound), same threshold defaults.
    """
    if bands is None:
        bands = IndexBands()
    if thresholds is None:
        thresholds = ClassThresholds()

    classes = set(classes_to_mask)
    invalid_classes = classes - {"water", "cloud", "shadow", "vegetation"}
    if invalid_classes:
        raise ValueError(
            f"unknown classes: {sorted(invalid_classes)}; "
            "allowed: water, cloud, shadow, vegetation"
        )

    if cube.ndim != 3:
        raise ValueError(f"cube must be (C, H, W); got shape {cube.shape}")
    if validity.shape != cube.shape:
        raise ValueError(
            f"validity shape {validity.shape} must match cube shape {cube.shape}"
        )
    if wavelengths.shape[0] != cube.shape[0]:
        raise ValueError(
            f"wavelengths length {wavelengths.shape[0]} must match cube band "
            f"count {cube.shape[0]}"
        )

    # Per-pixel spatial validity from the per-band validity stack.
    spatial_valid = (validity.sum(axis=0) > 0).astype(np.uint8)  # (H, W)

    b_red = find_band(wavelengths, bands.red_nm)
    b_green = find_band(wavelengths, bands.green_nm)
    b_nir = find_band(wavelengths, bands.nir_nm)
    b_vnir_end = find_band(wavelengths, bands.vnir_brightness_end_nm)
    b_vnir_end = max(b_vnir_end, 0)

    red = cube[b_red].astype(np.float32)
    green = cube[b_green].astype(np.float32)
    nir = cube[b_nir].astype(np.float32)

    # NDVI = (NIR - Red) / (NIR + Red); zero-out invalid pixels so
    # downstream thresholds don't fire on noise.
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi[spatial_valid == 0] = 0.0

    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi[spatial_valid == 0] = 0.0

    # Brightness = mean VNIR reflectance from band 0 to the VNIR upper
    # edge (typically ~910 nm). Index inclusive.
    brightness = cube[: b_vnir_end + 1].mean(axis=0).astype(np.float32)
    brightness[spatial_valid == 0] = 0.0

    # Threshold masks. uint8 for cheap on-disk + downstream OR.
    mask_water = (ndwi > thresholds.ndwi_water).astype(np.uint8)
    mask_cloud = (brightness > thresholds.brightness_cloud).astype(np.uint8)
    mask_shadow = (brightness < thresholds.brightness_shadow).astype(np.uint8)
    mask_vegetation = (ndvi > thresholds.ndvi_vegetation).astype(np.uint8)

    # Mask out invalid pixels so they don't show up in any class — the
    # shadow threshold (< 0.02) trivially fires on the zeros we wrote
    # into invalid pixels above.
    mask_water *= spatial_valid
    mask_cloud *= spatial_valid
    mask_shadow *= spatial_valid
    mask_vegetation *= spatial_valid

    # Build keep_mask = spatial_valid AND NOT (union of selected classes).
    exclusion = np.zeros_like(spatial_valid, dtype=bool)
    if "water" in classes:
        exclusion |= mask_water.astype(bool)
    if "cloud" in classes:
        exclusion |= mask_cloud.astype(bool)
    if "shadow" in classes:
        exclusion |= mask_shadow.astype(bool)
    if "vegetation" in classes:
        exclusion |= mask_vegetation.astype(bool)

    keep_mask = ((spatial_valid == 1) & ~exclusion).astype(np.uint8)

    n_valid = int(spatial_valid.sum())
    n_kept = int(keep_mask.sum())
    diagnostics = {
        "band_indices": {
            "red": int(b_red),
            "green": int(b_green),
            "nir": int(b_nir),
            "vnir_brightness_end": int(b_vnir_end),
        },
        "band_wavelengths_nm": {
            "red": float(wavelengths[b_red]),
            "green": float(wavelengths[b_green]),
            "nir": float(wavelengths[b_nir]),
            "vnir_brightness_end": float(wavelengths[b_vnir_end]),
        },
        "thresholds": {
            "ndwi_water": thresholds.ndwi_water,
            "brightness_cloud": thresholds.brightness_cloud,
            "brightness_shadow": thresholds.brightness_shadow,
            "ndvi_vegetation": thresholds.ndvi_vegetation,
        },
        "classes_to_mask": sorted(classes),
        "pixel_counts": {
            "spatial_valid": n_valid,
            "water": int(mask_water.sum()),
            "cloud": int(mask_cloud.sum()),
            "shadow": int(mask_shadow.sum()),
            "vegetation": int(mask_vegetation.sum()),
            "kept": n_kept,
        },
        "kept_pct_of_valid": (
            round(100.0 * n_kept / n_valid, 3) if n_valid > 0 else 0.0
        ),
    }

    return SceneSegmentation(
        ndvi=ndvi,
        ndwi=ndwi,
        brightness=brightness,
        mask_water=mask_water,
        mask_cloud=mask_cloud,
        mask_shadow=mask_shadow,
        mask_vegetation=mask_vegetation,
        keep_mask=keep_mask,
        diagnostics=diagnostics,
    )
