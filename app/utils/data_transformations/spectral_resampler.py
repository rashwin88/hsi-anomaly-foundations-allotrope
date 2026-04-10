"""
Spectral resampling onto a common wavelength grid.

Resamples a hyperspectral cube from its native (sensor-specific) wavelengths
onto a target wavelength grid using PCHIP interpolation. This ensures all
sensors produce identical band counts and wavelengths, enabling mixed-sensor
training with consistent tensor shapes.

Applied as the final stage of vendable construction, after band filtering,
spatial masking, and spectral gap-filling. At this point, every valid pixel
has a complete spectrum — no gaps to handle.

Invalid pixels (all bands = 0 in the validity cube) are left as zeros in
the resampled cube and marked invalid in the resampled validity cube.
"""

import logging
from typing import Tuple, List

import numpy as np
from scipy.interpolate import PchipInterpolator

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

logger = logging.getLogger(__name__)


def resample_to_common_grid(
    cube: np.ndarray,
    validity_cube: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
    spectral_families: List[SpectralFamily],
) -> Tuple[np.ndarray, np.ndarray, List[SpectralFamily]]:
    """
    Resample a hyperspectral cube from native wavelengths to a common grid.

    Uses PCHIP interpolation along the spectral axis. Only valid pixels
    (fully valid column) are resampled; invalid pixels remain as zeros.

    Args:
        cube: Reflectance cube (C_src, H, W) float32.
        validity_cube: Validity mask (C_src, H, W) int8, 1=valid.
        source_wavelengths: Native wavelengths (C_src,) in nm.
        target_wavelengths: Target grid (C_tgt,) in nm, ascending.
        spectral_families: Per-band family assignment, length C_src.

    Returns:
        Tuple of:
        - Resampled cube (C_tgt, H, W) float32
        - Resampled validity cube (C_tgt, H, W) int8
        - Resampled spectral family list, length C_tgt
    """
    C_src, H, W = cube.shape
    C_tgt = len(target_wavelengths)
    num_pixels = H * W

    logger.info(
        "Spectral resampling: %d native bands → %d target bands "
        "(%.1f–%.1f nm, %.1f nm spacing).",
        C_src, C_tgt,
        target_wavelengths[0], target_wavelengths[-1],
        target_wavelengths[1] - target_wavelengths[0],
    )

    # Sort source wavelengths ascending for PCHIP
    sort_idx = np.argsort(source_wavelengths)
    src_wl_sorted = source_wavelengths[sort_idx].astype(np.float64)
    tgt_wl = target_wavelengths.astype(np.float64)

    # Pre-sort the cubes
    cube_sorted = cube[sort_idx]          # (C_src, H, W)
    valid_sorted = validity_cube[sort_idx]  # (C_src, H, W)

    # Allocate output
    out_cube = np.zeros((C_tgt, H, W), dtype=np.float32)
    out_validity = np.zeros((C_tgt, H, W), dtype=np.int8)

    # Identify fully-valid pixel columns (post-interpolation, validity is binary)
    pixel_valid = valid_sorted[0].ravel().astype(bool)  # (N,)
    valid_cols = np.nonzero(pixel_valid)[0]

    if len(valid_cols) == 0:
        logger.info("Spectral resampling: no valid pixels to resample.")
        return out_cube, out_validity, _assign_families(tgt_wl, spectral_families, src_wl_sorted)

    logger.info(
        "Spectral resampling: %d valid pixels (%.1f%%) to resample.",
        len(valid_cols), len(valid_cols) / num_pixels * 100.0,
    )

    # Reshape for columnar processing
    cube_2d = cube_sorted.reshape(C_src, num_pixels)  # (C_src, N)
    out_2d = out_cube.reshape(C_tgt, num_pixels)       # (C_tgt, N)

    # Vectorized PCHIP: pass all valid pixels as 2D array
    valid_spectra = cube_2d[:, valid_cols].astype(np.float64)  # (C_src, n_valid)

    interp = PchipInterpolator(src_wl_sorted, valid_spectra, axis=0, extrapolate=False)
    resampled = interp(tgt_wl)  # (C_tgt, n_valid)

    # Handle NaN from extrapolation outside native range — clamp to edge values
    nan_mask = np.isnan(resampled)
    if nan_mask.any():
        below = tgt_wl < src_wl_sorted[0]
        above = tgt_wl > src_wl_sorted[-1]
        for j in range(C_tgt):
            if not nan_mask[j].any():
                continue
            nc = nan_mask[j]
            if below[j]:
                resampled[j, nc] = valid_spectra[0, nc]
            elif above[j]:
                resampled[j, nc] = valid_spectra[-1, nc]

    out_2d[:, valid_cols] = resampled.astype(np.float32)

    # Valid pixels get all-1 validity in the resampled cube
    out_validity_2d = out_validity.reshape(C_tgt, num_pixels)
    out_validity_2d[:, valid_cols] = 1

    # Assign spectral families to target wavelengths
    out_families = _assign_families(tgt_wl, spectral_families, src_wl_sorted)

    logger.info("Spectral resampling complete. Output cube shape: (%d, %d, %d)", C_tgt, H, W)
    return out_cube, out_validity, out_families


def _assign_families(
    target_wl: np.ndarray,
    source_families: List[SpectralFamily],
    source_wl_sorted: np.ndarray,
) -> List[SpectralFamily]:
    """
    Assign a spectral family (VNIR/SWIR) to each target wavelength by
    nearest-neighbor lookup from the source wavelengths.
    """
    sort_idx = np.argsort(source_wl_sorted)
    sorted_families = [source_families[i] for i in sort_idx] if not np.all(np.diff(source_wl_sorted) > 0) else source_families

    # For already-sorted source, just do nearest neighbor
    nearest_idx = np.searchsorted(source_wl_sorted, target_wl, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(source_wl_sorted) - 1)

    # Refine: check if the left or right neighbor is closer
    for i in range(len(nearest_idx)):
        idx = nearest_idx[i]
        if idx > 0 and idx < len(source_wl_sorted):
            if abs(source_wl_sorted[idx - 1] - target_wl[i]) < abs(source_wl_sorted[idx] - target_wl[i]):
                nearest_idx[i] = idx - 1

    return [sorted_families[idx] for idx in nearest_idx]
