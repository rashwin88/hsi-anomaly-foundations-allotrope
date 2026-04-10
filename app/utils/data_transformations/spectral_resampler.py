"""
Spectral resampling onto a common wavelength grid.

Resamples a hyperspectral cube from its native (sensor-specific) wavelengths
onto a target wavelength grid using linear interpolation. This ensures all
sensors produce identical band counts and wavelengths, enabling mixed-sensor
training with consistent tensor shapes.

Linear interpolation is used because the native bands are densely sampled
(6-12nm apart) and the target grid is 10nm — the sample points barely move,
so PCHIP offers no practical quality gain but costs significantly more memory.

Applied as the final stage of vendable construction, after band filtering,
spatial masking, and spectral gap-filling. At this point, every valid pixel
has a complete spectrum — no gaps to handle.

Invalid pixels (all bands = 0 in the validity cube) are left as zeros in
the resampled cube and marked invalid in the resampled validity cube.
"""

import logging
from typing import Tuple, List

import numpy as np

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

    Uses vectorized linear interpolation along the spectral axis with
    constant edge extrapolation. Only valid pixels (fully valid column)
    are resampled; invalid pixels remain as zeros.

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

    # Sort source wavelengths ascending
    sort_idx = np.argsort(source_wavelengths)
    src_wl_sorted = source_wavelengths[sort_idx].astype(np.float64)
    tgt_wl = target_wavelengths.astype(np.float64)

    # Pre-sort the cubes
    cube_sorted = cube[sort_idx]            # (C_src, H, W)
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

    # Reshape for columnar processing — stay in float32 to save memory
    cube_2d = cube_sorted.reshape(C_src, num_pixels)  # (C_src, N)
    out_2d = out_cube.reshape(C_tgt, num_pixels)       # (C_tgt, N)
    valid_spectra = cube_2d[:, valid_cols]              # (C_src, n_valid), float32

    # Vectorized linear interpolation: loop over 165 target wavelengths,
    # each iteration processes all valid pixels via numpy broadcasting.
    for i, twl in enumerate(tgt_wl):
        idx_right = np.searchsorted(src_wl_sorted, twl, side="right")
        idx_right = min(idx_right, C_src - 1)
        idx_left = max(idx_right - 1, 0)

        if idx_left == idx_right:
            # Edge: target at or beyond source boundary — constant extrapolation
            out_2d[i, valid_cols] = valid_spectra[idx_left]
        else:
            # Linear interpolation
            wl_left = src_wl_sorted[idx_left]
            wl_right = src_wl_sorted[idx_right]
            t = np.float32((twl - wl_left) / (wl_right - wl_left))
            out_2d[i, valid_cols] = (
                valid_spectra[idx_left] * (1.0 - t) + valid_spectra[idx_right] * t
            )

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

    nearest_idx = np.searchsorted(source_wl_sorted, target_wl, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(source_wl_sorted) - 1)

    for i in range(len(nearest_idx)):
        idx = nearest_idx[i]
        if idx > 0 and idx < len(source_wl_sorted):
            if abs(source_wl_sorted[idx - 1] - target_wl[i]) < abs(source_wl_sorted[idx] - target_wl[i]):
                nearest_idx[i] = idx - 1

    return [sorted_families[idx] for idx in nearest_idx]
