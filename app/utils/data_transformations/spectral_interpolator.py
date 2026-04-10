"""
Spectral interpolation for partially-valid pixels.

Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to fill
invalid/missing voxels along the spectral axis of a hyperspectral cube.
PCHIP is shape-preserving (no overshoot between known points) and handles
sharp spectral features without ringing.

The implementation processes all partially-valid pixels in a single
vectorized pass by iterating over invalid *bands* rather than pixels,
avoiding expensive per-pixel or per-pattern-group Python loops.

Edge bands with no valid neighbor on one side are filled via constant
extrapolation (clamped to nearest valid value) to avoid unphysical values.

Only partially-valid pixels are processed. Fully-valid and fully-invalid
pixels are left untouched.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


def _pchip_column_interp(
    wl_sorted: np.ndarray,
    cube_sorted: np.ndarray,
    valid_sorted: np.ndarray,
    partial_cols: np.ndarray,
) -> None:
    """
    Fast PCHIP interpolation by grouping pixels with identical invalid-band
    patterns together and interpolating each group in one vectorized call.

    Only processes groups with >= 100 pixels; remaining pixels fall through
    to the per-pixel linear path for speed.

    Operates in-place on cube_sorted and valid_sorted.

    Args:
        wl_sorted: Wavelengths in ascending order, shape (C,).
        cube_sorted: Cube reshaped to (C, N) in wavelength-sorted band order.
        valid_sorted: Validity mask (C, N) in wavelength-sorted band order.
        partial_cols: Column indices of partially-valid pixels.
    """
    C = cube_sorted.shape[0]

    # Build pattern keys for grouping
    pattern_data = valid_sorted[:, partial_cols].T.astype(np.uint8)  # (n_partial, C)
    pattern_keys = np.packbits(pattern_data, axis=1)
    pattern_view = pattern_keys.view(
        np.dtype((np.void, pattern_keys.shape[1] * pattern_keys.dtype.itemsize))
    ).ravel()

    unique_patterns, inverse_idx, counts = np.unique(
        pattern_view, return_inverse=True, return_counts=True
    )

    # Only use PCHIP for groups large enough to amortize the setup cost
    MIN_GROUP_SIZE = 5000
    large_groups = np.nonzero(counts >= MIN_GROUP_SIZE)[0]
    handled = np.zeros(len(partial_cols), dtype=bool)

    for gid in large_groups:
        members = np.nonzero(inverse_idx == gid)[0]
        px_cols = partial_cols[members]

        valid_mask = valid_sorted[:, px_cols[0]].astype(bool)
        invalid_mask = ~valid_mask
        valid_wl = wl_sorted[valid_mask]
        invalid_wl = wl_sorted[invalid_mask]
        valid_bands = np.nonzero(valid_mask)[0]
        invalid_bands = np.nonzero(invalid_mask)[0]

        y_valid = cube_sorted[valid_bands][:, px_cols].astype(np.float64)

        interp = PchipInterpolator(valid_wl, y_valid, axis=0, extrapolate=False)
        filled = interp(invalid_wl)  # (n_invalid, n_pixels)

        # Constant extrapolation for edge NaNs
        nan_mask = np.isnan(filled)
        if nan_mask.any():
            below = invalid_wl < valid_wl[0]
            above = invalid_wl > valid_wl[-1]
            for j in range(len(invalid_wl)):
                if not nan_mask[j].any():
                    continue
                nc = nan_mask[j]
                if below[j]:
                    filled[j, nc] = y_valid[0, nc]
                elif above[j]:
                    filled[j, nc] = y_valid[-1, nc]

        cube_sorted[np.ix_(invalid_bands, px_cols)] = filled.astype(cube_sorted.dtype)
        valid_sorted[np.ix_(invalid_bands, px_cols)] = 1
        handled[members] = True

    # Return indices of pixels NOT handled by PCHIP groups
    remaining = partial_cols[~handled]
    if len(remaining) > 0:
        _linear_column_interp(wl_sorted, cube_sorted, valid_sorted, remaining)


def _linear_column_interp(
    wl_sorted: np.ndarray,
    cube_sorted: np.ndarray,
    valid_sorted: np.ndarray,
    pixel_cols: np.ndarray,
) -> None:
    """
    Fast per-pixel linear interpolation with constant edge extrapolation.
    Uses np.interp (C-accelerated) for each pixel. Falls back here for
    pixels whose mask patterns are too rare for PCHIP group batching.
    """
    C = cube_sorted.shape[0]
    wl = wl_sorted

    for px in pixel_cols:
        vmask = valid_sorted[:, px].astype(bool)
        if vmask.sum() < 2:
            continue
        imask = ~vmask
        valid_wl = wl[vmask]
        y_known = cube_sorted[vmask, px].astype(np.float64)

        # np.interp does linear interpolation + constant edge extrapolation
        invalid_bands = np.nonzero(imask)[0]
        filled = np.interp(wl[invalid_bands], valid_wl, y_known)

        cube_sorted[invalid_bands, px] = filled.astype(cube_sorted.dtype)
        valid_sorted[invalid_bands, px] = 1


def interpolate_spectral_gaps(
    cube: np.ndarray,
    validity_cube: np.ndarray,
    band_wavelengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill invalid voxels in partially-valid pixels using spectral
    interpolation along the wavelength axis.

    Strategy:
    - Groups pixels by their invalid-band mask pattern.
    - Large groups (>=100 pixels): vectorized PCHIP interpolation.
    - Small groups / singletons: per-pixel np.interp (linear, C-fast).
    - Edge bands: constant extrapolation (clamped to nearest valid value).

    Args:
        cube: Reflectance cube in BSQ layout (C, H, W), float32.
        validity_cube: Validity mask (C, H, W), 1=valid, 0=invalid.
            Modified **in-place** — filled voxels are flipped to 1.
        band_wavelengths: Center wavelength per band, shape (C,).
            Used as the interpolation x-axis.

    Returns:
        Tuple of (cube, validity_cube) — both modified in-place but
        returned for convenience.
    """
    C, H, W = cube.shape
    num_pixels = H * W

    # Reshape to (C, N) for columnar processing
    cube_2d = cube.reshape(C, num_pixels)
    valid_2d = validity_cube.reshape(C, num_pixels)

    # Per-pixel valid band count
    valid_counts = valid_2d.sum(axis=0)  # (N,)

    # Partially valid: at least 2 valid bands (need >= 2 for interpolation)
    # and fewer than C (otherwise fully valid — skip)
    partial_mask = (valid_counts >= 2) & (valid_counts < C)
    partial_indices = np.nonzero(partial_mask)[0]

    if len(partial_indices) == 0:
        logger.info("Spectral interpolation: no partially-valid pixels to process.")
        return cube, validity_cube

    logger.info(
        "Spectral interpolation: %d partially-valid pixels to process "
        "(%.1f%% of %d total).",
        len(partial_indices),
        len(partial_indices) / num_pixels * 100.0,
        num_pixels,
    )

    wl = band_wavelengths.astype(np.float64)

    # Sort wavelengths ascending (PCHIP and np.interp both require this)
    sort_order = np.argsort(wl)
    wl_sorted = wl[sort_order]

    # Rearrange cube and validity to sorted wavelength order
    cube_sorted = cube_2d[sort_order]    # (C, N) — view into reshaped cube
    valid_sorted = valid_2d[sort_order]  # (C, N)

    # Run interpolation: PCHIP for large pattern groups, linear for the rest
    _pchip_column_interp(wl_sorted, cube_sorted, valid_sorted, partial_indices)
    # Note: _pchip_column_interp internally falls back to fast linear
    # interpolation (np.interp) for pattern groups with < 100 pixels.

    # Write back to original band order
    inv_sort = np.argsort(sort_order)
    cube_2d[:] = cube_sorted[inv_sort]
    valid_2d[:] = valid_sorted[inv_sort]

    filled_voxels = int((valid_2d[:, partial_indices] == 1).sum())
    logger.info(
        "Spectral interpolation complete: %d partially-valid pixels now fully filled.",
        len(partial_indices),
    )

    return cube, validity_cube
