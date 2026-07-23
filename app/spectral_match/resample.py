"""Gaussian SRF resampling of a high-resolution lab spectrum onto a sensor grid.

For each target band ``i`` with centre ``lambda_i`` and FWHM ``fwhm_i``:

    sigma_i = fwhm_i / (2 * sqrt(2 * ln 2))
    w_j     = exp(-(lib_wl_j - lambda_i)^2 / (2 * sigma_i^2))
    out_i   = sum(w_j * lib_refl_j) / sum(w_j)   over |lib_wl_j - lambda_i| <= 3 * sigma_i

The 3-sigma cutoff captures 99.7% of the SRF mass. ``np.searchsorted`` keeps
the inner loop O(log M) per target band instead of O(M).

Output bands are NaN where the library did not cover the band well enough
(fewer than ``min_lib_points_per_band`` finite samples inside the window).
Callers must keep that validity mask around; it feeds the per-pair masking
logic the SAM matcher does at runtime.
"""
from __future__ import annotations

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_resample_to_target(
    lib_wl: np.ndarray,
    lib_refl: np.ndarray,
    target_wl: np.ndarray,
    target_fwhm: np.ndarray,
    n_sigma_window: float = 3.0,
    min_lib_points_per_band: int = 3,
) -> np.ndarray:
    lib_wl = np.asarray(lib_wl, dtype=np.float64)
    lib_refl = np.asarray(lib_refl, dtype=np.float64)
    target_wl = np.asarray(target_wl, dtype=np.float64)
    target_fwhm = np.asarray(target_fwhm, dtype=np.float64)

    if lib_wl.shape != lib_refl.shape:
        raise ValueError(
            f"lib_wl and lib_refl must match: {lib_wl.shape} vs {lib_refl.shape}"
        )
    if target_wl.shape != target_fwhm.shape:
        raise ValueError(
            f"target_wl and target_fwhm must match: "
            f"{target_wl.shape} vs {target_fwhm.shape}"
        )

    sigma = target_fwhm * _FWHM_TO_SIGMA
    out = np.full(len(target_wl), np.nan, dtype=np.float64)
    finite_mask = np.isfinite(lib_refl)

    for i in range(len(target_wl)):
        sig_i = sigma[i]
        if sig_i <= 0:
            continue
        lam_i = target_wl[i]
        half_window = n_sigma_window * sig_i

        lo = np.searchsorted(lib_wl, lam_i - half_window, side="left")
        hi = np.searchsorted(lib_wl, lam_i + half_window, side="right")
        if hi - lo < min_lib_points_per_band:
            continue

        wl_slice = lib_wl[lo:hi]
        refl_slice = lib_refl[lo:hi]
        finite_slice = finite_mask[lo:hi]
        if int(finite_slice.sum()) < min_lib_points_per_band:
            continue

        diff = wl_slice - lam_i
        w = np.exp(-0.5 * (diff / sig_i) ** 2)
        w_eff = w * finite_slice
        wsum = w_eff.sum()
        if wsum <= 0:
            continue
        refl_clean = np.where(finite_slice, refl_slice, 0.0)
        out[i] = float((w_eff * refl_clean).sum() / wsum)

    return out.astype(np.float32)
