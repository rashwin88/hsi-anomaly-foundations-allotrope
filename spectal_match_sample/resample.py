"""
Gaussian Spectral Response Function (SRF) resampling.

Given a high-resolution lab spectrum (USGS splib07 has ~2151 bands at ASD
resolution, ~5 nm spacing) and a target sensor's band centres + FWHMs
(PRISMA, ~149 bands after drops, ~10 nm spacing), this convolves the lab
spectrum with the sensor's per-band Gaussian SRFs.

Why Gaussian and not simple interp1d
-------------------------------------
Linear interpolation samples the lab spectrum at one wavelength per target
band. That's wrong when the lab spectrum has narrow features (sharp absorption
bands in some minerals) that the sensor would *integrate* over a 10-nm-wide
slot. The Gaussian SRF correctly averages the lab spectrum across the full
sensor passband, weighted by the sensor's response, which is what the sensor
itself does to incoming photons.

For each target band i with centre lambda_i and FWHM_i:
    sigma_i  = FWHM_i / (2 * sqrt(2 ln 2))
    w_j      = exp( -(lib_wl_j - lambda_i)^2 / (2 sigma_i^2) )
    out_i    = sum( w_j * lib_refl_j ) / sum( w_j )      for |lib_wl_j - lambda_i| <= 3 sigma_i

The 3-sigma cutoff captures 99.7% of the SRF mass.

Performance
-----------
We use np.searchsorted to find the 3-sigma window bounds in O(log M) per
band, instead of boolean-masking the whole (M,) array each call. That's
the practical speedup: most splib07 spectra have ~2151 bands, but the
3-sigma window covers only ~10 of them, so the inner loop touches ~10
floats instead of 2151.

If you're on Colab and the library load is still slow (>60 s for ~2400
files), the bottleneck is Google Drive file open latency, NOT this
function. Copy ASCIIdata_splib07b_cv* to /content/ once before running.
"""
from __future__ import annotations

import numpy as np

# 1 / (2 sqrt(2 ln 2))  --  precomputed constant for FWHM -> sigma
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_resample_to_target(
    lib_wl: np.ndarray,
    lib_refl: np.ndarray,
    target_wl: np.ndarray,
    target_fwhm: np.ndarray,
    n_sigma_window: float = 3.0,
    min_lib_points_per_band: int = 3,
) -> np.ndarray:
    """
    Resample a library spectrum onto a target sensor's bands using Gaussian SRFs.

    Parameters
    ----------
    lib_wl : (M,) ndarray
        Library wavelengths in nm. Must be sorted ascending. NaNs not allowed.
    lib_refl : (M,) ndarray
        Library reflectances. NaN values are treated as missing and skipped.
    target_wl : (N,) ndarray
        Target sensor band centres in nm.
    target_fwhm : (N,) ndarray
        Target sensor per-band FWHMs in nm.
    n_sigma_window : float
        SRF support cutoff in sigmas. Default 3.0 captures 99.7% of mass.
    min_lib_points_per_band : int
        If fewer library points fall in the target band's window, the
        output for that band is NaN (caller can interpolate).

    Returns
    -------
    np.ndarray
        (N,) float32. NaN where the library didn't cover the band well enough.
    """
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

    sigma = target_fwhm * _FWHM_TO_SIGMA  # (N,)
    out = np.full(len(target_wl), np.nan, dtype=np.float64)
    finite_mask = np.isfinite(lib_refl)   # (M,)

    for i in range(len(target_wl)):
        sig_i = sigma[i]
        if sig_i <= 0:
            continue
        lam_i = target_wl[i]
        half_window = n_sigma_window * sig_i

        # O(log M) bound finding instead of O(M) boolean mask.
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
        # Zero out NaN positions in the weights so wsum + dot product
        # ignore them, without making a copy of the slices.
        w_eff = w * finite_slice
        wsum = w_eff.sum()
        if wsum <= 0:
            continue
        refl_clean = np.where(finite_slice, refl_slice, 0.0)
        out[i] = float((w_eff * refl_clean).sum() / wsum)

    return out.astype(np.float32)
