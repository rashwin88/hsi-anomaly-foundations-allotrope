"""
Savitzky-Golay smoothing for anomaly-pixel spectra.

We smooth ONLY the spectra extracted at anomaly pixels, not the whole
cube. The reasoning:
    1. SG is a noise filter; we want it where we're going to compute
       SAM, which is the anomaly spectra.
    2. Smoothing the whole cube is ~10000x more work and changes the
       cube the user already validated.

The library spectra are NOT SG-smoothed: USGS splib07 is already lab-clean
and the Gaussian SRF resampling we apply later is itself a strong smoothing
step.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def sg_smooth_spectra(
    spectra: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter along the band axis.

    Parameters
    ----------
    spectra : (n_pix, n_bands) float ndarray
    window : int
        Window length in bands. Must be odd; auto-clamped if it would
        exceed n_bands. If clamping makes it < 5, the filter is skipped.
    polyorder : int
        Polynomial order. Must be < window.

    Returns
    -------
    np.ndarray
        Same shape as input. If smoothing was skipped, returns input unchanged.
    """
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2-D (n_pix, n_bands), got {spectra.shape}")

    n_pix, n_bands = spectra.shape
    win = min(window, n_bands)
    if win % 2 == 0:
        win -= 1
    if win < 5 or polyorder >= win:
        # Smoothing not viable; pass through. Caller can detect this by
        # comparing input/output identity if they care.
        return spectra

    smoothed = savgol_filter(spectra, window_length=win, polyorder=polyorder, axis=1)
    # SG can produce small negative values near sharp features; clip to a
    # physically plausible reflectance range.
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32, copy=False)
