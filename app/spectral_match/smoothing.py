"""Savitzky-Golay smoothing for hyperspectral spectra (with NaN passthrough).

SG smoothing fits a low-order polynomial across a moving window of bands.
For our matcher we want to suppress sensor noise without flattening real
absorption features.

We treat NaN bands as missing: the smoother is applied along the spectral
axis using ``scipy.signal.savgol_filter`` on a NaN-filled copy where NaNs
are linearly interpolated *first*, and then the original NaN positions
are restored on the output so the validity mask is preserved.

Window / polyorder defaults (7, 2) come from the spectral-match theory
walkthrough and are tuned for ~5 nm sampling; for very narrow-band sensors
the caller should bump the window up.
"""
from __future__ import annotations

import numpy as np


def _interp_nans_inplace(row: np.ndarray) -> np.ndarray:
    nan = ~np.isfinite(row)
    if not nan.any():
        return row
    if nan.all():
        return row
    idx = np.arange(row.size)
    row[nan] = np.interp(idx[nan], idx[~nan], row[~nan])
    return row


def savgol_smooth(
    spectra: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along axis 1 of a (P, B) spectra array.

    NaN bands are linearly interpolated for the smoother, then restored as
    NaN in the output so the per-pixel validity mask is preserved.
    """
    if spectra.ndim != 2:
        raise ValueError(f"expected 2-D spectra, got {spectra.shape}")
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be odd and >= 3")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for SG smoothing; "
            "either install scipy or pass smoothing.disabled=true"
        ) from exc

    out = np.array(spectra, dtype=np.float32, copy=True)
    nan_mask = ~np.isfinite(out)
    if nan_mask.any():
        for r in range(out.shape[0]):
            _interp_nans_inplace(out[r])

    out = savgol_filter(out, window_length=window_length, polyorder=polyorder, axis=1)
    if nan_mask.any():
        out[nan_mask] = np.nan
    return out.astype(np.float32, copy=False)
