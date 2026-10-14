"""
Put a sample's bands onto a target wavelength list, so samples from different sensors, or from
scenes that number their bands differently, can share one batch.

Used by the spectral-stem trainer as a per-sample transform inside the DataLoader workers, when
a run sets `data.band_list_path` (app/foundation_models/trainers/spectral_stem_mae_trainer.py).

Why: the wavelength-token stems take any band list, but the reduced stem needs one list per
batch, and a batch is a stack. Matching by wavelength - never by band position - and filling
target bands the sample lacks with nodata turns "different band lists" into "dead bands", which
the stems already skip exactly. Nothing is interpolated or resampled.

Gotchas:
  - Target bands with no source band within `tolerance_nm` become nodata (-32768) - masked, not
    zero. Source bands with no target are dropped.
  - Wavelengths must be in nm; both lists may be in any order (EnMAP's VNIR/SWIR overlap is not
    monotonic).
"""

import json
from pathlib import Path

import numpy as np

NODATA = -32768
# EnMAP bands are 6-12 nm apart; scenes agree to 0.0 nm on shared bands. 0.5 nm only absorbs
# rounding in the metadata and never pairs two neighbouring bands.
TOLERANCE_NM = 0.5


def load_band_list(path: str | Path) -> np.ndarray:
    """(T,) float32 target wavelengths from a JSON file with a `wavelengths_nm` list - e.g. the
    stats.json written beside a shard set (SpectralEarth's 202, or every EnMAP wavelength)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    wl = np.asarray(data["wavelengths_nm"], dtype=np.float32)
    if wl.ndim != 1 or len(wl) == 0:
        raise ValueError(f"{path}: wavelengths_nm must be a non-empty list")
    return wl


def align_bands(pixels: np.ndarray, wavelengths_nm: np.ndarray, target_nm: np.ndarray,
                tolerance_nm: float = TOLERANCE_NM) -> np.ndarray:
    """(N, H, W) int16 at `wavelengths_nm` -> (T, H, W) int16 at `target_nm`. Each target band
    takes the nearest source band within tolerance, else nodata."""
    src, tgt = np.asarray(wavelengths_nm, np.float64), np.asarray(target_nm, np.float64)
    if pixels.shape[0] != len(src):
        raise ValueError(f"{pixels.shape[0]} bands of pixels, {len(src)} wavelengths")
    dist = np.abs(tgt[:, None] - src[None, :])                    # (T, N)
    nearest = dist.argmin(axis=1)
    found = dist[np.arange(len(tgt)), nearest] <= tolerance_nm
    out = np.full((len(tgt),) + pixels.shape[1:], NODATA, dtype=pixels.dtype)
    out[found] = pixels[nearest[found]]
    return out
