"""
USGS splib07 material matching - naming what an anomalous pixel is made of.

The last step of the hyperspectral chain. Once anomaly_detection_prep has
committed a mask, the spectral_library_match Action compares each flagged
pixel's spectrum against 519 curated lab spectra and reports the closest
materials. Driven from
backend/allotrope/action_types/_spectral_library_match_run.py.

The pipeline, one module per step:
  library.py    load the slim bundle; build and cache a per-sensor copy
  resample.py   convolve lab spectra onto the sensor's bands via a Gaussian SRF
  smoothing.py  Savitzky-Golay smoothing that preserves the NaN validity pattern
  sam.py        Spectral Angle Mapper, bucketed by valid-band pattern

Why SAM rather than a distance: it measures the ANGLE between two spectra
treated as vectors, so it is invariant to overall brightness. Illumination,
slope and shadow change a spectrum's magnitude but not its shape, and shape is
what identifies a material.

The per-sensor cache is content-addressed by a hash of (sensor, wavelengths,
fwhm, bad-band mask, chapters, min coverage, library version), so changing any
setting invalidates it automatically and re-running is a no-op.

Note `export` is deliberately NOT re-exported here - import it by path.
The algorithmic spec lives in spectal_match_sample/WALKTHROUGH.md.
"""

from app.spectral_match.resample import gaussian_resample_to_target
from app.spectral_match.library import (
    LibraryEntry,
    SlimBundle,
    build_cache_for_vendable,
    load_slim_bundle,
    build_sensor_cache,
    load_sensor_cache,
    sensor_cache_key,
)
from app.spectral_match.smoothing import savgol_smooth
from app.spectral_match.sam import MatchResult, match_pixels

__all__ = [
    "gaussian_resample_to_target",
    "LibraryEntry",
    "SlimBundle",
    "build_cache_for_vendable",
    "load_slim_bundle",
    "build_sensor_cache",
    "load_sensor_cache",
    "sensor_cache_key",
    "savgol_smooth",
    "MatchResult",
    "match_pixels",
]
