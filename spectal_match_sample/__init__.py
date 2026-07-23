"""Spectral processing primitives: SG smoothing, library loading, SRF resampling, SAM."""
from .preprocess import sg_smooth_spectra
from .library import LibraryEntry, load_splib07_library
from .resample import gaussian_resample_to_target
from .sam import compute_sam_matrix

__all__ = [
    "sg_smooth_spectra",
    "LibraryEntry",
    "load_splib07_library",
    "gaussian_resample_to_target",
    "compute_sam_matrix",
]
