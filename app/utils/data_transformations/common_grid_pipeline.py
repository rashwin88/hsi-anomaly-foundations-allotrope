"""
Native hyperspectral cube -> vendable on the 165-band common grid, in one call.

Used by SpectralEarthPatch.to_vendable(). It composes the same four steps, in the same
order and with the same BandFilterConfig, that EnmapDatasetBuilder.vend_dataset runs
inline: SpectralBandFilter -> pixel-column invalidation -> the max-invalid-voxel rule ->
interpolate_spectral_gaps -> resample_to_common_grid. A SpectralEarth patch is EnMAP L2A
with 22 bands removed, so going through identical steps is what makes its common-grid
spectra comparable with the EnMAP and PRISMA shards already trained on.

The EnMAP builder was deliberately left as is rather than refactored onto this helper:
it is the production onboarding path, and this composition adds no logic of its own —
it only calls the shared components. If the two ever need to diverge, that is the place
to notice it.

Gotchas:
  - `pixel_reject` (rows, cols) bool is applied where the builder applies provider
    quality masks: the whole column is invalidated before interpolation.
  - Inputs are not modified; the cube is copied before in-place interpolation.
  - With `common_wavelength_grid=None` the cube stays on its (filtered) native bands.
"""

import numpy as np

from app.models.dataset.vendables import BandFilterConfig, VendableHyperspectralDataset
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter
from app.utils.data_transformations.spectral_interpolator import interpolate_spectral_gaps
from app.utils.data_transformations.spectral_resampler import resample_to_common_grid


def _band_scores(validity: np.ndarray) -> list[float]:
    return (validity.reshape(validity.shape[0], -1).mean(axis=1) * 100.0).tolist()


def to_common_grid(
    cube: np.ndarray,
    validity: np.ndarray,
    wavelengths_nm: list[float],
    families: list[SpectralFamily],
    fwhm_nm: list[float],
    config: BandFilterConfig,
    pixel_reject: np.ndarray | None = None,
) -> VendableHyperspectralDataset:
    """Filter, mask, gap-fill and resample a (C, H, W) float32 reflectance cube.

    Returns a VendableHyperspectralDataset whose invalid pixels are 0.0 with validity 0
    in every band, the same convention as the EnMAP vendables and the training shards.
    """
    flags = [1] * len(wavelengths_nm)
    good = SpectralBandFilter(
        band_wavelengths=list(wavelengths_nm),
        band_validity_flags=flags,
        exclusion_ranges=config.exclusion_ranges,
        spectral_families=list(families),
        edge_bands_to_trim=config.edge_bands_to_trim,
        band_level_validity_scores=_band_scores(validity),
        min_valid_pixel_pct=config.min_valid_pixel_pct,
    ).get_good_band_indices()
    idx = np.array(good)
    cube = cube[idx].astype(np.float32, copy=True)
    validity = validity[idx].astype(np.int8, copy=True)
    wl = [wavelengths_nm[i] for i in good]
    fam = [families[i] for i in good]
    fwhm = [fwhm_nm[i] for i in good]

    if pixel_reject is not None:
        validity[:, pixel_reject] = 0

    too_sparse = (1.0 - validity.mean(axis=0)) > config.max_invalid_voxel_fraction
    validity[:, too_sparse] = 0

    interpolate_spectral_gaps(cube=cube, validity_cube=validity, band_wavelengths=np.array(wl))

    if config.common_wavelength_grid is not None:
        cube, validity, fam = resample_to_common_grid(
            cube=cube, validity_cube=validity, source_wavelengths=np.array(wl),
            target_wavelengths=config.common_wavelength_grid, spectral_families=fam,
        )
        wl, fwhm = config.common_wavelength_grid.tolist(), []   # FWHM has no meaning after resampling

    cube[validity == 0] = 0.0
    return VendableHyperspectralDataset(
        normalized_hyperspectral_cube=cube,
        validity_cube=validity,
        spectral_family_order=fam,
        band_cw_order=wl,
        band_fwhm_order=fwhm,
        band_validity_by_position=[1] * len(wl),
        band_level_validity_score=_band_scores(validity),
    )
