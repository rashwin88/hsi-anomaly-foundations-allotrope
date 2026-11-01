"""
Dataset builder for "any hyperspectral file" (`sensor_type="generic_hsi"`): GeoTIFF / COG, ENVI, HDF5 / NetCDF.

Called by scene onboarding (backend/allotrope_worker/scene_onboard.py) and by
band_filter_apply; returns the VendableHyperspectralDataset every HSI action
consumes. Reads through GenericHsiHelper, so bands arrive in ascending
wavelength, scaled per the pre-flight decision.

Why it exists: the safety net for evaluation data no native reader knows
(GHOSt GeoTIFFs, OSK ENVI, ...); rationale in docs/lld/generic-hsi.md. The
pipeline is the Tanager / EnMAP in-RAM one: band filter -> spatial masking
-> PCHIP gap fill -> optional common-grid resample. There are no provider
quality masks, so `quality_masks_to_apply` names are skipped with a warning
(band_filter_apply lists them as `quality_masks_skipped`).

Gotchas:
  - Validity = not no-data AND finite AND not an all-zero spectrum AND the
    band is good (an ENVI `bbl`). Invalid voxels are written as 0.0.
  - A not-reflectance cube (DN, radiance) is vended unscaled with
    `units="not_reflectance"`; anomaly_scoring then runs only the unit-free
    classical detectors on it.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.models.dataset.vendables import BandFilterConfig, VendableHyperspectralDataset
from app.models.file_processing.sources import FileSourceConfig
from app.models.hyperspectral_concepts.band import (
    HyperpectralBandInformation,
    HyperSpectralBand,
    WavelengthMeasurementUnits,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.images.cube_representation import CubeRepresentation
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter
from app.utils.data_transformations.spectral_interpolator import interpolate_spectral_gaps
from app.utils.data_transformations.spectral_resampler import resample_to_common_grid
from app.utils.files.generic_hsi_helper import GenericHsiHelper, read_cube

logger = logging.getLogger("GenericHsiDatasetBuilder")
logger.setLevel(logging.INFO)

# Same VNIR/SWIR split PRISMA, EnMAP, AVIRIS-NG and Tanager use in this codebase.
_VNIR_SWIR_BOUNDARY_NM = 1000.0
NOT_REFLECTANCE = "not_reflectance"


def _family(wavelength_nm: float) -> SpectralFamily:
    return SpectralFamily.VNIR if wavelength_nm < _VNIR_SWIR_BOUNDARY_NM else SpectralFamily.SWIR


def _band_scores(validity: np.ndarray) -> List[float]:
    per_band = validity.reshape(validity.shape[0], -1)
    return [float(v) for v in per_band.mean(axis=1, dtype=np.float64) * 100.0]


class GenericHsiDatasetBuilder(DatasetBuilder):
    """`FileSourceConfig.source_path` is the picked cube file (`pick_generic_file`)."""

    def __init__(self, file_source_configuration: FileSourceConfig):
        super().__init__(file_source_configuration=file_source_configuration)
        self._file_helper = self.initialize_helper()
        self._band_information = self.extract_band_information()
        self._stac_item_cached: Optional[Item] = None
        self.quality_masks_skipped: List[str] = []

    @property
    def file_helper(self) -> GenericHsiHelper:
        return self._file_helper

    @property
    def band_information(self) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        return self._band_information

    @property
    def stac_item(self) -> Item:
        if self._stac_item_cached is None:
            self._stac_item_cached = self._build_stac_item()
        return self._stac_item_cached

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        return CubeRepresentation.BSQ

    @property
    def preflight(self) -> dict:
        """The pre-flight report (generic_hsi_preflight.py)."""
        return self.file_helper.file_metadata.preflight

    def initialize_helper(self) -> GenericHsiHelper:
        return GenericHsiHelper(file_source_config=self.file_source_config)

    def extract_band_information(self) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """VNIR/SWIR split at 1000 nm; band_index restarts within each family."""
        md = self.file_helper.file_metadata
        tables = {f: HyperpectralBandInformation(bands_by_index={}, bands_by_wavelength={})
                  for f in (SpectralFamily.VNIR, SpectralFamily.SWIR)}
        for cw, fwhm, good in zip(md.wavelengths_nm, md.fwhm_nm, md.good_band):
            info = tables[_family(cw)]
            band = HyperSpectralBand(
                wavelength=cw, wavelength_measurement_unit=WavelengthMeasurementUnits.NANO_METERS,
                band_index=len(info.bands_by_index), full_width_at_half_maximum=fwhm, is_valid=bool(good),
            )
            info.bands_by_index[band.band_index] = band
            info.bands_by_wavelength[cw] = band
        return tables

    def cloud_cover_pct(self) -> Optional[float]:
        """None: a generic file carries no cloud mask."""
        return None

    def vend_dataset(self, band_filter_config: Optional[BandFilterConfig] = None, **kwargs) -> VendableHyperspectralDataset:
        """(C, H, W) cube, ascending wavelength; with `band_filter_config`, the EnMAP band pipeline runs."""
        md = self.file_helper.file_metadata
        logger.info("Building generic_hsi vendable %s: %d bands x %d x %d (%s)",
                    md.scene_id, md.bands, md.height, md.width, md.scale_decision)
        cube, ok = read_cube(md, range(md.bands))
        good = np.asarray(md.good_band, dtype=bool)
        validity = (ok & good[:, None, None]).astype(np.int8)
        del ok
        families = [_family(cw) for cw in md.wavelengths_nm]
        band_cw, band_fwhm = list(md.wavelengths_nm), list(md.fwhm_nm)
        band_valid = [int(g) for g in md.good_band]

        if band_filter_config is not None:
            band_filter = SpectralBandFilter(
                band_wavelengths=band_cw, band_validity_flags=band_valid,
                exclusion_ranges=band_filter_config.exclusion_ranges, spectral_families=families,
                edge_bands_to_trim=band_filter_config.edge_bands_to_trim,
                band_level_validity_scores=_band_scores(validity),
                min_valid_pixel_pct=band_filter_config.min_valid_pixel_pct,
            )
            keep = band_filter.get_good_band_indices()
            s = band_filter.summary()
            logger.info("Band filter: %d -> %d bands. Dropped: %d flag, %d wavelength, %d edge, %d coverage.",
                        s["total_bands"], s["surviving"], s["dropped_by_flags"]["count"],
                        s["dropped_by_wavelength"]["count"], s["dropped_by_edge"]["count"],
                        s["dropped_by_coverage"]["count"])
            idx = np.asarray(keep)
            cube, validity = cube[idx], validity[idx]
            families = [families[i] for i in keep]
            band_cw = [band_cw[i] for i in keep]
            band_fwhm = [band_fwhm[i] for i in keep]
            band_valid = [band_valid[i] for i in keep]
            for name in band_filter_config.quality_masks_to_apply:
                logger.warning("generic_hsi scene has no %r mask; skipped", name)
                self.quality_masks_skipped.append(name)

            invalid_fraction = 1.0 - validity.sum(axis=0) / max(validity.shape[0], 1)
            bad_pixels = invalid_fraction > band_filter_config.max_invalid_voxel_fraction
            if bad_pixels.any():
                validity[:, bad_pixels] = 0
                logger.info("Spatial masking: %d pixels invalidated (>%.0f%% voxels invalid).",
                            int(bad_pixels.sum()), band_filter_config.max_invalid_voxel_fraction * 100.0)
            interpolate_spectral_gaps(cube=cube, validity_cube=validity,
                                      band_wavelengths=np.asarray(band_cw, dtype=np.float64))
            if band_filter_config.common_wavelength_grid is not None:
                cube, validity, families = resample_to_common_grid(
                    cube=cube, validity_cube=validity, source_wavelengths=np.asarray(band_cw, dtype=np.float64),
                    target_wavelengths=band_filter_config.common_wavelength_grid, spectral_families=families,
                )
                band_cw = band_filter_config.common_wavelength_grid.tolist()
                band_fwhm = []      # FWHM is not defined after resampling (EnMAP does the same)
                band_valid = [1] * len(band_cw)

        logger.info("generic_hsi vendable assembled: cube %s", cube.shape)
        return VendableHyperspectralDataset(
            normalized_hyperspectral_cube=cube, validity_cube=validity, spectral_family_order=families,
            band_cw_order=band_cw, band_fwhm_order=band_fwhm, band_validity_by_position=band_valid,
            band_level_validity_score=_band_scores(validity),
            units=None if md.reflectance else NOT_REFLECTANCE,
        )

    def _build_stac_item(self) -> Item:
        """Item id = file stem (plus `@<override>`); bbox from the georef, None without one."""
        md = self.file_helper.file_metadata
        # pystac refuses datetime=None without a range; Scene.acquisition_at is nullable,
        # so construct with a placeholder and clear it (as the Tanager builder does).
        item = Item(
            id=md.scene_id, geometry=None, bbox=md.bbox,
            datetime=md.acquisition_at or datetime.now(timezone.utc),
            properties={
                "platform": md.platform or "Other hyperspectral",
                "instrument": md.platform or "unknown",
                "product_type": md.driver,
                "proj:epsg": md.epsg,
                "generic_hsi:georef_source": md.georef_source,
                "generic_hsi:scale_decision": md.scale_decision,
                "generic_hsi:reflectance": md.reflectance,
                "generic_hsi:fwhm_source": md.fwhm_source,
            },
        )
        item.datetime = md.acquisition_at
        return item
