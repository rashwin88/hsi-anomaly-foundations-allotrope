"""
Dataset builder for Planet Tanager ortho surface reflectance (HDF-EOS5 `.h5` or ENVI `.hdr`).

Called by scene onboarding (backend/allotrope_worker/scene_onboard.py) and by
band_filter_apply; returns the VendableHyperspectralDataset every HSI action
consumes. Reads through TanagerHelper, so bands arrive in ascending wavelength.

Why native, not dressed up as AVIRIS-NG like GHOSt: the file already carries
everything the product needs - per-band FWHM and a bad-band list, a nodata
layer, Planet's cloud and cirrus masks, and an exact UTM grid. A costume would
drop the masks and grid, and would label Planet scenes as JPL ones in the
Library. Rationale in docs/lld/tanager.md.

The pipeline is EnMAP's in-RAM one (enmap_dataset_builder.py): band filter ->
quality-mask invalidation -> spatial masking -> PCHIP gap fill -> optional
common-grid resample. A 426 x 1020 x 895 scene is 1.55 GB of float32, the same
order as an EnMAP tile, so there is no AVIRIS-style memmap spill.

Validity of a voxel = value != _FillValue AND finite AND nodata_pixels == 0 AND
the band is in good_wavelengths. The finite check is deliberate: the AVIRIS-NG
builder counted NaN as valid, which broke spectral matching on GHOSt. An ENVI
scene has no nodata layer and no cloud/cirrus masks: validity drops the
nodata term, and named quality masks are skipped with a warning that
band_filter_apply records (`quality_masks_skipped`).

Gotchas:
  - Reflectance is already 0-1; there is no DN scaling. Invalid voxels are set
    to 0.0, as EnMAP's transformer does, so no -9999 reaches a percentile.
  - beta_cloud_mask / beta_cirrus_mask use 1 = flagged and 255 = fill; only 1
    counts. They are applied only through quality_masks_to_apply.
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
from app.utils.files.tanager_helper import (
    LAYER_CIRRUS,
    LAYER_CLOUD,
    LAYER_NODATA,
    TanagerHelper,
    grid_epsg,
)


logger = logging.getLogger("TanagerDatasetBuilder")
logger.setLevel(logging.INFO)


# Same VNIR/SWIR split PRISMA, EnMAP and AVIRIS-NG use in this codebase.
_VNIR_SWIR_BOUNDARY_NM = 1000.0

# Bands per validity pass: bounds the temporary bool arrays to ~50 MB.
_VALIDITY_BLOCK_BANDS = 16

# quality_masks_to_apply names this builder understands -> Data Fields layer.
QUALITY_MASK_LAYERS = {"cloud": LAYER_CLOUD, "cirrus": LAYER_CIRRUS}


def _family(wavelength_nm: float) -> SpectralFamily:
    return SpectralFamily.VNIR if wavelength_nm < _VNIR_SWIR_BOUNDARY_NM else SpectralFamily.SWIR


class TanagerDatasetBuilder(DatasetBuilder):
    """`FileSourceConfig.source_path` is the ortho_sr `.h5` file."""

    def __init__(self, file_source_configuration: FileSourceConfig):
        super().__init__(file_source_configuration=file_source_configuration)
        self._file_helper = self.initialize_helper()
        self._band_information = self.extract_band_information()
        self._stac_item_cached: Optional[Item] = None
        # quality_masks_to_apply names this scene could not honour (ENVI: all of them).
        self.quality_masks_skipped: List[str] = []

    @property
    def file_helper(self) -> TanagerHelper:
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

    def initialize_helper(self) -> TanagerHelper:
        return TanagerHelper(file_source_config=self.file_source_config)

    def extract_band_information(self) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """VNIR/SWIR split at 1000 nm; band_index restarts within each family, as PRISMA and AVIRIS-NG do."""
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

    def vend_dataset(
        self,
        band_filter_config: Optional[BandFilterConfig] = None,
        **kwargs,
    ) -> VendableHyperspectralDataset:
        """Build the vendable: (C, H, W) reflectance, ascending wavelength.

        Without `band_filter_config` the cube carries all native bands with
        Planet's bad bands marked invalid. With it, the EnMAP pipeline runs.
        """
        md = self.file_helper.file_metadata
        logger.info("Building Tanager vendable %s: %d bands x %d x %d", md.scene_id, md.bands, md.height, md.width)

        cube = self.file_helper.extract_specific_bands(bands=[], mode="all")
        if self.file_helper.has_planet_layers:
            pixel_valid = self.file_helper.extract_layer(LAYER_NODATA) == 0
        else:
            pixel_valid = np.ones((md.height, md.width), dtype=bool)   # ENVI: fill and NaN carry nodata
        good = np.asarray(md.good_band, dtype=bool)
        validity = np.empty(cube.shape, dtype=np.int8)
        for b0 in range(0, md.bands, _VALIDITY_BLOCK_BANDS):
            block = cube[b0:b0 + _VALIDITY_BLOCK_BANDS]
            data_ok = (block != md.fill_value) & np.isfinite(block) & pixel_valid
            block[~data_ok] = 0.0
            validity[b0:b0 + _VALIDITY_BLOCK_BANDS] = data_ok & good[b0:b0 + _VALIDITY_BLOCK_BANDS, None, None]
        logger.info("Tanager: %.1f%% of pixels have data; %d / %d bands good",
                    pixel_valid.mean() * 100.0, int(good.sum()), md.bands)

        families: List[SpectralFamily] = [_family(cw) for cw in md.wavelengths_nm]
        band_cw = list(md.wavelengths_nm)
        band_fwhm = list(md.fwhm_nm)
        band_valid = [int(g) for g in md.good_band]
        scores = _band_scores(validity)

        if band_filter_config is not None:
            band_filter = SpectralBandFilter(
                band_wavelengths=band_cw,
                band_validity_flags=band_valid,
                exclusion_ranges=band_filter_config.exclusion_ranges,
                spectral_families=families,
                edge_bands_to_trim=band_filter_config.edge_bands_to_trim,
                band_level_validity_scores=scores,
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

            self._apply_quality_masks(validity, band_filter_config.quality_masks_to_apply)

            # Spatial masking: pixel columns with too many invalid voxels go entirely.
            invalid_fraction = 1.0 - validity.sum(axis=0) / validity.shape[0]
            bad_pixels = invalid_fraction > band_filter_config.max_invalid_voxel_fraction
            if bad_pixels.any():
                validity[:, bad_pixels] = 0
                logger.info("Spatial masking: %d pixels invalidated (>%.0f%% voxels invalid).",
                            int(bad_pixels.sum()), band_filter_config.max_invalid_voxel_fraction * 100.0)

            interpolate_spectral_gaps(
                cube=cube, validity_cube=validity, band_wavelengths=np.asarray(band_cw, dtype=np.float64),
            )

            if band_filter_config.common_wavelength_grid is not None:
                cube, validity, families = resample_to_common_grid(
                    cube=cube, validity_cube=validity,
                    source_wavelengths=np.asarray(band_cw, dtype=np.float64),
                    target_wavelengths=band_filter_config.common_wavelength_grid,
                    spectral_families=families,
                )
                band_cw = band_filter_config.common_wavelength_grid.tolist()
                band_fwhm = []      # FWHM is not defined after resampling (EnMAP does the same)
                band_valid = [1] * len(band_cw)
            scores = _band_scores(validity)

        logger.info("Tanager vendable assembled: cube %s", cube.shape)
        return VendableHyperspectralDataset(
            normalized_hyperspectral_cube=cube,
            validity_cube=validity,
            spectral_family_order=families,
            band_cw_order=band_cw,
            band_fwhm_order=band_fwhm,
            band_validity_by_position=band_valid,
            band_level_validity_score=scores,
        )

    def cloud_cover_pct(self) -> Optional[float]:
        """Percent of data pixels Planet's beta cloud mask flags (1); None without data pixels or for ENVI."""
        if not self.file_helper.has_planet_layers:
            return None
        data = self.file_helper.extract_layer(LAYER_NODATA) == 0
        if not data.any():
            return None
        cloud = self.file_helper.extract_layer(LAYER_CLOUD) == 1
        return float((cloud & data).sum() / data.sum() * 100.0)

    def _apply_quality_masks(self, validity: np.ndarray, names: List[str]) -> None:
        """Zero every validity column flagged (== 1) by any named mask. In place."""
        flagged = np.zeros(validity.shape[1:], dtype=bool)
        for name in names:
            if not self.file_helper.has_planet_layers:
                logger.warning("Tanager ENVI scene has no %r mask; skipped", name)
                self.quality_masks_skipped.append(name)
                continue
            layer = QUALITY_MASK_LAYERS.get(name)
            if layer is None:
                logger.warning("Tanager has no %r quality mask (has %s); ignored", name, sorted(QUALITY_MASK_LAYERS))
                continue
            hit = self.file_helper.extract_layer(layer) == 1
            logger.info("Quality mask %s: %d pixels flagged", name, int(hit.sum()))
            flagged |= hit
        if flagged.any():
            validity[:, flagged] = 0

    def _build_stac_item(self) -> Item:
        """Item id = Planet item id; bbox = WGS-84 envelope of the grid corners (None for an ungridded ENVI scene)."""
        from rasterio.warp import transform as warp_transform

        md = self.file_helper.file_metadata
        epsg, bbox = None, None
        if md.grid is not None:
            epsg = grid_epsg(md.grid)
            (x0, y0), (x1, y1) = md.grid.upper_left_m, md.grid.lower_right_m
            lons, lats = warp_transform(f"EPSG:{epsg}", "EPSG:4326", [x0, x1, x1, x0], [y0, y0, y1, y1])
            bbox = [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]
        # pystac refuses datetime=None without a start/end range, which a scene
        # named only by its file stem does not have; Scene.acquisition_at is
        # nullable, so construct with a placeholder and clear it.
        item = Item(
            id=md.scene_id,
            geometry=None,
            bbox=bbox,
            datetime=md.acquisition_at or datetime.now(timezone.utc),
            properties={
                "platform": "Tanager",
                "instrument": "Tanager",
                "product_type": "ortho_sr",
                "proj:epsg": epsg,
                "planet:strip_id": md.strip_id,
                "tanager:file_format": md.file_format,
                "tanager:reflectance_scale_source": md.reflectance_scale_source,
                "tanager:fwhm_derived": md.fwhm_derived,
            },
        )
        item.datetime = md.acquisition_at
        return item


def _band_scores(validity: np.ndarray) -> List[float]:
    """Percent of valid pixels per band."""
    per_band = validity.reshape(validity.shape[0], -1)
    return [float(v) for v in per_band.mean(axis=1, dtype=np.float64) * 100.0]
