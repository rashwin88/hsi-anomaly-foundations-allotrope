"""
Concrete implementation of the EnMAP L2A Dataset Builder.
Takes a folder path containing all scene files and produces
a VendableEnmapHyperspectralDataset.
"""

import logging
from typing import Dict, Optional

import numpy as np
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.models.dataset.vendables import VendableEnmapHyperspectralDataset, BandFilterConfig
from app.models.file_processing.sources import FileSourceConfig
from app.models.images.cube_representation import CubeRepresentation
from app.models.hyperspectral_concepts.band import (
    HyperpectralBandInformation,
    HyperSpectralBand,
    WavelengthMeasurementUnits,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.hyperspectral_concepts.file_components import EnmapFileComponents
from app.utils.files.enmap_helper import EnmapHelper
from app.utils.data_transformations.enmap_l2a_dn_to_surface_reflectance_transformer import (
    EnmapL2aDnToSurfaceReflectanceTransformer,
)
from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter
from app.utils.data_transformations.spectral_interpolator import interpolate_spectral_gaps
from app.utils.data_transformations.spectral_resampler import resample_to_common_grid
from app.utils.stac.stac_utils.stac_items import StacCreator
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier


logger = logging.getLogger("EnmapDatasetBuilder")
logger.setLevel(logging.INFO)


class EnmapDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for EnMAP L2A hyperspectral data.

    Responsibilities:
    - Read EnMAP XML metadata and build band lookup tables with VNIR/SWIR assignment.
    - Convert int16 DN values to surface reflectance using uniform gain/offset.
    - Assemble a normalized hyperspectral cube, validity masks, and quality layers.
    - Produce a vendable dataset in BSQ representation (rasterio native).
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        super().__init__(file_source_configuration=file_source_configuration)
        logger.info("Creating STAC item for EnMAP dataset.")
        self._stac_item = StacCreator(
            file_path=self.file_source_config.source_path
        ).build_stack()
        logger.info("Initializing EnMAP helper and extracting band metadata.")
        self._file_helper: EnmapHelper = self.initialize_helper()
        self._band_information = self.extract_band_information()
        logger.info("Band metadata loaded for VNIR/SWIR.")
        self.dn_to_reflectance_transformer = EnmapL2aDnToSurfaceReflectanceTransformer()

    @property
    def stac_item(self) -> Item:
        return self._stac_item

    @property
    def file_helper(self) -> EnmapHelper:
        return self._file_helper

    @property
    def band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        return self._band_information

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        return CubeRepresentation.BSQ

    def initialize_helper(self) -> EnmapHelper:
        return EnmapHelper(
            file_source_config=self.file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.ENMAP_HYPERSPECTRAL),
        )

    def extract_band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """
        Builds band info from XML metadata.
        Uses detector_boundary to assign VNIR/SWIR family per band.
        """
        metadata = self.file_helper.file_metadata
        vnir_channels = set(metadata.detector_boundary.vnir_expected_channels)

        vnir_by_index, vnir_by_wavelength = {}, {}
        swir_by_index, swir_by_wavelength = {}, {}

        for i, band_char in enumerate(metadata.band_characterisation):
            band = HyperSpectralBand(
                wavelength=band_char.wavelength_center,
                wavelength_measurement_unit=WavelengthMeasurementUnits.NANO_METERS,
                band_index=i,
                full_width_at_half_maximum=band_char.fwhm,
                is_valid=True,
            )
            if band_char.band_number in vnir_channels:
                vnir_by_index[i] = band
                vnir_by_wavelength[band_char.wavelength_center] = band
            else:
                swir_by_index[i] = band
                swir_by_wavelength[band_char.wavelength_center] = band

        return {
            SpectralFamily.VNIR: HyperpectralBandInformation(
                bands_by_index=vnir_by_index,
                bands_by_wavelength=vnir_by_wavelength,
            ),
            SpectralFamily.SWIR: HyperpectralBandInformation(
                bands_by_index=swir_by_index,
                bands_by_wavelength=swir_by_wavelength,
            ),
        }

    def vend_dataset(
        self,
        band_filter_config: Optional[BandFilterConfig] = None,
        **kwargs,
    ) -> VendableEnmapHyperspectralDataset:
        """
        Builds the full vendable EnMAP dataset.

        Args:
            band_filter_config: If provided, applies spectral band filtering
                (atmospheric exclusion, bad band removal, edge trimming,
                coverage-aware pruning). Pass ``BandFilterConfig()`` for defaults.
        """
        metadata = self.file_helper.file_metadata
        logger.info("Building vendable EnMAP hyperspectral dataset.")

        # 1. Read raw spectral cube (BSQ: 224 x H x W, int16)
        raw_cube = self.file_helper.extract_specific_bands(bands=[], mode="all")
        logger.info("Raw cube shape: %s, dtype: %s", raw_cube.shape, raw_cube.dtype)

        # 2. Build nodata mask (1=valid, 0=nodata)
        nodata_mask = (raw_cube != metadata.spatial_info.background_value).astype(
            np.int8
        )

        # 3. Transform DN to surface reflectance
        sr_cube = self.dn_to_reflectance_transformer.transform(
            input_data=raw_cube,
            nodata_value=metadata.spatial_info.background_value,
        )
        logger.info("Surface reflectance cube shape: %s", sr_cube.shape)

        # 4. Read pixel validity mask (224 x H x W, uint8)
        #    EnMAP QL_PIXELMASK convention: 0=valid, 1=flagged (invalid).
        #    Invert so 1=valid, 0=invalid to match our internal convention.
        pixel_mask_raw = self.file_helper.extract_pixel_mask()
        pixel_mask = (1 - pixel_mask_raw).astype(np.uint8)

        # 5. Combine validity: nodata AND pixel mask
        overall_validity = nodata_mask * pixel_mask.astype(np.int8)

        # 6. Build per-band metadata arrays
        spectral_family_order = []
        band_cw_order = []
        band_fwhm_order = []
        band_validity_by_position = []
        vnir_channels = set(metadata.detector_boundary.vnir_expected_channels)

        for band_char in metadata.band_characterisation:
            if band_char.band_number in vnir_channels:
                spectral_family_order.append(SpectralFamily.VNIR)
            else:
                spectral_family_order.append(SpectralFamily.SWIR)
            band_cw_order.append(band_char.wavelength_center)
            band_fwhm_order.append(band_char.fwhm)
            # Ashwin's Note:
            # Band validity as a concept does not exist in EnMap as opposed to PRISMA
            # So all bands are marked as valid.
            band_validity_by_position.append(1)

        # 7. Read quality masks
        logger.info("Extracting Quality Mask")
        cloud_mask = self.file_helper.extract_quality_mask(
            EnmapFileComponents.QUALITY_CLOUD
        )
        cirrus_mask = self.file_helper.extract_quality_mask(
            EnmapFileComponents.QUALITY_CIRRUS
        )
        haze_mask = self.file_helper.extract_quality_mask(
            EnmapFileComponents.QUALITY_HAZE
        )
        cloud_shadow_mask = self.file_helper.extract_quality_mask(
            EnmapFileComponents.QUALITY_CLOUDSHADOW
        )
        snow_mask = self.file_helper.extract_quality_mask(
            EnmapFileComponents.QUALITY_SNOW
        )

        # Compute per-band valid pixel percentage from the validity cube (C, H, W).
        pixels_per_band = overall_validity.shape[1] * overall_validity.shape[2]
        band_level_validity_score = [
            float(overall_validity[b].sum() / pixels_per_band * 100.0)
            for b in range(overall_validity.shape[0])
        ]

        # ── Apply band filtering if configured ──
        if band_filter_config is not None:
            logger.info("Applying spectral band filtering.")
            band_filter = SpectralBandFilter(
                band_wavelengths=band_cw_order,
                band_validity_flags=band_validity_by_position,
                exclusion_ranges=band_filter_config.exclusion_ranges,
                spectral_families=spectral_family_order,
                edge_bands_to_trim=band_filter_config.edge_bands_to_trim,
                band_level_validity_scores=band_level_validity_score,
                min_valid_pixel_pct=band_filter_config.min_valid_pixel_pct,
            )
            good_idx = band_filter.get_good_band_indices()
            summary = band_filter.summary()
            logger.info(
                "Band filter: %d → %d bands. Dropped: %d flag, %d wavelength, "
                "%d edge, %d coverage.",
                summary["total_bands"],
                summary["surviving"],
                summary["dropped_by_flags"]["count"],
                summary["dropped_by_wavelength"]["count"],
                summary["dropped_by_edge"]["count"],
                summary["dropped_by_coverage"]["count"],
            )

            idx_arr = np.array(good_idx)
            sr_cube = sr_cube[idx_arr]
            overall_validity = overall_validity[idx_arr]
            spectral_family_order = [spectral_family_order[i] for i in good_idx]
            band_cw_order = [band_cw_order[i] for i in good_idx]
            band_fwhm_order = [band_fwhm_order[i] for i in good_idx]
            band_validity_by_position = [band_validity_by_position[i] for i in good_idx]
            band_level_validity_score = [band_level_validity_score[i] for i in good_idx]

            # ── Quality mask invalidation (EnMAP-specific) ──
            quality_mask_map = {
                "cloud": cloud_mask,
                "cirrus": cirrus_mask,
                "haze": haze_mask,
                "cloud_shadow": cloud_shadow_mask,
                "snow": snow_mask,
            }
            composite_quality_mask = np.zeros(
                (overall_validity.shape[1], overall_validity.shape[2]), dtype=bool
            )
            applied_masks = []
            for mask_name in band_filter_config.quality_masks_to_apply:
                mask_arr = quality_mask_map.get(mask_name)
                if mask_arr is not None:
                    flagged = mask_arr > 0
                    n_flagged = int(flagged.sum())
                    if n_flagged > 0:
                        composite_quality_mask |= flagged
                        applied_masks.append((mask_name, n_flagged))

            n_quality_invalidated = int(composite_quality_mask.sum())
            total_pixels = overall_validity.shape[1] * overall_validity.shape[2]
            if n_quality_invalidated > 0:
                overall_validity[:, composite_quality_mask] = 0
                mask_detail = ", ".join(f"{n}={c}" for n, c in applied_masks)
                logger.info(
                    "Quality mask invalidation: %d / %d pixels (%.1f%%) flagged. "
                    "Breakdown: %s.",
                    n_quality_invalidated,
                    total_pixels,
                    n_quality_invalidated / total_pixels * 100.0,
                    mask_detail,
                )
            else:
                logger.info("Quality mask invalidation: no pixels flagged.")

            # ── Spatial masking: invalidate pixel columns with too many bad voxels ──
            num_bands = overall_validity.shape[0]
            valid_count_per_pixel = overall_validity.sum(axis=0)  # (H, W)
            invalid_fraction = 1.0 - (valid_count_per_pixel / num_bands)
            bad_pixel_mask = invalid_fraction > band_filter_config.max_invalid_voxel_fraction  # (H, W)
            num_masked = int(bad_pixel_mask.sum())
            total_pixels = bad_pixel_mask.shape[0] * bad_pixel_mask.shape[1]
            if num_masked > 0:
                overall_validity[:, bad_pixel_mask] = 0
                logger.info(
                    "Spatial masking: %d / %d pixels (%.1f%%) invalidated "
                    "(>%.0f%% voxels invalid).",
                    num_masked,
                    total_pixels,
                    num_masked / total_pixels * 100.0,
                    band_filter_config.max_invalid_voxel_fraction * 100.0,
                )
            else:
                logger.info("Spatial masking: no pixels exceeded the invalid voxel threshold.")

            # ── Spectral interpolation: fill gaps in partially-valid pixels ──
            interpolate_spectral_gaps(
                cube=sr_cube,
                validity_cube=overall_validity,
                band_wavelengths=np.array(band_cw_order, dtype=np.float64),
            )

            # ── Spectral resampling to common wavelength grid ──
            if band_filter_config.common_wavelength_grid is not None:
                sr_cube, overall_validity, spectral_family_order = resample_to_common_grid(
                    cube=sr_cube,
                    validity_cube=overall_validity,
                    source_wavelengths=np.array(band_cw_order, dtype=np.float64),
                    target_wavelengths=band_filter_config.common_wavelength_grid,
                    spectral_families=spectral_family_order,
                )
                band_cw_order = band_filter_config.common_wavelength_grid.tolist()
                band_fwhm_order = []  # FWHM not meaningful after resampling
                band_validity_by_position = [1] * len(band_cw_order)

            # Recompute band-level validity scores after full pipeline
            pixels_per_band = overall_validity.shape[1] * overall_validity.shape[2]
            band_level_validity_score = [
                float(overall_validity[b].sum() / pixels_per_band * 100.0)
                for b in range(overall_validity.shape[0])
            ]

            # Recompute channel indices from filtered families
            vnir_channel_indices = [
                i for i, f in enumerate(spectral_family_order) if f == SpectralFamily.VNIR
            ]
            swir_channel_indices = [
                i for i, f in enumerate(spectral_family_order) if f == SpectralFamily.SWIR
            ]
        else:
            vnir_channel_indices = metadata.detector_boundary.vnir_expected_channels
            swir_channel_indices = metadata.detector_boundary.swir_expected_channels

        logger.info("Vendable EnMAP dataset assembled. Cube shape: %s", sr_cube.shape)
        return VendableEnmapHyperspectralDataset(
            normalized_hyperspectral_cube=sr_cube,
            validity_cube=overall_validity,
            spectral_family_order=spectral_family_order,
            band_cw_order=band_cw_order,
            band_fwhm_order=band_fwhm_order,
            band_validity_by_position=band_validity_by_position,
            band_level_validity_score=band_level_validity_score,
            cloud_mask=cloud_mask,
            cirrus_mask=cirrus_mask,
            haze_mask=haze_mask,
            cloud_shadow_mask=cloud_shadow_mask,
            snow_mask=snow_mask,
            vnir_channel_indices=vnir_channel_indices,
            swir_channel_indices=swir_channel_indices,
        )
