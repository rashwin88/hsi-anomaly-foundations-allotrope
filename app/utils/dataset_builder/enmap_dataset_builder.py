"""
Concrete implementation of the EnMAP L2A Dataset Builder.
Takes a folder path containing all scene files and produces
a VendableEnmapHyperspectralDataset.
"""

import logging
from typing import Dict

import numpy as np
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.models.dataset.vendables import VendableEnmapHyperspectralDataset
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

    def vend_dataset(self, **kwargs) -> VendableEnmapHyperspectralDataset:
        """
        Builds the full vendable EnMAP dataset.
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
        logger.info(f"Extracting Quality Mask")
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
            # TODO: I think these will not be needed we already have the spectral family order anyway
            vnir_channel_indices=metadata.detector_boundary.vnir_expected_channels,
            swir_channel_indices=metadata.detector_boundary.swir_expected_channels,
        )
