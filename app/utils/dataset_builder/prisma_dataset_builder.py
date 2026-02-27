"""
Concrete implementation of the Prisma Dataset Builder
"""

import logging
from typing import Dict, Union, List

import numpy as np
from pystac import Item

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.abstract_classes.file_helper import FileHelper
from app.models.dataset.vendables import VendableHyperspectralDataset
from app.models.file_processing.sources import FileSourceConfig
from app.models.images.cube_representation import CubeRepresentation
from app.utils.files.he5_helper import HE5Helper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.hyperspectral_concepts.band import (
    HyperpectralBandInformation,
    HyperSpectralBand,
    WavelengthMeasurementUnits,
)
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

from app.utils.data_transformations.prs_l2d_dn_to_surface_reflectance_transformer import (
    PrsL2dDnToSurfaceReflectanceTransformer,
)
from app.utils.stac.stac_utils.stac_items import StacCreator

logger = logging.getLogger("PrismaDatasetBuilder")
logger.setLevel(logging.INFO)


class PrismaDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for PRISMA L2D hyperspectral data.

    Responsibilities:
    - Read PRISMA metadata and build band lookup tables for SWIR/VNIR.
    - Convert DN values to surface reflectance using the configured transformer.
    - Assemble a normalized hyperspectral cube and validity masks for downstream use.
    - Produce a vendable dataset in a canonical BSQ representation.
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        """
        Initializes the builder and prepares metadata and helpers.
        """
        super().__init__(file_source_configuration=file_source_configuration)
        # Create the STAC item as early as possible for metadata access.
        logger.info("Creating STAC item for PRISMA dataset.")
        self._stac_item = StacCreator(
            file_path=self.file_source_config.source_path
        ).build_stack()
        # Load the helper for HE5 access and parse band metadata.
        logger.info("Initializing HE5 helper and extracting band metadata.")
        self._file_helper: HE5Helper = self.initialize_helper()
        self._band_information = self.extract_band_information()
        logger.info("Band metadata loaded for SWIR/VNIR.")
        self.dn_to_reflectance_transformer = PrsL2dDnToSurfaceReflectanceTransformer()
        logger.info("Loaded all transformations")
        self.cube_reshaper = ImageCubeOperations()

    @property
    def stac_item(self) -> Item:
        return self._stac_item

    @property
    def file_helper(self) -> FileHelper:
        return self._file_helper

    @property
    def band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation] | None:
        return self._band_information

    @property
    def default_cube_representation(self) -> CubeRepresentation:
        return CubeRepresentation.BIL

    def initialize_helper(self) -> HE5Helper:
        return HE5Helper(
            file_source_config=self.file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL),
        )

    def _transformation_pipeline(
        self,
        input_data: Union[np.ndarray, np.ma.MaskedArray],
        band_mapping: List[SpectralFamily],
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        A transformation pipeline for the dataset.
        """
        # Convert to reflectance values
        dn_to_ref_transformed = self.dn_to_reflectance_transformer.transform(
            input_data=input_data,
            cube_representation=self.default_cube_representation,
            band_mapping=band_mapping,
            file_metadata=self.file_helper.file_metadata,
            masking_indicator=0,
        )
        return dn_to_ref_transformed

    def extract_band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """
        Constructs the hyperspectral band lookup tables using file metadata.

        Produces a mapping keyed by spectral family with band info indexed
        both by position and center wavelength.
        """

        logger.info("Extracting band information from metadata.")
        output_dict = {}

        measurement_units = WavelengthMeasurementUnits.NANO_METERS

        # Collect file attributes
        file_attributes = self.file_helper.file_metadata.root_metadata.file_attributes

        # This dataset should be easy to produce and we can effectively just do this in two steps
        # First we collect SWIR bands and then we collect VNIR

        swir: SpectralFamily = SpectralFamily.SWIR

        # zip together different pieces of information
        information_package = zip(
            file_attributes.get("List_Cw_Swir"),
            file_attributes.get("List_Cw_Swir_Flags"),
            file_attributes.get("List_Fwhm_Swir"),
        )

        # Loop through the information package and construct the SWIR tables.
        swir_by_wavelength = {}
        swir_by_index = {}

        index = 0
        for cw, cw_flag, fwhm in information_package:
            band = HyperSpectralBand(
                wavelength=cw,
                wavelength_measurement_unit=measurement_units,
                band_index=index,
                full_width_at_half_maximum=fwhm,
                is_valid=bool(cw_flag),
            )
            swir_by_index[index] = band
            swir_by_wavelength[cw] = band
            index += 1

        output_dict[swir] = HyperpectralBandInformation(
            bands_by_index=swir_by_index, bands_by_wavelength=swir_by_wavelength
        )

        # Now do the same for VNIR
        vnir: SpectralFamily = SpectralFamily.VNIR

        del information_package

        # zip together different pieces of information
        information_package = zip(
            file_attributes.get("List_Cw_Vnir"),
            file_attributes.get("List_Cw_Vnir_Flags"),
            file_attributes.get("List_Fwhm_Vnir"),
        )

        # Loop through the information package and construct the VNIR tables.
        vnir_by_wavelength = {}
        vnir_by_index = {}

        index = 0
        for cw, cw_flag, fwhm in information_package:
            band = HyperSpectralBand(
                wavelength=cw,
                wavelength_measurement_unit=measurement_units,
                band_index=index,
                full_width_at_half_maximum=fwhm,
                is_valid=bool(cw_flag),
            )
            vnir_by_index[index] = band
            vnir_by_wavelength[cw] = band
            index += 1

        output_dict[vnir] = HyperpectralBandInformation(
            bands_by_index=vnir_by_index, bands_by_wavelength=vnir_by_wavelength
        )

        logger.info("Band information extracted for SWIR and VNIR.")
        return output_dict

    def vend_dataset(self, **kwargs) -> VendableHyperspectralDataset:
        """
        Vends the full hyperspectral dataset that is usable by downstream applications.
        For convention and ease of use, we force it to be a BSQ format in representation.
        """

        # Assemble a normalized cube that merges SWIR and VNIR and build validity masks.
        logger.info("Building vendable hyperspectral dataset.")

        # First we will construct the normalized cube
        # We will arrange as SWIR followd by VNIR
        spectral_family_by_position = []
        band_validity_by_position = []
        band_cw_by_position = []

        output_cubes = []
        error_pixel_cubes = []
        invalid_value_masks = []

        processing_order = [SpectralFamily.SWIR, SpectralFamily.VNIR]

        for family in processing_order:
            logger.info("Processing spectral family: %s", family)
            bands = self.band_information.get(family)
            indices = sorted([*bands.bands_by_index.keys()])
            # Record ordering, validity flags, and wavelength metadata for downstream use.
            for index in indices:
                spectral_family_by_position.append(family)
                band_detail = bands.bands_by_index.get(index)
                band_validity_by_position.append(int(band_detail.is_valid))
                band_cw_by_position.append(band_detail.wavelength)

            # Pull the unnormalized cube for the family and compute validity masks.
            unnormalized_cube = self.file_helper.extract_specific_bands(
                bands=[],
                masking_needed=False,
                spectral_family=family,
                mode="all",
            )
            # Validity mask where values are non-zero (1 = valid).
            invalid_value_masks.append((unnormalized_cube != 0.0).astype(np.int8))
            # Normalize DN values to reflectance.
            normalized_cube = self._transformation_pipeline(
                input_data=unnormalized_cube,
                band_mapping=[family] * unnormalized_cube.shape[1],
            )
            output_cubes.append(normalized_cube)
            # Error pixels are 0 when valid (1 = valid).
            error_pixels = self.file_helper.extract_error_matrices(
                bands=[], spectral_family=family, mode="all"
            )
            error_pixel_cubes.append((error_pixels == 0.0).astype(np.int8))
            logger.info(
                "Family %s processed. Cube shape: %s", family, unnormalized_cube.shape
            )

        logger.info("Concatenating cubes and assembling masks.")
        output_cube = np.concatenate(output_cubes, axis=1)
        logger.info("Output Cube Intermediate BIL Shape %s", output_cube.shape)
        invalid_value_cube = np.concatenate(invalid_value_masks, axis=1)
        logger.info(
            "Invalid Value Cube Intermediate BIL Shape %s", invalid_value_cube.shape
        )
        error_pixel_cube = np.concatenate(error_pixel_cubes, axis=1)
        logger.info(
            "Error Pixel Cube Intermediate BIL Shape %s", error_pixel_cube.shape
        )

        # Broadcast per-band validity into a cube (1 = valid).
        band_validity = np.asarray(band_validity_by_position, dtype=np.uint8)
        valid_band_cube = np.broadcast_to(
            band_validity[None, :, None], output_cube.shape
        )

        # Combine all validity signals into a single mask.
        overall_validity_mask = valid_band_cube * error_pixel_cube * invalid_value_cube

        logger.info("Vendable dataset assembled. Cube shape: %s", output_cube.shape)
        # Reshape and produce vendable
        return VendableHyperspectralDataset(
            normalized_hyperspectral_cube=self.cube_reshaper.convert_cube(
                cube=output_cube,
                from_format=self.default_cube_representation,
                to_format=CubeRepresentation.BSQ,
            ),
            validity_cube=self.cube_reshaper.convert_cube(
                cube=overall_validity_mask,
                from_format=self.default_cube_representation,
                to_format=CubeRepresentation.BSQ,
            ),
            spectral_family_order=spectral_family_by_position,
            band_cw_order=band_cw_by_position,
        )
