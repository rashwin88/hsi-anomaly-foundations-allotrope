"""
Concrete implementation of the Prisma Dataset Builder
"""

import logging
from typing import Dict

from app.abstract_classes.dataset_builder import DatasetBuilder
from app.abstract_classes.file_helper import FileHelper
from app.models.file_processing.sources import FileSourceConfig
from app.models.images.cube_representation import CubeRepresentation
from app.utils.files.he5_helper import HE5Helper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.hyperspectral_concepts.band import (
    HyperpectralBandInformation,
    HyperSpectralBand,
    WavelengthMeasurementUnits,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily


class PrismaDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for prisma data
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        """
        Class Constructor
        """
        super().__init__(file_source_configuration=file_source_configuration)
        # Load in the file helper
        self._file_helper: HE5Helper = self.initialize_helper()
        self._band_information = self.extract_band_information()

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

    def extract_band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation]:
        """
        Constructs the Hyperspectral band information look up by using the metadata
        """

        output_dict = {}

        measurement_units = WavelengthMeasurementUnits.NANO_METERS

        # Collect file attributes
        file_attributes = self.file_helper.file_metadata.file_attributes

        # This dataset should be easy to produce and we can effectively just do this in two steps
        # First we collect SWIR bands and then we collect VNIR

        swir: SpectralFamily = SpectralFamily.SWIR

        # zip together different pieces of information
        information_package = zip(
            file_attributes.get("List_Cw_Swir"),
            file_attributes.get("List_Cw_Swir_Flags"),
            file_attributes.get("List_Fwhm_Swir"),
        )

        # Loop through the information package and construct the swir dict.
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

        # Loop through the information package and construct the swir dict.
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

        output_dict[vnir] = HyperpectralBandInformation(
            bands_by_index=vnir_by_index, bands_by_wavelength=vnir_by_wavelength
        )

        return output_dict
