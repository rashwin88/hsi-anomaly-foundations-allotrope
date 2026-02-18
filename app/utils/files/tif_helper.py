"""
Implements a concrete TIFHelper class
"""

from typing import Dict, List, Optional, Literal
import logging

import numpy as np
import rasterio
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.file_metadata_models import TIFMetadata, TIFProperty
from app.models.hyperspectral_concepts.file_components import ThermalComponents
from app.models.hyperspectral_concepts.references import ReferenceDefinition
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.abstract_classes.file_helper import FileHelper


logger = logging.getLogger("TIFHelper")
logger.setLevel(logging.INFO)


class TIFHelper(FileHelper):
    """
    A helper class for working with tif files.
    """

    def __init__(
        self,
        file_source_config: FileSourceConfig,
        template: Dict[ThermalComponents, ReferenceDefinition],
    ):
        """
        Class constructor for file loading
        """
        super().__init__(file_source_config=file_source_config, template=template)
        self._file_source_config: FileSourceConfig = file_source_config
        self._file_metadata: TIFMetadata = self._construct_metadata_structure()

    @property
    def file_metadata(self) -> TIFMetadata:
        """
        Returns the file metadata
        """
        return self._file_metadata

    @property
    def template(self) -> Dict:
        return self._template

    def _construct_metadata_structure(self) -> TIFMetadata:
        """
        Constructs a metadata structure for the TIF file
        """
        # Initialize the property dictionary
        property_dict: Dict[str, TIFProperty] = {}

        logger.debug(
            "Constructing metadata structure for TIF file: %s",
            self.file_source_config.source_path,
        )

        # First pull information from profile
        logger.debug(
            "Pulling information from profile: %s", self.file_source_config.source_path
        )
        profile = rasterio.open(self.file_source_config.source_path).profile
        for key, value in profile.items():
            property_dict[key] = TIFProperty(name=key, value=value)

        # Now pull information from tags
        logger.debug(
            "Pulling information from tags: %s", self.file_source_config.source_path
        )
        tags = rasterio.open(self.file_source_config.source_path).tags()
        for key, value in tags.items():
            property_dict[key] = TIFProperty(name=key, value=value)

        # Now pull information from bounds
        logger.debug(
            "Pulling information from bounds: %s", self.file_source_config.source_path
        )
        bounds = rasterio.open(self.file_source_config.source_path).bounds
        property_dict["bounds"] = TIFProperty(name="bounds", value=bounds)
        return TIFMetadata(metadata=property_dict)

    # spectral family need not be relevant for TIF files and thermal datasets
    # But since the abstract class defines the method, it is kept here are optional with a default of none
    # The value will not be used.
    def extract_specific_bands(
        self,
        bands: List[int] | int = 1,
        masking_needed: bool = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> np.ma.MaskedArray | np.ndarray:
        """
        Extracts bands from the TIF file.
        Refer to the base class for method docstring.
        """
        if isinstance(bands, int):
            bands = [bands]
        logger.debug(
            "Extracting bands: %s from %s",
            bands,
            self.file_source_config.source_path,
        )
        try:
            with rasterio.open(self.file_source_config.source_path) as src:
                # Read only specific bands or read in all the bands.
                # Notice here that the masking concepts are slightly different and need
                # some refinement. The mask is already coming out as part of the dataset pull.
                # This is great, but needs some more work to understand fully.
                if mode == "specific":
                    band_data = src.read(bands, masked=masking_needed)
                else:
                    band_data = src.read(masked=masking_needed)
            return band_data
        except Exception as e:
            logger.error(
                "Error extracting bands %s from %s : %s",
                bands,
                self.file_source_config.source_path,
                e,
            )
            raise e
