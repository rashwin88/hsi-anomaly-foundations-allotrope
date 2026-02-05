from typing import Dict, List
import logging

import numpy as np
import rasterio
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.file_metadata_models import TIFMetadata, TIFProperty
from app.models.images.cube_representation import CubeRepresentation
from app.models.products.products import Product
from app.abstract_classes.file_helper import FileHelper


logger = logging.getLogger("TIFHelper")
logger.setLevel(logging.INFO)


class TIFHelper(FileHelper):
    """
    A helper class for working with tif files.
    """

    def __init__(self, file_source_config: FileSourceConfig, product: Product = None):
        """
        Class constructor for file loading
        """
        # Initialize file Source Config
        self._file_source_config: FileSourceConfig = file_source_config
        self._product = product
        self._file_metadata: TIFMetadata = self._construct_metadata_structure()

    @property
    def file_source_config(self) -> FileSourceConfig:
        """
        Return the file source config
        """
        return self._file_source_config

    @property
    def product(self) -> Product:
        """
        Returns the product
        """
        return self._product

    @property
    def file_metadata(self) -> TIFMetadata:
        """
        Returns the file metadata
        """
        return self._file_metadata

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

    def extract_bands(
        self, bands: List[int] | int, masking_needed: bool = True
    ) -> np.ma.MaskedArray | np.ndarray:
        """Extracts bands from the TIF file.

        Args:
            bands: List[int] : The bands to extract
            masking_needed: bool = True : Whether to mask the bands in which case the return type is a MaskedArray

        Returns:
            np.ma.MaskedArray | np.ndarray : The extracted bands of the image
        """
        if isinstance(bands, int):
            bands = [bands]
        logger.debug(
            f"Extracting bands: {bands} from {self.file_source_config.source_path}"
        )
        try:
            with rasterio.open(self.file_source_config.source_path) as src:
                band_data = src.read(bands, masked=masking_needed)
            return band_data
        except Exception as e:
            logger.error(
                f"Error extracting bands {bands} from {self.file_source_config.source_path} : {e}"
            )
            raise e


if __name__ == "__main__":
    import logging
    import sys

    logger = logging.getLogger("ScriptLogger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    FILE_NAME = "raw_files/Thermal/LC09_L2SP_147049_20251121_20251122_02_T1_ST_B10.TIF"
    file_source_config = FileSourceConfig(source_path=FILE_NAME)
    tif_helper = TIFHelper(file_source_config)
    print(tif_helper.file_metadata)
