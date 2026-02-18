"""
Defines an abstract class for dataset builder
"""

from typing import Dict, Union
from abc import ABC, abstractmethod
from pystac import Item


# models imported
from app.models.file_processing.sources import FileSourceConfig
from app.models.images.cube_representation import CubeRepresentation

from app.models.hyperspectral_concepts.band import (
    HyperpectralBandInformation,
)
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.dataset.vendables import VendableHyperspectralDataset

# Other abstract classes
from app.abstract_classes.file_helper import FileHelper

# utility classes
from app.utils.files.he5_helper import HE5Helper
from app.utils.files.tif_helper import TIFHelper


class DatasetBuilder(ABC):
    """
    Abstract class definition for dataset builder.

    Assume that we have a file that is supplied to us. In this case, the file can be of any format,
    it can be from any sensor and so on. The way we process these files and collect data and standardize them
    will differ depending upon the type of sensor and type of file. But, the methods we use will remain common.

    This abstract class defines what a dataset builder must do to make a dataset usable.
    """

    def __init__(self, file_source_configuration: FileSourceConfig):
        """
        Initialize with a file source configuration
        always
        """
        self.file_source_config = file_source_configuration

    @property
    @abstractmethod
    def file_helper(self) -> FileHelper:
        """
        A file helper to deal with the complexities of files
        """
        pass

    @property
    @abstractmethod
    def band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation] | None:
        """
        The band information contained in the dataset neatly organized if available
        """
        pass

    @abstractmethod
    def extract_band_information(
        self,
    ) -> Dict[SpectralFamily, HyperpectralBandInformation] | None:
        """
        Extrtacts band information from the dataset if applicable
        """
        pass

    @property
    @abstractmethod
    def stac_item(self) -> Item:
        """
        The stac item for the dataset
        """
        pass

    @property
    @abstractmethod
    def default_cube_representation(self) -> CubeRepresentation:
        """
        The default cube representation for the dataset. Will be different depending on sensor, provider and
        dataset category. Thermal is BSQ, Hyperspectral is usually BIL
        """
        pass

    @abstractmethod
    def initialize_helper(self) -> Union[HE5Helper, TIFHelper]:
        """
        Initializes the helper and populates the helper property
        """
        pass

    @abstractmethod
    def vend_dataset(self) -> Union[VendableHyperspectralDataset]:
        """
        Constructs the complete dataset in a form that can be used in downstream
        applications in full.
        """
        pass
