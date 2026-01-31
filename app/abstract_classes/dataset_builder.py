"""
Defines an abstract class for dataset builder
"""

from typing import Union, List, Optional
from abc import ABC, abstractmethod

# models imported
from app.models.file_processing.sources import FileSourceConfig
from app.models.dataset.dataset_categories import DatasetCategory
from app.models.images.cube_representation import CubeRepresentation
from app.models.intermediate_concepts.band_by_index_response import BandByIndexResponse
from app.models.intermediate_concepts.band_by_wavelength_response import BandByWavelengthResponse
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

# utility classes
from app.utils.files.he5_helper import HE5Helper
from app.utils.files.tif_helper import TIFHelper
import numpy as np


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
    def dataset_category(self) -> DatasetCategory:
        """
        The category of dataset that we are loading.
        In the case of the specific problem we are solving, it can be Thermal or Hyperspectral
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

    @property
    @abstractmethod
    def file_helper(self):
        """
        The helper is the basic file helper that will be the entry point for all
        file level operations
        """
        pass

    @abstractmethod
    def initialize_helper(self) -> Union[HE5Helper, TIFHelper]:
        """
        Initializes the helper and populates the helper property
        """
        pass

    @abstractmethod
    def collect_raw_bands_by_index(
        self,
        bands: List[int],
        output_representation: CubeRepresentation,
        masking_required: Optional[bool] = None,
        spectral_family: Optional[SpectralFamily] = None,
        normalization_needed: Optional[bool] = False
    ) -> BandByIndexResponse:
        """
        Collects the raw bands from the image when specific index bands are given.

        Args:
            bands (List[int]): The list of band indices that we are requesting from the dataset.
            output_representation (CubeRepresentation): The output cube's desired representation.
        """
        pass

    @abstractmethod
    def collect_raw_bands_by_wavelength(
        self,
        wavelengths: List[float],
        output_representation: CubeRepresentation,
        error_threshold_in_percentage: float = 0.05,
        masking_required: Optional[int] = None,
        spectral_family: Optional[SpectralFamily] = None,
        normalization_needed: Optional[bool] = None
    )
