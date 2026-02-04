"""
Defines the abstract notion of a file helper
Very useful foundational class to abstract away all the individual complexities of
dealing with very hierarchical and complicated datasets
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, List, Optional, Literal, Dict
import numpy as np

from pydantic import BaseModel


# Local models
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.file_categories import FileCategory
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.products.products import Product
from app.models.hyperspectral_concepts.references import ReferenceDefinition
from app.models.hyperspectral_concepts.file_components import (
    HyperspectralFileComponents,
)


# using a generic pattern here
T = TypeVar("T", bound=BaseModel)


class FileHelper(ABC, Generic[T]):
    """
    Abstract notion of a file helper
    """

    def __init__(self, file_source_config: FileSourceConfig):
        self.file_source_config = file_source_config

    @property
    @abstractmethod
    def file_category(self) -> FileCategory:
        """
        The file category for the helper
        """
        pass

    def access_dataset(self, path: str) -> Any:
        """
        Most times the datasets are loaded in a lazy manner. This allows pulling actual
        content from the dataset. Subclasses will over-ride this but donot have to in reality
        """
        pass

    @abstractmethod
    def _construct_metadata_structure(self) -> T:
        """
        Constructs the metadata structure for the file
        """
        pass

    @property
    @abstractmethod
    def file_metadata(self) -> T:
        """
        The actual file metadata that is extracted
        """
        pass

    @abstractmethod
    def extract_specific_bands(
        self,
        bands: List[int],
        masking_needed: Optional[bool] = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> np.ndarray | np.ma.MaskedArray:
        """
        Extracts bands from the dataset. This is different from actually accessing different elements of the dataset.
        Think of extracting a band and operating upon an element of the dataset itself.

        Args:
            bands (List[int]): A list of all the bands that need to extracted from the dataset.
            masking_needed (Optional[bool]): Whether masking needs to be applied on the dataset that is extracted
            spectral_family (Optional[SpectralFamily]): The spectral family of the bands to be extracted.
            mode (Literal) : Specifies if all or specific bands need to be extracted.
        """
        pass

    @property
    @abstractmethod
    def product(self) -> Product:
        """
        The product from which this file is produced
        """
        pass

    @property
    @abstractmethod
    def template(self) -> Dict[HyperspectralFileComponents, ReferenceDefinition]:
        """
        The reference template corresponding to the product
        """
        pass
