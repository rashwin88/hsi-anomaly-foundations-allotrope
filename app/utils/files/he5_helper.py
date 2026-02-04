"""
Implementation of the FileHelper for
HE5 files
"""

import logging
from typing import Dict, Any, Literal, List, Optional

import numpy as np
import h5py
from h5py import Dataset

# Abstract classes
from app.abstract_classes.file_helper import FileHelper
from app.models.products.products import Product
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.file_processing.sources import FileSourceConfig
from app.models.file_processing.file_metadata_models import (
    He5ComponentMetadata,
    He5Metadata,
)
from app.models.hyperspectral_concepts.file_components import (
    HyperspectralFileComponents,
)
from app.models.hyperspectral_concepts.references import ReferenceDefinition

from app.templates.template_mappings import TEMPLATE_MAPPINGS


logger = logging.getLogger("He5Helper")
logger.setLevel(logging.INFO)


class HE5Helper(FileHelper):
    """
    A helper class for working with HE5 files.
    Operates on multiple file sources.

    ** Abstract Class : FileHelper **
    """

    def __init__(self, file_source_config: FileSourceConfig, product: Product = None):
        """
        Class constructor for file loading
        """
        # Initialize the file loader helper
        super().__init__(file_source_config=file_source_config)

        self.product = product
        # The raw structure in itself can be used to access the metadata and the actual content in the file.
        self.raw_structure: h5py.File = h5py.File(
            self.file_source_config.source_path, "r"
        )
        self.file_metadata: He5Metadata = self._construct_metadata_structure()
        # Get the template mappings
        self.template: Dict[HyperspectralFileComponents, ReferenceDefinition] = (
            TEMPLATE_MAPPINGS.get(self.product)
        )

        # Additional things needed for proper file handling
        self.masked_pixel_value: int = 0

    def access_dataset(self, path: str) -> Any:
        """
        Given a path, returns the dataset at that path
        """
        # First check if the path is a dataset
        if path not in self.file_metadata.components:
            raise TypeError(f"Path {path} not found in the file")
        if not isinstance(self.raw_structure[path], Dataset):
            raise TypeError(f"Path {path} is not a dataset")
        return self.raw_structure[path][()]

    def _get_clean_attrs(self, key_path: str | None = None) -> Dict[str, Any]:
        """
        Gets clean attributes for a given key path
        """
        clean_attrs: Dict = {}
        if key_path is None:
            # meant for root level metadata tags if any
            metadata = self.raw_structure
        else:
            metadata = self.raw_structure[key_path]
        for k, v in metadata.attrs.items():
            if isinstance(v, bytes):
                clean_attrs[k] = v.decode("utf-8", errors="ignore")
            else:
                clean_attrs[k] = v
        return clean_attrs

    def _construct_metadata_structure(self) -> He5Metadata:
        """
        Constructs a metadata structure for the He5 file
        """
        output = He5Metadata()
        metadata_paths = []
        metadata_structure = {}

        # First we need to get the root metadata
        root_meta = He5ComponentMetadata()
        root_meta.type = type(self.raw_structure)
        root_meta.shape = None
        root_meta.is_scalar = False
        # Root level metadata extracted
        root_meta.file_attributes = self._get_clean_attrs()
        output.root_metadata = root_meta

        # Visit each key in the raw structure and add it to the metadata paths
        self.raw_structure.visit(lambda name: metadata_paths.append(name))
        output.components = metadata_paths

        # Now for each path build up the metdata structure
        for path in metadata_paths:
            metadata = He5ComponentMetadata()
            metadata.type = type(self.raw_structure[path])
            metadata_structure[path] = metadata
            # Check if this is a dataset
            if isinstance(self.raw_structure[path], Dataset):
                metadata.shape = self.raw_structure[path].shape
                # We can add a small check to see if it a tensor or a scalar
                if metadata.shape == ():
                    metadata.is_scalar = True
                else:
                    metadata.is_scalar = False
            metadata.file_attributes = self._get_clean_attrs(path)
            output.component_metadata[path] = metadata
        return output

    def extract_specific_bands(
        self,
        bands: List[int],
        masking_needed: Optional[bool] = False,
        spectral_family: Optional[SpectralFamily] = None,
        mode: Literal["all", "specific"] = "specific",
    ) -> np.ndarray | np.ma.MaskedArray:
        """
        Extracts bands from the dataset.
        Refer to base class for documentation.
        """
        # First we access the dataset and store it.
        # To do that we need the spectral family
        if spectral_family == SpectralFamily.SWIR:
            path = self.template.get(
                HyperspectralFileComponents.SWIR_CUBE_DATA
            ).file_name
        elif spectral_family == SpectralFamily.VNIR:
            path = self.template.get(
                HyperspectralFileComponents.VNIR_CUBE_DATA
            ).file_name
        else:
            raise KeyError(
                "Mapping Key for Spectral Family not found",
            )

        # Now that we have the path we can perform further operations
        # Access the data in the path
        raw_cube = self.access_dataset(path)
        # Slice out only the bands in the cube that matter


## Local testing
if __name__ == "__main__":
    file_source_config = FileSourceConfig(
        source_path="raw_files/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
    )
    he5_helper = HE5Helper(file_source_config)
    root_attributes = he5_helper.file_metadata.root_metadata.file_attributes
    ## understanding VNIR
    vnir_count = len(root_attributes["List_Cw_Vnir"])
    vnir_non_zero_count = sum(
        1 for x in root_attributes["List_Cw_Vnir_Flags"] if x == 1
    )
    vnir_fwhm_count = len(root_attributes["List_Fwhm_Vnir"])
    vnir_cube_shape = he5_helper.file_metadata.component_metadata[
        "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
    ].shape
    print(f"VNIR Count: {vnir_count}")
    print(f"VNIR Non Zero Count: {vnir_non_zero_count}")
    print(f"VNIR FWHM Count: {vnir_fwhm_count}")
    print(f"VNIR Cube Shape: {vnir_cube_shape}")
    ## understanding SWIR
    swir_count = len(root_attributes["List_Cw_Swir"])
    swir_non_zero_count = sum(
        1 for x in root_attributes["List_Cw_Swir_Flags"] if x == 1
    )
    swir_fwhm_count = len(root_attributes["List_Fwhm_Swir"])
    swir_cube_shape = he5_helper.file_metadata.component_metadata[
        "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
    ].shape
    print(f"SWIR Count: {swir_count}")
    print(f"SWIR Non Zero Count: {swir_non_zero_count}")
    print(f"SWIR FWHM Count: {swir_fwhm_count}")
    print(f"SWIR Cube Shape: {swir_cube_shape}")
