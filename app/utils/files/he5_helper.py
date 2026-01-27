import h5py
from pydantic import ValidationError
from app.models.file_processing.sources import FileSourceConfig
from app.utils.files.file_loader_helper import FileLoaderHelper
from typing import Dict, List, Any
from app.models.file_processing.file_metadata_models import (
    FullMetadata,
    ComponentMetadata,
)
from h5py import Group, Dataset

import logging

logger = logging.getLogger("He5Helper")
logger.setLevel(logging.INFO)


class HE5Helper:
    """
    A helper class for working with HE5 files.
    Operates on multiple file sources.
    """

    def __init__(self, file_source_config: FileSourceConfig):
        """
        Class constructor for file loading
        """

        # Initialize the file loader helper
        self.file_source_config: FileSourceConfig = file_source_config

        # The raw strcture in itself can be used to access the metadata and the actual content in the file.
        self.raw_structure: h5py.File = h5py.File(
            self.file_source_config.source_path, "r"
        )
        self.file_metadata: FullMetadata = self._construct_metadata_structure()

    def get_dataset(self, path: str) -> Any:
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
        clean_attrs = {}
        if key_path is None:
            metadata = self.raw_structure
        else:
            metadata = self.raw_structure[key_path]
        for k, v in metadata.attrs.items():
            if isinstance(v, bytes):
                clean_attrs[k] = v.decode("utf-8", errors="ignore")
            else:
                clean_attrs[k] = v
        return clean_attrs

    def _construct_metadata_structure(self) -> FullMetadata:
        """

        Constructs a metadata structure for the He5 file
        """
        output = FullMetadata()
        metadata_paths = []
        metadata_structure = {}

        # First we need to get the root metadata

        root_meta = ComponentMetadata()
        root_meta.type = type(self.raw_structure)
        root_meta.shape = None
        root_meta.is_scalar = False
        root_meta.file_attributes = self._get_clean_attrs()
        output.root_metadata = root_meta

        # Visit each key in the raw structure and add it to the metadata paths
        self.raw_structure.visit(lambda name: metadata_paths.append(name))
        output.components = metadata_paths

        # Now for each path build up the metdata structure
        for path in metadata_paths:
            metadata = ComponentMetadata()
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


## Local testing

if __name__ == "__main__":
    import pprint

    file_source_config = FileSourceConfig(
        source_path="raw_files/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
    )
    he5_helper = HE5Helper(file_source_config)
    pprint.pprint([*he5_helper.file_metadata.root_metadata.file_attributes.keys()])
