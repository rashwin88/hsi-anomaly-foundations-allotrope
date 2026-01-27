import h5py
from app.models.file_processing.sources import FileSource, FileSourceConfig
from app.utils.files.file_loader_helper import FileLoaderHelper
from typing import Dict, List, Any
from app.models.file_processing.file_metadata_models import FileMetadata
from h5py import Group, Dataset

import logging

logger = logging.getLogger("He5Helper")
logger.setLevel(logging.INFO)


class HE5Helper:
    """
    A helper class for working with HE5 files.
    Operates on multiple file sources.
    """

    def __init__(self, file_source: FileSource, file_source_config: FileSourceConfig):
        """
        Class constructor for file loading
        """

        # Initialize the file loader helper
        logger.info(f"Initializing file loader helper for file source: {file_source}")
        self.file_loader_helper = FileLoaderHelper(file_source)
        self.file_source_config = file_source_config
        self.file_metadata = self._get_file_metadata()

        # We may now use this to construct a master meta data object for the He5 file
        # This will be the basis for all processing that happens.

    def _get_file_metadata(self):
        """
        Gets all the required file metadata
        """
        logger.info(
            f"Getting file metadata for file: {self.file_source_config.source_path}"
        )
        file_metadata = h5py.File(self.file_source_config.source_path, "r")
        return file_metadata

    def _construct_metadata_structure(self) -> Dict[str, Any]:
        """

        Constructs a metadata structure for the He5 file
        """
        metadata_paths = []
        metadata_structure = {}
        # Populate
        self.file_metadata.visit(lambda name: metadata_paths.append(name))

        # Now for each path build up the metdata structure
        for path in metadata_paths:
            metadata = FileMetadata()
            metadata.type = type(self.file_metadata[path])
            metadata_structure[path] = metadata

            # Check if this is a dataset
            if isinstance(self.file_metadata[path], Dataset):
                metadata.shape = self.file_metadata[path].shape
                # We can add a small check to see if it a tensor or a scalar
                if metadata.shape == ():
                    metadata.is_scalar = True
                else:
                    metadata.is_scalar = False
        return metadata_structure


## Local testing

if __name__ == "__main__":
    file_source = FileSource.LOCAL_FILE
    file_source_config = FileSourceConfig(
        source_path="raw_files/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
    )
    he5_helper = HE5Helper(file_source, file_source_config)
    print(he5_helper.file_metadata)
    print(he5_helper._construct_metadata_structure())
