"""
Defines sources for files
"""

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from app.models.file_processing.file_categories import FileCategory


class FileDownloaderType(Enum):
    """
    Defines the type of file downloader to use
    """

    GOOGLE_DRIVE = "google_drive"
    S3 = "s3"
    DOWNLOAD_LINK = "download_link"


class FileSourceConfig(BaseModel):
    """
    Configuration for the local file path
    """

    source_path: str = Field(description="The source of the file")
    file_category: Optional[FileCategory] = Field(
        default=None, description="The category of the file"
    )

    # Directly infer the file category in case it is not specified.
    # Can be used in downstream processing to switch processing methods.
    @model_validator(mode="after")
    def infer_file_category(self) -> "FileSourceConfig":
        """
        Infers the file category based on the file extension
        """
        if self.file_category is None:
            # .h5 / .hdf5: Planet Tanager's HDF-EOS5 product (TanagerHelper).
            # .hdf / .nc / .nc4 / .tiff: the generic_hsi reader (GenericHsiHelper).
            if self.source_path.lower().endswith((".he5", ".h5", ".hdf5", ".hdf", ".nc", ".nc4", ".cdf")):  # pylint: disable=no-member
                self.file_category = FileCategory.HDFS
            elif self.source_path.lower().endswith((".tif", ".tiff")):  # pylint: disable=no-member
                self.file_category = FileCategory.TIF
            elif self.source_path.lower().endswith(".hdr"):  # pylint: disable=no-member
                # Tanager delivered as ENVI: the builder is handed the header (tanager_envi.py).
                self.file_category = FileCategory.ENVI_HEADER
            elif "ENMAP" in self.source_path and "L2A" in self.source_path:
                self.file_category = FileCategory.ENMAP_FOLDER
            elif os.path.isdir(self.source_path) and any(
                    f.lower().endswith(".hdr") for f in os.listdir(self.source_path)):
                # AVIRIS-NG (and GHOSt converted to it): the builder is handed the flight-line folder (known issue 13)
                self.file_category = FileCategory.ENVI_FOLDER
            else:
                raise ValueError("Unrecognized File Extension")
        return self
