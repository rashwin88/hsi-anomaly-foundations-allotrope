"""
Defines sources for files
"""

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
            if self.source_path.lower().endswith(".he5"):  # pylint: disable=no-member
                self.file_category = FileCategory.HDFS
            elif self.source_path.lower().endswith(".tif"):  # pylint: disable=no-member
                self.file_category = FileCategory.TIF
            else:
                raise ValueError("Unrecognized File Extension")
        return self
