from enum import Enum
from pydantic import BaseModel, Field


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
