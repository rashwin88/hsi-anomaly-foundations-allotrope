from enum import Enum
from pydantic import BaseModel, Field


class FileSource(Enum):
    GOOGLE_DRIVE = "google_drive"
    LOCAL_FILE = "local_file"
    S3 = "s3"
    DOWNLOAD_LINK = "downloader"


class FileSourceConfig(BaseModel):
    source_path: str = Field(description="The source of the file")
