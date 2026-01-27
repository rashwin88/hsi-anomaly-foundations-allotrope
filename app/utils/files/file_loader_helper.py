"""
Defines a simple file loader helper class
that vends file loader method
"""

from app.errors.NotImplementedError import NotImplementedError
from app.models.file_processing.sources import FileSourceConfig
import tqdm


class FileLoaderHelper:
    """
    A helper class for loading files from various sources.
    """

    def __init__(self, file_path: str):
        """
        Class constructor
        """
        self.file_path = file_path
        # Map the file source to a file loader method
        # using a dictionary
        self.file_loader_methods = {
            FileSource.GOOGLE_DRIVE: self._google_drive_source,
            FileSource.S3: self._s3_source,
            FileSource.DOWNLOAD_LINK: self._download_link_source,
        }

        self.file_loader = self.file_loader_methods[self.file_path]

    def _local_source(self, file_source_config: FileSourceConfig):
        """
        Local source file loader method
        """
        # read in the file from the local source
        with open(file_source_config.source_path, "rb") as f:
            return f.read()

    def _google_drive_source(self):
        """
        Google drive source file loader method
        """
        raise NotImplementedError(
            "Google drive source file loader method not implemented"
        )

    def _s3_source(self):
        """
        S3 source file loader method
        """
        raise NotImplementedError("S3 source file loader method not implemented")

    def _download_link_source(self):
        """
        Download link source file loader method
        """
        raise NotImplementedError(
            "Download link source file loader method not implemented"
        )
