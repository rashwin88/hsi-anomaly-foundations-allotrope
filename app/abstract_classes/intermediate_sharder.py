"""
Abstraction for intermediate sharding
"""

from abc import ABC, abstractmethod
from typing import Generator, List, Dict


class IntermediateSharder(ABC):
    """
    Defines an intermediate sharder
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def source_folder(self) -> str:
        """
        The source folder from which the files will be obtained
        """
        pass

    @property
    @abstractmethod
    def destination_folder(self) -> str:
        """
        The destination folder where the intermediate shards will be written
        """
        pass

    @abstractmethod
    def s3_searcher(self) -> List:
        """
        Searches S3 and returns a list of all the possible files or their prefixes
        """
        pass

    @abstractmethod
    def s3_downloader(self, key: str) -> Dict | None:
        """
        Downlaods the target file from S3
        """
        pass

    @abstractmethod
    def patch_generator(self, manifest: Dict) -> Generator:
        """
        Generates patches
        """
        pass

    @abstractmethod
    def sharder(self, scenes: int = None) -> None:
        """
        Generates the intermediate shards
        """
        pass
