"""
Abstraction for intermediate sharding
"""

from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Literal


class IntermediateSharder(ABC):
    """
    Defines an intermediate sharder.

    Subclasses must define the sensor name and accept a split + patch dimensions.
    The S3 destination prefix is computed automatically as:
        patches/{sensor}/{split}/intermediate/w{width}_h{height}_s{stride}/
    """

    SENSOR: str  # Subclasses must set this (e.g. "landsat", "enmap", "prisma")

    def __init__(self):
        pass

    @staticmethod
    def build_prefix(
        sensor: str,
        split: str,
        stage: str,
        width: int,
        height: int,
        stride: int,
    ) -> str:
        """
        Builds a structured S3 prefix from the given parameters.
        Example: patches/landsat/train/intermediate/w128_h128_s64/
        """
        return f"patches/{sensor}/{split}/{stage}/w{width}_h{height}_s{stride}/"

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
        Downloads the target file from S3
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
