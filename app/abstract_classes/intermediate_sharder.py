"""
The contract for stage-1 patch generation, shared across sensors.

Implemented by LandsatIntermediateSharder, PrismaIntermediateSharder and
EnmapIntermediateSharder. Each follows the same five steps - discover scenes in
S3, download one, build its vendable, cut patches, write and upload a shard -
differing only in which builder and cutter they use.

The value here is build_prefix(), which is the single definition of the S3
layout:

    patches/{sensor}/{split}/{stage}/w{width}_h{height}_s{stride}/

Both the writer and the training-time reader derive their paths from this one
function, so the layout cannot drift between them. Encoding patch geometry in
the path also means several patch sizes coexist without collision, and a trainer
asking for 128px shards physically cannot receive 256px ones.

"Intermediate" distinguishes this from the final stage: shards written here hold
patches from a single scene, and FinalPatchShuffler later mixes across scenes so
a training batch is not 32 crops of the same field.

Offline tooling. Nothing in backend/ imports this - it runs from
scripts/generate_*_patches.py.
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
