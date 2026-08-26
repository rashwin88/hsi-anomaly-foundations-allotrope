"""
The contract for stage-1 patch generation, shared across sensors.

Implemented by LandsatIntermediateSharder, PrismaIntermediateSharder,
EnmapIntermediateSharder and EnmapSegmentationSharder. Each follows the same
five steps - list the available scenes, make one readable locally, build its
vendable, cut patches, write and publish a shard - differing only in which
builder and cutter they use.

Where scenes come from is not this contract's business. Implementations take a
`SceneStorage` (app/utils/patch_generation/scene_storage.py), so the same
sharder runs against S3 or a mounted disk. This docstring used to say "discover
scenes in S3", and the methods below were called `s3_searcher` and
`s3_downloader` - an abstraction naming the thing it was supposed to abstract
over, which is why sharding could not run anywhere but AWS.

The value here is build_prefix(), which is the single definition of the shard
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
    The destination prefix is computed automatically as:
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
        Builds the structured shard prefix from the given parameters.
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
    def list_scenes(self) -> List:
        """
        Every scene id available from the storage backend.
        """
        pass

    @abstractmethod
    def prepare_scene(self, key: str) -> Dict | None:
        """
        Make one scene readable locally and return a manifest describing where
        its parts landed. The manifest's shape is the implementation's choice:
        EnMAP returns {"scene_folder": ...}, PRISMA {"he5": ...}, Landsat names
        the ST_B10 and QA_PIXEL files the builder needs.
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
