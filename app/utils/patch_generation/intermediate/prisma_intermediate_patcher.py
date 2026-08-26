"""
Generates intermediate patches for PRISMA hyperspectral scenes.

Discovers .he5 files from S3, builds vendable datasets with band filtering,
patches them into tiles, filters by spatial validity, and writes webdataset
shards to S3.
"""

import logging
from typing import List, Dict, Generator, Literal, Optional
import random
import os

from tqdm import tqdm
import webdataset as wds

from app.abstract_classes.intermediate_sharder import IntermediateSharder
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.models.file_processing.sources import FileSourceConfig
from app.models.dataset.vendables import BandFilterConfig
from app.utils.patch_generation.hyperspectral_patcher import patch_hyperspectral_vendable
from app.utils.patch_generation.generate_patch_plan import (
    PatchPlanGenerator,
    PatchRequest,
)
from app.utils.general_utils import s3_config
from app.utils.patch_generation.scene_storage import SceneStorage, S3SceneStorage


S3_BUCKET: str = s3_config.BUCKET


class PrismaIntermediateSharder(IntermediateSharder):
    """
    PRISMA intermediate sharder.

    Discovers all .he5 scenes from S3, splits them deterministically into
    train/test, then patches only the scenes belonging to the requested split.

    S3 destination prefix is computed automatically as:
        patches/prisma/{split}/intermediate/w{width}_h{height}_s{stride}/
    """

    SENSOR = "prisma"

    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        split: Literal["train", "test"] = "train",
        test_fraction: float = 0.2,
        seed: int = 42,
        width: int = 64,
        height: int = 64,
        stride: int = 32,
        patch_validity_threshold: float = 0.5,
        band_filter_config: Optional[BandFilterConfig] = None,
        max_scenes: Optional[int] = None,
        storage: Optional[SceneStorage] = None,
    ):
        self._source_folder = source_folder
        self._destination_folder = destination_folder
        self.split = split
        self.width = width
        self.height = height
        self.stride = stride
        self.patch_validity_threshold = patch_validity_threshold
        self.band_filter_config = band_filter_config if band_filter_config is not None else BandFilterConfig()
        self.target_size = 1 * 1024 * 1024 * 1024  # 1 GB per shard

        # Split inputs are stored, not acted on. Discovery happens on first
        # access to `_scene_keys` — see the property below.
        self._seed = seed
        self._test_fraction = test_fraction
        self._max_scenes = max_scenes
        self._scene_keys_cache: Optional[List[str]] = None

        self.destination_prefix = self.build_prefix(
            sensor=self.SENSOR,
            split=self.split,
            stage="intermediate",
            width=self.width,
            height=self.height,
            stride=self.stride,
        )

        self.shard_pattern = f"{self.destination_folder}intermediate_shard_%04d.tar"

        # Built after destination_prefix: the default backend needs it as
        # shard_prefix, or publish_shard silently does nothing. scene_suffix
        # marks PRISMA as a one-file-per-scene sensor.
        self.storage = storage or S3SceneStorage(
            scene_prefix="prisma/",
            shard_prefix=self.destination_prefix,
            scene_suffix=".he5",
        )

    @property
    def source_folder(self) -> str:
        return self._source_folder

    @property
    def destination_folder(self) -> str:
        return self._destination_folder

    @property
    def _scene_keys(self) -> List[str]:
        """Scenes belonging to this split, discovered on first access.

        This used to run in `__init__`, so merely constructing a sharder
        required a network and credentials — which is why none of these
        classes had tests. Behaviour is otherwise identical: same sort,
        same seeded shuffle, same cap, same cut.
        """
        if self._scene_keys_cache is None:
            self._scene_keys_cache = self._split_scenes()
        return self._scene_keys_cache

    def _split_scenes(self) -> List[str]:
        all_scenes = sorted(self.list_scenes())
        rng = random.Random(self._seed)
        rng.shuffle(all_scenes)
        if self._max_scenes is not None and self._max_scenes < len(all_scenes):
            all_scenes = all_scenes[: self._max_scenes]
        split_idx = int(len(all_scenes) * (1 - self._test_fraction))
        chosen = (
            all_scenes[:split_idx] if self.split == "train" else all_scenes[split_idx:]
        )
        print(
            f"[PRISMA] Split '{self.split}': {len(chosen)} scenes "
            f"(of {len(all_scenes)} total, seed={self._seed}, "
            f"test_fraction={self._test_fraction})"
        )
        return chosen

    def list_scenes(self) -> List:
        """
        Lists all .he5 files under the prisma/ prefix in S3.
        """
        return self.storage.list_scenes()

    def publish_hook(self, local_path: str) -> None:
        """Hand a finished shard to storage, then delete the local copy.

        Deletion is conditional on `shard_exists` confirming it landed:
        `publish_shard` is a no-op on a backend with no destination, so
        deleting unconditionally would discard shards silently.
        """
        self.storage.publish_shard(local_path)
        if self.storage.shard_exists(os.path.basename(local_path)):
            os.remove(local_path)
        else:
            print(f"[PRISMA] shard NOT published, keeping local copy: {local_path}")

    def prepare_scene(self, key: str) -> Dict:
        """
        Downloads a single .he5 file from S3 to the source folder.
        """
        return {"he5": self.storage.fetch_scene(key, self.source_folder)}

    def patch_generator(self, manifest: Dict) -> Generator:
        """
        Builds a vendable dataset from the downloaded .he5 file, generates a
        patching plan, and yields patches with ascending wavelength order.
        """
        builder = PrismaDatasetBuilder(
            file_source_configuration=FileSourceConfig(
                source_path=manifest.get("he5")
            )
        )
        vendable = builder.vend_dataset(band_filter_config=self.band_filter_config)

        generator = PatchPlanGenerator()
        plan = generator.generate_patching_plan(
            request=PatchRequest(
                input_cube=vendable.normalized_hyperspectral_cube.shape,
                height=self.height,
                width=self.width,
                stride=self.stride,
            )
        )
        return patch_hyperspectral_vendable(
            vendable=vendable,
            patching_plan=plan,
            scene_id=builder.stac_item.id,
            sensor="prisma",
        )

    def sharder(self, scenes: int = None):
        """
        Orchestrates the intermediate sharding process.
        scenes arg is ignored — scene cap is applied at init via max_scenes.
        """
        processed_patches = 0
        valid_patches = 0
        with wds.ShardWriter(
            self.shard_pattern, maxsize=self.target_size, post=self.publish_hook
        ) as sink:
            scene_keys = list(self._scene_keys)
            random.shuffle(scene_keys)

            for scene in tqdm(scene_keys, desc="PRISMA Scene"):
                try:
                    print(f"Processing Scene: {scene}")
                    manifest = self.prepare_scene(scene)
                    patches = self.patch_generator(manifest)
                    for patch_sample in tqdm(patches, desc="Patch"):
                        # Post-interpolation: validity is pixel-level binary.
                        # Band 0 is sufficient as spatial validity proxy.
                        spatial_validity = patch_sample["validity_cube.npy"][0]
                        valid_fraction = spatial_validity.sum() / spatial_validity.size
                        if valid_fraction > self.patch_validity_threshold:
                            sink.write(patch_sample)
                            valid_patches += 1
                        processed_patches += 1
                    # Cleanup goes through the backend, never os.remove
                    # here: it cannot tell a temp download from source data.
                    self.storage.release_scene(manifest.get("he5", ""))
                except Exception as err:
                    print(f"Failed to process scene {scene}: {err}")
        print(f"[PRISMA] Processed Patches: {processed_patches}")
        print(f"[PRISMA] Valid Patches: {valid_patches}")
