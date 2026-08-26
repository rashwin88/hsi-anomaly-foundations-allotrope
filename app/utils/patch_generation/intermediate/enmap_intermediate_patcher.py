"""
Generates intermediate patches for EnMAP hyperspectral scenes.

Discovers scene folders from S3, downloads all files per scene, builds
vendable datasets with band filtering, patches them into tiles, filters
by spatial validity, and writes webdataset shards to S3.
"""

import logging
from typing import List, Dict, Generator, Literal, Optional
import random
import os

from tqdm import tqdm
import webdataset as wds

from app.abstract_classes.intermediate_sharder import IntermediateSharder
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
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


class EnmapIntermediateSharder(IntermediateSharder):
    """
    EnMAP intermediate sharder.

    Discovers all scene folders from S3, splits them deterministically into
    train/test, then patches only the scenes belonging to the requested split.

    S3 destination prefix is computed automatically as:
        patches/enmap/{split}/intermediate/w{width}_h{height}_s{stride}/
    """

    SENSOR = "enmap"

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
        # access to `_scene_prefixes` — see the property below.
        self._seed = seed
        self._test_fraction = test_fraction
        self._max_scenes = max_scenes
        self._scene_prefixes_cache: Optional[List[str]] = None

        self.destination_prefix = self.build_prefix(
            sensor=self.SENSOR,
            split=self.split,
            stage="intermediate",
            width=self.width,
            height=self.height,
            stride=self.stride,
        )

        self.shard_pattern = f"{self.destination_folder}intermediate_shard_%04d.tar"

        # Storage is built after destination_prefix because the default S3
        # backend needs it: with shard_prefix unset, publish_shard is a no-op.
        # Pass a LocalSceneStorage to shard from, and to, a mounted disk.
        self.storage = storage or S3SceneStorage(
            scene_prefix="enmap/", shard_prefix=self.destination_prefix
        )

    @property
    def source_folder(self) -> str:
        return self._source_folder

    @property
    def destination_folder(self) -> str:
        return self._destination_folder

    @property
    def _scene_prefixes(self) -> List[str]:
        """Scenes belonging to this split, discovered on first access.

        This used to run in `__init__`, so merely constructing a sharder
        required a network and credentials — which is why none of these
        classes had tests. Behaviour is otherwise identical: same sort,
        same seeded shuffle, same cap, same cut.
        """
        if self._scene_prefixes_cache is None:
            self._scene_prefixes_cache = self._split_scenes()
        return self._scene_prefixes_cache

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
            f"[EnMAP] Split '{self.split}': {len(chosen)} scenes "
            f"(of {len(all_scenes)} total, seed={self._seed}, "
            f"test_fraction={self._test_fraction})"
        )
        return chosen

    def list_scenes(self) -> List:
        """Every available scene id, via the injected storage backend.

        Ids are bare folder names, not `enmap/<id>/` prefixes — both backends
        speak that vocabulary so this class never learns which it holds.
        """
        return self.storage.list_scenes()

    def publish_hook(self, local_path: str) -> None:
        """Hand a finished shard to storage, then delete the local copy.

        `wds.ShardWriter` calls this with each completed shard.

        Deletion is conditional on `shard_exists` confirming the shard landed.
        `publish_shard` is a *no-op* on a backend with no destination
        configured, so deleting unconditionally — which is what
        `s3_upload_and_cleanup` did — would discard shards that were never
        stored, silently and irrecoverably.
        """
        self.storage.publish_shard(local_path)
        if self.storage.shard_exists(os.path.basename(local_path)):
            os.remove(local_path)
        else:
            print(
                f"[EnMAP] shard NOT published, keeping local copy: {local_path}. "
                "Does the storage backend have a destination configured?"
            )

    def prepare_scene(self, key: str) -> Dict:
        """Make one scene readable locally and return its manifest.

        The scene folder name is preserved, because `FileSourceConfig`
        auto-detects the sensor from it.
        """
        return {"scene_folder": self.storage.fetch_scene(key, self.source_folder)}

    def patch_generator(self, manifest: Dict) -> Generator:
        """
        Builds a vendable dataset from the downloaded scene folder, generates a
        patching plan, and yields patches with ascending wavelength order.
        """
        builder = EnmapDatasetBuilder(
            file_source_configuration=FileSourceConfig(
                source_path=manifest.get("scene_folder")
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
            sensor="enmap",
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
            scene_prefixes = list(self._scene_prefixes)
            random.shuffle(scene_prefixes)

            for scene in tqdm(scene_prefixes, desc="EnMAP Scene"):
                try:
                    print(f"Processing Scene: {scene}")
                    manifest = self.prepare_scene(scene)
                    patches = self.patch_generator(manifest)
                    for patch_sample in tqdm(patches, desc="Patch"):
                        spatial_validity = patch_sample["validity_cube.npy"][0]
                        valid_fraction = spatial_validity.sum() / spatial_validity.size
                        if valid_fraction > self.patch_validity_threshold:
                            sink.write(patch_sample)
                            valid_patches += 1
                        processed_patches += 1
                    # Cleanup goes through the backend, never rmtree here: it
                    # cannot tell a temp download from a user's own folder.
                    self.storage.release_scene(manifest.get("scene_folder", ""))
                except Exception as err:
                    print(f"Failed to process scene {scene}: {err}")
        print(f"[EnMAP] Processed Patches: {processed_patches}")
        print(f"[EnMAP] Valid Patches: {valid_patches}")
