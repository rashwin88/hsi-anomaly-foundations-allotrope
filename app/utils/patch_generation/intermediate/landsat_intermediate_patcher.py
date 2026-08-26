"""
Stage 1 of Landsat training-data prep: whole scenes in, patch shards out.

Runs from scripts/generate_landsat_patches.py, never from the product. Models
train on small patches, not whole scenes, so this walks every Landsat scene in
S3 and emits WebDataset .tar shards of fixed-size crops.

Per scene: download -> LandsatDataBuilder.vend_dataset() -> cut patches on a
sliding window -> drop patches that are mostly invalid -> write a shard ->
upload -> delete the local copy. The download/delete cycle is why this streams
rather than materialising the corpus on disk.

    patches/landsat/{split}/intermediate/w{width}_h{height}_s{stride}/

Train/test is split by SCENE, not by patch, and derived deterministically from a
seed. That matters: patches from one scene overlap heavily, so splitting at the
patch level would leak nearly-identical crops across the boundary and flatter
every evaluation you run.

"Intermediate" because shards at this stage hold patches from a single scene.
Stage 2 (FinalPatchShuffler) mixes across scenes, so a training batch is not 32
crops of the same field.
"""

import logging
from typing import List, Dict, Generator, Literal, Optional
import random
import os

from tqdm import tqdm
import webdataset as wds


from app.abstract_classes.intermediate_sharder import IntermediateSharder
from app.utils.dataset_builder.landsat_dataset_builder import LandsatDataBuilder
from app.models.file_processing.sources import FileSourceConfig
from app.utils.patch_generation.landsat_patcher import patch_landsat_vendable
from app.utils.patch_generation.generate_patch_plan import (
    PatchPlanGenerator,
    PatchRequest,
)
from app.utils.general_utils import s3_config
from app.utils.patch_generation.scene_storage import SceneStorage, S3SceneStorage


S3_BUCKET: str = s3_config.BUCKET


class LandsatIntermediateSharder(IntermediateSharder):
    """
    Landsat intermediate sharder.

    Discovers all scenes from S3, splits them deterministically into train/test
    based on the provided seed and test_fraction, then patches only the scenes
    belonging to the requested split.

    S3 destination prefix is computed automatically as:
        patches/landsat/{split}/intermediate/w{width}_h{height}_s{stride}/
    """

    SENSOR = "landsat"

    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        split: Literal["train", "test"] = "train",
        test_fraction: float = 0.2,
        seed: int = 42,
        width: int = 128,
        height: int = 128,
        stride: int = 64,
        storage: Optional[SceneStorage] = None,
    ):
        self._source_folder = source_folder
        self._destination_folder = destination_folder
        self.split = split
        self.width = width
        self.height = height
        self.stride = stride
        self.target_size = 1 * 1024 * 1024 * 1024  # 1 GB per shard

        # Split inputs are stored, not acted on. Discovery happens on first
        # access to `_scene_prefixes` — see the property below.
        self._seed = seed
        self._test_fraction = test_fraction
        self._scene_prefixes_cache: Optional[List[str]] = None

        # Build the structured S3 prefix
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
        # shard_prefix, or publish_shard silently does nothing.
        self.storage = storage or S3SceneStorage(
            scene_prefix="landsat/", shard_prefix=self.destination_prefix
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
        same seeded shuffle, same cut. Note Landsat has no `max_scenes`
        cap, unlike the PRISMA and EnMAP sharders.
        """
        if self._scene_prefixes_cache is None:
            self._scene_prefixes_cache = self._split_scenes()
        return self._scene_prefixes_cache

    def _split_scenes(self) -> List[str]:
        all_scenes = sorted(self.list_scenes())
        rng = random.Random(self._seed)
        rng.shuffle(all_scenes)
        split_idx = int(len(all_scenes) * (1 - self._test_fraction))
        chosen = (
            all_scenes[:split_idx] if self.split == "train" else all_scenes[split_idx:]
        )
        print(
            f"Split '{self.split}': {len(chosen)} scenes "
            f"(of {len(all_scenes)} total, seed={self._seed}, "
            f"test_fraction={self._test_fraction})"
        )
        return chosen

    def list_scenes(self) -> List:
        """
        Runs through the S3 bucket and returns specific folders representing scenes
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
            print(f"[Landsat] shard NOT published, keeping local copy: {local_path}")

    def prepare_scene(self, key: str) -> Dict:
        """Make one scene readable locally and pick out the two files we use.

        Landsat's manifest names files rather than the folder: the builder
        wants the ST_B10 band and the QA_PIXEL mask specifically.

        Note the files now land in a per-scene subfolder rather than flat in
        `source_folder`, because that is what `fetch_scene` returns. That
        also removes a latent collision — two scenes' files previously shared
        one directory.
        """
        scene_folder = self.storage.fetch_scene(key, self.source_folder)
        manifest = {"scene_folder": scene_folder}
        for file_name in sorted(os.listdir(scene_folder)):
            if "ST_B10" in file_name:
                manifest["b10"] = os.path.join(scene_folder, file_name)
            elif "QA_PIXEL" in file_name:
                manifest["qa_pixel"] = os.path.join(scene_folder, file_name)
        return manifest

    def patch_generator(self, manifest: Dict) -> Generator:
        """
        Actual patch generator which yields an iterable
        """

        # get the Builder
        builder = LandsatDataBuilder(
            file_source_configuration=FileSourceConfig(source_path=manifest.get("b10"))
        )
        # Build the vendable
        vendable = builder.vend_dataset(
            provider_qa_pixel_source=manifest.get("qa_pixel")
        )

        # Create the generator
        generator = PatchPlanGenerator()
        plan = generator.generate_patching_plan(
            request=PatchRequest(
                input_cube=vendable.normalized_thermal_cube.shape,
                height=self.height,
                width=self.width,
                stride=self.stride,
            )
        )
        patcher = patch_landsat_vendable(
            vendable=vendable, patching_plan=plan, stac_item=builder.stac_item
        )

        return patcher

    def sharder(self, scenes: int = None):
        """
        Orchestrates the intermediate sharding process.
        Only processes scenes belonging to this instance's split.
        """
        processed_patches = 0
        valid_patches = 0
        with wds.ShardWriter(
            self.shard_pattern, maxsize=self.target_size, post=self.publish_hook
        ) as sink:
            scene_prefixes = list(self._scene_prefixes)
            # Shuffle within the split for shard diversity
            random.shuffle(scene_prefixes)
            # Truncate to save time
            if scenes:
                scene_prefixes = scene_prefixes[0 : min(scenes, len(scene_prefixes))]
            # Loop through
            for scene in tqdm(scene_prefixes, desc="Scene Number"):
                try:
                    print(f"Processing Scene: {scene}")
                    manifest = self.prepare_scene(scene)
                    # create the patcher
                    patches = self.patch_generator(manifest)
                    for patch_sample in tqdm(patches, desc="Patch"):
                        # Calculate to see if the patch needs to be kept or not
                        valid_pixels = patch_sample.get("pure_validity_mask.npy").sum()
                        b, h, w = patch_sample.get("pure_validity_mask.npy").shape

                        if valid_pixels / (b * h * w) > 0.5:
                            sink.write(patch_sample)
                            valid_patches += 1
                        processed_patches += 1
                    # Cleanup goes through the backend, never os.remove here:
                    # it cannot tell a temp download from source data.
                    self.storage.release_scene(manifest.get("scene_folder", ""))
                except Exception as err:
                    print(f"Failed to download scene {scene}. Moving on.")
        print(f"Processed Patches: {processed_patches}")
        print(f"Valid Patches : {valid_patches}")


