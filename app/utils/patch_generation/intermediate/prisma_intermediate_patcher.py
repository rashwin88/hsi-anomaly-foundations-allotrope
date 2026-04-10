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
from functools import partial

import boto3
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
from app.utils.general_utils.s3_upload_and_delete import s3_upload_and_cleanup


S3_BUCKET: str = "allotrope-raw-data-india"


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
    ):
        self.s3_client = boto3.client("s3", region_name="ap-south-1")
        self.paginator = self.s3_client.get_paginator("list_objects_v2")
        self._source_folder = source_folder
        self._destination_folder = destination_folder
        self.split = split
        self.width = width
        self.height = height
        self.stride = stride
        self.patch_validity_threshold = patch_validity_threshold
        self.band_filter_config = band_filter_config if band_filter_config is not None else BandFilterConfig()
        self.target_size = 1 * 1024 * 1024 * 1024  # 1 GB per shard

        # Discover all scenes and split deterministically
        all_scenes = sorted(self.s3_searcher())
        rng = random.Random(seed)
        rng.shuffle(all_scenes)
        split_idx = int(len(all_scenes) * (1 - test_fraction))
        if split == "train":
            self._scene_keys = all_scenes[:split_idx]
        else:
            self._scene_keys = all_scenes[split_idx:]

        print(
            f"[PRISMA] Split '{split}': {len(self._scene_keys)} scenes "
            f"(of {len(all_scenes)} total, seed={seed}, test_fraction={test_fraction})"
        )

        self.destination_prefix = self.build_prefix(
            sensor=self.SENSOR,
            split=self.split,
            stage="intermediate",
            width=self.width,
            height=self.height,
            stride=self.stride,
        )

        self.shard_pattern = f"{self.destination_folder}intermediate_shard_%04d.tar"

        self.upload_hook = partial(
            s3_upload_and_cleanup,
            bucket_name=S3_BUCKET,
            s3_prefix=self.destination_prefix,
            client=self.s3_client,
        )

    @property
    def source_folder(self) -> str:
        return self._source_folder

    @property
    def destination_folder(self) -> str:
        return self._destination_folder

    def s3_searcher(self) -> List:
        """
        Lists all .he5 files under the prisma/ prefix in S3.
        """
        output = []
        page_iterator = self.paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix="prisma/",
            PaginationConfig={"PageSize": 500},
        )
        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith(".he5"):
                    output.append(key)
        return output

    def s3_downloader(self, key: str) -> Dict:
        """
        Downloads a single .he5 file from S3 to the source folder.
        """
        file_name = self.source_folder + key.split("/")[-1]
        self.s3_client.download_file(
            Bucket=S3_BUCKET, Key=key, Filename=file_name
        )
        return {"he5": file_name}

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
        """
        processed_patches = 0
        valid_patches = 0
        with wds.ShardWriter(
            self.shard_pattern, maxsize=self.target_size, post=self.upload_hook
        ) as sink:
            scene_keys = list(self._scene_keys)
            random.shuffle(scene_keys)
            if scenes:
                scene_keys = scene_keys[: min(scenes, len(scene_keys))]

            for scene in tqdm(scene_keys, desc="PRISMA Scene"):
                try:
                    print(f"Processing Scene: {scene}")
                    manifest = self.s3_downloader(scene)
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
                    # Cleanup downloaded file
                    he5_path = manifest.get("he5", "")
                    if os.path.exists(he5_path):
                        os.remove(he5_path)
                except Exception as err:
                    print(f"Failed to process scene {scene}: {err}")
        print(f"[PRISMA] Processed Patches: {processed_patches}")
        print(f"[PRISMA] Valid Patches: {valid_patches}")
