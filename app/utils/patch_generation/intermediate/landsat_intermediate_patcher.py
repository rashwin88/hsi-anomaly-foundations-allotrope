"""
Generates intermediate patches for landsat
"""

import logging
from typing import List, Dict, Generator
import random
import os
from functools import partial

import boto3
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
from app.utils.general_utils.s3_upload_and_delete import s3_upload_and_cleanup


S3_BUCKET: str = "allotrope-raw-data-india"


class LandsatIntermediateSharder(IntermediateSharder):
    """
    Landsat intermediate sharder
    """

    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        destination_prefix: str,
        width: int = 128,
        height: int = 128,
        stride: int = 64,
    ):
        """
        Class Constructor
        """

        self.s3_client = boto3.client("s3", region_name="ap-south-1")
        self.paginator = self.s3_client.get_paginator("list_objects_v2")
        self._source_folder = source_folder
        self._destination_folder = destination_folder
        # The destination prefix is where we want to write to s3 to.
        self.destination_prefix = destination_prefix
        self.width = width
        self.height = height
        self.stride = stride
        self.target_size = 1 * 1024 * 1024 * 1024  # 1 GB per shard
        self.shard_pattern = f"{self.destination_folder}intermediate_shard_%04d.tar"

        # Build the upload hook
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
        Runs through the S3 bucket and returns specific fodlers representing scenes
        """
        output = []
        # Build the page iterator
        page_iterator = self.paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix="landsat/",
            Delimiter="/",
            PaginationConfig={"PageSize": 500},
        )
        # Add to the output list
        for page in page_iterator:
            for folder in page.get("CommonPrefixes"):
                output.append(folder.get("Prefix"))
        return output

    def s3_downloader(self, key: str) -> Dict:
        """
        Downloads the files to the destination folder.
        Here the key refers to the scene prefix
        """
        # We begin with a key and we download all the files from that key
        objects = self.s3_client.list_objects(Bucket=S3_BUCKET, Prefix=key)

        # List everything we want to download
        downloadable = [content.get("Key") for content in objects.get("Contents")]
        print(downloadable)
        download_result = {}
        # Download to the destination bucket
        for download in downloadable:
            file_name = self.source_folder + download.split("/")[-1]
            self.s3_client.download_file(
                Bucket=S3_BUCKET, Key=download, Filename=file_name
            )
            if "ST_B10" in download:
                download_result["b10"] = file_name
            elif "QA_PIXEL" in download:
                download_result["qa_pixel"] = file_name
        return download_result

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
        Orcherstrates the intermediate sharding process.
        """
        processed_patches = 0
        valid_patches = 0
        with wds.ShardWriter(
            self.shard_pattern, maxsize=self.target_size, post=self.upload_hook
        ) as sink:
            # Search S3
            scene_prefixes = self.s3_searcher()
            # Randomly mix the scene prefixes because there may be some regularity here
            random.shuffle(scene_prefixes)
            # Truncate to save time
            if scenes:
                scene_prefixes = scene_prefixes[0 : min(scenes, len(scene_prefixes))]
            # Loop through
            for scene in tqdm(scene_prefixes, desc="Scene Number"):
                try:
                    print(f"Processing Scene: {scene}")
                    manifest = self.s3_downloader(scene)
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
                    # Delete the files in the manifest
                    for file_path in manifest.values():
                        if isinstance(file_path, str) and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                print(f"Failed to delete {file_path}: {e}")
                except Exception as err:
                    print(f"Failed to download scene {scene}. Moving on.")
        print(f"Processed Patches: {processed_patches}")
        print(f"Valid Patches : {valid_patches}")


if __name__ == "__main__":
    # Configure root logger to show all logs (DEBUG and above) in the console.
    import sys

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console_handler.setFormatter(formatter)

    patcher = LandsatIntermediateSharder(
        source_folder="/home/ubuntu/",
        destination_folder="/home/ubuntu/",
        destination_prefix="patches/intermediate/test1",
    )
    patcher.sharder(scenes=2)
