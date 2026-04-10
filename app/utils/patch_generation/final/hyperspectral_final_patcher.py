"""
Final shuffler for hyperspectral patches.

Reads intermediate shards from multiple sensors (PRISMA + EnMAP),
shuffles patches across all scenes and sensors, and writes mixed
final shards under a unified 'hyperspectral' prefix.
"""

import os
import boto3
from typing import List, Literal
from functools import partial

from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm
import torch

from app.abstract_classes.intermediate_sharder import IntermediateSharder
from app.utils.general_utils.s3_upload_and_delete import s3_upload_and_cleanup

S3_BUCKET: str = "allotrope-raw-data-india"
FINAL_SHARD_PATTERN: str = "final_shard_%05d.tar"
TARGET_GB = 1024**3


class HyperspectralFinalShuffler:
    """
    Reads intermediate shards from multiple hyperspectral sensors,
    shuffles patches across all sources, and writes mixed final shards.

    S3 paths:
        Sources: patches/{sensor}/{split}/intermediate/w{w}_h{h}_s{s}/  (per sensor)
        Dest:    patches/hyperspectral/{split}/final/w{w}_h{h}_s{s}/    (mixed)
    """

    def __init__(
        self,
        split: Literal["train", "test"],
        width: int = 64,
        height: int = 64,
        stride: int = 32,
        sensors: List[str] = None,
        shard_temp_location: str = "/tmp/",
        worker_count: int = 10,
        shuffle_size: int = 10,
        patch_write_count: int = 500_000,
    ):
        self.s3_client = boto3.client("s3", region_name="ap-south-1")
        self.paginator = self.s3_client.get_paginator("list_objects_v2")

        if sensors is None:
            sensors = ["prisma", "enmap"]
        self.sensors = sensors

        # Build source prefixes for each sensor
        self._source_keys = {}
        for sensor in sensors:
            self._source_keys[sensor] = IntermediateSharder.build_prefix(
                sensor=sensor,
                split=split,
                stage="intermediate",
                width=width,
                height=height,
                stride=stride,
            )

        # Destination is unified under 'hyperspectral'
        self._destination_key = IntermediateSharder.build_prefix(
            sensor="hyperspectral",
            split=split,
            stage="final",
            width=width,
            height=height,
            stride=stride,
        )

        self.shard_temp_location = f"{shard_temp_location}{FINAL_SHARD_PATTERN}"

        # Build pipe URLs for each sensor's intermediate shards
        intermediate_urls = []
        for sensor, source_key in self._source_keys.items():
            shard_range = self._compute_shard_ranges(source_key)
            if shard_range is not None:
                url = f"pipe: aws s3 cp s3://{S3_BUCKET}/{source_key}{shard_range} -"
                intermediate_urls.append(url)
                print(f"  [{sensor}] {url}")

        if not intermediate_urls:
            raise ValueError("No intermediate shards found for any sensor.")

        self.upload_hook = partial(
            s3_upload_and_cleanup,
            bucket_name=S3_BUCKET,
            s3_prefix=self._destination_key,
            client=self.s3_client,
        )

        self.worker_count = worker_count
        self.shuffle_size = shuffle_size
        self.patch_write_count = patch_write_count

        # Create a WebDataset that reads from all sensor sources
        self.dataset = (
            wds.WebDataset(intermediate_urls, resampled=True)
            .shuffle(self.shuffle_size, initial=self.shuffle_size)
            .decode()
        )

        self.dataloader = DataLoader(
            self.dataset, num_workers=self.worker_count, batch_size=None
        )

        print(
            f"HyperspectralFinalShuffler ready\n"
            f"  Sensors:     {sensors}\n"
            f"  Destination: s3://{S3_BUCKET}/{self._destination_key}\n"
            f"  Patches to write: {self.patch_write_count}"
        )

    def _compute_shard_ranges(self, source_key: str):
        """
        Computes the shard range string for a given source prefix.
        Returns None if no shards are found.
        """
        output = []
        page_iterator = self.paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=source_key,
            PaginationConfig={"PageSize": 500},
        )
        for page in page_iterator:
            for key in page.get("Contents", []):
                output.append(key.get("Key"))

        if not output:
            return None

        shard_numbers = [
            key.split("/")[-1].split(".")[0].split("_")[-1] for key in output
        ]

        sample_key = output[0]
        shard_elements = sample_key.split("/")[-1].split(".")[0].split("_")
        shard_identifier = "_".join(shard_elements[:2])

        return f"{shard_identifier}_{{{min(shard_numbers)}..{max(shard_numbers)}}}.tar"

    def write_shards(self):
        """
        Shuffles patches across all sensors and writes mixed final shards.
        """
        processed_patches = 0

        with wds.ShardWriter(
            self.shard_temp_location, maxsize=TARGET_GB, post=self.upload_hook
        ) as sink:
            for i, patch_dict in tqdm(enumerate(self.dataloader)):
                if i >= self.patch_write_count:
                    print("Reached total patch count. Halting.")
                    break
                for key, value in patch_dict.items():
                    if isinstance(value, torch.Tensor):
                        patch_dict[key] = value.cpu().numpy()
                sink.write(patch_dict)
                processed_patches += 1
        print(f"[Hyperspectral Final] Processed Patches: {processed_patches}")
