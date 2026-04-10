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

        # Build one pipe URL per shard file for each sensor.
        # We enumerate individual shard keys rather than using brace expansion
        # because wds.WebDataset with a list of pipe: URLs does not reliably
        # expand braces inside pipe commands.
        intermediate_urls = []
        for sensor, source_key in self._source_keys.items():
            shard_keys = self._list_shard_keys(source_key)
            if shard_keys:
                for key in shard_keys:
                    intermediate_urls.append(
                        f"pipe: aws s3 cp s3://{S3_BUCKET}/{key} -"
                    )
                print(f"  [{sensor}] {len(shard_keys)} shards from {source_key}")

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

    def _list_shard_keys(self, source_key: str) -> list[str]:
        """
        Lists all shard S3 keys under a source prefix.
        Returns empty list if no shards found.
        """
        output = []
        page_iterator = self.paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=source_key,
            PaginationConfig={"PageSize": 500},
        )
        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith(".tar"):
                    output.append(key)
        return sorted(output)

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
