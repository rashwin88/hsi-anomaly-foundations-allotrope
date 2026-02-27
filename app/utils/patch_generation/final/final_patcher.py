"""
Final patching is basically a mixing operation with some sort of
shuffle buffer in play. Final patching is basically provider independent IMO
"""

from multiprocessing import process
import os
import boto3
from functools import partial

from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm
import torch

from app.utils.general_utils.s3_upload_and_delete import s3_upload_and_cleanup

S3_BUCKET: str = "allotrope-raw-data-india"
FINAL_SHARD_PATTERN: str = "final_shard_%05d.tar"
TARGET_GB = 1024**3


class FinalPatchShuffler:
    """
    Implements a final patch shuffler
    """

    def __init__(
        self,
        intermediate_patch_s3_key: str,  # Needs the / at the end
        final_s3_key: str,
        shard_temp_location="/Users/ashwinravi/Desktop/",
        worker_count: int = 10,
        shuffle_size: int = 10,
        patch_write_count: int = 10_000,
    ):
        """
        We only need the intermediate location and the final location with no other information.
        """

        self.s3_client = boto3.client("s3", region_name="ap-south-1")
        self.paginator = self.s3_client.get_paginator("list_objects_v2")
        self._source_key = intermediate_patch_s3_key
        self._destination_key = final_s3_key
        self.shard_temp_location = f"{shard_temp_location}{FINAL_SHARD_PATTERN}"

        # We immediately need to compute the shard ranges
        self._shard_ranges = self._compute_shard_ranges()
        # Compute the final pipe here
        self.intermediate_urls = (
            f"pipe: aws s3 cp s3://{S3_BUCKET}/{self._source_key}{self._shard_ranges} -"
        )

        # Create the upoload hook
        self.upload_hook = partial(
            s3_upload_and_cleanup,
            bucket_name=S3_BUCKET,
            s3_prefix=self._destination_key,
            client=self.s3_client,
        )

        self.worker_count = worker_count
        self.shuffle_size = shuffle_size
        self.patch_write_count = patch_write_count

        # Compute the dataset
        self.dataset = (
            wds.WebDataset(self.intermediate_urls, resampled=True)
            .shuffle(self.shuffle_size, initial=self.shuffle_size)
            .decode()
        )

        # Create the dataloader
        self.dataloader = DataLoader(
            self.dataset, num_workers=self.worker_count, batch_size=None
        )

    def _compute_shard_ranges(self):
        """
        Computes the exact shard range for piping
        """
        output = []

        # Build the page iterator
        page_iterator = self.paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=self._source_key,
            PaginationConfig={"PageSize": 500},
        )

        for page in page_iterator:
            for key in page.get("Contents"):
                output.append(key.get("Key"))
        # we now have all the keys and we can try things out
        shard_numbers = [
            key.split("/")[-1].split(".")[0].split("_")[-1] for key in output
        ]

        # Identify the key for the intermediate shards
        sample_key = output[0]
        shard_elements = sample_key.split("/")[-1].split(".")[0].split("_")
        shard_identifier = "_".join(shard_elements[:2])

        # Put everything together
        final_range = (
            f"{shard_identifier}_{{{min(shard_numbers)}..{max(shard_numbers)}}}.tar"
        )

        return final_range

    def write_shards(self):
        """
        Shuffles and writes the shards
        """
        processed_patches = 0

        with wds.ShardWriter(
            self.shard_temp_location, maxsize=TARGET_GB, post=self.upload_hook
        ) as sink:
            for i, patch_dict in tqdm(enumerate(self.dataloader)):
                if i >= self.patch_write_count:
                    print("Reached Total Patch Count. Halting")
                    break
                ## A Key element here is to convert every tensor back to numpy.
                # This is important because of tensors cannpt be handled natively by webdataset but dataloader sort of converts them to tensors
                # Automatically
                for key, value in patch_dict.items():
                    if isinstance(value, torch.Tensor):
                        patch_dict[key] = value.cpu().numpy()
                sink.write(patch_dict)
                processed_patches += 1
        print(f"Processed Patches: {processed_patches}")


if __name__ == "__main__":
    finals = FinalPatchShuffler(
        intermediate_patch_s3_key="patches/intermediate/s200w128h128s64/",
        final_s3_key="patches/final/test2/",
    )
    finals.write_shards()
