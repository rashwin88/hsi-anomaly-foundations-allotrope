"""
Sync hyperspectral shards from S3 to local disk (hot storage).

Downloads a specified number of train and test shards for a given
sensor/patch-size configuration. Skips shards that already exist locally.

Use this to pre-populate hot storage before training, or to download
shards for computing pixel normalization stats.

Usage:
    # Sync 50 train + 10 test shards to /tmp/hot_shards_hsi/
    python -m scripts.sync_shards_to_local \
        --local-dir /tmp/hot_shards_hsi/ \
        --train-shards 50 \
        --test-shards 10

    # Sync from intermediate shards (per-sensor, for stats computation)
    python -m scripts.sync_shards_to_local \
        --local-dir /tmp/hot_shards_hsi/ \
        --train-shards 20 \
        --test-shards 0 \
        --provider prisma \
        --stage intermediate

    # Then compute stats:
    python -m scripts.compute_hyperspectral_pixel_stats \
        --shard-dir /tmp/hot_shards_hsi/train/128/ \
        --output /home/ubuntu/constants/hyperspectral.json
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.general_utils import s3_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

S3_BUCKET = s3_config.BUCKET
S3_REGION = s3_config.REGION

SHARD_KEY_TEMPLATE = "patches/{provider}/{split}/{stage}/w{size}_h{size}_s{stride}/"


def list_shard_keys(client, prefix: str) -> list[str]:
    """List all .tar shard keys under a prefix."""
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 500}):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".tar"):
                keys.append(key)
    return sorted(keys)


def download_shard(client, key: str, local_path: str) -> str:
    """Download a single shard, skip if exists."""
    if os.path.exists(local_path):
        return f"SKIP {os.path.basename(local_path)}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.download_file(S3_BUCKET, key, local_path)
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    return f"OK   {os.path.basename(local_path)} ({size_mb:.0f} MB)"


def sync_split(
    client,
    provider: str,
    split: str,
    stage: str,
    size: int,
    local_dir: str,
    max_shards: int,
    max_workers: int,
):
    """Sync up to max_shards for a given split."""
    if max_shards <= 0:
        return

    prefix = SHARD_KEY_TEMPLATE.format(
        provider=provider, split=split, stage=stage,
        size=size, stride=size // 2,
    )
    logger.info(f"Listing shards: s3://{S3_BUCKET}/{prefix}")
    keys = list_shard_keys(client, prefix)

    if not keys:
        logger.warning(f"No shards found at {prefix}")
        return

    keys = keys[:max_shards]
    # Match trainer's expected path: {local_dir}/{split}/{stage}/w{size}_h{size}_s{stride}/
    local_split_dir = os.path.join(local_dir, split, stage, f"w{size}_h{size}_s{size // 2}")
    os.makedirs(local_split_dir, exist_ok=True)

    existing = len([f for f in os.listdir(local_split_dir) if f.endswith(".tar")])
    logger.info(
        f"  {split}: {len(keys)} shards to sync, {existing} already local → {local_split_dir}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for key in keys:
            fname = os.path.basename(key)
            local_path = os.path.join(local_split_dir, fname)
            futures[executor.submit(download_shard, client, key, local_path)] = fname

        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if done % 10 == 0 or done == len(futures):
                logger.info(f"  [{done}/{len(futures)}] {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync hyperspectral shards from S3 to local disk"
    )
    parser.add_argument(
        "--local-dir", type=str, default="/tmp/hot_shards_hsi/",
        help="Local directory to sync shards into (default: /tmp/hot_shards_hsi/)",
    )
    parser.add_argument(
        "--train-shards", type=int, default=50,
        help="Number of train shards to sync (default: 50)",
    )
    parser.add_argument(
        "--test-shards", type=int, default=10,
        help="Number of test shards to sync (default: 10)",
    )
    parser.add_argument(
        "--provider", type=str, default="hyperspectral",
        help="Data provider prefix: 'hyperspectral' (mixed final), 'prisma', or 'enmap' (default: hyperspectral)",
    )
    parser.add_argument(
        "--stage", type=str, default="final",
        help="Shard stage: 'final' or 'intermediate' (default: final)",
    )
    parser.add_argument(
        "--size", type=int, default=128,
        help="Patch size (default: 128)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Parallel download threads (default: 10)",
    )

    args = parser.parse_args()
    client = s3_config.client()

    logger.info(f"Syncing to: {args.local_dir}")
    logger.info(f"Provider: {args.provider}, stage: {args.stage}, size: {args.size}")

    sync_split(client, args.provider, "train", args.stage, args.size,
               args.local_dir, args.train_shards, args.max_workers)

    sync_split(client, args.provider, "test", args.stage, args.size,
               args.local_dir, args.test_shards, args.max_workers)

    # Summary
    for split in ["train", "test"]:
        split_dir = os.path.join(args.local_dir, split, args.stage, f"w{args.size}_h{args.size}_s{args.size // 2}")
        if os.path.isdir(split_dir):
            tars = [f for f in os.listdir(split_dir) if f.endswith(".tar")]
            total_gb = sum(os.path.getsize(os.path.join(split_dir, f)) for f in tars) / 1024**3
            print(f"  {split}: {len(tars)} shards, {total_gb:.1f} GB → {split_dir}")

    train_dir = os.path.join(args.local_dir, "train", args.stage, f"w{args.size}_h{args.size}_s{args.size // 2}")
    print(f"\nTo compute pixel stats:")
    print(f"  python -m scripts.compute_hyperspectral_pixel_stats \\")
    print(f"      --shard-dir {train_dir} \\")
    print(f"      --output /home/ubuntu/constants/hyperspectral.json")


if __name__ == "__main__":
    main()
