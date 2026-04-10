"""
Compute per-band pixel statistics (mean and std) from hyperspectral shards.

Reads webdataset shards from a local directory (hot storage) or S3,
computes the running mean and std across all valid pixels for each of the
165 spectral bands, and writes the result to a JSON file.

Only valid pixels contribute (validity_cube band 0 == 1). Invalid pixels
are excluded to avoid biasing the statistics toward zero.

Uses Welford's online algorithm for numerically stable single-pass
computation — no need to hold all data in memory.

Usage:
    # From hot storage (local shards)
    python -m scripts.compute_hyperspectral_pixel_stats \
        --shard-dir /tmp/hot_shards_hsi/train/128/ \
        --output /home/ubuntu/constants/hyperspectral.json

    # From S3 (streaming)
    python -m scripts.compute_hyperspectral_pixel_stats \
        --s3-prefix patches/hyperspectral/train/final/w128_h128_s64/ \
        --output /home/ubuntu/constants/hyperspectral.json

    # Limit number of patches for a quick estimate
    python -m scripts.compute_hyperspectral_pixel_stats \
        --shard-dir /tmp/hot_shards_hsi/train/128/ \
        --output /tmp/hyperspectral_stats.json \
        --max-patches 10000
"""

import argparse
import json
import glob
import logging
import sys

import numpy as np
import webdataset as wds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def compute_stats(dataset, max_patches: int = None) -> dict:
    """
    Compute per-band mean and std using Welford's online algorithm.

    Returns dict with 'mean' and 'std' lists (length = num bands).
    """
    n = None           # per-band valid pixel count
    mean = None        # per-band running mean
    m2 = None          # per-band running sum of squared deviations

    num_patches = 0

    for sample in dataset:
        pixels = sample["pixels.npy"]          # (C, H, W)
        validity = sample["validity_cube.npy"]  # (C, H, W)

        C = pixels.shape[0]

        if n is None:
            n = np.zeros(C, dtype=np.float64)
            mean = np.zeros(C, dtype=np.float64)
            m2 = np.zeros(C, dtype=np.float64)

        # Spatial validity mask — band 0 as proxy (all bands agree)
        spatial_valid = validity[0] > 0  # (H, W)

        if not spatial_valid.any():
            continue

        # Extract valid pixels: (C, num_valid)
        valid_pixels = pixels[:, spatial_valid].astype(np.float64)
        batch_n = valid_pixels.shape[1]

        # Welford's online update (per-band)
        for c in range(C):
            for val in valid_pixels[c]:
                n[c] += 1
                delta = val - mean[c]
                mean[c] += delta / n[c]
                delta2 = val - mean[c]
                m2[c] += delta * delta2

        num_patches += 1

        if num_patches % 500 == 0:
            logger.info(
                "Processed %d patches, ~%d valid pixels per band",
                num_patches, int(n[0]),
            )

        if max_patches is not None and num_patches >= max_patches:
            logger.info("Reached max_patches=%d, stopping.", max_patches)
            break

    if n is None or n.min() < 2:
        raise ValueError("Not enough valid pixels to compute statistics.")

    std = np.sqrt(m2 / (n - 1))

    logger.info(
        "Done. %d patches, %d–%d valid pixels per band.",
        num_patches, int(n.min()), int(n.max()),
    )
    logger.info("Mean range: [%.6f, %.6f]", mean.min(), mean.max())
    logger.info("Std range:  [%.6f, %.6f]", std.min(), std.max())

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_patches": num_patches,
        "valid_pixels_per_band": n.tolist(),
    }


def build_dataset(shard_dir: str = None, s3_prefix: str = None):
    """Build a webdataset from local shards or S3."""
    if shard_dir is not None:
        shard_files = sorted(glob.glob(f"{shard_dir}/*.tar"))
        if not shard_files:
            raise FileNotFoundError(f"No .tar files found in {shard_dir}")
        logger.info("Reading %d local shards from %s", len(shard_files), shard_dir)
        return wds.WebDataset(shard_files, shardshuffle=False).decode()

    if s3_prefix is not None:
        bucket = "allotrope-raw-data-india"
        url = f"pipe: aws s3 cp s3://{bucket}/{s3_prefix} - 2>/dev/null"
        logger.info("Streaming from S3: %s", s3_prefix)
        return wds.WebDataset(url, shardshuffle=False).decode()

    raise ValueError("Provide either --shard-dir or --s3-prefix")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-band pixel statistics from hyperspectral shards"
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default=None,
        help="Local directory containing .tar shard files (hot storage)",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="S3 prefix for shard files (streaming mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Limit number of patches to process (for quick estimates)",
    )

    args = parser.parse_args()

    dataset = build_dataset(shard_dir=args.shard_dir, s3_prefix=args.s3_prefix)
    stats = compute_stats(dataset, max_patches=args.max_patches)

    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Stats written to %s", args.output)
    logger.info(
        "  %d bands, mean range [%.4f, %.4f], std range [%.4f, %.4f]",
        len(stats["mean"]),
        min(stats["mean"]), max(stats["mean"]),
        min(stats["std"]), max(stats["std"]),
    )


if __name__ == "__main__":
    main()
