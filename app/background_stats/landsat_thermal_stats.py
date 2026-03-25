"""
Normalizing temperature data is crucial for even simple encoder decoder 
frameworks to learn meaningful representations. This is a simple implementation 
of a mean and standard deviation calculator for the landsat thermal dataset. 
The dataset needs to be in hot storage for this to work.
"""

import argparse
import json
import sys
import pathlib
from pathlib import Path

import numpy as np
import torch
import webdataset as wds

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from app.models.training.training_config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Compute pixel normalization stats")
    parser.add_argument("config", type=str, help="Path to training config JSON")
    parser.add_argument("--max-patches", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="configs/pixel_stats.json")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Single patch size to use. Default: smallest in config.")
    return parser.parse_args()

def compute_stats(config: TrainingConfig, patch_size: int, max_patches: int):
    """
    We use the Welford - free approach of accumulating x and x^2 and counts in float64
    """

    hot = config.hot_storage
    data = config.data
    stride = patch_size // 2

    # Check if shards are present in the local dir
    print("Checking for local shards...")
    local_dir = (
        Path(hot.local_cache_dir)
        / f"train/{data.stage}/w{patch_size}_h{patch_size}_s{stride}"
    )
    shard_paths = sorted(local_dir.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {local_dir}")
    
    print("Building URLs ...")
    urls = [str(p) for p in shard_paths]
    in_channels = config.model_config_.in_channels

    # Set up float64 accumulators
    print("Setting up accumulators...")
    sum_x = np.zeros(in_channels, dtype=np.float64)
    sum_x2 = np.zeros(in_channels, dtype=np.float64)
    count = np.zeros(in_channels, dtype=np.float64)

    print("Setting up dataset and loader")
    dataset = wds.WebDataset(urls, shardshuffle=False, handler=wds.handlers.warn_and_continue)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    total_patches = 0

    for sample in loader:
        pixels = sample["pixels.npy"]
        pv = sample["pure_validity_mask.npy"]
        cq = sample["custom_quality_mask.npy"]

        # Note: mask is always pv * cq
        mask = (pv *cq).float()

        # Convert to numpy float 64
        # Not squeeze to maintain intuition
        px = pixels.numpy().astype(np.float64) # C, H, W
        # Masks must never be squeezed
        mk = mask.numpy().astype(np.float64) # C, H, W

        # Accumulate per channel
        # Doesnt matter for thermal but will matter for Hyperspectral
        for c in range(in_channels):
            sum_x[c] += (px[c] * mk[c]).sum()
            sum_x2[c] += (px[c] ** 2 * mk[c]).sum()
            count[c] += mk[c].sum()

        total_patches +=1
        if total_patches % 100 == 0:
            running_mean = sum_x /np.maxnimum(count, 1)
            print(f"  [{total_patches}] running mean = {running_mean}")
        
        if total_patches >= max_patches:
            break

    mean = sum_x / np.maximum(count, 1)                        # (C,)
    var = (sum_x2 / np.maximum(count, 1)) - mean ** 2          # (C,)
    std = np.sqrt(np.maximum(var, 1e-8))
    
    print(f"\nDone. {total_patches} patches, {int(count.sum())} valid pixels")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_patches": total_patches,
        "num_valid_pixels": int(count.sum()),
        "patch_size": patch_size,
    }                       

def main():
    args = parse_args()

    with open(args.config) as f:
        raw = json.load(f)
    config = TrainingConfig(**raw)

    if not config.hot_storage.enabled:
        print("Warning: hot_storage not enabled in config, but using its local_cache_dir")

    patch_sizes = config.data.patch_sizes
    size = args.patch_size or min(patch_sizes)
    print(f"Computing stats from patch size {size}, max {args.max_patches} patches")
    print(f"Reading from: {config.hot_storage.local_cache_dir}")

    stats = compute_stats(config, size, args.max_patches)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {output_path}")