"""
End-to-end Landsat patch generation for all target patch sizes.

Runs intermediate sharding (Stage 1) for all sizes and splits first,
then final sharding (Stage 2) for all sizes and splits.

Final patch write counts are scaled proportionally: ~6.5M for 64x64
and reducing by (size/64)^2 for larger sizes, since larger patches
produce fewer patches per scene.

Scene images are approximately 1200x1200 pixels.

Usage:
    python -m scripts.generate_landsat_patches
    python -m scripts.generate_landsat_patches --sizes 64 128
    python -m scripts.generate_landsat_patches --seed 99 --test-fraction 0.15
    python -m scripts.generate_landsat_patches --final-workers 20
"""

import argparse
import logging
import sys
import time

from app.utils.patch_generation.intermediate.landsat_intermediate_patcher import (
    LandsatIntermediateSharder,
)
from app.utils.patch_generation.final.final_patcher import FinalPatchShuffler


# stride = size // 2 (50% overlap)
# patch_write_count scaled from ~6.5M at 64x64, down by (size/64)^2
PATCH_CONFIGS = {
    64:  {"width": 64,  "height": 64,  "stride": 32,  "patch_write_count": 6_500_000},
    128: {"width": 128, "height": 128, "stride": 64,  "patch_write_count": 1_625_000},
    256: {"width": 256, "height": 256, "stride": 128, "patch_write_count": 406_250},
    512: {"width": 512, "height": 512, "stride": 256, "patch_write_count": 101_562},
}

SPLITS = ["train", "test"]


def run_intermediate(
    sizes: list[int],
    seed: int,
    test_fraction: float,
    source_folder: str,
    destination_folder: str,
    max_scenes: int | None,
):
    for size in sizes:
        config = PATCH_CONFIGS[size]
        for split in SPLITS:
            label = f"[Intermediate] {split} w{config['width']}_h{config['height']}_s{config['stride']}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            start = time.time()
            sharder = LandsatIntermediateSharder(
                source_folder=source_folder,
                destination_folder=destination_folder,
                split=split,
                test_fraction=test_fraction,
                seed=seed,
                width=config["width"],
                height=config["height"],
                stride=config["stride"],
            )
            sharder.sharder(scenes=max_scenes)
            elapsed = time.time() - start
            print(f"  {label} done in {elapsed:.1f}s")


def run_final(
    sizes: list[int],
    shard_temp_location: str,
    worker_count: int,
    shuffle_size: int,
):
    for size in sizes:
        config = PATCH_CONFIGS[size]
        patch_count = config["patch_write_count"]
        for split in SPLITS:
            label = f"[Final] {split} w{config['width']}_h{config['height']}_s{config['stride']}"
            print(f"\n{'='*60}")
            print(f"  {label}  (patch_write_count={patch_count:,})")
            print(f"{'='*60}")

            start = time.time()
            shuffler = FinalPatchShuffler(
                sensor="landsat",
                split=split,
                width=config["width"],
                height=config["height"],
                stride=config["stride"],
                shard_temp_location=shard_temp_location,
                worker_count=worker_count,
                shuffle_size=shuffle_size,
                patch_write_count=patch_count,
            )
            shuffler.write_shards()
            elapsed = time.time() - start
            print(f"  {label} done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Landsat patches at multiple scales"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Patch sizes to generate (default: 64 128 256 512)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of scenes for test (default: 0.2)",
    )
    parser.add_argument(
        "--source-folder",
        type=str,
        default="/home/ubuntu/",
        help="Local temp dir for scene downloads",
    )
    parser.add_argument(
        "--destination-folder",
        type=str,
        default="/home/ubuntu/",
        help="Local temp dir for shard files",
    )
    parser.add_argument(
        "--shard-temp-location",
        type=str,
        default="/home/ubuntu/",
        help="Local temp dir for final shard files",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit scenes per split (for testing)",
    )
    parser.add_argument(
        "--final-workers",
        type=int,
        default=10,
        help="DataLoader workers for final sharding (default: 10)",
    )
    parser.add_argument(
        "--final-shuffle",
        type=int,
        default=10,
        help="Shuffle buffer size for final sharding (default: 10)",
    )
    parser.add_argument(
        "--skip-intermediate",
        action="store_true",
        help="Skip Stage 1, only run final sharding",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip Stage 2, only run intermediate sharding",
    )

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    valid_sizes = set(PATCH_CONFIGS.keys())
    for s in args.sizes:
        if s not in valid_sizes:
            parser.error(f"Invalid size {s}. Choose from {sorted(valid_sizes)}")

    print(f"Patch sizes: {sorted(args.sizes)}")
    print(f"Seed: {args.seed}, Test fraction: {args.test_fraction}")
    print("Final patch write counts:")
    for s in sorted(args.sizes):
        print(f"  {s}x{s}: {PATCH_CONFIGS[s]['patch_write_count']:,}")

    total_start = time.time()

    # Stage 1: Intermediate sharding — sequential, single-threaded
    if not args.skip_intermediate:
        print("\n" + "#" * 60)
        print("  STAGE 1: INTERMEDIATE SHARDING")
        print("#" * 60)
        run_intermediate(
            sizes=args.sizes,
            seed=args.seed,
            test_fraction=args.test_fraction,
            source_folder=args.source_folder,
            destination_folder=args.destination_folder,
            max_scenes=args.max_scenes,
        )

    # Stage 2: Final sharding — uses DataLoader workers
    if not args.skip_final:
        print("\n" + "#" * 60)
        print(f"  STAGE 2: FINAL SHARDING ({args.final_workers} DataLoader workers)")
        print("#" * 60)
        run_final(
            sizes=args.sizes,
            shard_temp_location=args.shard_temp_location,
            worker_count=args.final_workers,
            shuffle_size=args.final_shuffle,
        )

    total_elapsed = time.time() - total_start
    print(f"\nAll done. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
