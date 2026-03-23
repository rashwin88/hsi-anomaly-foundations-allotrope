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
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _run_one_intermediate(
    size: int,
    split: str,
    seed: int,
    test_fraction: float,
    base_folder: str,
    max_scenes: int | None,
):
    """Single intermediate sharding job. Runs in its own process with its own temp dir."""
    config = PATCH_CONFIGS[size]
    label = f"[Intermediate] {split} w{config['width']}_h{config['height']}_s{config['stride']}"

    # Each worker gets its own temp dir to avoid file clobbering
    work_dir = tempfile.mkdtemp(prefix=f"patch_{split}_{size}_", dir=base_folder)
    print(f"  {label} — using temp dir: {work_dir}")

    start = time.time()
    sharder = LandsatIntermediateSharder(
        source_folder=work_dir + "/",
        destination_folder=work_dir + "/",
        split=split,
        test_fraction=test_fraction,
        seed=seed,
        width=config["width"],
        height=config["height"],
        stride=config["stride"],
    )
    sharder.sharder(scenes=max_scenes)
    elapsed = time.time() - start

    # Clean up the temp dir
    os.rmdir(work_dir) if not os.listdir(work_dir) else None
    print(f"  {label} done in {elapsed:.1f}s")
    return label, elapsed


def run_intermediate(
    sizes: list[int],
    seed: int,
    test_fraction: float,
    base_folder: str,
    max_scenes: int | None,
    max_workers: int = 4,
):
    jobs = [(size, split) for size in sizes for split in SPLITS]

    if max_workers <= 1:
        # Sequential fallback
        for size, split in jobs:
            _run_one_intermediate(size, split, seed, test_fraction, base_folder, max_scenes)
        return

    print(f"  Launching {len(jobs)} intermediate jobs across {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_intermediate, size, split, seed, test_fraction, base_folder, max_scenes
            ): (size, split)
            for size, split in jobs
        }
        for future in as_completed(futures):
            size, split = futures[future]
            try:
                label, elapsed = future.result()
                print(f"  COMPLETED: {label} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {split} {size}x{size}: {e}")


def _run_one_final(
    size: int,
    split: str,
    shard_temp_location: str,
    worker_count: int,
    shuffle_size: int,
):
    """Single final sharding job. Each job gets its own temp dir to avoid clobbering."""
    config = PATCH_CONFIGS[size]
    patch_count = config["patch_write_count"]
    label = f"[Final] {split} w{config['width']}_h{config['height']}_s{config['stride']}"
    print(f"  {label}  (patch_write_count={patch_count:,})")

    # Each job writes to its own temp dir to avoid multiple workers
    # writing to the same shard file when running in parallel
    job_temp_dir = tempfile.mkdtemp(
        prefix=f"final_{split}_{size}_", dir=shard_temp_location
    )

    start = time.time()
    shuffler = FinalPatchShuffler(
        sensor="landsat",
        split=split,
        width=config["width"],
        height=config["height"],
        stride=config["stride"],
        shard_temp_location=job_temp_dir + "/",
        worker_count=worker_count,
        shuffle_size=shuffle_size,
        patch_write_count=patch_count,
    )
    shuffler.write_shards()
    elapsed = time.time() - start

    # Clean up empty temp dir (shards are uploaded to S3 and deleted by the hook)
    if os.path.exists(job_temp_dir) and not os.listdir(job_temp_dir):
        os.rmdir(job_temp_dir)

    print(f"  {label} done in {elapsed:.1f}s")
    return label, elapsed


def run_final(
    sizes: list[int],
    shard_temp_location: str,
    worker_count: int,
    shuffle_size: int,
    max_workers: int = 4,
):
    jobs = [(size, split) for size in sizes for split in SPLITS]

    if max_workers <= 1:
        for size, split in jobs:
            _run_one_final(size, split, shard_temp_location, worker_count, shuffle_size)
        return

    # Scale down DataLoader workers per job to avoid oversubscription
    workers_per_job = max(1, worker_count // max_workers)
    print(f"  Launching {len(jobs)} final jobs across {max_workers} workers ({workers_per_job} DataLoader workers each)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_final, size, split, shard_temp_location, workers_per_job, shuffle_size
            ): (size, split)
            for size, split in jobs
        }
        for future in as_completed(futures):
            size, split = futures[future]
            try:
                label, elapsed = future.result()
                print(f"  COMPLETED: {label} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {split} {size}x{size}: {e}")


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
        "--parallel",
        type=int,
        default=4,
        help="Max parallel jobs for size x split combos (default: 4, use 1 for sequential)",
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
            base_folder=args.source_folder,
            max_scenes=args.max_scenes,
            max_workers=args.parallel,
        )

    # Stage 2: Final sharding — uses DataLoader workers
    if not args.skip_final:
        print("\n" + "#" * 60)
        print(f"  STAGE 2: FINAL SHARDING ({args.final_workers} DataLoader workers, {args.parallel} parallel jobs)")
        print("#" * 60)
        run_final(
            sizes=args.sizes,
            shard_temp_location=args.shard_temp_location,
            worker_count=args.final_workers,
            shuffle_size=args.final_shuffle,
            max_workers=args.parallel,
        )

    total_elapsed = time.time() - total_start
    print(f"\nAll done. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
