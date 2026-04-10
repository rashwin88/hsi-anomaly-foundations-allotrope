"""
End-to-end hyperspectral patch generation for PRISMA and EnMAP scenes.

Stage 1 (Intermediate): Per-sensor sharding — downloads scenes from S3,
builds vendable datasets with configurable band filtering, patches into
tiles, filters by spatial validity, writes webdataset shards.

Stage 2 (Final): Cross-sensor shuffling — reads intermediate shards from
both PRISMA and EnMAP, shuffles patches together, writes mixed final shards
under a unified 'hyperspectral' prefix.

BandFilterConfig thresholds are exposed as CLI arguments so band-level
and pixel-level filtering can be tuned per sharding run.

Usage:
    python -m scripts.generate_hyperspectral_patches
    python -m scripts.generate_hyperspectral_patches --sensors prisma --sizes 64
    python -m scripts.generate_hyperspectral_patches --min-valid-pixel-pct 30 --max-invalid-voxel-fraction 0.3
    python -m scripts.generate_hyperspectral_patches --skip-final --max-scenes 2
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from app.models.dataset.vendables import BandFilterConfig, DEFAULT_COMMON_WAVELENGTH_GRID
from app.utils.patch_generation.intermediate.prisma_intermediate_patcher import (
    PrismaIntermediateSharder,
)
from app.utils.patch_generation.intermediate.enmap_intermediate_patcher import (
    EnmapIntermediateSharder,
)
from app.utils.patch_generation.final.hyperspectral_final_patcher import (
    HyperspectralFinalShuffler,
)


# stride = size // 2 (50% overlap)
# Patch counts based on 100 scenes per sensor, ~270 valid patches/scene:
#   Train (80 scenes × 2 sensors × 270): ~43,200 available → 43,000 final
#   Test  (20 scenes × 2 sensors × 270): ~10,800 available → 10,000 final
PATCH_CONFIGS = {
    64: {
        "width": 64, "height": 64, "stride": 32,
        "patch_write_count_train": 170_000,
        "patch_write_count_test": 42_000,
    },
    128: {
        "width": 128, "height": 128, "stride": 64,
        "patch_write_count_train": 43_000,
        "patch_write_count_test": 10_000,
    },
}

DEFAULT_MAX_SCENES = 100

SENSORS = ["prisma", "enmap"]
SPLITS = ["train", "test"]

SHARDER_CLASSES = {
    "prisma": PrismaIntermediateSharder,
    "enmap": EnmapIntermediateSharder,
}


def _run_one_intermediate(
    sensor: str,
    size: int,
    split: str,
    seed: int,
    test_fraction: float,
    base_folder: str,
    max_scenes: int | None,
    patch_validity_threshold: float,
    band_filter_config: BandFilterConfig,
):
    config = PATCH_CONFIGS[size]
    label = f"[Intermediate] {sensor} {split} w{config['width']}_h{config['height']}_s{config['stride']}"

    work_dir = tempfile.mkdtemp(prefix=f"patch_{sensor}_{split}_{size}_", dir=base_folder)
    print(f"  {label} — using temp dir: {work_dir}")

    start = time.time()
    sharder_cls = SHARDER_CLASSES[sensor]
    sharder = sharder_cls(
        source_folder=work_dir + "/",
        destination_folder=work_dir + "/",
        split=split,
        test_fraction=test_fraction,
        seed=seed,
        width=config["width"],
        height=config["height"],
        stride=config["stride"],
        patch_validity_threshold=patch_validity_threshold,
        band_filter_config=band_filter_config,
        max_scenes=max_scenes,
    )
    sharder.sharder()
    elapsed = time.time() - start

    if os.path.exists(work_dir) and not os.listdir(work_dir):
        os.rmdir(work_dir)
    print(f"  {label} done in {elapsed:.1f}s")
    return label, elapsed


def run_intermediate(
    sensors: list[str],
    sizes: list[int],
    seed: int,
    test_fraction: float,
    base_folder: str,
    max_scenes: int | None,
    patch_validity_threshold: float,
    band_filter_config: BandFilterConfig,
    max_workers: int = 2,
):
    jobs = [
        (sensor, size, split)
        for sensor in sensors
        for size in sizes
        for split in SPLITS
    ]

    if max_workers <= 1:
        for sensor, size, split in jobs:
            _run_one_intermediate(
                sensor, size, split, seed, test_fraction,
                base_folder, max_scenes, patch_validity_threshold,
                band_filter_config,
            )
        return

    print(f"  Launching {len(jobs)} intermediate jobs across {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_intermediate, sensor, size, split, seed, test_fraction,
                base_folder, max_scenes, patch_validity_threshold, band_filter_config,
            ): (sensor, size, split)
            for sensor, size, split in jobs
        }
        for future in as_completed(futures):
            sensor, size, split = futures[future]
            try:
                label, elapsed = future.result()
                print(f"  COMPLETED: {label} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {sensor} {split} {size}x{size}: {e}")


def _run_one_final(
    size: int,
    split: str,
    sensors: list[str],
    shard_temp_location: str,
    worker_count: int,
    shuffle_size: int,
):
    config = PATCH_CONFIGS[size]
    patch_count = config[f"patch_write_count_{split}"]
    label = f"[Final] {split} w{config['width']}_h{config['height']}_s{config['stride']}"
    print(f"  {label}  (patch_write_count={patch_count:,}, sensors={sensors})")

    job_temp_dir = tempfile.mkdtemp(
        prefix=f"final_hsi_{split}_{size}_", dir=shard_temp_location
    )

    start = time.time()
    shuffler = HyperspectralFinalShuffler(
        split=split,
        width=config["width"],
        height=config["height"],
        stride=config["stride"],
        sensors=sensors,
        shard_temp_location=job_temp_dir + "/",
        worker_count=worker_count,
        shuffle_size=shuffle_size,
        patch_write_count=patch_count,
    )
    shuffler.write_shards()
    elapsed = time.time() - start

    if os.path.exists(job_temp_dir) and not os.listdir(job_temp_dir):
        os.rmdir(job_temp_dir)

    print(f"  {label} done in {elapsed:.1f}s")
    return label, elapsed


def run_final(
    sizes: list[int],
    sensors: list[str],
    shard_temp_location: str,
    worker_count: int,
    shuffle_size: int,
    max_workers: int = 2,
):
    # Final stage iterates over (size, split) — sensors are mixed within each job
    jobs = [(size, split) for size in sizes for split in SPLITS]

    if max_workers <= 1:
        for size, split in jobs:
            _run_one_final(size, split, sensors, shard_temp_location, worker_count, shuffle_size)
        return

    workers_per_job = max(1, worker_count // max_workers)
    print(f"  Launching {len(jobs)} final jobs across {max_workers} workers ({workers_per_job} DataLoader workers each)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_final, size, split, sensors, shard_temp_location,
                workers_per_job, shuffle_size,
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
        description="Generate hyperspectral patches from PRISMA and EnMAP scenes"
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        type=str,
        default=SENSORS,
        choices=SENSORS,
        help="Sensors to process (default: prisma enmap)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128],
        help="Patch sizes to generate (default: 64 128)",
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
        "--shard-temp-location",
        type=str,
        default="/home/ubuntu/",
        help="Local temp dir for final shard files",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=DEFAULT_MAX_SCENES,
        help=f"Max scenes per sensor (split across train/test) (default: {DEFAULT_MAX_SCENES})",
    )

    # BandFilterConfig thresholds
    parser.add_argument(
        "--min-valid-pixel-pct",
        type=float,
        default=20.0,
        help="Band-level: min %% of valid pixels to keep a band (default: 20.0)",
    )
    parser.add_argument(
        "--max-invalid-voxel-fraction",
        type=float,
        default=0.4,
        help="Pixel-level: max fraction of invalid bands before pixel is fully invalidated (default: 0.4)",
    )
    parser.add_argument(
        "--edge-bands-to-trim",
        type=int,
        default=3,
        help="Bands to trim from each detector edge (default: 3)",
    )
    parser.add_argument(
        "--patch-validity-threshold",
        type=float,
        default=0.5,
        help="Patch-level: min fraction of valid pixels to keep a patch (default: 0.5)",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip resampling to common wavelength grid (keeps native band counts)",
    )

    # Parallelism and shuffling
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
        default=2,
        help="Max parallel jobs (default: 2, lower than Landsat due to memory)",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    valid_sizes = set(PATCH_CONFIGS.keys())
    for s in args.sizes:
        if s not in valid_sizes:
            parser.error(f"Invalid size {s}. Choose from {sorted(valid_sizes)}")

    # Build BandFilterConfig from CLI args
    common_grid = None if args.no_resample else DEFAULT_COMMON_WAVELENGTH_GRID
    band_filter_config = BandFilterConfig(
        min_valid_pixel_pct=args.min_valid_pixel_pct,
        max_invalid_voxel_fraction=args.max_invalid_voxel_fraction,
        edge_bands_to_trim=args.edge_bands_to_trim,
        common_wavelength_grid=common_grid,
    )

    print(f"Sensors: {args.sensors}")
    print(f"Patch sizes: {sorted(args.sizes)}")
    print(f"Seed: {args.seed}, Test fraction: {args.test_fraction}")
    print(f"Common wavelength grid: {len(common_grid)} bands" if common_grid is not None else "Common wavelength grid: disabled (native bands)")
    print(f"BandFilterConfig: min_valid_pixel_pct={band_filter_config.min_valid_pixel_pct}, "
          f"max_invalid_voxel_fraction={band_filter_config.max_invalid_voxel_fraction}, "
          f"edge_bands_to_trim={band_filter_config.edge_bands_to_trim}")
    print(f"Patch validity threshold: {args.patch_validity_threshold}")
    print(f"Max scenes per sensor: {args.max_scenes}")
    print("Final patch write counts:")
    for s in sorted(args.sizes):
        cfg_s = PATCH_CONFIGS[s]
        print(f"  {s}x{s}: train={cfg_s['patch_write_count_train']:,}, test={cfg_s['patch_write_count_test']:,}")

    total_start = time.time()

    # Stage 1: Intermediate sharding — only the sensors requested
    if not args.skip_intermediate:
        print("\n" + "#" * 60)
        print(f"  STAGE 1: INTERMEDIATE SHARDING ({args.sensors})")
        print("#" * 60)
        run_intermediate(
            sensors=args.sensors,
            sizes=args.sizes,
            seed=args.seed,
            test_fraction=args.test_fraction,
            base_folder=args.source_folder,
            max_scenes=args.max_scenes,
            patch_validity_threshold=args.patch_validity_threshold,
            band_filter_config=band_filter_config,
            max_workers=args.parallel,
        )

    # Stage 2: Final sharding — ALWAYS mixes all sensors that have intermediates
    # (regardless of --sensors flag, which only controls Stage 1)
    if not args.skip_final:
        all_sensors = list(SENSORS)  # always ["prisma", "enmap"]
        print("\n" + "#" * 60)
        print(f"  STAGE 2: FINAL SHARDING (mixing {all_sensors})")
        print("#" * 60)
        run_final(
            sizes=args.sizes,
            sensors=all_sensors,
            shard_temp_location=args.shard_temp_location,
            worker_count=args.final_workers,
            shuffle_size=args.final_shuffle,
            max_workers=args.parallel,
        )

    total_elapsed = time.time() - total_start
    print(f"\nAll done. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
