"""
Generate EnMAP segmentation training shards.

Separate from `generate_hyperspectral_patches.py` on purpose: that script
produces Indradhanu's reconstruction shards and this one must not disturb
them. Output lands under the `enmap_seg` prefix, patches carry the provider
quality layers as `label_*.npy`, and the train/test split is stratified on
cloud cover rather than random.

Intended to run on Colab with Google Drive mounted:

    python -m scripts.generate_segmentation_patches \
        --scene-root "/content/drive/MyDrive/PS_11/EnMAP_datasets/vinoth" \
        --shard-dir  "/content/drive/MyDrive/PS_11/EnMAP_datasets/seg_shards"

Write shards to Colab's local disk and publish to Drive: one large copy per
scene beats thousands of small writes to a network mount. One shard per
scene means a killed session resumes rather than restarts.
"""

import argparse

from app.utils.patch_generation.intermediate.enmap_segmentation_patcher import (
    EnmapSegmentationSharder,
)
from app.utils.general_utils import s3_config
from app.utils.patch_generation.scene_storage import (
    LocalSceneStorage,
    S3SceneStorage,
    SceneStorage,
)

SPLITS = ("train", "test")


def run(
    storage: SceneStorage,
    work_dir: str,
    width: int = 128,
    height: int = 128,
    stride: int = 64,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_scenes: int | None = None,
    max_scenes_per_split: int | None = None,
) -> None:
    """Shard both splits. Safe to re-run: finished scenes are skipped."""
    for split in SPLITS:
        sharder = EnmapSegmentationSharder(
            storage=storage,
            work_dir=work_dir,
            split=split,
            test_fraction=test_fraction,
            seed=seed,
            width=width,
            height=height,
            stride=stride,
            max_scenes=max_scenes,
        )
        print(f"\n=== {split}: {len(sharder.scene_ids)} scenes -> "
              f"{sharder.destination_prefix} ===")
        sharder.sharder(scenes=max_scenes_per_split)


def build_storage(args: argparse.Namespace) -> SceneStorage:
    """Construct the backend named by --storage."""
    if args.storage == "local":
        if not args.scene_root:
            raise SystemExit("--scene-root is required with --storage local")
        return LocalSceneStorage(scene_root=args.scene_root, shard_dir=args.shard_dir)
    return S3SceneStorage(
        bucket=args.bucket,
        scene_prefix=args.scene_prefix,
        shard_prefix=args.shard_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--storage", choices=["local", "s3"], default="local")

    local = parser.add_argument_group("local storage")
    local.add_argument("--scene-root", help="Folder holding ENMAP01-* scene folders")
    local.add_argument(
        "--shard-dir",
        help="Where finished shards are published. Omit to leave them in --work-dir.",
    )

    s3 = parser.add_argument_group("s3 storage")
    s3.add_argument("--bucket", default=s3_config.BUCKET)
    s3.add_argument("--scene-prefix", default="enmap/")
    s3.add_argument("--shard-prefix", default=None)

    parser.add_argument(
        "--work-dir",
        default="/tmp/enmap_seg/",
        help="Scratch space for shards before publishing. Use fast local disk.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="Cap scenes before splitting"
    )
    parser.add_argument(
        "--max-scenes-per-split", type=int, default=None, help="Cap per split; for smoke tests"
    )

    args = parser.parse_args()
    run(
        storage=build_storage(args),
        work_dir=args.work_dir,
        width=args.width,
        height=args.height,
        stride=args.stride,
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_scenes=args.max_scenes,
        max_scenes_per_split=args.max_scenes_per_split,
    )


if __name__ == "__main__":
    main()
