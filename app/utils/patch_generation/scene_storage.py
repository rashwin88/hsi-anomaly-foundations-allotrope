"""Storage seam for segmentation sharding.

The reconstruction sharders reach for boto3 directly; segmentation sharding
must also run on Colab with Drive mounted, where scenes are ordinary local
paths. Rationale: docs/lld/segmentation-sharding.md.
"""

import glob
import os
import shutil
from typing import Protocol


class SceneStorage(Protocol):
    """Where scenes are read from and finished shards are written to."""

    def list_scenes(self) -> list[str]:
        """Every available scene id, sorted. Ids are opaque to the caller."""
        ...

    def fetch_scene(self, scene_id: str, dest_dir: str) -> str:
        """Make a scene's files readable locally; return its folder path.

        Local backends copy nothing and return the path they already have.
        """
        ...

    def publish_shard(self, local_path: str) -> None:
        """Hand off a finished .tar — upload, move, or do nothing."""
        ...

    def release_scene(self, path: str) -> None:
        """Discard a fetched scene once its patches are written.

        S3 removes the temporary download so the disk does not fill across a
        few hundred scenes. Local backends do nothing at all — the path is
        the user's own data, not a copy.

        The sharder must NEVER delete a scene folder itself: it cannot tell
        which case it is in, and getting it wrong destroys source data.
        """
        ...

    def shard_exists(self, name: str) -> bool:
        """True if a shard of this name has already been published.

        Drives resume: a scene whose shard is already there is skipped, so
        a killed Colab session costs one scene rather than the whole run.
        Backends with no publish destination return False — never skip
        work on the strength of a destination that was never configured.
        """
        ...


class LocalSceneStorage:
    """Scenes already on a mounted filesystem (Colab + Drive, or a copy).

    `shard_dir` is where finished shards are published to. Leaving it None
    keeps them where they were written. On Colab, write shards to fast local
    disk and set `shard_dir` to Drive: one large copy per scene beats
    thousands of small writes to a network mount.
    """

    def __init__(
        self,
        scene_root: str,
        shard_dir: str | None = None,
        pattern: str = "ENMAP01*",
    ):
        self.scene_root = scene_root
        self.shard_dir = shard_dir
        self.pattern = pattern

    def list_scenes(self) -> list[str]:
        return sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(self.scene_root, self.pattern))
            if os.path.isdir(p)
        )

    def fetch_scene(self, scene_id: str, dest_dir: str) -> str:
        # Already readable — nothing to download. `dest_dir` is ignored.
        return os.path.join(self.scene_root, scene_id)

    def publish_shard(self, local_path: str) -> None:
        if self.shard_dir is None:
            return
        os.makedirs(self.shard_dir, exist_ok=True)
        dest = os.path.join(self.shard_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)

    def release_scene(self, path: str) -> None:
        # Nothing was copied, so there is nothing to discard. Deleting here
        # would destroy the scenes on the user's own disk or Drive.
        return

    def shard_exists(self, name: str) -> bool:
        if self.shard_dir is None:
            return False
        return os.path.exists(os.path.join(self.shard_dir, name))


class S3SceneStorage:
    """Scenes in S3 — mirrors what the reconstruction sharders already do.

    Scene ids are bare folder names, not full prefixes, so both backends
    speak the same vocabulary and the sharder never learns which it has.
    """

    def __init__(
        self,
        bucket: str = "allotrope-raw-data-india",
        scene_prefix: str = "enmap/",
        shard_prefix: str | None = None,
        region_name: str = "ap-south-1",
    ):
        import boto3  # deferred: local runs must not need AWS installed

        self.bucket = bucket
        self.scene_prefix = scene_prefix
        self.shard_prefix = shard_prefix
        self.client = boto3.client("s3", region_name=region_name)
        # Folders this backend created — the only ones release_scene may remove.
        self._fetched: set[str] = set()
        self.paginator = self.client.get_paginator("list_objects_v2")

    def list_scenes(self) -> list[str]:
        pages = self.paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.scene_prefix,
            Delimiter="/",
            PaginationConfig={"PageSize": 500},
        )
        return sorted(
            folder["Prefix"].rstrip("/").split("/")[-1]
            for page in pages
            for folder in page.get("CommonPrefixes", [])
        )

    def fetch_scene(self, scene_id: str, dest_dir: str) -> str:
        # Scene folder name is preserved — FileSourceConfig auto-detects on it.
        local_folder = os.path.join(dest_dir, scene_id)
        os.makedirs(local_folder, exist_ok=True)
        objects = self.client.list_objects(
            Bucket=self.bucket, Prefix=f"{self.scene_prefix}{scene_id}/"
        )
        for content in objects.get("Contents", []):
            file_name = content.get("Key", "").split("/")[-1]
            if not file_name:
                continue
            self.client.download_file(
                Bucket=self.bucket,
                Key=content["Key"],
                Filename=os.path.join(local_folder, file_name),
            )
        self._fetched.add(local_folder)
        return local_folder

    def release_scene(self, path: str) -> None:
        # Only ever remove a directory this backend downloaded into. A path
        # we did not create is somebody else's data.
        if path in self._fetched and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            self._fetched.discard(path)

    def publish_shard(self, local_path: str) -> None:
        if self.shard_prefix is None:
            return
        self.client.upload_file(
            Filename=local_path,
            Bucket=self.bucket,
            Key=f"{self.shard_prefix}{os.path.basename(local_path)}",
        )

    def shard_exists(self, name: str) -> bool:
        if self.shard_prefix is None:
            return False
        try:
            self.client.head_object(
                Bucket=self.bucket, Key=f"{self.shard_prefix}{name}"
            )
            return True
        except self.client.exceptions.ClientError:
            return False
