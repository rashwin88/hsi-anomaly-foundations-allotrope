"""Storage seam for segmentation sharding.

The reconstruction sharders reach for boto3 directly; segmentation sharding
must also run on Colab with Drive mounted, where scenes are ordinary local
paths. Rationale: docs/lld/segmentation-sharding.md.
"""

import glob
import os
import shutil
from typing import Protocol

from app.utils.general_utils import s3_config


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

    def fetch_metadata(self, scene_id: str, dest_dir: str) -> str:
        """Make only a scene's metadata readable locally; return its folder.

        The stratified split needs each scene's cover percentages, which live
        in METADATA.XML — and only its first 64 KB at that. Going through
        `fetch_scene` would pull the whole ~300 MB scene to read them, so
        splitting 212 scenes would move ~64 GB to use ~14 MB.

        Local backends are a no-op and return the folder they already have,
        exactly like `fetch_scene`.
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
        dirs_only: bool = True,
    ):
        # dirs_only=False for sensors whose "scene" is a single file rather
        # than a folder — PRISMA ships one .he5 per scene, where EnMAP and
        # Landsat ship a folder of TIFs.
        self.scene_root = scene_root
        self.shard_dir = shard_dir
        self.pattern = pattern
        self.dirs_only = dirs_only

    def list_scenes(self) -> list[str]:
        keep = os.path.isdir if self.dirs_only else os.path.isfile
        return sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(self.scene_root, self.pattern))
            if keep(p)
        )

    def fetch_scene(self, scene_id: str, dest_dir: str) -> str:
        # Already readable — nothing to download. `dest_dir` is ignored.
        return os.path.join(self.scene_root, scene_id)

    def fetch_metadata(self, scene_id: str, dest_dir: str) -> str:
        # Nothing to download, so there is nothing cheaper to download.
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
        bucket: str = s3_config.BUCKET,
        scene_prefix: str = "enmap/",
        shard_prefix: str | None = None,
        region_name: str = s3_config.REGION,
        metadata_suffix: str = "METADATA.XML",
        scene_suffix: str | None = None,
    ):
        import boto3  # deferred: local runs must not need AWS installed

        self.bucket = bucket
        self.scene_prefix = scene_prefix
        self.shard_prefix = shard_prefix
        self.metadata_suffix = metadata_suffix
        # Set for sensors whose "scene" is one object rather than a folder
        # of them — PRISMA's ".he5". Changes both listing and fetching.
        self.scene_suffix = scene_suffix
        self.client = boto3.client("s3", region_name=region_name)
        # Folders this backend created — the only ones release_scene may remove.
        self._fetched: set[str] = set()
        self.paginator = self.client.get_paginator("list_objects_v2")

    def list_scenes(self) -> list[str]:
        if self.scene_suffix is not None:
            pages = self.paginator.paginate(
                Bucket=self.bucket,
                Prefix=self.scene_prefix,
                PaginationConfig={"PageSize": 500},
            )
            return sorted(
                obj["Key"].split("/")[-1]
                for page in pages
                for obj in page.get("Contents", [])
                if obj.get("Key", "").endswith(self.scene_suffix)
            )
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
        return self._download(scene_id, dest_dir, suffix=None)

    def fetch_metadata(self, scene_id: str, dest_dir: str) -> str:
        return self._download(scene_id, dest_dir, suffix=self.metadata_suffix)

    def _download(self, scene_id: str, dest_dir: str, suffix: str | None) -> str:
        """Download a scene's files, or only those ending in `suffix`.

        For a single-object sensor (`scene_suffix` set) there is nothing to
        filter and the returned path is the file itself, not a folder.
        """
        if self.scene_suffix is not None:
            os.makedirs(dest_dir, exist_ok=True)
            local_path = os.path.join(dest_dir, scene_id)
            self.client.download_file(
                Bucket=self.bucket,
                Key=f"{self.scene_prefix}{scene_id}",
                Filename=local_path,
            )
            self._fetched.add(local_path)
            return local_path

        local_folder = os.path.join(dest_dir, scene_id)
        os.makedirs(local_folder, exist_ok=True)
        objects = self.client.list_objects(
            Bucket=self.bucket, Prefix=f"{self.scene_prefix}{scene_id}/"
        )
        for content in objects.get("Contents", []):
            file_name = content.get("Key", "").split("/")[-1]
            if not file_name or (suffix is not None and not file_name.endswith(suffix)):
                continue
            self.client.download_file(
                Bucket=self.bucket,
                Key=content["Key"],
                Filename=os.path.join(local_folder, file_name),
            )
        self._fetched.add(local_folder)
        return local_folder

    def release_scene(self, path: str) -> None:
        # Only ever remove something this backend downloaded. A path we did
        # not create is somebody else's data. A scene may be a folder or,
        # for single-object sensors, one file.
        if path not in self._fetched:
            return
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
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
