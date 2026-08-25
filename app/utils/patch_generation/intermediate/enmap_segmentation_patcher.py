"""
Intermediate sharder for EnMAP **segmentation** training data.

Produces the same patches as `EnmapIntermediateSharder` plus the provider
quality layers as `label_*.npy` keys - the targets for the cloud/water
segmentation head. Writes under its own `enmap_seg` prefix, so Indradhanu's
reconstruction shards are untouched.

Three deliberate departures from the reconstruction sharder:

1. **Subclasses the ABC, not `EnmapIntermediateSharder`.** That class builds
   a boto3 client and performs a network listing inside `__init__`, so its
   constructor cannot run on Colab at all. See
   `docs/tech-debt/s3-coupling-in-sharding.md`.
2. **No I/O in `__init__`.** Scene discovery is lazy, so constructing this
   object is free and testable.
3. **Storage is injected.** Scenes may come from Drive or from S3; this class
   does not know which.
"""

import os
import random
from collections import defaultdict
from typing import Literal, Optional

import numpy as np

from app.abstract_classes.intermediate_sharder import IntermediateSharder
from app.models.dataset.vendables import (
    DEFAULT_COMMON_WAVELENGTH_GRID,
    BandFilterConfig,
)
from app.utils.files.enmap_scene_cover import read_scene_cover
from app.utils.patch_generation.scene_storage import SceneStorage


def scene_stratum(cover: dict[str, float]) -> str:
    """Which bucket a scene belongs to for the train/test split.

    Stratifying on cloud, then snow, because those are the scarce classes.
    A 212-scene screen on 2026-08-25 gave, of scenes with any cloud at all,
    only 37 - and none above 14%. Splitting those at random can leave a test
    set with no cloud in it, which is how an earlier experiment ended up
    reporting a macro-F1 that rested on three scenes out of twelve.

    Thresholds are deliberately low. `cloud_high` means "5% or more", which
    sounds trivial and is the top quartile of what the archive actually
    holds. Revisit once the cloudier scenes land - with a fuller range these
    boundaries should move up.
    """
    cloud = cover.get("cloudCover", 0.0)
    snow = cover.get("snowCover", 0.0)
    if cloud >= 5.0:
        return "cloud_high"
    if cloud > 0.0:
        return "cloud_low"
    if snow > 0.0:
        return "snow"
    return "clear"


class EnmapSegmentationSharder(IntermediateSharder):
    """EnMAP scenes -> patches carrying provider labels."""

    SENSOR = "enmap_seg"

    def __init__(
        self,
        storage: SceneStorage,
        work_dir: str,
        split: Literal["train", "test"] = "train",
        test_fraction: float = 0.2,
        seed: int = 42,
        width: int = 128,
        height: int = 128,
        stride: int = 64,
        patch_validity_threshold: float = 0.5,
        band_filter_config: Optional[BandFilterConfig] = None,
        max_scenes: Optional[int] = None,
    ):
        self.storage = storage
        self.work_dir = work_dir
        self.split = split
        self.test_fraction = test_fraction
        self.seed = seed
        self.width = width
        self.height = height
        self.stride = stride
        self.patch_validity_threshold = patch_validity_threshold
        # Both forced, not defaulted, because getting either wrong produces
        # shards that look fine and are unusable.
        #
        # quality_masks_to_apply=[]: with masks applied the builder zeroes
        #   validity at every cloud/shadow/haze pixel, so the labels are
        #   consumed before patching and cannot be recovered - and the patch
        #   filter then discards any patch more than half cloud, which is the
        #   training data.
        # common_wavelength_grid: the field defaults to None, which disables
        #   resampling and yields EnMAP's ~188 native bands. The segmentation
        #   head reuses Indradhanu's encoder, which takes exactly 165 channels
        #   on the common grid. A smoke test on 2026-08-25 produced 188-band
        #   shards before this was set.
        self.band_filter_config = (band_filter_config or BandFilterConfig()).model_copy(
            update={
                "quality_masks_to_apply": [],
                "common_wavelength_grid": DEFAULT_COMMON_WAVELENGTH_GRID,
            }
        )
        self.max_scenes = max_scenes

        # Populated on first access — see chunk 13's stratified split. Kept out
        # of __init__ so constructing this object touches neither disk nor net.
        self._scene_ids: Optional[list[str]] = None

        self.destination_prefix = self.build_prefix(
            sensor=self.SENSOR,
            split=split,
            stage="intermediate",
            width=width,
            height=height,
            stride=stride,
        )

    @property
    def scene_ids(self) -> list[str]:
        """Scene ids belonging to this split, computed once on first access.

        Costly under `S3SceneStorage`: reading a scene's cover percentages
        needs its METADATA.XML, and `fetch_scene` downloads the whole scene.
        Cheap under `LocalSceneStorage` (~3 ms/scene), which is the intended
        path. See docs/tech-debt/s3-coupling-in-sharding.md.
        """
        if self._scene_ids is None:
            self._scene_ids = self._split_scenes()
        return self._scene_ids

    def _split_scenes(self) -> list[str]:
        """Split each stratum independently, so scarce classes reach both sides."""
        candidates = self.s3_searcher()
        if self.max_scenes is not None:
            candidates = candidates[: self.max_scenes]

        strata: dict[str, list[str]] = defaultdict(list)
        for scene_id in candidates:
            try:
                cover = read_scene_cover(self.storage.fetch_scene(scene_id, self.work_dir))
            except FileNotFoundError:
                # No METADATA.XML means the builder cannot read it either.
                print(f"[enmap_seg] skipping {scene_id}: no METADATA.XML")
                continue
            strata[scene_stratum(cover)].append(scene_id)

        rng = random.Random(self.seed)
        chosen: list[str] = []
        for name in sorted(strata):
            # Sort before shuffling so filesystem order cannot change the split.
            members = sorted(strata[name])
            rng.shuffle(members)
            cut = int(len(members) * (1 - self.test_fraction))
            chosen.extend(members[:cut] if self.split == "train" else members[cut:])
        return sorted(chosen)

    def s3_searcher(self) -> list[str]:
        """Every available scene id.

        Named for the ABC's contract, which predates the storage seam. This
        reaches S3 only if the injected backend does; under
        `LocalSceneStorage` it is a filesystem glob and no AWS call is made.
        """
        return self.storage.list_scenes()

    def s3_downloader(self, key: str) -> dict:
        """Make one scene readable locally and return its manifest.

        `key` is a bare scene id. The manifest shape - `{"scene_folder": ...}`
        - matches the reconstruction sharder, because `patch_generator`
        reads it the same way.

        NOTE: under `LocalSceneStorage` the returned path is the user's own
        scene folder, not a temporary copy. Callers must not delete it.
        """
        return {"scene_folder": self.storage.fetch_scene(key, self.work_dir)}

    def patch_generator(self, manifest: dict):
        """Build the vendable and cut patches, labels attached.

        Mirrors `EnmapIntermediateSharder.patch_generator` — deliberate
        duplication, since that class's constructor cannot be called here.
        The differences are `include_labels=True` and the label-preserving
        band filter set in __init__.

        `sensor="enmap"` stays literal: the data is EnMAP. Only the storage
        prefix is `enmap_seg`.
        """
        from app.models.file_processing.sources import FileSourceConfig
        from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
        from app.utils.patch_generation.generate_patch_plan import (
            PatchPlanGenerator,
            PatchRequest,
        )
        from app.utils.patch_generation.hyperspectral_patcher import (
            patch_hyperspectral_vendable,
        )

        builder = EnmapDatasetBuilder(
            file_source_configuration=FileSourceConfig(
                source_path=manifest.get("scene_folder")
            )
        )
        vendable = builder.vend_dataset(band_filter_config=self.band_filter_config)
        plan = PatchPlanGenerator().generate_patching_plan(
            request=PatchRequest(
                input_cube=vendable.normalized_hyperspectral_cube.shape,
                height=self.height,
                width=self.width,
                stride=self.stride,
            )
        )
        return patch_hyperspectral_vendable(
            vendable=vendable,
            patching_plan=plan,
            scene_id=builder.stac_item.id,
            sensor="enmap",
            include_labels=True,
            # Halves the dominant term (10.8 -> 5.4 MB/patch at 165x128x128).
            # Reflectance is ~0-1, where float16 errs by ~1e-5 — far below
            # sensor noise. The trainer must cast to float32.
            pixel_dtype=np.float16,
        )

    def shard_name(self, scene_id: str) -> str:
        """One shard per scene, named after it — that is what makes resume work."""
        return f"{self.SENSOR}_{scene_id}.tar"

    def sharder(self, scenes: int = None) -> None:
        """Write one shard per scene, skipping any already published.

        A tar per scene rather than a rolling ShardWriter: a killed Colab
        session then costs the scene in flight, not the whole run. Shard
        sizes come out uneven, which the final shuffle stage evens out.

        Scene cleanup goes through `storage.release_scene`, never `rmtree`.
        Under a local backend the scene folder is the user's own data — the
        reconstruction sharder deletes it unconditionally, which would
        destroy source scenes here.
        """
        import webdataset as wds  # heavy; keep this module importable without it

        os.makedirs(self.work_dir, exist_ok=True)
        todo = self.scene_ids[:scenes] if scenes else self.scene_ids

        for scene_id in todo:
            name = self.shard_name(scene_id)
            if self.storage.shard_exists(name):
                print(f"[enmap_seg] skip {scene_id} (shard exists)")
                continue

            local_tar = os.path.join(self.work_dir, name)
            manifest = self.s3_downloader(scene_id)
            try:
                kept = 0
                # Hand TarWriter a stream, not a path: it routes paths through
                # its URL opener, which reads a Windows "C:\..." drive letter
                # as a scheme and fails with "no gopen handler defined".
                with open(local_tar, "wb") as raw, wds.TarWriter(raw) as sink:
                    for patch in self.patch_generator(manifest):
                        validity = patch["validity_cube.npy"][0]
                        if validity.mean() > self.patch_validity_threshold:
                            sink.write(patch)
                            kept += 1
                self.storage.publish_shard(local_tar)
                print(f"[enmap_seg] {scene_id}: {kept} patches")
            except Exception as err:  # one bad scene must not end the run
                print(f"[enmap_seg] FAILED {scene_id}: {err}")
            finally:
                self.storage.release_scene(manifest["scene_folder"])
                if os.path.exists(local_tar):
                    os.remove(local_tar)

    @property
    def source_folder(self) -> str:
        return self.work_dir

    @property
    def destination_folder(self) -> str:
        return self.work_dir
