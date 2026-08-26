# Segmentation sharding

**Files:** `app/utils/patch_generation/scene_storage.py`,
`app/utils/patch_generation/intermediate/enmap_segmentation_patcher.py`,
`app/utils/files/enmap_scene_cover.py`, `scripts/generate_segmentation_patches.py`
**Tests:** `tests/test_utils/test_patch_generation/test_scene_storage.py`,
`tests/test_utils/test_patch_generation/test_hyperspectral_patcher.py`

## Purpose

Produce EnMAP training shards carrying the provider's quality layers as labels, for a
supervised cloud/water segmentation head. The existing hyperspectral sharder cannot do
this, and the reasons are structural rather than a missing flag.

## The three things that made a separate lane necessary

**1. The labels are consumed before patching.** `EnmapDatasetBuilder` uses the quality
masks to zero validity — `overall_validity[:, composite_quality_mask] = 0`
(`enmap_dataset_builder.py:273`). After that a shard cannot distinguish cloud from
nodata from a dead band. Fixed by forcing `quality_masks_to_apply=[]`.

**2. The validity filter then deletes the training set.** With masks applied, cloud pixels
are invalid, and `enmap_intermediate_patcher.py:205-207` drops any patch below
`patch_validity_threshold` (0.5). **Any patch more than half cloud is silently discarded.**
Correct for reconstruction; fatal for a cloud detector. Fixing (1) fixes this too.

**3. The parent's constructor cannot be called.** It builds a boto3 client and performs a
network listing inside `__init__` (`enmap_intermediate_patcher.py:62,75`), so it cannot run
on Colab at all. `EnmapSegmentationSharder` therefore subclasses the **ABC**, not the
concrete EnMAP sharder, and duplicates ~20 lines of vendable-building. See
`docs/tech-debt/s3-coupling-in-sharding.md`.

## Interfaces

```python
class SceneStorage(Protocol):
    def list_scenes(self) -> list[str]: ...
    def fetch_scene(self, scene_id: str, dest_dir: str) -> str: ...
    def release_scene(self, path: str) -> None: ...
    def publish_shard(self, local_path: str) -> None: ...
    def shard_exists(self, name: str) -> bool: ...

# LocalSceneStorage(scene_root, shard_dir=None, pattern="ENMAP01*")
# S3SceneStorage(bucket, scene_prefix, shard_prefix=None, region_name=...)

def scene_stratum(cover: dict[str, float]) -> str          # cloud_high|cloud_low|snow|clear
def read_scene_cover(scene_folder, head_bytes=65536) -> dict[str, float]
```

## Data flow

```
storage.list_scenes()  ->  cap to max_scenes
   -> read_scene_cover() per scene   (64 KB head read, ~3 ms on Drive)
   -> scene_stratum()                -> strata
   -> per-stratum shuffle(seed) and cut at 1 - test_fraction
   -> scene_ids for this split                              [cached, lazy]

per scene:  shard_exists? -> skip
            fetch_scene -> build vendable -> plan -> patches (include_labels=True)
            -> filter on validity.mean() > threshold
            -> one .tar -> publish_shard -> release_scene
```

Output: `patches/enmap_seg/{split}/intermediate/w128_h128_s64/`.

## Invariants

- **`quality_masks_to_apply` is `[]`.** Forced with `model_copy(update=...)`, overriding
  whatever the caller passed. Honouring a caller's value would silently destroy the labels
  and the run would still look successful.
- **Labels are stored raw.** Cirrus stays 0-3; class codes stay 0/1/2/3. Thresholding is
  the trainer's decision and baking it in costs a re-shard to undo.
- **The sharder never deletes a scene folder.** Cleanup goes through
  `storage.release_scene`, which is a no-op locally. See failure modes.
- **Shard names are scene-derived** (`enmap_seg_<scene_id>.tar`). This is what makes resume
  work; a rolling shard counter would not.
- **Scenes are sorted before shuffling**, so the seed reproduces a split across machines
  rather than inheriting filesystem enumeration order.
- **`enmap_seg` is a storage prefix, not a sensor.** `meta.json` still records
  `sensor="enmap"`, because the data is EnMAP.

## Failure modes

**The one that would destroy data.** The reconstruction sharder ends each scene with
`shutil.rmtree(scene_folder, ignore_errors=True)`
(`enmap_intermediate_patcher.py:212-214`). Under `LocalSceneStorage` that path is the
user's own scene folder on Drive, not a temp download. `release_scene` exists so the
backend decides: local does nothing, S3 removes only paths recorded in `_fetched` at
download time. `ignore_errors=True` means a wrong target fails silently, so guarding the
input is the only defence. `test_release_does_not_delete_the_source` is mutation-checked —
inject the rmtree and it fails.

**A scene without METADATA.XML** is skipped with a printed reason during splitting; the
builder could not read it either.

**A scene that raises during patching** is caught, reported by name, and the run continues.
The partial tar is removed in `finally`, so a resume never mistakes it for finished work —
but it also means a failed scene leaves nothing to inspect.

**No publish destination configured** makes `shard_exists` return `False`, so work is
redone rather than skipped. Deliberate: the opposite would let a misconfigured run complete
in seconds having produced nothing and look like success.

**`--max-scenes-per-split` truncates after the split**, so it can leave splits lopsided. It
is a smoke-test flag. Use `--max-scenes` to build a smaller real dataset — that caps before
splitting and preserves the strata.

## Decisions

**Stratified split.** Cloud appeared in 37 of 212 scenes screened on 2026-08-25, none above
14%. A random split can strand nearly all of it on one side — which happened in an earlier
cloud-mask experiment, where nine of twelve validation scenes held no cloud and the
headline macro-F1 rested on the other three. Thresholds (`cloud_high` = 5%) are low because
5% is roughly the top quartile of the current archive, and should rise once cloudier scenes
land.

**One shard per scene, not a rolling `ShardWriter`.** Sharding runs on Colab, where
sessions are killed at 12-24 hours. Per-scene shards make the run resumable at the cost of
uneven shard sizes and scene-ordered output, both of which the final stage fixes —
see below. (An earlier version of this file said the *existing* `FinalShuffler` would
even them out. It would not: it reads over S3 and there is no S3 in this path.)

**`TarWriter` is handed a stream, not a path.** It routes paths through its URL opener,
which reads a Windows `C:\...` drive letter as a scheme and fails with `no gopen handler
defined`. Passing an open file object avoids `gopen` entirely and works on both platforms.

**Heavy imports are inside `patch_generator` and `sharder`.** The module imports with no
rasterio, torch or webdataset present, which is what makes it testable on a machine that
cannot import the reconstruction sharder at all.

**Rejected: a `segmentation_mode` flag on `EnmapIntermediateSharder`.** It would put
Indradhanu's data-production path inside the blast radius of every change here. A separate
class costs duplication and buys the guarantee that the reconstruction lane is untouched.

## Verified end to end, 2026-08-25

One real EnMAP scene through the full path on the Windows dev box: 5 patches, 165 bands
at 460-2450 nm matching `DEFAULT_COMMON_WAVELENGTH_GRID`, all six `label_*` keys present,
`label_classes` showing 63.1% land / 0.1% water / 36.7% off-swath for an edge patch.
Source scene folders intact afterwards.

It found two real bugs that no amount of reading would have:

- **`stac_items.py` split paths on `"/"` only**, so on Windows an EnMAP *folder* was
  rejected as an unsupported file type. Fixed to `os.path.basename`; unchanged on Linux.
- **The common wavelength grid was never set.** `BandFilterConfig.common_wavelength_grid`
  defaults to `None`, which disables resampling — the reconstruction script passes it
  explicitly and this sharder did not. Shards came out at **188 native bands instead of
  165**, and nothing would have complained until the first forward pass hit
  `Conv2d(165, 32)`. Now forced alongside `quality_masks_to_apply`.

## The final stage — `LocalFinalShuffler`

**File:** `app/utils/patch_generation/final/local_final_shuffler.py`
**Tests:** `tests/test_utils/test_patch_generation/test_local_final_shuffler.py`

One shard per scene is right for resume and wrong for training: each holds ~270
spatially sequential tiles of one scene, and the trainers shuffle only at *shard*
level (no `.shuffle(buffer)` anywhere in `foundation_trainer.py`). A batch would be
drawn from very few scenes.

```
sorted *.tar  -> shard order shuffled (seed)
   -> interleave group_size shards round-robin
      -> window shuffle of shuffle_size samples
         -> roll into shard_size_bytes tars
```

**`group_size` is the mixing parameter, not a speed one.** A scene shard holds ~270
patches, so a 200-sample buffer cannot mix across one on its own — the interleave is
what puts several scenes in flight. This was originally a DataLoader `worker_count`,
which meant mixing silently collapsed if someone set workers to 0 while debugging.
Making it an explicit interleave width removes that trap.

**Reads with plain `tarfile`, not webdataset.** webdataset routes every shard name
through its URL opener, which has no handler for a bare path and reads a Windows
`C:\...` drive letter as a scheme. Three workarounds failed — a stream instead of a
path (works for `TarWriter`, not for `ShardWriter`, which builds its own), and
`file://` URIs (work through `gopen` directly, fail inside the pipeline). For a
local-filesystem job the URL layer buys nothing. Reading tars directly also drops the
torch dependency, which matters on a CPU Colab session, and — the deciding factor —
makes the stage runnable on the dev machine, so its tests are real.

**One complete pass, `resampled=False`.** `FinalShuffler` samples shards *with
replacement* to reach a target patch count, which is why the hyperspectral trainer
suppresses a `"duplicate file name in tar"` warning. For a fixed dataset every patch
should appear exactly once; the tests assert both no loss and no duplication.

**Records move as opaque bytes.** No decode/re-encode round trip. A test asserts
`pixels.npy` is byte-identical before and after.

**Inputs are never touched.** Output must be verifiable before ~89 GB of input is
deleted.

### Failure modes

Reading is single-threaded. On a mounted Drive that may be slow for a large corpus;
the fix is threads around `_records`, not a return to the URL layer.

`group_size` set to 1 disables cross-scene mixing while still producing plausible
output. `test_output_is_actually_mixed` is mutation-checked against exactly this.

## Known cost

Under `S3SceneStorage`, stratification needs each scene's METADATA.XML, and `fetch_scene`
downloads the whole scene — ~64 GB to split 212 scenes. Cheap on a local/Drive mount
(~3 ms/scene), which is the intended path. A metadata-only fetch method is the fix; noted
in the tech-debt entry.
