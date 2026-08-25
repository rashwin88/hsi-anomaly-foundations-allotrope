# S3 coupling in the patch-generation pipeline

**Recorded:** 2026-08-25, while building the segmentation sharder.
**Status:** deliberately unpaid. Reasons below; re-read them before deciding they still hold.

The patch-generation pipeline assumes S3 is the only place scenes and shards can live. It
works, and has produced every shard Indradhanu trained on. It became a problem the moment
sharding needed to run somewhere else — Colab with Google Drive mounted, where every scene
is an ordinary local path.

## The extent, counted

| debt | extent |
|---|---|
| `"allotrope-raw-data-india"` as a module-level constant | **8 files** |
| `boto3.client("s3", ...)` constructed inline | **8 call sites**, 6 with the region as a literal |
| Scene discovery + network I/O performed inside `__init__` | all **3** intermediate sharders |
| Near-identical sharders that have diverged by copy-paste | 205 / 218 / 244 lines |
| An ABC whose method names name their implementation | `s3_searcher`, `s3_downloader` |
| Tests covering any intermediate sharder | **0** |

The constants: `batch_visualization_thermal.py`, `shard_pipe_expression_builder.py`,
`final_patcher.py`, `hyperspectral_final_patcher.py`, and the three
`*_intermediate_patcher.py` files, plus `scripts/sync_shards_to_local.py`.

The ABC is the sharpest example. `app/abstract_classes/intermediate_sharder.py` exists to
define a storage-agnostic contract, and its docstring opens *"discover scenes in S3"*. The
abstraction names the thing it was supposed to abstract over.

## What it blocks

**Running sharding anywhere but AWS.** Constructing `EnmapIntermediateSharder` builds a
boto3 client at line 62 and performs a network listing at line 75 — before any work is
requested. A Colab run that never touches S3 still needs credentials, or it fails in the
constructor.

That is what forced `EnmapSegmentationSharder` to subclass the ABC rather than the concrete
EnMAP sharder: the parent constructor could not be called at all. The cost was duplicating
~15 lines of vendable-building.

Secondary: I/O in constructors is why none of these classes are unit-tested. You cannot
construct one without a network.

## Why not now

Three reasons, all still live at time of writing:

1. **`SceneStorage` has never run.** Migrating three working sharders onto an unproven
   interface means a mistake costs four edits instead of one. Prove it once, then
   propagate.
2. **It touches Indradhanu's data pipeline.** The segmentation work was deliberately scoped
   to leave the reconstruction path untouched — a subclass rather than a flag, for exactly
   this reason.
3. **No safety net.** Zero tests cover the intermediate sharders, and they cannot be
   imported on a machine without AWS and rasterio. A 650-line refactor would be verified by
   reading alone.

Reason 1 expires as soon as the segmentation sharder completes a real run. Reason 3 is
partly addressable independently — see below.

## The fix, sized

Roughly **12-15 chunks**, in dependency order:

1. One config module holding bucket + region — deletes 8 constants and 8 inline clients
2. Rename the ABC's methods to `list_scenes` / `fetch_scene`, updating 3 implementers
3. Move discovery out of the constructors into an explicit `prepare()` or a lazy property
4. Each sharder accepts a `SceneStorage` (`app/utils/patch_generation/scene_storage.py`)
5. Delete the per-file `boto3` imports

Steps 1 and 2 are mechanical. Step 3 is the one with real behaviour change, because callers
currently rely on the scene list existing immediately after construction.

**Cheap and available now, independent of the rest:** tests for the intermediate sharders,
using `LocalSceneStorage` against a synthetic scene folder. Step 3 becomes far safer with
those in place, and they need no AWS.
`tests/test_utils/test_patch_generation/test_hyperspectral_patcher.py` is the pattern —
synthetic inputs, no `large_files` marker, runs anywhere.

## Smaller items noted in passing

- `S3SceneStorage.shard_exists` catches `ClientError` broadly, so a permissions failure
  reads as "shard absent" and the work is redone. Errs toward doing work rather than
  skipping it, which is the safe direction, but it delays surfacing an auth problem.
- `EnmapSegmentationSharder` implements the ABC's `s3_searcher` / `s3_downloader` while
  never touching S3. Misleading, and fixed by step 2 above.
