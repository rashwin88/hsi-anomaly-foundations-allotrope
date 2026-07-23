# 7. The Training-Time Pipe Expression

This section closes the loop. The final shuffler has written
shards under `patches/{sensor}/{split}/final/w*_h*_s*/`. At training
time, the trainer needs to construct the exact URL that will stream
those shards into `wds.WebDataset`. That URL is built by
`shard_pipe_expression_builder` — the read-side counterpart of
`IntermediateSharder.build_prefix`.

## What the code does

[`shard_pipe_expression_builder`](../../app/utils/general_utils/shard_pipe_expression_builder.py)
at [shard_pipe_expression_builder.py:16](../../app/utils/general_utils/shard_pipe_expression_builder.py)
takes a `data_key` (e.g. `patches/hyperspectral/train/final/w64_h64_s32/`)
and:

1. Paginates S3 to enumerate the shard keys via
   `get_all_objects_paginated` at
   [shard_pipe_expression_builder.py:26-31](../../app/utils/general_utils/shard_pipe_expression_builder.py).
2. Strips numeric suffixes to find each shard's number at
   [shard_pipe_expression_builder.py:36-38](../../app/utils/general_utils/shard_pipe_expression_builder.py):

   ```python
   shard_numbers = [
       key.split("/")[-1].split(".")[0].split("_")[-1] for key in object_list
   ]
   ```

   This pulls `"00042"` out of `final_shard_00042.tar`.
3. Reconstructs the shard identifier (everything before the
   number) from the first object as a template at
   [shard_pipe_expression_builder.py:41-43](../../app/utils/general_utils/shard_pipe_expression_builder.py).
4. Returns the brace-expansion URL at
   [shard_pipe_expression_builder.py:46-51](../../app/utils/general_utils/shard_pipe_expression_builder.py):

   ```
   pipe: aws s3 cp s3://allotrope-raw-data-india/<prefix>final_shard_{00000..00042}.tar -
   ```

This is what the trainer passes to `wds.WebDataset(...)`.

### Call sites

The same function is used by:

- [`batch_visualization_thermal.py`](../../app/utils/general_utils/batch_visualization_thermal.py)
  for grid visualization of thermal patches loaded directly from S3
  shards.
- [`foundation_trainer.py`](../../app/abstract_classes/foundation_trainer.py)
  at [foundation_trainer.py:337](../../app/abstract_classes/foundation_trainer.py),
  which is where the training loop pulls its data from.

Both call sites pass a `data_key` that looks identical to what
`IntermediateSharder.build_prefix` produces — that symmetry is the
whole point.

### The differences vs `_compute_shard_ranges`

The shuffler has its own near-identical helper, `_compute_shard_ranges`
on `FinalPatchShuffler` at
[final_patcher.py:111-142](../../app/utils/patch_generation/final/final_patcher.py),
which does the same job for intermediate shards. The two functions
are *almost* the same; the writer-side `_compute_shard_ranges`
hard-codes that the prefix is two underscore-joined tokens
(`intermediate_shard`), while the reader-side
`shard_pipe_expression_builder` is more general:

```python
# shard_pipe_expression_builder.py:43
shard_identifier = "_".join(shard_elements[: len(shard_elements) - 1])

# final_patcher.py:135
shard_identifier = "_".join(shard_elements[:2])
```

The reader-side version handles `final_shard_00042.tar` and
`intermediate_shard_0123.tar` interchangeably; the writer-side
version assumes `intermediate_shard_*`. This is fine because the
writer only ever reads intermediate shards.

## Theory in plain language

### Why `pipe: aws s3 cp ... -`?

Why use the shell pipe rather than mounting the bucket or using the
S3 SDK from inside the loader?

- **Decoupling**. `wds.WebDataset` only needs a tar stream on
  stdin. The `aws` CLI is battle-tested, handles auth and retries,
  and runs in a subprocess so its memory does not pile up in the
  loader workers. Each `aws s3 cp` is its own process, with its own
  socket pool and credentials cache; if one stalls or dies, the
  others keep going.
- **Brace expansion**. The shell expands `{00000..00042}` into 43
  separate `aws s3 cp` invocations, one per shard. Combined with
  `num_workers > 1` on the `DataLoader`, multiple shards stream in
  parallel. WebDataset's internal worker logic ensures each shard
  is assigned to at most one worker per epoch, so there's no
  duplicate work.
- **Broken-pipe warnings are expected.** When a worker finishes
  draining the shard it needs (e.g. the batch is full and the
  iteration breaks), it closes the pipe; `aws s3 cp` then emits a
  broken-pipe warning to stderr. This is documented in the project
  memory under `feedback_visualization_rendering` and is not an
  error. If you grep your training logs for "broken pipe" you will
  see it once per finished shard.

### The shared-prefix invariant

The fact that the *writer*
(`IntermediateSharder.build_prefix`) and the *reader*
(`shard_pipe_expression_builder`) agree on the same
`patches/{sensor}/{split}/{stage}/w{w}_h{h}_s{s}/` convention is
what ties the whole pipeline together: if a trainer's config says
`(width=64, height=64, stride=32)` for hyperspectral, the URL is
computed deterministically and the right shards are pulled — no
manifest file, no out-of-band registry, just the prefix.

This means that:

- **There is no version mismatch failure mode.** You cannot point
  the trainer at a "stale" manifest because there is no manifest.
- **Adding a new geometry never breaks the old one.** A
  `(width=96, height=96, stride=48)` run lives under its own
  prefix, listable and inspectable independently.
- **The trainer can audit its own data.** Before training, the
  `data_key` is computable from config; you can list it, sample one
  shard, eyeball its contents.

### Brace expansion vs URL lists

`shard_pipe_expression_builder` uses brace expansion because the
single-sensor `final` prefix has a contiguous shard-number range
(`final_shard_{00000..00042}.tar`) and a single pipe URL is the
cleanest form.

By contrast, `HyperspectralFinalShuffler` builds a *list* of pipe
URLs in its `__init__`
(see [hyperspectral_final_patcher.py:84-92](../../app/utils/patch_generation/final/hyperspectral_final_patcher.py))
because the source shards come from two different sensor prefixes
and braces do not expand across prefixes. The trainer-side
`shard_pipe_expression_builder` does not need that complication
because the *final* prefix is unified under `hyperspectral/` — by
the time the trainer reads, the source-sensor distinction has
collapsed.

### Why not fsspec / s3fs?

A reasonable question: why not use `fsspec` or `s3fs` to mount S3
as a virtual filesystem and let WebDataset open shards as if they
were local? Two reasons:

1. **Process isolation.** `aws s3 cp` runs out-of-process, so its
   network buffers, retries, and memory are independent of the
   Python loader. `s3fs` runs in-process and competes with the
   loader for CPython resources.
2. **Maturity.** The `aws` CLI has been the canonical S3 client for
   over a decade; bug reports and retry strategies are
   well-understood. `s3fs` is fine but has its own failure modes
   (caching pitfalls, FUSE-vs-non-FUSE differences) that we did
   not want to introduce.

## Worked example: a hyperspectral final URL

Suppose the final shuffler has written 43 shards under
`patches/hyperspectral/train/final/w64_h64_s32/`. The trainer
calls:

```python
from app.utils.general_utils.shard_pipe_expression_builder import shard_pipe_expression_builder
url = shard_pipe_expression_builder(
    data_key="patches/hyperspectral/train/final/w64_h64_s32/"
)
```

`url` ends up as:

```
pipe: aws s3 cp s3://allotrope-raw-data-india/patches/hyperspectral/train/final/w64_h64_s32/final_shard_{00000..00042}.tar -
```

Passed to `wds.WebDataset(url)`, the shell expands the braces into
43 invocations of `aws s3 cp`, each piping one tar to stdin of a
loader worker. The worker decodes the tar member groups
(`pixels.npy`, `validity_cube.npy`, `meta.json`, `wavelengths.npy`)
into a dict, applies the `decode()` step, and yields it to the
training loop.

## Sequence diagram

```mermaid
sequenceDiagram
    participant Trainer as foundation_trainer.py
    participant Builder as shard_pipe_expression_builder
    participant S3 as S3 (final prefix)
    participant Shell as aws s3 cp (brace expanded)
    participant WDS as wds.WebDataset
    participant Loop as training loop

    Trainer->>Builder: shard_pipe_expression_builder(data_key)
    Builder->>S3: paginate list objects
    S3-->>Builder: list of shard keys
    Builder->>Builder: extract min/max numbers, build template
    Builder-->>Trainer: pipe URL with braces
    Trainer->>WDS: wds.WebDataset(url, ...)
    WDS->>Shell: spawn aws s3 cp per expanded shard
    Shell->>S3: GET shard tar
    S3-->>Shell: stream bytes
    Shell-->>WDS: stdin pipe
    WDS-->>Loop: dict per patch (pixels, validity, ...)
```
