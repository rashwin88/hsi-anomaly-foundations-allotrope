# 27 · Training data: shards, patches, epochs

> **The one thing this part teaches:** how a satellite scene becomes millions of
> 128x128 training patches, and what the word "epoch" means in this project
> (it is not what you think).

---

## The pipeline, end to end

```
raw satellite file
      |  DatasetBuilder                    app/utils/dataset_builder/
      v
  vendable  (165, H, W)                    common grid, part 03
      |  PatchPlanGenerator                decides WHERE to cut
      v
  a list of coordinates
      |  hyperspectral_patcher             does the cutting
      v
  patches as dictionaries
      |  packed into .tar files
      v
  shards on S3
      |  DataLoader
      v
  batches of tensors -> the model
```

Each stage has one job. Below, each in turn.

---

## Deciding where to cut

**Source:**
[`app/utils/patch_generation/generate_patch_plan.py`](../app/utils/patch_generation/generate_patch_plan.py)

This module produces **coordinates only** — no pixels are touched. Given a cube
size, a patch size and a stride, it returns the top-left corner of every patch.

Keeping the geometry separate from the cutting means you can test the geometry
without loading a gigabyte of imagery, and you can replay the same plan over any
cube of matching shape.

### Stride and overlap

**Stride** is how far you move between patches.

```
stride = patch size   ->   patches touch, no overlap
stride = size / 2     ->   50% overlap
stride = size / 4     ->   75% overlap
```

The training convention throughout this project is `stride = size // 2`. With
128-pixel patches, that is a stride of 64.

Why overlap at all? Because a pixel near the edge of a patch has less context
around it than one in the middle, and reconstructs worse. With 50% overlap,
every interior pixel appears in several patches at several offsets, so the model
sees it centred sometimes and at the edge other times.

### The edge-snapping rule

What happens when the last step would run off the edge? The final patch is
**snapped back** so that it ends flush with the boundary:

```python
if row + request.height > cube_height:
    row_coords.append(cube_height - request.height)
    break
```

Full coverage, no partial patches, no padding. The cost is that the last patch
overlaps its neighbour more than the stride implies.

### Worked example

A dimension of 300 pixels, `size = 128`, `stride = 64`:

```
start at 0        0 + 128 = 128  <= 300     ok, keep 0
step to 64       64 + 128 = 192  <= 300     ok, keep 64
step to 128     128 + 128 = 256  <= 300     ok, keep 128
step to 192     192 + 128 = 320  >  300     too far!
                snap back to 300 - 128 = 172, keep 172, then stop

coords = [0, 64, 128, 172]        4 patches
```

Note the gap from 128 to 172 is only 44, not 64. That last patch overlaps its
neighbour more.

**The lesson:** the patch count is **not** `ceil(extent / stride)`. Do not
assume it is; the module's own docstring warns about this.

> **The same generator runs at inference**, tiling a whole scene the same way.
> That is why it lives in `app/utils/` and not in a training-only module.

---

## Doing the cutting

**Source:**
[`app/utils/patch_generation/hyperspectral_patcher.py`](../app/utils/patch_generation/hyperspectral_patcher.py)

For each coordinate, it slices out a patch and emits a dictionary:

```python
{
    "__key__": f"{scene_id}#row_coord:{r}#col_coord:{c}",
    "meta.json": {
        "scene_id": ..., "row_coords": r, "col_coords": c,
        "patch_height": 128, "patch_width": 128, "patch_stride": 64,
        "sensor": "prisma", "spectral_family_order": [...], "band_count": 165,
    },
    "pixels.npy":        cube[:, r:r+h, c:c+w].astype(np.float32),   # (165,128,128)
    "validity_cube.npy": validity[:, r:r+h, c:c+w].astype(np.int8),
    "wavelengths.npy":   wavelengths,                                 # (165,)
}
```

Those key names are a **contract**. The trainer reads them literally:

```python
pixels   = batch["pixels.npy"].to(self.device)          # (B, 165, H, W)
validity = batch["validity_cube.npy"].to(self.device)
```

Rename `pixels.npy` in the patcher and every existing shard becomes unreadable.

Notice the types: `float32` for the values, `int8` for the mask. The mask only
ever holds 0 or 1, so a single byte per entry is plenty, and it is four times
smaller than float32.

The patcher also guarantees ascending wavelength order, sorting as a fallback if
a vendable somehow arrives unresampled.

---

## Shards on S3

Individual patch files would be a disaster — millions of tiny objects, each
needing its own network request. Instead, patches are packed into `.tar` archives
called **shards**, each perhaps a gigabyte, streamed sequentially.

(This is the `webdataset` convention, which is why the keys look like filenames
inside a tar.)

The storage location is templated:

```json
"shard_key_template": "patches/{provider}/{split}/{stage}/w{size}_h{size}_s{stride}/"
```

Filled in for Indradhanu's configuration:

```
patches/hyperspectral/train/final/w128_h128_s64/
patches/hyperspectral/test/final/w128_h128_s64/
```

Bucket `allotrope-raw-data-india`, region `ap-south-1`.

### Look at that path again

The provider is `"hyperspectral"` — **not** `"prisma"` or `"enmap"`.

PRISMA and EnMAP patches sit in the **same shards**, interleaved. Thanks to the
common grid (part 03), the model genuinely cannot tell them apart at the tensor
level. This is the payoff for all that resampling work, made concrete in a
directory name.

### The dangerous default

The config schema carries this comment on the field:

```python
# Important to set this properly The default is landsat which is dangerous.
```

The default template points at thermal data. Forget to override it and you will
happily start a training run that feeds single-band Landsat patches into a
165-band model — and the failure will be a confusing shape error deep inside the
first forward pass, or worse, silently wrong data.

---

## Hot storage

Streaming every batch from S3 across the internet is slow and costs money in
egress charges. So shards are copied to the training machine's local disk once
and kept there:

```json
"hot_storage": {
  "enabled": true,
  "local_cache_dir": "/tmp/hot_shards_hsi/",
  "train_shards_per_size": 500,
  "test_shards_per_size": 100,
  "skip_sync_if_exists": true,
  "max_parallel_downloads": 20
}
```

**There is no rotation.** Those 500 shards are the training set for the entire
run. Variety across epochs comes from `shardshuffle` (30 in this config), which
reorders shards each pass so the model does not see them in the same sequence.

To get genuinely fresh data you must delete the cache directory or set
`skip_sync_if_exists: false`. The docstring says so explicitly.

> **Trade-off to be aware of:** 500 shards is a large but finite sample. Over 200
> epochs the model sees each shard 200 times. `shardshuffle` changes the order
> but not the contents.

---

## What an "epoch" means here

In most tutorials, one epoch = one complete pass over the training set.

**Not here.** The corpus is far too large to pass over even once in reasonable
time. Instead an epoch is a **fixed sample budget**:

```json
"train_samples_per_epoch": { "128": 40000 },
"test_samples_per_epoch":  { "128":  5000 }
```

Read as: *"consume 40,000 training patches of size 128, then 5,000 validation
patches, then call that an epoch."*

An epoch is therefore a **unit of scheduling**, not a unit of coverage. It is
the interval at which:

- the learning-rate scheduler steps,
- validation runs,
- checkpoints may be saved,
- the SAM ramp advances (part 25).

### Why the dictionary keyed by size?

Because the thermal models train on several patch sizes simultaneously — 64,
128, 256 and 512 — with a separate budget for each. Indradhanu uses only 128, so
its dictionary has one entry. The machinery is shared.

### Only surviving samples count

```python
for batch in loader:
    if valid_samples >= sample_cap:
        break
    loss, num_kept = self.compute_loss(batch, self.model)
    if num_kept == 0:
        continue
    ...
    valid_samples += num_kept
```

Patches thrown out by the 40% validity filter (part 23) do **not** consume
budget. So "40,000 samples" means 40,000 patches that actually contributed
gradients — a consistent amount of learning per epoch, regardless of how much
nodata happened to be in the shards.

---

## The scale of the current run

```
batch_size    128
num_workers   40
samples/epoch 40,000
epochs        200
```

```
40,000 x 200 = 8,000,000 patch presentations
```

Each patch is a `(165, 128, 128)` float32 array:

```
165 * 128 * 128 * 4 bytes = 10,813,440 bytes = about 10.8 MB
```

(That figure matches `torchinfo`'s "Input size (MB): 10.81" for a single patch.)

So a batch of 128 patches is about **1.4 GB** of input data alone, before any
activations.

This is why `num_workers: 40` and `max_parallel_downloads: 20` exist. **Feeding
this model is an I/O problem before it is a GPU problem.** If the data pipeline
stalls, an expensive GPU sits idle.

---

## Common confusions

**"Does the model see every patch of every scene?"**
No. It sees 40,000 randomly-ordered patches per epoch, drawn from whichever
shards are cached. Coverage is statistical, not exhaustive.

**"Are train and test patches from different scenes?"**
They come from different shard prefixes (`train/` and `test/`), built by the
dataset builder. That separation is what makes validation meaningful.

**"Why store wavelengths in every patch when they are always identical?"**
Robustness and self-description. A shard is readable without any external
context, and a mis-resampled patch is detectable.

**"Can I train on 256-pixel patches instead?"**
Yes — add a `256` entry to both sample dictionaries and make sure shards exist
at that size. The path template includes the size, so they are separate
directories.

---

## Check yourself

1. What does `stride = size // 2` give you, and why is it worth the extra
   patches?
2. For a 500-pixel dimension with `size = 128, stride = 64`, list the
   coordinates.
3. Why do PRISMA and EnMAP patches share a shard directory?
4. Define "epoch" as this codebase means it.
5. Why does `num_workers` need to be as high as 40?

<details>
<summary>Answers</summary>

1. 50% overlap. Every interior pixel appears in several patches at different
   offsets — sometimes centred, sometimes at an edge — which averages away
   position-dependent reconstruction quality.
2. `0, 64, 128, 192, 256, 320` (`320 + 128 = 448 <= 500`), then 384 would give
   `384 + 128 = 512 > 500`, so snap to `500 - 128 = 372` and stop. Coordinates:
   `[0, 64, 128, 192, 256, 320, 372]` — 7 patches.
3. Because both are resampled onto the same 165-band common grid, so their
   patches are indistinguishable at the tensor level. One model trains on both.
4. A fixed sample budget — 40,000 training patches and 5,000 validation patches
   — not a full pass over the data. It is the interval for LR stepping,
   validation, checkpointing and the SAM ramp.
5. Each patch is about 10.8 MB, so a batch of 128 is roughly 1.4 GB. Decoding
   and assembling that fast enough to keep a GPU busy needs many parallel worker
   processes.

</details>

---

**Next:** the loop that consumes all this, in
[28-training-loop.md](28-training-loop.md)
