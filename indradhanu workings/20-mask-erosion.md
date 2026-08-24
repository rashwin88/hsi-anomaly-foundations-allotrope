# 20 · Mask erosion: throwing away the edges

> **The one thing this part teaches:** pixels near a cloud edge or a scene
> boundary produce false alarms, so a border strip is deliberately discarded.

**Source:** `TokenMasking.erode_mask` in
[`token_masking.py`](../app/foundation_models/components/token_masking.py)

---

## The problem, concretely

Picture a pixel that is perfectly valid, but sits two pixels away from the edge
of a cloud.

Its token covers a 4x4 block. Some of that block is cloud — invalid, zeroed,
sitting at -1.22 in normalised space (part 09). So the token's value is built
partly from real ground and partly from "no data".

The model's prediction there will be poor. Not because anything unusual is on
the ground, but because the input was contaminated.

### What this looks like in practice

Every cloud outline, every water boundary, every ragged swath edge in the scene
lights up as a **bright ring** in the anomaly map.

Analysts have seen this. It is the single most common false-positive pattern in
this system, and it is entirely an artefact.

> **Analogy.** Photographs from cheap lenses are soft and distorted at the very
> corners. If you were measuring something precisely, you would crop the border
> rather than trust it. Erosion is that crop, applied around every hole as well
> as the outside edge.

---

## The fix

Shrink — "erode" — the valid region inward by a buffer, and exclude that buffer
from:

- the loss during training,
- the score during inference.

The pixels are still reconstructed. They just do not count.

---

## The implementation trick

Here is a neat piece of code. Eroding the *valid* region is the same as
**growing** the *invalid* region and then flipping.

And growing a binary region is just a **max-pool with stride 1**.

```python
padding = kernel_size // 2

invalid = 1.0 - mask                     # flip: invalid becomes 1

dilated_invalid = F.max_pool2d(invalid,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)

eroded_mask = 1.0 - dilated_invalid      # flip back
```

Why does max-pool grow a region? Because max-pool asks, for each position:
**"is there a 1 anywhere in my window?"** If yes, output 1.

So any pixel within reach of an invalid pixel becomes invalid too.

- `stride = 1` means the output is the same size as the input (no downsampling).
- `padding = kernel_size // 2` keeps the alignment centred.

**Erosion radius = `kernel_size // 2`.** Kernel 5 removes 2 pixels; kernel 15
removes 7.

---

## Worked example, 1-D, kernel_size = 5

Radius is `5 // 2 = 2`, so we expect 2 pixels removed from each edge of the
valid region.

```
index:      0  1  2  3  4  5  6  7  8  9 10 11
mask:       0  0  1  1  1  1  1  1  1  1  0  0
                  ^--------- valid region ---^
```

**Step 1 — flip:**

```
invalid:    1  1  0  0  0  0  0  0  0  0  1  1
```

**Step 2 — for each index, take the maximum over the window `[i-2, i+2]`**
(clipped at the array ends):

```
i= 0   window covers 0,1,2        values 1,1,0        max = 1
i= 1   window covers 0,1,2,3      values 1,1,0,0      max = 1
i= 2   window covers 0..4         values 1,1,0,0,0    max = 1
i= 3   window covers 1..5         values 1,0,0,0,0    max = 1
i= 4   window covers 2..6         values 0,0,0,0,0    max = 0
i= 5   window covers 3..7         all zeros           max = 0
i= 6   window covers 4..8         all zeros           max = 0
i= 7   window covers 5..9         all zeros           max = 0
i= 8   window covers 6..10        includes index 10   max = 1
i= 9   window covers 7..11        includes 10 and 11  max = 1
i=10   window covers 8..11        includes 10 and 11  max = 1
i=11   window covers 9..11        includes 10 and 11  max = 1
```

```
dilated:    1  1  1  1  0  0  0  0  1  1  1  1
```

**Step 3 — flip back:**

```
eroded:     0  0  0  0  1  1  1  1  0  0  0  0
                        ^---valid---^
```

### Check the result

```
before:  valid from index 2 to index 9      (8 pixels)
after:   valid from index 4 to index 7      (4 pixels)
```

Exactly 2 removed from each end, as the radius promised.

In two dimensions the same thing happens in both directions, so a cloud hole
grows a 2-pixel collar all the way around it.

---

## The two very different defaults

This is the part people find confusing, so here it is plainly:

| Where | Field | Default | What it does |
|---|---|---|---|
| **Training** | `HyperspectralSegFormerMAEConfig.erosion_kernel_size` | **1** | a 1x1 max-pool is the identity — **no erosion at all** |
| **Inference** | `InferenceConfig.erosion_kernel_size` | **15** | removes a 7-pixel collar |

Why the asymmetry? Because the two situations genuinely differ.

**During training**, patches are cut from the interiors of scenes and are mostly
fully valid. Applying a 7-pixel erosion to every 128-pixel patch would discard a
significant fraction of your supervision for very little benefit — a fully-valid
patch has no contaminated border to remove.

**During inference**, we run over the *whole* scene, including its ragged edges
and every cloud in it. Here a false ring around every cloud is far more costly
than a few missing pixels of coverage.

The field is exposed per-Action, so an analyst dealing with a messy cloudy scene
can raise it:

```python
erosion_ks_override = ovr.get("erosion_kernel_size")
if erosion_ks_override is not None:
    ic_kwargs["erosion_kernel_size"] = int(erosion_ks_override)
```

> **Note on `kernel_size = 1`:** `padding = 1 // 2 = 0`, and a max-pool over a
> 1x1 window returns the value itself. So the whole operation is a no-op. That
> is a tidy way to express "off" without a special case.

---

## Three different erosions — do not confuse them

There are three separate erosion operations in this pipeline. They stack, and
they are easy to mix up.

### 1. Validity erosion during training

Where: `compute_loss` in the trainer.
Kernel: `cfg.erosion_kernel_size`, default 1 (off).
Purpose: keep contaminated border pixels out of the loss.

```python
eroded_mask = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)
loss_mask   = pixel_pred_mask * eroded_mask
```

### 2. Validity erosion during inference

Where: `predict_full_scene` in the inferencer.
Kernel: `InferenceConfig.erosion_kernel_size`, default 15.
Purpose: keep contaminated border pixels out of the accumulated reconstruction.

```python
eroded_mask = TokenMasking.erode_mask(mask.unsqueeze(0), kernel_size=erosion_ks).squeeze(0)
```

### 3. Region-of-interest erosion during scoring

Where: `_anomaly_scoring_run.py`.
Kernel: `keep_mask_erosion_kernel_size`, default 1 (off).
Purpose: strip the rim of the *analyst's* chosen region — the output of
`scene_segmentation` or `cloud_mask`, not the sensor's validity.

Different implementation entirely — SciPy rather than torch:

```python
from scipy.ndimage import binary_erosion
eroded_keep_mask = binary_erosion(keep_mask.astype(bool), structure=structure)
```

### The summary

| # | Erodes | Where | Default |
|---|---|---|---|
| 1 | sensor validity | training loss | 1 (off) |
| 2 | sensor validity | inference accumulation | 15 |
| 3 | analyst's region of interest | scoring | 1 (off) |

**A pixel must survive every applicable erosion to contribute a score.**

---

## Common confusions

**"Does erosion remove pixels from the output image?"**
No. The image is the same size. Those pixels simply do not contribute to the
loss or the score, and at inference the accumulator gives them zero weight.

**"Is erosion the same as the validity mask?"**
No. Erosion *derives* a stricter mask from the validity mask. Validity says
"this is not a measurement"; erosion says "this is a measurement, but it is too
close to something that is not one".

**"Why is training's default off?"**
Training patches come from scene interiors and are mostly fully valid, so
eroding would cost supervision for little gain. The 40%-validity patch filter
already excludes the worst patches.

**"If I see rings around clouds, what do I change?"**
Raise the inference `erosion_kernel_size` (default 15), and/or set
`keep_mask_erosion_kernel_size` on the Action. Both must be odd numbers.

---

## Check yourself

1. Why do pixels near a cloud edge produce false anomalies?
2. Explain the dilate-then-invert trick in one sentence.
3. Work the 1-D example with `kernel_size = 3` on
   `mask = [0, 1, 1, 1, 1, 0]`. What comes out?
4. Why is training's erosion default 1 while inference's is 15?
5. Name the three erosions and what each one erodes.

<details>
<summary>Answers</summary>

1. Their token's receptive field overlaps invalid pixels, so the token is built
   partly from "no data" and the reconstruction there is unreliable — producing a
   large residual for a purely technical reason.
2. Eroding the valid region equals growing the invalid region (a stride-1
   max-pool) and then flipping the result back.
3. Radius `3//2 = 1`. `invalid = [1,0,0,0,0,1]`; dilated
   `= [1,1,0,0,1,1]`; eroded `= [0,0,1,1,0,0]`. One pixel removed from each end
   of the valid region.
4. Training patches come from scene interiors and are mostly fully valid, so
   erosion would waste supervision. Inference covers whole scenes with real cloud
   and swath edges, where a false ring costs more than a few missing pixels.
5. (1) sensor validity in the training loss; (2) sensor validity during
   inference accumulation; (3) the analyst's region of interest during scoring.

</details>

---

**Next:** rebuilding the full-resolution picture, in
[21-the-decoder.md](21-the-decoder.md)
