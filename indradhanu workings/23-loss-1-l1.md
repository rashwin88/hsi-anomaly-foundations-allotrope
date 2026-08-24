# 23 · Loss, part 1: L1 and the loss mask

> **The one thing this part teaches:** L1 measures average absolute error in
> reflectance units, and it is computed only at pixels that were hidden, valid,
> and away from any border.

**Source:** `compute_loss` in
[`hyperspectral_segformer_mae_trainer.py`](../app/foundation_models/trainers/hyperspectral_segformer_mae_trainer.py)

---

## What a "loss" is, briefly

A loss is a single number saying how wrong the model currently is. Training
works by nudging every parameter in the direction that makes that number
smaller.

Two properties matter:

- **smaller must mean better**, always;
- it must be **differentiable**, so we can compute which direction to nudge.

---

## What L1 is

Mean absolute error. Take the difference, ignore its sign, average.

```
L1 = mean over graded pixels of [ mean over bands of |x_hat - x| ]
```

Note the order of operations, which the code follows exactly:

1. average across the 165 bands first, giving **one number per pixel**;
2. then average those numbers over the graded pixels.

> **Why "L1"?** From the L1 norm, the mathematical name for "sum of absolute
> values". Its sibling L2 is "square root of sum of squares"; squaring the L2
> distance gives you MSE. That is the whole naming story.

---

## Worked example, one pixel

Three bands instead of 165, for readability:

```
x     = [0.100, 0.200, 0.300]      what the satellite recorded
x_hat = [0.120, 0.180, 0.330]      what the model predicted
```

**Differences:**

```
0.120 - 0.100 = +0.020
0.180 - 0.200 = -0.020
0.330 - 0.300 = +0.030
```

**Absolute values** (drop the signs — an error of -0.02 is just as wrong as
+0.02):

```
0.020
0.020
0.030
```

**Mean across bands:**

```
(0.020 + 0.020 + 0.030) / 3
= 0.070 / 3
= 0.02333
```

That pixel contributes **0.02333** to the L1 term. With real data the same
arithmetic runs over 165 numbers instead of 3.

### Is 0.02333 good or bad?

Reflectance sits around 0.02 to 0.30, so an average error of 0.023 is
substantial — you would see it. A well-trained model reaches about **0.008**,
according to the training configuration's own baseline table. So this made-up
pixel is a poor reconstruction, roughly what you would see early in training.

---

## The loss mask: which pixels count

A pixel is graded **only if all three of these hold**:

1. it was a **prediction target** — its token was hidden from the encoder;
2. it is **valid** — a real measurement;
3. it is **interior** — not within the erosion buffer of an invalid region.

Conditions 2 and 3 are both captured by `eroded_mask` (part 20). So:

```python
pixel_pred_mask = self._pred_mask_to_pixel_mask(pred_mask, H, W)
eroded_mask     = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)
loss_mask       = pixel_pred_mask * eroded_mask          # (B, 1, H, W)
```

**Multiplying two 0/1 masks is a logical AND.** `1 * 1 = 1`; anything with a 0
gives 0.

### A worked mini-table

For four example pixels:

| pixel | hidden? | valid & interior? | `loss_mask` | graded? |
|---|---|---|---|---|
| A | 1 | 1 | `1 * 1 = 1` | **yes** |
| B | 1 | 0 | `1 * 0 = 0` | no — it is nodata or on a border |
| C | 0 | 1 | `0 * 1 = 0` | no — the model could see it |
| D | 0 | 0 | `0 * 0 = 0` | no |

Only A counts.

---

## From token grid to pixel grid

`pred_mask` is `(B, N)` — one flag per token on a 32x32 grid. The loss lives on
the 128x128 pixel grid. Somebody must blow it up 4x.

```python
token_grid = pred_mask.reshape(B, 1, H_tokens, W_tokens)
return F.interpolate(token_grid, size=(H, W), mode="nearest")
```

**`mode="nearest"` is essential.** Nearest-neighbour upsampling by 4 copies each
token's flag to exactly the 16 pixels that token covered:

```
token flag:  1

becomes:     1 1 1 1
             1 1 1 1
             1 1 1 1
             1 1 1 1
```

If you used bilinear here, values at block boundaries would come out as 0.5 or
0.75 — fractional. Then `loss_mask == 1` would silently reject them, and the
grading region would develop a strange lattice of gaps.

Nearest-neighbour keeps the mask strictly 0 or 1. Use it for masks, always.

---

## Putting it together

```python
per_pixel_l1      = (x_hat - pixels).abs()                    # (B, 165, H, W)
per_pixel_l1_mean = per_pixel_l1.mean(dim=1, keepdim=True)    # (B, 1, H, W)
valid_l1          = per_pixel_l1_mean[loss_mask == 1]         # 1-D, graded only
l1_loss           = valid_l1.mean()                           # a single number
```

Line by line:

1. absolute difference, everywhere, all bands;
2. average across bands (`dim=1`), giving one value per pixel;
3. **boolean indexing** — `[loss_mask == 1]` pulls out just the graded pixels
   into a flat 1-D list;
4. mean of that list.

Step 3 is a neat trick. Because it extracts only the graded pixels, the mean in
step 4 automatically has the right denominator. No dividing by a mask sum, no
risk of counting zeros as if they were real values.

(The SAM term in part 24 cannot do this, because it needs to keep its spatial
shape. It divides by an explicit count instead.)

---

## Why L1 rather than MSE

MSE (mean squared error) is the more common default. Here is the comparison:

| | L1 | MSE |
|---|---|---|
| formula | mean of `|error|` | mean of `error^2` |
| a 2x bigger error counts | 2x | **4x** |
| a 10x bigger error counts | 10x | **100x** |
| dominated by outliers? | no | **yes** |

### Why that matters here specifically

The training corpus is **real satellite imagery**. It has not been curated. It
contains:

- genuine anomalies (the very things we want the model to find surprising),
- sensor glitches and dead detectors,
- cloud edges that the masks did not quite catch.

Under MSE, a handful of pixels with huge errors would dominate every gradient
update. The model would spend its capacity learning to reconstruct **exactly the
things we want it to fail at**.

Under L1, each pixel's influence is proportional to its error, not to its square.
Outliers pull, but they do not take over. The model learns the bulk
distribution — normality — which is precisely what part 01 said we wanted.

> **Evidence, not theory.** This repo contains models that differ *only* in this
> choice: Antardhana uses MSE and Tirohita uses L1, same architecture otherwise.
> That comparison is why the SegFormer models use L1.

---

## The 40% rule: dropping bad patches entirely

Before any loss is computed, whole patches are discarded:

```python
MIN_VALID_PIXEL_FRACTION = 0.4

valid_fractions = mask.flatten(1).float().mean(dim=1)
keep = valid_fractions >= MIN_VALID_PIXEL_FRACTION
num_kept = keep.sum().item()
```

`mask.flatten(1).float().mean(dim=1)` gives, for each patch, the fraction of its
pixels that are valid — the same "average of 0/1 values is a proportion" trick
from part 17.

A patch that is less than 40% valid is dropped from the batch entirely.

**Why?** Such a patch is mostly nodata. Training on it teaches the model to
reconstruct holes, and its few real pixels are all near a border and therefore
contaminated anyway.

### `num_kept` matters more than you would think

The trainer counts **surviving samples**, not batches:

```python
total_loss += loss.item() * num_kept
valid_samples += num_kept
```

Two consequences:

- the epoch's sample budget (part 27) counts only patches that actually
  contributed;
- the epoch's average loss is weighted correctly — a batch where 3 patches
  survived does not count the same as one where 128 did.

And if a whole batch is filtered out, `compute_loss` returns a zero tensor with
`num_kept = 0`, which the training loop skips entirely:

```python
if num_kept == 0:
    continue  # entire batch filtered out, skip
```

---

## Gotcha: training and inference use different thresholds

Training discards patches below **40%** valid. Inference discards patches below
**10%** valid:

```python
# in predict_full_scene
MIN_VALID_FRACTION = 0.1
```

And the comment above it says:

> *"Matches training: patches with < 40% valid pixels were discarded."*

The comment says 40; the constant is 0.1. They do not match.

The looser inference threshold is defensible — at inference you want scores near
scene edges, and you would rather have an imperfect score than none. But if you
are ever debugging edge artefacts, know that these two numbers differ, and that
one comment is wrong about it.

(Same lesson as part 11's stale kernel comment: **verify against source**.)

---

## Common confusions

**"Why average bands first and then pixels? Isn't averaging an average wrong?"**
It would be, if the groups had different sizes. Here every pixel has exactly 165
bands, so a mean of per-pixel means equals the overall mean. It is done this way
so that a per-pixel value exists for the mask to select.

**"Does the loss include the pixels the model could see?"**
No. That is exactly what `pixel_pred_mask` excludes. Grading visible pixels
would reward copying.

**"Is L1 the same as the L1 anomaly score used later?"**
Same formula, different job. Here it is a training objective averaged to one
number; at scoring time (part 30) it is kept per pixel as a map.

---

## Check yourself

1. Compute L1 for `x = [0.10, 0.20]` and `x_hat = [0.13, 0.16]`.
2. List the three conditions for a pixel to be graded, and which mask supplies
   each.
3. Why must the token-to-pixel upsample be nearest-neighbour?
4. Give the robustness argument for L1 over MSE, referring to what is actually
   in the training data.
5. What is the 40% rule, and what does `num_kept` affect?

<details>
<summary>Answers</summary>

1. Differences +0.03 and -0.04; absolute values 0.03 and 0.04; mean
   `0.07 / 2 = 0.035`.
2. Hidden (from `pixel_pred_mask`), valid and interior (both from
   `eroded_mask`). The two are multiplied to give `loss_mask`.
3. Because bilinear would produce fractional values at block edges, and
   `loss_mask == 1` would silently reject them, punching holes in the graded
   region. Nearest keeps the mask strictly 0 or 1.
4. The corpus contains real anomalies, sensor glitches and missed cloud edges.
   MSE squares errors, so those few huge residuals would dominate the gradient
   and the model would learn to reconstruct exactly what it should find
   surprising.
5. Patches less than 40% valid are dropped from the batch before any loss is
   computed. `num_kept` weights the epoch's average loss and counts toward the
   epoch's sample budget, so filtered patches cost nothing.

</details>

---

**Next:** the second, more interesting error measurement, in
[24-loss-2-sam.md](24-loss-2-sam.md)
