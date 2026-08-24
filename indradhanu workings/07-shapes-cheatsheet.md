# 07 · Notation and the complete shape cheat sheet

> **The one thing this part teaches:** how to read a shape like `(B, N, C)` and
> know instantly what each number means.

Print this part. Every later part refers back to it. You are not expected to
memorise the tables — you are expected to know they exist and to look things up.

---

## First: what is a "tensor"?

A tensor is just a rectangular block of numbers. That is all.

- A single number is a 0-D tensor.
- A list of numbers is a 1-D tensor. Shape `(5,)`.
- A grid of numbers is a 2-D tensor. Shape `(3, 4)` = 3 rows, 4 columns.
- A stack of grids is a 3-D tensor. Shape `(165, 128, 128)`.
- A batch of stacks is a 4-D tensor. Shape `(8, 165, 128, 128)`.

When you read `(B, C, H, W)`, read it left to right as nested containers:

> "B items, each containing C layers, each of which is H rows by W columns."

The **last** dimension changes fastest in memory. That will matter in exactly
one place (the conversions below).

---

## The symbols

| Symbol | Stands for | Typical value here |
|---|---|---|
| `B` | **B**atch size — how many patches processed at once | 128 training, 8 inference |
| `C` | **C**hannels — input spectral bands | **165** |
| `D` | **D**imension after compression | **32** (was 24 in v0.1.0) |
| `H`, `W` | **H**eight and **W**idth in pixels | 128 and 128 |
| `N` | **N**umber of tokens at a stage | 1024 at stage 1 |
| `C_i` | channel width at stage `i` | 32, 64, 160, 256 |
| `R` | **R**eduction ratio for attention | 8, 4, 2, 1 |
| `E` | **E**xpansion ratio inside the feed-forward net | 4 |
| `x` | the observed cube | `(B, 165, H, W)` |
| `x_hat` | the model's reconstruction | `(B, 165, H, W)` |

> **Why `B`?** Neural networks process many examples simultaneously for speed —
> a GPU is far more efficient on 128 patches at once than on one patch 128
> times. That group is a "batch". Almost every tensor in the model has `B` as
> its first dimension, and almost every operation treats the batch items
> independently. When reading code, you can usually **mentally ignore `B`** and
> think about a single patch.

---

## The two layouts, and converting between them

Tensors in this model constantly flip between two arrangements of the same data.

| Layout | Shape | Used by |
|---|---|---|
| **spatial** | `(B, C, H, W)` | convolutions, PixelShuffle, the loss |
| **sequence** | `(B, N, C)` | attention, LayerNorm, linear layers |

Spatial says "a picture with C layers". Sequence says "a list of N tokens, each
described by C numbers". Same numbers, different arrangement.

### The conversion, line by line

```python
# spatial -> sequence
x = x.flatten(2).transpose(1, 2)
```

Step 1, `flatten(2)`: "flatten everything from dimension 2 onwards into one
dimension."

```
(B, C, H, W)  ->  (B, C, H*W)  =  (B, C, N)
```

The two spatial axes get squashed into a single list of positions.

Step 2, `transpose(1, 2)`: "swap dimensions 1 and 2."

```
(B, C, N)  ->  (B, N, C)
```

And back the other way:

```python
# sequence -> spatial
x = x.transpose(1, 2).reshape(B, C, H, W)
```

### Why the transpose is genuinely needed

A common beginner question: why not just `reshape` in one go?

Because `reshape` preserves the order of the numbers in memory, and the two
layouts want a **different order**. Memory after a convolution is laid out as
`[B][C][H][W]` — all of channel 0's pixels, then all of channel 1's pixels.
Attention wants `[B][N][C]` — token 0's 32 numbers, then token 1's 32 numbers.

Those are genuinely different orderings, so an axis swap is unavoidable.
`reshape` alone would silently scramble the data.

---

## The complete forward-pass shape table

For one training patch, with `C = 165`, `D = 32`, `H = W = 128`.

| Step | Module | Output shape | Notes |
|---|---|---|---|
| input | — | `(B, 165, 128, 128)` | reflectance |
| mask | `x * mask` | `(B, 165, 128, 128)` | invalid pixels become 0.0 |
| normalise | `PixelNormalize` | `(B, 165, 128, 128)` | z-score, per band |
| compress | `SpectralCompressor` | `(B, 32, 128, 128)` | 1x1 conv + BatchNorm |
| **S1** patch embed | `OverlapPatchEmbedding(k=4,s=4,p=0)` | `(B, 1024, 32)` | grid 32x32 |
| S1 hide tokens | `remove_tokens` | `(B, ~358, 32)` | at mask_ratio 0.65 |
| S1 blocks x2 | `SegFormerBlock` | `(B, ~358, 32)` | `R=8` |
| S1 restore | `restore_tokens` | `(B, 1024, 32)` | zeros in the gaps |
| S1 to 2-D | reshape | `(B, 32, 32, 32)` = **F1** | |
| **S2** patch embed | `(k=3,s=2,p=1)` | `(B, 256, 64)` | grid 16x16 |
| S2 blocks x2 | | `(B, 256, 64)` | `R=4` |
| S2 to 2-D | | `(B, 64, 16, 16)` = **F2** | |
| **S3** patch embed | | `(B, 64, 160)` | grid 8x8 |
| S3 blocks x2 | | `(B, 64, 160)` | `R=2` |
| S3 to 2-D | | `(B, 160, 8, 8)` = **F3** | |
| **S4** patch embed | | `(B, 16, 256)` | grid 4x4 |
| S4 blocks x2 | | `(B, 16, 256)` | `R=1`, full attention |
| S4 to 2-D | | `(B, 256, 4, 4)` = **F4** | |
| decoder project | 4 x `Conv1x1 -> 256` | four maps, 256 ch each | |
| decoder upsample | bilinear to F1's size | 4 x `(B, 256, 32, 32)` | |
| decoder concat | `torch.cat` | `(B, 1024, 32, 32)` | 4 x 256 = 1024 |
| decoder fuse | `Conv1x1` + GELU | `(B, 256, 32, 32)` | |
| decoder refine | `Conv3x3` + GELU | `(B, 256, 32, 32)` | |
| decoder sub-pixel | `Conv3x3` | `(B, 512, 32, 32)` | 512 = 32 x 16 |
| PixelShuffle(4) | | `(B, 32, 128, 128)` | |
| decompress | `SpectralDecompressor` | `(B, 165, 128, 128)` | |
| denormalise | `PixelDenormalize` | `(B, 165, 128, 128)` | back to reflectance |

Notice the symmetry: it starts at `(B, 165, 128, 128)` and ends at
`(B, 165, 128, 128)`. **Reconstruction models are shape-preserving.** If they
were not, you could not subtract the output from the input.

---

## Token-grid arithmetic

Each stage shrinks the grid. The rule, when padding is `kernel // 2`:

```
H_out = H_in // stride
```

Stage 1 is special (`kernel = 4, stride = 4, padding = 0`), but it also gives
`H_out = H_in / 4`. Part 11 explains why it is set up differently.

Worked for `H = W = 128`:

```
Stage 1:  128 / 4  = 32   ->  32 x 32 = 1024 tokens
Stage 2:   32 / 2  = 16   ->  16 x 16 =  256 tokens
Stage 3:   16 / 2  =  8   ->   8 x  8 =   64 tokens
Stage 4:    8 / 2  =  4   ->   4 x  4 =   16 tokens
```

Each stage has **a quarter** as many tokens as the one before (half in each
direction).

### A pattern worth noticing

Attention shrinks the tokens it compares against by a factor `R` in each
direction. So the number it actually attends to is `N / R^2`:

| Stage | grid | `N` | `R` | reduced grid | `N / R^2` |
|---|---|---|---|---|---|
| 1 | 32 x 32 | 1024 | 8 | 4 x 4 | **16** |
| 2 | 16 x 16 | 256 | 4 | 4 x 4 | **16** |
| 3 | 8 x 8 | 64 | 2 | 4 x 4 | **16** |
| 4 | 4 x 4 | 16 | 1 | 4 x 4 | **16** |

Every stage ends up attending over exactly **16** things. That is not luck — the
`[8, 4, 2, 1]` schedule was chosen to make it true. Part 13.

---

## Every mask in the system

Masks are 0/1 arrays that select things. There are more of them here than you
expect, so keep this table handy.

| Name | Shape | 1 means | Made by |
|---|---|---|---|
| pixel validity `mask` | `(B, 1, H, W)` | real measurement | the vendable |
| `token_mask` | `(B, N)` | this token is real data | `pixel_mask_to_token_mask` |
| `keep_mask` | `(B, N)` | show this token to the encoder | `generate_prediction_mask` |
| `pred_mask` | `(B, N)` | hide this token and grade it | same |
| `pixel_pred_mask` | `(B, 1, H, W)` | `pred_mask` blown up to pixels | the trainer |
| `eroded_mask` | `(B, 1, H, W)` | valid **and** away from a border | `erode_mask` |
| `loss_mask` | `(B, 1, H, W)` | grade the model here | `pixel_pred_mask * eroded_mask` |

Two rules that always hold:

```
keep_mask = 1 - pred_mask          there is no third state
loss_mask = pixel_pred_mask * eroded_mask     multiplying 0/1 masks = logical AND
```

> **Why `(B, 1, H, W)` and not `(B, H, W)` for the pixel masks?** So they line up
> with the `(B, C, H, W)` cube for broadcasting — see below. The `1` is a
> placeholder in the channel slot.

---

## Broadcasting, briefly

When you multiply two arrays of different shapes, numpy and torch will stretch
any dimension of size 1 to match:

```
   (B, 165, H, W)      the cube
 * (B,   1, H, W)      the mask
 = (B, 165, H, W)      the mask was reused for all 165 bands
```

This is used constantly: one mask value per pixel applied to all 165 bands, or
one mean per band applied to all pixels. Without broadcasting you would have to
copy the mask 165 times.

---

## A habit worth adopting

When reading this code, **annotate every line with the shape**. The repo already
does this — almost every line in `app/foundation_models/` carries a trailing
comment:

```python
x = x.flatten(2)          # x: (B, c_out, N)
x = x.transpose(1, 2)     # x: (B, N, c_out)
```

Follow the same convention in anything you add. It is the house style
(`docs/10-code-style.md`) and it is the single biggest aid to reading tensor
code.

---

## Common confusions

**"Is `N` the number of pixels?"**
No. `N` is the number of *tokens*. At stage 1 each token covers a 4x4 pixel
block, so `N = (H/4) * (W/4)`, which is 1/16 of the pixel count.

**"Do `C` and `C_i` mean the same thing?"**
No. `C` is the input band count (165). `C_i` is the width of the token vectors
inside stage `i` (32, 64, 160 or 256). They are unrelated numbers that both get
called "channels".

**"Why does the batch dimension keep disappearing in the explanations?"**
Because every operation treats batch items independently. Ignoring `B` is a
legitimate simplification while you are learning.

---

## Check yourself

1. Read this shape aloud in plain English: `(8, 165, 128, 128)`.
2. Convert `(4, 64, 16, 16)` into sequence form. What are `N` and `C`?
3. How many tokens does stage 3 have for a 128x128 patch?
4. What is the relationship between `keep_mask` and `pred_mask`?
5. What shape must a pixel mask have to multiply cleanly against a
   `(B, 165, H, W)` cube?

<details>
<summary>Answers</summary>

1. "Eight patches, each with 165 bands, each band 128 rows by 128 columns."
2. `(4, 256, 64)`. `N = 16*16 = 256` tokens, `C = 64` features per token.
3. 64 (an 8x8 grid).
4. `keep_mask = 1 - pred_mask`. They are exact complements; every token is one
   or the other.
5. `(B, 1, H, W)` — the `1` broadcasts across all 165 bands.

</details>

---

**Next:** the small amount of maths you actually need, in
[08-math-warmup.md](08-math-warmup.md)
