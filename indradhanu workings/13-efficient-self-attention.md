# 13 · Efficient Self-Attention (ESA)

> **The one thing this part teaches:** keep the questions detailed, but summarise
> the answers. Queries stay at full resolution; keys and values are shrunk first.

**Source:**
[`app/foundation_models/components/efficient_self_attention.py`](../app/foundation_models/components/efficient_self_attention.py)

---

## The cost, laid out

Plain attention builds an `N x N` score matrix — one entry for every pair of
tokens. Here are the four stages:

| Stage | tokens `N` | `N x N` entries |
|---|---|---|
| 1 | 1024 | **1,048,576** |
| 2 | 256 | 65,536 |
| 3 | 64 | 4,096 |
| 4 | 16 | 256 |

Stage 1 is over a million entries — and it is a thousand times worse than stage
4. It also runs at the finest resolution, where every other tensor is largest.

Something has to give.

---

## The trick, in one sentence

> Every token still gets its own updated representation (so the **queries** stay
> at full resolution), but the things it compares against are **summarised
> first** by a factor `R` in each direction.

Cost changes from:

```
N x N          ->          N x (N / R^2)
```

With `R = 8` at stage 1, `R^2 = 64`, so the score matrix shrinks 64-fold.

> **Analogy.** You have a detailed question about a thousand-page report. Rather
> than reading all thousand pages, you read a fifteen-page executive summary —
> one paragraph per chapter. Your question stays as specific as it was; the
> material you consult has been condensed.

---

## How the shrinking is done

With a strided convolution, on the key/value path only:

```python
self.reduction = nn.Conv2d(embed_dim, embed_dim,
                           kernel_size=reduction_ratio,
                           stride=reduction_ratio)
self.reduction_norm = nn.LayerNorm(embed_dim)
```

`kernel = stride = R` means non-overlapping pooling (same pattern as part 11's
stage 1). Each output token summarises an `R x R` block of input tokens. And
because it is a convolution, **how** to summarise is learned, not a fixed
average.

---

## The full data flow

```
Q path (unchanged, full resolution):

   (B, N, C)  --Linear-->  (B, N, C)  --reshape-->  (B, h, N, d)


K/V path (shrunk):

   (B, N, C)
      --transpose + reshape-->  (B, C, H, W)     back to a 2-D grid
      --Conv2d(k=R, s=R)----->  (B, C, H/R, W/R) shrunk
      --flatten + transpose-->  (B, N/R^2, C)    back to a sequence
      --LayerNorm------------>  (B, N/R^2, C)
      --Linear--------------->  K and V, each (B, h, N/R^2, d)


Attention:

   scores  = Q @ K^T * scale     ->  (B, h, N, N/R^2)
   weights = softmax(scores, -1)
   out     = weights @ V         ->  (B, h, N, d)
```

The key sentence for understanding what `scores` means now:

> `scores[b, h, i, j]` = how much should full-resolution token `i` attend to
> summary token `j`, where `j` represents an `R x R` neighbourhood?

**Fine-grained questions, spatially aggregated answers.**

Note that the output shape is `(B, h, N, d)` — still one vector per original
token. Nothing was lost on the query side. Every token still gets its own
personalised update.

---

## The reduction schedule, worked

The configuration is `reduction_ratios = [8, 4, 2, 1]`. For a 128x128 patch:

| Stage | grid | `N` | `R` | shrunk grid | `N / R^2` | score matrix | plain would be |
|---|---|---|---|---|---|---|---|
| 1 | 32x32 | 1024 | 8 | 32/8 = 4x4 | **16** | 1024 x 16 = 16,384 | 1,048,576 |
| 2 | 16x16 | 256 | 4 | 16/4 = 4x4 | **16** | 256 x 16 = 4,096 | 65,536 |
| 3 | 8x8 | 64 | 2 | 8/2 = 4x4 | **16** | 64 x 16 = 1,024 | 4,096 |
| 4 | 4x4 | 16 | 1 | 4x4 | **16** | 16 x 16 = 256 | 256 |

Stage 1 goes from 1,048,576 entries to 16,384 — **64 times cheaper**.

### Notice the pattern

Look down the "shrunk grid" column: **4x4, 4x4, 4x4, 4x4**. Every single stage
attends over exactly 16 summary tokens.

That is not luck. The schedule `[8, 4, 2, 1]` was chosen precisely so that:

```
Stage 1:  32 / 8 = 4
Stage 2:  16 / 4 = 4
Stage 3:   8 / 2 = 4
Stage 4:   4 / 1 = 4
```

Every stage sees the whole patch summarised into a 4x4 overview, regardless of
how detailed its own view is. Stage 4 needs no reduction at all, because 16
tokens is already the target.

---

## The fallback path — the subtle bit

The reduction convolution has to rebuild a full 2-D grid, and that requires the
token count to exactly equal `H * W`.

But after tokens are hidden at stage 1 (part 19), it does not. If 65% of 1024
tokens are removed, only about 358 remain — and you cannot arrange 358 things
into a 32x32 grid. The reshape would crash.

The code detects this and steps around it:

```python
can_reduce = (self.reduction_ratio > 1) and (N == H * W)

if can_reduce:
    ...   # reshape, convolve, flatten
else:
    reduced_x = x        # no reduction; full attention over what is left
```

So **during masked training and masked inference, stage 1 quietly runs plain
full attention over the surviving tokens.**

Is that expensive? Let us check. With 358 surviving tokens:

```
358 x 358 = 128,164 entries
```

against the 1,048,576 of unmasked full attention. It is about one eighth of the
cost, because we removed most of the tokens. Perfectly affordable.

(The source comment uses the older 50% mask ratio and says
`512 * 512 = 262K` — same reasoning, different mask ratio.)

> **Why this matters to you.** The model genuinely behaves differently with and
> without `keep_mask`. That is intentional and documented, but it does mean a
> forward pass with no mask is not identical machinery to one with a mask.

---

## Parameter count for one ESA

The pieces: four `Linear(C, C)` layers (q, k, v and the output projection);
plus, when `R > 1`, the reduction convolution and its LayerNorm.

### Stage 1, worked in full

`C = 32`, `R = 8`:

```
q, k, v, proj    each: 32*32 + 32 = 1,056
                 four:              4,224

reduction conv   Conv2d(32, 32, k=8, s=8)
                 32 * 32 * 8 * 8 + 32
                 = 32 * 32 * 64 + 32
                 = 65,536 + 32     =  65,568

reduction norm   2 * 32            =      64
                                     -------
                            total     69,856
```

### The surprise in that number

The reduction convolution is **65,568** of the 69,856 — over 93% of the whole
attention module, and roughly **fifteen times** the size of all four projections
combined.

Why? Because an 8x8 kernel with 32 input and 32 output channels is
`32 x 32 x 64` weights. Kernel area grows as the square of `R`.

**The practical lesson:** `R = 8` buys a 64x saving in compute but costs a lot
in parameters. Stage 1 blocks are not nearly as cheap as their small
`embed_dim = 32` would suggest. This is a genuine trade-off, not a free lunch.

---

## Common confusions

**"Does reducing the keys lose information?"**
Yes, some. Fine spatial detail on the *answer* side is blurred. The bet — which
SegFormer's authors validated across many tasks — is that queries need to be
precise while answers can be summarised.

**"Is the reduction the same as pooling?"**
It is learned pooling. Average pooling would compute a fixed mean; this
convolution learns what to keep from each `R x R` block.

**"Why does stage 4 have `R = 1`? Is attention disabled?"**
No — `R = 1` means no reduction, i.e. **ordinary full attention**. With only 16
tokens there is nothing to save.

**"Does the fallback make masked training less accurate?"**
It makes it *less compressed*, if anything — full attention over the survivors is
strictly more information than reduced attention would be. The cost is compute,
and the compute is small because most tokens are gone.

---

## Check yourself

1. Which of query, key and value stay at full resolution, and why?
2. For stage 2 (`N = 256`, `R = 4`), how many summary tokens are there, and how
   big is the score matrix?
3. Why does every stage end up attending over exactly 16 tokens?
4. When does the reduction get skipped, and why is that safe?
5. Which single component dominates stage 1's attention parameter count?

<details>
<summary>Answers</summary>

1. Queries. Every token must still receive its own updated representation, so
   there must be one query per token.
2. `256 / 16 = 16` summary tokens; score matrix `256 x 16 = 4,096` entries.
3. Because the reduction ratios `[8, 4, 2, 1]` were chosen so that each stage's
   grid divided by its `R` gives 4x4 in every case.
4. When `N != H * W`, which happens at stage 1 after tokens are hidden. Safe
   because most tokens have been removed, so plain full attention over the
   survivors is cheap.
5. The reduction convolution: 65,568 of 69,856 parameters, because an 8x8 kernel
   over 32 channels is large.

</details>

---

**Next:** the other half of a transformer block, in
[14-mix-ffn.md](14-mix-ffn.md)
