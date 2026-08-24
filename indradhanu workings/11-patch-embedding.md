# 11 · Patch embedding: turning pixels into tokens

> **The one thing this part teaches:** a transformer cannot read pixels. It reads
> a *list of vectors*. Patch embedding is the step that chops the picture into
> blocks and summarises each block as one vector.

**Source:**
[`app/foundation_models/components/overlap_patch_embedding.py`](../app/foundation_models/components/overlap_patch_embedding.py)

---

## What a token is

Transformers were invented for text. There, a token is a word or word fragment,
and a sentence is a list of tokens.

For images, we need the equivalent. The answer:

> **A token is a small square block of the picture, summarised as a list of
> numbers.**

In this model, at stage 1, a token is a **4 pixel by 4 pixel block**, summarised
as **32 numbers**.

Concretely:

```
input:  32 channels, 128 x 128 pixels
        |
        | cut into 4x4 blocks; summarise each block as 32 numbers
        v
output: 1024 tokens, each 32 numbers long        tensor (B, 1024, 32)
```

Where does 1024 come from?

```
128 / 4 = 32 blocks across
128 / 4 = 32 blocks down
32 x 32 = 1024 blocks total
```

> **Common confusion.** "A 4x4 block summarised as 32 numbers" sounds like it
> should compress. Count it: the block holds `4 x 4 x 32 channels = 512` numbers
> and the token holds 32. It *is* a big compression — 16 to 1. That is the
> price of turning a picture into a manageable sequence.

---

## How the summarising is done

With a convolution. Specifically:

```python
self.proj = nn.Conv2d(c_in, c_out,
                      kernel_size=patch_size,
                      stride=desired_compression,
                      padding=padding)
```

To follow this you need two convolution concepts.

**Kernel size** is how big the sliding window is. `kernel_size=4` means the
window looks at a 4x4 patch of pixels at a time.

**Stride** is how far the window jumps between positions. `stride=4` means it
moves 4 pixels each step.

When kernel and stride are **equal**, the windows tile the image perfectly:
they touch but never overlap, and every pixel is covered exactly once.

```
kernel=4, stride=4:

 pixels: | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | ...
 window:  \___1___/  \___2___/  \____3___/
```

Each window position produces one output vector — one token.

---

## The three lines after the convolution

```python
x = x.flatten(2)          # (B, C, H', W') -> (B, C, N)
x = x.transpose(1, 2)     #               -> (B, N, C)
out = self.norm(x)        # LayerNorm across each token's C features
```

The first two are the spatial-to-sequence conversion from part 07. The third is
a LayerNorm.

**LayerNorm here normalises each token independently, across its own 32
features.** Not across the batch. Not across tokens. Just: take this one token's
32 numbers, make them zero-mean and unit-spread.

> **Why LayerNorm and not BatchNorm?** BatchNorm normalises each channel using
> statistics from the whole batch, which couples every example to whichever
> other examples happened to be in the batch. For sequences that is the wrong
> coupling — and it makes the model behave differently depending on batch
> composition. LayerNorm keeps every token self-contained.

---

## What the module returns

```python
return out, H_dash, W_dash
```

Three things: the tokens, and the **grid dimensions**.

Why return the grid size? Because later modules need to rebuild the 2-D layout.
Attention's reduction step and the feed-forward network's depthwise convolution
both physically reshape the token list back into a grid, do something spatial,
and flatten it again.

Once you flatten `(32, 32)` into `1024`, that shape information is gone unless
somebody carries it. So it gets carried, explicitly, through every function
signature you will see:

```python
def forward(self, x, H, W):
```

---

## Overlapping versus non-overlapping — the key design choice

There are two ways to place the windows.

**Non-overlapping** (`kernel = stride`): windows tile exactly. Each pixel
belongs to exactly one token.

**Overlapping** (`kernel > stride`): windows share pixels with their
neighbours. Each pixel influences several tokens.

```
kernel=3, stride=2 (overlapping):

 positions: | 0 1 2 |
              | 2 3 4 |
                | 4 5 6 |
                  ^ shared
```

A plain Vision Transformer uses non-overlapping. SegFormer normally uses
overlapping, because sharing pixels preserves continuity across token boundaries
and avoids visible seams.

This model uses **both**, deliberately. From
[`seg_former_encoder.py`](../app/foundation_models/components/seg_former_encoder.py):

```python
patch_size = 4 if i == 0 else 3
stride     = 4 if i == 0 else 2
padding    = 0 if i == 0 else None      # None means kernel // 2
```

| Stage | kernel | stride | padding | overlap | why |
|---|---|---|---|---|---|
| 1 | 4 | 4 | 0 | **none** | correctness of the hiding game |
| 2 | 3 | 2 | 1 | 1 token | continuity; no hiding happens here |
| 3 | 3 | 2 | 1 | 1 token | same |
| 4 | 3 | 2 | 1 | 1 token | same |

### Why stage 1 must not overlap — this is important

Stage 1 is the only place where tokens get hidden (part 19).

Now suppose stage 1's windows overlapped. Token A's window and token B's window
would share some pixels. Hide token B, keep token A — and some of B's pixels are
still present, inside A.

The model would be able to partially peek at the very thing we are grading it
on. Every leaked pixel makes the exam a little less honest, and the anomaly
score a little less meaningful.

With `kernel = stride = 4` and `padding = 0`, each token sees **exactly its own
4x4 block and nothing else**. Zero leakage. Perfect isolation.

The encoder's own comment states it in one line:

> *"NON-OVERLAPPING — zero information leakage for MAE token removal. Each token
> sees exactly its own 4x4 block."*

Stages 2 to 4 never hide anything, so they are free to overlap and gain the
continuity benefit.

---

## Padding, and the size arithmetic

**Padding** means adding a border of extra values around the image before
convolving, so that the output comes out a convenient size.

Two rules to remember:

```
With padding = kernel // 2  (stages 2-4):      H_out = H_in // stride
With kernel = stride, padding = 0 (stage 1):   H_out = H_in / stride  exactly
```

Both give a clean division. Worked for `H = 128`:

```
Stage 1:  128 / 4 = 32
Stage 2:   32 / 2 = 16
Stage 3:   16 / 2 =  8
Stage 4:    8 / 2 =  4
```

So the grid goes 32, 16, 8, 4 — each stage halving in each direction, which
quarters the token count: 1024, 256, 64, 16.

---

## Parameter counts — derive them, then check

Two formulas:

```
Conv2d(c_in, c_out, k)  :  c_in * c_out * k * k + c_out
LayerNorm(c)            :  2 * c        (a scale and a shift per feature)
```

### Stage 1, worked in full

`c_in = 32` (the compressed bands), `c_out = 32`, `k = 4`:

```
conv weights : 32 * 32 * 4 * 4  = 32 * 32 * 16 = 16,384
conv bias    : 32               =                    32
                                                 --------
                                                   16,416
LayerNorm    : 2 * 32           =                      64
                                                 --------
                                        total      16,480
```

### All four stages

| Stage | conv | conv params | LayerNorm | total | torchinfo says |
|---|---|---|---|---|---|
| 1 | 32 -> 32, k=4 | 16,416 | 64 | **16,480** | 16,480 |
| 2 | 32 -> 64, k=3 | 32*64*9 + 64 = 18,496 | 128 | **18,624** | 18,624 |
| 3 | 64 -> 160, k=3 | 64*160*9 + 160 = 92,320 | 320 | **92,640** | 92,640 |
| 4 | 160 -> 256, k=3 | 160*256*9 + 256 = 368,896 | 512 | **369,408** | 369,408 |

Every one matches the real model exactly. Open
`research/model_break_down/05_hyperspectral_segformer_mae.md`, find the
`torchinfo` block, and confirm it yourself. Four matches in a row is not a
coincidence — it means you have genuinely understood the layer.

### What compression bought us

Without the spectral compressor, stage 1's convolution would be `165 -> 32`:

```
165 * 32 * 16 + 32 = 84,480 + 32 = 84,512
```

versus 16,416 with compression. **Five times smaller**, on the layer that runs at
the finest resolution and therefore the most times. That is the payoff from
part 10, quantified.

---

## Gotcha: a stale comment in the source

Two places in this codebase describe stage 1 as using **kernel 7**:

- the header of `overlap_patch_embedding.py`:
  *"Example for Stage 1: kernel=7, stride=4"*
- the docstring of `erode_mask` in `token_masking.py`:
  *"Should match OPE kernel_size (7 for Stage 1)"*

The encoder actually builds stage 1 with **kernel 4**. Go and look:

```python
patch_size = 4 if i == 0 else 3
```

The prose is left over from an earlier design that used overlapping stage-1
patches. It was presumably changed to fix exactly the leakage problem described
above, and the comments were not updated.

This is a live example of the project's own rule:

> **Verify claims against source.** The code is right; the comment is stale.

Do not "fix" the code to match the comment. Fix the comment, if anything.

---

## Common confusions

**"Is a token the same as a pixel?"**
No. One token covers 16 pixels (a 4x4 block) at stage 1.

**"Does every stage cut the *original image* into blocks?"**
No — only stage 1 sees the image. Stages 2, 3 and 4 cut up the *previous
stage's output grid*. By stage 4 each token traces back to a 32x32 region of the
original picture.

**"Why is it called 'embedding'?"**
Borrowed from language modelling, where a word is "embedded" into a vector of
numbers. Same idea: turn something that is not a vector into a vector.

**"Could you use a plain reshape instead of a convolution?"**
Yes, and that is roughly what the original Vision Transformer did. Using a
convolution means the summarising is *learned* rather than a fixed rearrangement.

---

## Check yourself

1. How many tokens does stage 1 produce for a 128x128 patch, and why?
2. What do `kernel_size` and `stride` mean, and what happens when they are
   equal?
3. Why must stage 1 be non-overlapping while stages 2–4 need not be?
4. Compute the parameters of `Conv2d(64, 160, kernel_size=3)` plus its
   LayerNorm.
5. Why does the module return `H'` and `W'` as well as the tokens?

<details>
<summary>Answers</summary>

1. 1024. `128/4 = 32` blocks in each direction, and `32 x 32 = 1024`.
2. Kernel is the window size; stride is the step between windows. Equal means
   the windows tile exactly — no overlap, no gaps.
3. Stage 1 is where tokens are hidden. Overlap would leak pixels from a hidden
   token into a visible one, corrupting the grading. No hiding happens at stages
   2–4, so overlap there is a free benefit.
4. `64 * 160 * 9 + 160 = 92,160 + 160 = 92,320`; LayerNorm `2 * 160 = 320`;
   total **92,640**.
5. Because flattening destroys the grid shape, and downstream modules need to
   rebuild a 2-D grid for their spatial operations.

</details>

---

**Next:** what the transformer actually does with these tokens, in
[12-attention.md](12-attention.md)
