# 22 · One patch, all the way through

> **The one thing this part teaches:** nothing new. This is parts 09 to 21
> joined into a single continuous walk, so the pieces click together.

Keep
[`hyperspectral_seg_former_mae.py`](../app/foundation_models/components/hyperspectral_seg_former_mae.py)
open beside this. It is 156 lines and you can now read every one of them.

---

## Setup for the walk

```
B = 4               a batch of four patches
C = 165             input bands
D = 32              compressed bands
H = W = 128         patch size in pixels
mask_ratio = 0.65   hide 65% of the valid tokens
```

---

## The entry point

```python
def forward(self, x, mask=None, keep_mask=None):
```

| Argument | Shape | What it is |
|---|---|---|
| `x` | `(4, 165, 128, 128)` | the observed reflectance |
| `mask` | `(4, 1, 128, 128)` | 1 where the pixel is a real measurement |
| `keep_mask` | `(4, 1024)` | 1 = show this token, 0 = hide it |

### Notice what is NOT here

No loss function. No anomaly score. No masking strategy. No decision about
*which* tokens to hide.

The model reconstructs whatever it is given, using whatever mask it is handed.
Deciding what to hide belongs to the trainer (random) and the inferencer
(checkerboard or complementary random).

The docstring of the thermal sibling states the principle:

> *"Masking strategy (random for training, checkerboard for inference) is NOT
> handled inside this model. This keeps the model architecture clean and
> reusable across different masking strategies."*

That separation is why the same class serves both.

---

## Step 1 · Zero the invalid pixels

```python
if mask is not None:
    x = x * mask
```

Shape unchanged: `(4, 165, 128, 128)`.

The mask is `(4, 1, 128, 128)`, so it broadcasts across all 165 bands (part 07).
Every invalid pixel becomes exactly `0.0` in every band.

---

## Step 2 · Normalise

```python
if self.normalize is not None:
    x = self.normalize(x)
```

Shape unchanged: `(4, 165, 128, 128)`.

Each band becomes a z-score using its own baked-in mean and standard deviation.
Valid pixels land roughly between -2 and +2. The zeroed pixels land at
`-mean/std`, about -1.22 for band 0 — recognisably out of the ordinary range.

---

## Step 3 · Compress

```python
x = self.compressor(x)
```

```
(4, 165, 128, 128)  ->  (4, 32, 128, 128)
```

A 1x1 convolution replaces each pixel's 165 numbers with 32 learned mixtures,
followed by a BatchNorm to stabilise the distribution.

**The picture is still 128x128.** Nothing spatial has happened yet — only the
band axis shrank.

---

## Step 4 · Encode

```python
features = self.encoder(x, keep_mask=keep_mask)
```

Inside, four stages run (part 16):

```
Stage 1
   patch embed  k=4 s=4 p=0    (4, 32, 128, 128) -> (4, 1024, 32)
   remove_tokens               -> (4, ~358, 32)      65% of valid tokens deleted
   2 x block                   -> (4, ~358, 32)      R=8, but falls back to
                                                     full attention (N != H*W)
   LayerNorm
   restore_tokens              -> (4, 1024, 32)      zeros in the gaps
   reshape                     -> (4, 32, 32, 32)    = F1

Stage 2
   patch embed  k=3 s=2 p=1    -> (4, 256, 64)
   2 x block                   -> (4, 256, 64)       R=4
   reshape                     -> (4, 64, 16, 16)    = F2

Stage 3
   patch embed                 -> (4, 64, 160)
   2 x block                                          R=2
   reshape                     -> (4, 160, 8, 8)     = F3

Stage 4
   patch embed                 -> (4, 16, 256)
   2 x block                                          R=1, true full attention
   reshape                     -> (4, 256, 4, 4)     = F4
```

Returns the list `[F1, F2, F3, F4]`.

---

## Step 5 · Decode

```python
x_hat = self.decoder(features)
```

```
project each to 256 channels
upsample all four to 32x32
concatenate                ->  (4, 1024, 32, 32)
fuse with 1x1 conv + GELU  ->  (4,  256, 32, 32)
refine with 3x3 + GELU     ->  (4,  256, 32, 32)
sub-pixel 3x3 conv         ->  (4,  512, 32, 32)     512 = 32 x 16
pixel_shuffle(4)           ->  (4,   32, 128, 128)
```

Back to full resolution, still in the compressed 32-band space.

---

## Step 6 · Decompress

```python
x_hat = self.decompressor(x_hat)
```

```
(4, 32, 128, 128)  ->  (4, 165, 128, 128)
```

A 1x1 convolution expands 32 back to 165. No BatchNorm, no activation (part 10).

---

## Step 7 · Denormalise

```python
if self.denormalize is not None:
    x_hat = self.denormalize(x_hat)
```

Shape unchanged: `(4, 165, 128, 128)`.

Multiply by the standard deviation, add the mean. Back into physical
reflectance units.

**This is the return value.** The model stops here.

---

## The shape story, on one line

```
(4,165,128,128) -> (4,32,128,128) -> tokens -> F1..F4 -> (4,32,128,128) -> (4,165,128,128)
                   ^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^
                   compressed in                         compressed out
```

Input and output shapes are identical. **Reconstruction models are
shape-preserving** — they have to be, or you could not subtract one from the
other.

---

## What the caller does next

The trainer takes over:

```python
x_hat = model(pixels, mask=mask, keep_mask=keep_mask)

pixel_pred_mask = self._pred_mask_to_pixel_mask(pred_mask, H, W)
eroded_mask     = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)
loss_mask       = pixel_pred_mask * eroded_mask

per_pixel_l1      = (x_hat - pixels).abs()                     # (B, 165, H, W)
per_pixel_l1_mean = per_pixel_l1.mean(dim=1, keepdim=True)     # (B, 1, H, W)
l1_loss           = per_pixel_l1_mean[loss_mask == 1].mean()
sam_loss          = self._sam_loss(x_hat, pixels, loss_mask)
loss              = l1_loss + sam_weight * sam_loss
```

Parts 23, 24 and 25 unpack exactly those lines.

---

## An important detail: which space is the loss in?

Read that snippet again. It compares `x_hat` — the **denormalised** output —
against `pixels`, the **raw input**.

Not the normalised versions. Not the compressed versions. Raw reflectance
against raw reflectance.

The training configuration states it explicitly:

```json
"note": "Loss is computed in RAW REFLECTANCE SPACE (not normalised).
         L1 is in reflectance units, SAM is in radians."
```

### Why this matters to you

It makes the numbers **physically meaningful**. When the configuration says a
well-trained model reaches an L1 of about 0.008, that means:

> On average, each band's predicted reflectance is off by 0.008.

And since real reflectance sits between 0.02 and 0.30, an error of 0.008 is a
few percent of a typical value. You can reason about whether that is good.

In normalised space, "0.008" would mean 0.008 standard deviations, which is a
number you would have to translate before it meant anything.

---

## Three questions to test the trace

**Does the token count change between stages?**
Yes: 1024, 256, 64, 16. Each patch embedding quarters it (halves in each
direction).

**Where is the only place tokens are deleted?**
Stage 1, and only when `keep_mask` was supplied. Note the `i == 0` guard in the
encoder loop.

**What shape is the thing you subtract from the input?**
Exactly the input's shape, `(B, 165, H, W)`.

---

## Where each part fits, as a map

```
Step 1  zero invalid           part 09
Step 2  normalise              part 09
Step 3  compress               part 10
Step 4  encode                 parts 11-19
          patch embedding      part 11
          attention            parts 12, 13
          Mix-FFN              part 14
          block                part 15
          four stages          part 16
          masking              parts 17, 18, 19
Step 5  decode                 part 21
Step 6  decompress             part 10
Step 7  denormalise            part 09
        (erosion)              part 20
```

If any box in that map still feels hazy, this is the moment to go back — the
remaining parts assume all of it.

---

## Common confusions

**"Where does the anomaly score get computed?"**
Not here. The model returns a reconstruction. Scoring happens outside, in the
Action (part 30).

**"Does the model behave differently with `keep_mask=None`?"**
Yes. With no mask, no tokens are removed, `N == H*W` holds, and both ESA's
spatial reduction and Mix-FFN's depthwise convolution take their standard paths.
It is a genuinely different code path — and a full-reconstruction pass, which is
not what you want for scoring.

**"Why compress and then decompress? Isn't that just undoing itself?"**
No, because the transformer sits between them and modifies the representation.
The compressor makes the transformer affordable; the decompressor turns its
answer back into bands.

---

## Check yourself

1. List the seven steps of `forward` in order.
2. Why is the masking strategy not part of the model class?
3. What is the shape after `self.compressor` for a batch of 4 at 128x128?
4. Which space is the loss computed in, and why does that make L1 interpretable?
5. Where in the trace do tokens get deleted, and where do they come back?

<details>
<summary>Answers</summary>

1. Zero invalid pixels; normalise; compress; encode; decode; decompress;
   denormalise.
2. So the same model class can serve random masking during training and
   checkerboard or complementary-random masking at inference, without change.
3. `(4, 32, 128, 128)` — the band axis shrank from 165 to 32; the picture is
   still 128x128.
4. Raw reflectance space. An L1 of 0.008 therefore means "each band is off by
   0.008 reflectance on average", which can be compared against typical values of
   0.02–0.30.
5. Deleted inside stage 1 of the encoder, after the patch embedding
   (`remove_tokens`). Restored as zeros after that stage's blocks and LayerNorm
   (`restore_tokens`), before the reshape to 2-D.

</details>

---

**Next:** measuring how wrong the reconstruction is, in
[23-loss-1-l1.md](23-loss-1-l1.md)
