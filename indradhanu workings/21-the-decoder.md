# 21 · The decoder: fuse four scales, then PixelShuffle

> **The one thing this part teaches:** the decoder brings four different-sized
> feature maps to a common size, merges them, and then expands to full
> resolution using a trick that gives every output pixel its own prediction.

**Source:**
[`app/foundation_models/components/seg_former_decoder.py`](../app/foundation_models/components/seg_former_decoder.py)

---

## What the decoder is handed

Four feature maps from the encoder, at four scales and four widths:

```
F1 : (B,  32, 32, 32)     finest    fine texture
F2 : (B,  64, 16, 16)
F3 : (B, 160,  8,  8)
F4 : (B, 256,  4,  4)     coarsest  global context
```

It must turn these into one prediction at the original resolution:

```
(B, 32, 128, 128)
```

It does so in five steps.

---

## Step 1 — put everything on the same number of channels

The four maps have 32, 64, 160 and 256 channels. You cannot stack things of
different widths, so first project each one to a common width,
`decoder_dim = 256`.

```python
self.linear_projections = nn.ModuleList([
    nn.Conv2d(embed_dim, decoder_dim, kernel_size=1) for embed_dim in embed_dims
])
```

Four separate 1x1 convolutions — a per-pixel matrix multiply (part 08), one per
input map.

```
F1: 32  -> 256
F2: 64  -> 256
F3: 160 -> 256
F4: 256 -> 256
```

---

## Step 2 — put everything at the same spatial size

Resize all four to **F1's** resolution, 32x32.

```python
proj = F.interpolate(proj, size=(target_h, target_w),
                     mode='bilinear', align_corners=False)
```

Bilinear interpolation is ordinary smooth image resizing — each output pixel is
a weighted blend of the nearest input pixels.

```
F1: 32x32 -> 32x32      (already there, no-op)
F2: 16x16 -> 32x32      2x up
F3:  8x8  -> 32x32      4x up
F4:  4x4  -> 32x32      8x up
```

> **Important:** the whole decoder works at 32x32, which is a **quarter** of the
> final resolution. That is deliberate — convolutions at 32x32 cost one
> sixteenth of what they cost at 128x128. The jump to full size happens once, at
> the very end, by a cheaper mechanism.

---

## Step 3 — stack them

```python
fused = torch.cat(unified, dim=1)     # (B, 4*256, 32, 32) = (B, 1024, 32, 32)
```

`dim=1` is the channel axis, so this glues them into one 1024-channel map.

At this point every spatial position holds 1024 numbers: 256 describing fine
texture there, 256 describing medium structure, 256 coarse, 256 global.

---

## Step 4 — merge

```python
fused = self.fuse_conv(fused)         # Conv1x1: 1024 -> 256
fused = self.fuse_act(fused)          # GELU
```

A 1x1 convolution mixes those 1024 numbers down to 256, learning how much to
weight each scale at each position. Then a nonlinearity.

Now: one 256-channel map at 32x32, carrying information from all four scales.

---

## Step 5 — refine, then jump to full resolution

```python
self.refine = nn.Sequential(
    nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
    nn.GELU(),
    nn.Conv2d(decoder_dim, out_channels * 4 * 4, kernel_size=3, padding=1),
)
...
out = F.pixel_shuffle(refined, 4)
```

The two 3x3 convolutions run at 32x32. Their **effective reach in original
pixels** is bigger than it looks:

```
a 3x3 window at quarter resolution = 3 x 4 = 12 original pixels across
```

Twelve pixels of context around every point, at one sixteenth of the compute.

The second convolution outputs `out_channels * 16` channels — one for every
sub-pixel of a 4x4 block. Then `pixel_shuffle` rearranges channels into space.

---

## PixelShuffle, worked

`pixel_shuffle(x, r)` turns `(B, C*r*r, H, W)` into `(B, C, H*r, W*r)`. It moves
information from the **channel** axis into the **spatial** axes.

The formula, if you want it precisely:

```
out[c, h*r + i, w*r + j]  =  in[c*r*r + i*r + j, h, w]
```

### The simplest possible example

`C = 1` output channel, `r = 2`, input spatial size 1x1, so the input has
`1 * 2 * 2 = 4` channels holding values `a, b, c, d`:

```
input:  4 channels, each a single pixel

        channel 0 = a
        channel 1 = b
        channel 2 = c
        channel 3 = d

output: 1 channel, 2x2 pixels

        [ a  b ]
        [ c  d ]
```

Four channel values became a 2x2 spatial block. Nothing was computed — the
numbers were only **rearranged**.

### The real case

`r = 4`, so each group of 16 channels becomes a 4x4 block. With
`out_channels = D = 32`:

```
input channels needed = 32 * 4 * 4 = 512

(B, 512, 32, 32)  --pixel_shuffle(4)-->  (B, 32, 128, 128)
```

Sanity check on the total count of numbers:

```
before: 512 * 32 * 32 = 524,288
after:   32 * 128 * 128 = 524,288
```

Identical. As promised — a rearrangement, not a computation.

---

## Why PixelShuffle instead of just upsampling?

This is the most important design decision in the decoder, and it is entirely
about **point anomalies**.

### Bilinear upsampling smears

To make a pixel, bilinear interpolation averages its neighbours. Blow up 4x, and
every output pixel is a blend of the input pixels around it.

Now suppose one single pixel on the ground is anomalous. Bilinear upsampling
would predict roughly the same value for it as for its 15 neighbours in the same
4x4 block — because they all come from the same coarse input pixel, blended.

The model's prediction at the anomaly would be dragged toward its ordinary
neighbours. The residual — the thing we are measuring — gets diluted.

### PixelShuffle gives every pixel its own channel

With PixelShuffle, output pixel 7 of a 4x4 block comes from **channel 7**, which
has its own weights all the way back through the refine convolutions.

A single anomalous pixel can therefore receive its own distinct predicted value.
The residual is not smeared.

> Since a single-pixel residual is exactly the signal this whole system is built
> to measure, this is not a cosmetic choice. The source comment puts it
> concretely: *"a 1-pixel anomaly at 45°C surrounded by 30°C background gets its
> own predicted value, not a smoothed average of the 4x4 block."*

### And it is cheap

The alternative — running convolutions at full 128x128 resolution — costs 16
times more. PixelShuffle does the expensive thinking at 32x32 and pays only a
rearrangement to get to full size.

---

## No activation on the last convolution

Notice the `refine` sequence ends with a bare `Conv2d`. No GELU, no ReLU, no
sigmoid.

Same reasoning as the decompressor in part 10: this output flows into
`SpectralDecompressor`, then `PixelDenormalize`, and must be free to represent
**any** value, including large negatives. Any squashing activation here would
make whole ranges of reflectance unreachable.

---

## Parameter count

```
projections    32 -> 256, k=1  :   32*256 + 256  =     8,448
               64 -> 256, k=1  :   64*256 + 256  =    16,640
              160 -> 256, k=1  :  160*256 + 256  =    41,216
              256 -> 256, k=1  :  256*256 + 256  =    65,792

fuse         1024 -> 256, k=1  : 1024*256 + 256  =   262,400

refine 1      256 -> 256, k=3  : 256*256*9 + 256 =   590,080
refine 2      256 -> 512, k=3  : 256*512*9 + 512 = 1,180,160
                                                  ----------
                                          total    2,164,736
```

**That is 39% of the entire model**, in a decoder routinely described as
"lightweight".

Where does it go? The last two lines. Those two 3x3 convolutions at 256 and 512
channels are 1.77 million parameters between them — 82% of the decoder.

> "Lightweight" here means *relative to a segmentation decoder*, which would run
> at full resolution with skip connections. It does not mean small.

Every number above matches the `torchinfo` table in
`research/model_break_down/05_hyperspectral_segformer_mae.md`, line for line.

---

## Common confusions

**"Why upsample everything to F1's size rather than to full resolution?"**
Cost. F1's size is a quarter of full resolution in each direction, so
convolutions there are 16x cheaper. The final jump is handled by PixelShuffle,
which is nearly free.

**"Is PixelShuffle learned?"**
No — it is a pure rearrangement with no parameters. What is learned is the
convolution *before* it, which produces the 16 sub-pixel channels.

**"Does the decoder know which positions were hidden?"**
Not explicitly. F1 arrives with zeros in those slots, which is an implicit
signal, but no separate mask is passed to the decoder.

**"Why is it fine that F1 has holes?"**
Because F2, F3 and F4 cover the same ground without holes, at coarser
resolution. The fusion in step 4 is what makes that work.

---

## Check yourself

1. List the five decoder steps in order.
2. Why does the decoder work at 32x32 instead of 128x128?
3. Perform `pixel_shuffle` with `r = 2` on a 1x1 input with 4 channels holding
   `[9, 8, 7, 6]`.
4. Give the anomaly-detection reason PixelShuffle is preferred to bilinear
   upsampling.
5. Which two layers account for most of the decoder's parameters?

<details>
<summary>Answers</summary>

1. Unify channels to 256; upsample all to F1's size; concatenate; fuse with a
   1x1 conv plus GELU; refine with two 3x3 convs and PixelShuffle to full
   resolution.
2. Convolutions at quarter resolution cost one sixteenth as much. The jump to
   full size is done once, cheaply, by rearrangement.
3. `[[9, 8], [7, 6]]` — a 2x2 block, filled in row-major order from the
   channels.
4. Bilinear averages neighbours, so a single anomalous pixel's prediction is
   dragged toward its ordinary neighbours and the residual is diluted.
   PixelShuffle gives every output pixel its own channel and its own weights.
5. The two 3x3 refine convolutions: 590,080 and 1,180,160 — together 82% of the
   decoder.

</details>

---

**Next:** the whole thing, end to end, in
[22-full-forward-trace.md](22-full-forward-trace.md)
