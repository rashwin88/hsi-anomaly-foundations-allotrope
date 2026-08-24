# 10 · The spectral compressor and decompressor

> **The one thing this part teaches:** 165 bands are squeezed into 32 learned
> mixtures before the heavy machinery runs, and expanded back afterwards. This
> is the *only* structural difference between Indradhanu and its thermal
> sibling.

**Source:**
[`app/foundation_models/components/spectral_compressor.py`](../app/foundation_models/components/spectral_compressor.py)
— 59 lines, most of them explanation.

---

## Start with the problem

The transformer behind this model is a small configuration called
`SegFormer-B0`. Its first stage works with **32-number** token vectors.

Now imagine handing it 165 input channels directly. Three things go wrong.

**1. The first layer becomes bloated.** Its first operation is a convolution
from the input channels to 32. With 165 inputs that is
`165 x 32 x 4 x 4 = 84,480` weights. With 32 inputs it is
`32 x 32 x 4 x 4 = 16,384`. Five times smaller, for a layer that has to run at
the finest resolution.

**2. Most of those 165 numbers are near-duplicates.** Band 62 and band 63 are
10 nanometres apart. Physically, almost nothing changes over 10 nm. Their values
are nearly identical in nearly every pixel. You are paying full price for
information you already have.

**3. Every intermediate tensor is five times bigger** in memory, which limits
batch size, which slows training.

---

## The idea

Learn a way to boil 165 numbers down to 32, run everything in that smaller
space, and expand back at the end.

```
x     : (B, 165, H, W)
y     = Conv2d(165, D, kernel_size=1)(x)         ->  (B, D, H, W)
        ... the whole transformer runs here ...
x_hat = Conv2d(D, 165, kernel_size=1)(y_hat)     ->  (B, 165, H, W)
```

Remember from part 08: **a 1x1 convolution is a per-pixel matrix multiply**. So
in plain words:

> At every pixel, replace its 165 numbers with 32 weighted mixtures of them.
> Then later, at every pixel, expand those 32 numbers back into 165.

`D` is 32 in the current model. It was 24 in the previous version.

> **An everyday analogy.** A sound engineer takes a 48-track recording and mixes
> it down to a stereo pair. They do not pick two tracks and throw away 46 — they
> build each output as a blend of everything, chosen so that the important parts
> survive. Then a listener's speakers "expand" that stereo back into a room full
> of sound. The compressor is the mixdown; the decompressor is the playback.
> And here, the mixing desk settings are *learned* rather than set by hand.

---

## The code, both halves

```python
class SpectralCompressor(nn.Module):
    def __init__(self, in_channels, compressed_channels):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(compressed_channels)

    def forward(self, x):
        x = self.compress(x)
        x = self.norm(x)
        return x


class SpectralDecompressor(nn.Module):
    def __init__(self, compressed_channels, out_channels):
        super().__init__()
        self.decompress = nn.Conv2d(compressed_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.decompress(x)
```

Look at how short those are. The compressor is two lines of real work; the
decompressor is one. This is not a complicated module — it is a simple module in
an important position.

---

## Worked example, small enough to do on paper

Pretend there are only 4 bands and we want 2. One pixel, already normalised:

```
x = [ 1.2, 1.0, -0.4, -0.6 ]
     band0 band1 band2 band3
```

Suppose training happened to arrive at these weights (all biases zero):

```
W = [[ 0.5,  0.5,  0.0,  0.0],     <- output 0's recipe
     [ 0.0,  0.0,  0.5,  0.5]]     <- output 1's recipe
```

Output 0, using row 0:

```
y0 = 0.5*1.2 + 0.5*1.0 + 0.0*(-0.4) + 0.0*(-0.6)
   = 0.60 + 0.50 + 0 + 0
   = 1.10
```

Output 1, using row 1:

```
y1 = 0.0*1.2 + 0.0*1.0 + 0.5*(-0.4) + 0.5*(-0.6)
   = 0 + 0 - 0.20 - 0.30
   = -0.50
```

```
y = [1.10, -0.50]
```

Four numbers became two. And notice the two survivors still describe the *shape*
of the original: "bright at short wavelengths, dark at long wavelengths". The
essential information survived; the redundancy did not.

The real thing does the same with 165 in and 32 out, and the recipes are found
by gradient descent rather than invented by me.

---

## Counting the parameters

Using `parameters = outputs x inputs + outputs` from part 08:

```
compressor
    Conv2d(165 -> 32, k=1)   : 165*32 + 32   =  5,312
    BatchNorm2d(32)          : 2 * 32        =     64
                                              -------
                                                5,376

decompressor
    Conv2d(32 -> 165, k=1)   : 32*165 + 165  =  5,445
                                              -------
    total for both                             10,821
```

Out of the model's 5,506,629 parameters, the entire spectral bottleneck is
**under 0.2%**. It is almost free, and it saves a great deal everywhere else.

(The exact per-layer figures are in the `torchinfo` table at the bottom of
`research/model_break_down/05_hyperspectral_segformer_mae.md`. Go and match
5,312 and 5,445 against it. It takes ten seconds and it is worth doing.)

---

## "It is MNF-shaped, but it is not MNF"

You will hear MNF mentioned constantly around hyperspectral work, so here is
what it is and how it relates.

**MNF** stands for Minimum Noise Fraction. It is a classical, non-neural
technique: you analyse the statistics of a cube, find the directions in which
the signal-to-noise ratio is highest, and keep only the leading few. It produces
a matrix that projects many bands down to a few.

Structurally that is **the same kind of object** as our compressor — a linear
projection with a matrix of shape (few, many). The differences are in how the
matrix is found and what it optimises:

| | MNF | SpectralCompressor |
|---|---|---|
| How the recipe is found | eigen-decomposition of a covariance matrix | gradient descent |
| What it optimises for | maximum signal-to-noise | minimum reconstruction error |
| When it runs | offline, usually per scene | once, during training, for all scenes |
| Learned jointly with the rest of the network? | no | **yes** |

That last row is the real advantage. MNF optimises a proxy (signal-to-noise);
the compressor optimises the thing we actually care about, and it does so while
the rest of the network is adapting to it.

> MNF appears for real elsewhere in this repo, inside the classical `mnf_rx`
> detector. Different code path, same underlying intuition: get into a smaller,
> better-behaved space before doing the hard work.

---

## The asymmetry — do NOT "tidy" this

Look at the two classes again:

- compressor: convolution **plus BatchNorm**
- decompressor: convolution, **no BatchNorm, no activation function**

This looks like somebody forgot to finish the second one. It is intentional, and
the docstring says so explicitly.

**Why the compressor has BatchNorm.** BatchNorm rescales its output so that,
across the batch, each channel has roughly zero mean and unit spread. The
compressor's job is to hand the transformer a stable, well-behaved input, and
this is exactly how you do that.

**Why the decompressor must not have one.** Its output goes straight into
`PixelDenormalize`, which multiplies by the standard deviation and adds the mean
to produce real reflectance. For that to work, the decompressor must be free to
output **any value at all**, including large negatives.

- A BatchNorm there would force every output band to zero mean and unit spread
  across the batch. Real reflectance is emphatically not that.
- An activation like ReLU would clip everything below zero, making it impossible
  to represent any pixel darker than its band's average.

> If you ever see a pull request titled "add missing BatchNorm to
> SpectralDecompressor for symmetry", reject it. The asymmetry is the design.

---

## Why D is 24–32, not 8 and not 128

**Why not much larger?** Stage 1 of the transformer embeds tokens at 32
dimensions. Feeding it far more than 32 input channels inflates that first
convolution for no benefit — you would be compressing again immediately.

**Why not much smaller?** Squeeze too hard and genuine spectral structure is
destroyed before the transformer ever sees it. The model then physically cannot
get spectral *shape* right, no matter how long you train.

The project's own configuration file records this as a diagnosis to look for:

```json
"SAM_stuck_above_0.3_after_epoch_30":
    "Compressor bottleneck too narrow (try compressed_channels=32)"
```

"SAM" is the shape-error measurement (part 24). And that is precisely the change
made between versions:

| | v0.1.0 | v0.2.0 (current) |
|---|---|---|
| `compressed_channels` | 24 | **32** |
| validation loss | 0.07694 | **0.04349** |

The manifest's own note on the older checkpoint:

```json
"note": "earlier config with D=24 (vs D=32 in v0.2). v0.2 wins on val loss
         across the same epoch range."
```

So this is not theory. Somebody hit the symptom, widened the bottleneck, and the
number improved.

---

## Common confusions

**"Is compressing lossy? Are we throwing information away?"**
Yes, it is lossy — 165 numbers cannot be perfectly stored in 32. The bet is that
what is lost is mostly redundancy and noise. The validation loss is how you
check that bet.

**"Does the compressor look at neighbouring pixels?"**
No. Kernel size is 1, so each pixel is processed completely independently. All
spatial reasoning happens later, in the transformer.

**"Is the decompressor the mathematical inverse of the compressor?"**
No, and it is not required to be. They are two independent sets of learned
weights that happen to be trained to work together. There is no constraint tying
them.

**"Why is it called 'spectral' compression?"**
Because it compresses along the *spectral* (band) axis. Nothing spatial is
compressed — the picture stays 128x128 throughout.

---

## Check yourself

1. In one sentence, what does the compressor do to a single pixel?
2. Compute the parameter count of `Conv2d(165, 24, kernel_size=1)`.
3. Give the two reasons the decompressor has no BatchNorm and no activation.
4. How is the compressor like MNF, and how is it different?
5. What symptom suggests `D` is too small, and what was actually done about it?

<details>
<summary>Answers</summary>

1. It replaces the pixel's 165 band values with 32 learned weighted mixtures of
   them.
2. `165 * 24 + 24 = 3960 + 24 = 3984`.
3. Its output feeds denormalisation and must be free to take any value; a
   BatchNorm would force zero-mean unit-spread output (which real reflectance is
   not), and an activation like ReLU would clip negatives.
4. Both are linear projections from many bands to few. MNF is found offline by
   eigen-decomposition to maximise signal-to-noise; the compressor is learned by
   gradient descent to minimise reconstruction error, jointly with the rest of
   the network.
5. SAM (spectral shape error) stuck high — above 0.3 rad after epoch 30. The fix
   was raising `compressed_channels` from 24 to 32, and validation loss improved
   from 0.077 to 0.043.

</details>

---

**Next:** turning a picture into a sequence of tokens, in
[11-patch-embedding.md](11-patch-embedding.md)
