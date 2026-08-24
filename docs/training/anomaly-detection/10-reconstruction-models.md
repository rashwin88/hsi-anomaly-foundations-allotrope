# Part 10 - Reconstruction models

> **The one thing this part teaches:** an encoder-decoder squeezes an image through a
> bottleneck too small to memorise it, so it learns to rebuild ordinary terrain and fails on
> anything else.

## The shape

```
input  (B, 1, 128, 128)
   |
 encoder        halve H and W, double channels, three times
   |
   v
bottleneck z  (B, 128, 16, 16)
   |
 decoder        mirror it back up
   |
   v
output (B, 1, 128, 128)
```

`B` is the **batch** - how many patches go through together. It is not a band count; on
thermal data there is one band, hence the `1`.

The bottleneck is the mechanism. To rebuild the image from `z`, the network must store
something more compact than the pixels themselves - the *structure* of ordinary terrain.
A thing it has never seen does not fit that structure and comes out wrong.

## Count the squeeze yourself

```
input      = 1 * 128 * 128 = 16,384 values
bottleneck = 128 *  16 *  16 = 32,768 values
```

The bottleneck is **larger** than the input. That surprises people, and it is worth sitting
with: the compression is not in the value count. It is that those 32,768 numbers are shared
across every patch the network ever sees, so they must encode reusable structure rather than
one patch's pixels. The constraint is capacity and training pressure, not raw arithmetic.

## Predict the parameter count

This is the exercise that will convince you that you understand the architecture. Do it
before reading on.

`SpatialEncoder(in_channels=1, base_channels=32, num_stages=3)`. Each stage is
`Conv2d(k=4, stride=2, padding=1)` then `BatchNorm2d`. Channels go `1 -> 32 -> 64 -> 128`.

A Conv2d has `in * out * k * k` weights plus `out` biases. A BatchNorm2d has `2 * out`.

```
stage 0:  conv  1 * 32 * 4 * 4 = 512      + 32 bias  =    544
          bn    2 * 32                              =     64
                                                      ------
                                                         608

stage 1:  conv  32 * 64 * 4 * 4 = 32,768  + 64 bias  = 32,832
          bn    2 * 64                              =    128
                                                      ------
                                                      32,960

stage 2:  conv  64 * 128 * 4 * 4 = 131,072 + 128 bias = 131,200
          bn    2 * 128                              =    256
                                                      -------
                                                     131,456

total = 608 + 32,960 + 131,456 = 165,024
```

Now check it:

```python
from app.foundation_models.components.spatial_encoder import SpatialEncoder
sum(p.numel() for p in SpatialEncoder(1, 32, 3).parameters())
# 165024
```

**165,024.** If your arithmetic matched, you understand the encoder. The full autoencoder,
with the mirrored decoder, is 329,665.

## Why (k=4, s=2, p=1)

That combination halves height and width exactly:

```
H_out = (H_in + 2*padding - kernel) / stride + 1
      = (128 + 2 - 4) / 2 + 1
      = 126 / 2 + 1
      = 63 + 1
      = 64
```

No fractional sizes, and the transposed convolution in the decoder doubles exactly, so
encoder and decoder mirror without `output_padding` fixes.

The consequence: **height and width must be divisible by `2^num_stages`.** Three stages means
divisible by 8. Feed it 100x100 and it fails.

## The decoder's last block is different, deliberately

Intermediate decoder blocks are `ConvTranspose2d -> BatchNorm -> GELU -> Dropout`. The final
block has **only the convolution**.

This looks like an oversight. It is not, and "tidying" it would break the model.

BatchNorm forces its output to zero mean and unit variance per channel. The final layer's job
is to emit real temperatures. A scene sitting at 300 K would be forced to zero mean and
become unrecoverable. GELU would clip negatives, and temperature anomalies can be negative.
The last layer must be free to output any value.

Verbatim from `app/foundation_models/components/spatial_decoder.py`:

> The final layer's job is to reconstruct the original input values (e.g. surface
> temperatures in Kelvin). BatchNorm would force the output to zero-mean unit-variance per
> channel - but real thermal data is NOT zero-mean or unit-variance.

## Normalisation, and a live trap

Models normalise their input: `(x - mean) / std`, per band, using statistics stored as
**registered buffers** - which means they travel inside the checkpoint file.

The shipped thermal figures are mean 24.5756, std 13.5744 - Celsius.

Now feed that model HotSat-1, which ships raw digital numbers around 5,000:

```
(5000 - 290) / 10 = 471.0
```

using a Celsius-fitted mean of 290 and std of 10. A network trained on inputs in roughly
-3 to +3 receives **471**. It does not degrade gracefully; it reconstructs noise, and since
the score is reconstruction error, everything looks anomalous and nothing usefully so.

The fix is `PixelStatsOverride`, which recomputes mean and std from the scene itself. See
`docs/lld/pixel-stats.md`. **The override wins outright over the baked figures** - it never
merges and never falls back, because falling back is precisely the bug.

## The roster

Seven architectures, each with a Sanskrit codename. **The backend selects models by
codename**, not class name.

| codename | slug | notes |
|---|---|---|
| Pratibimba | `spatial_autoencoder` | the baseline |
| Antardhana | `spatial_masked_autoencoder` | masked, L2 loss |
| Tirohita | `spatial_masked_autoencoder_l1` | masked, L1 loss |
| Asanskrita | `spatial_masked_autoencoder_l1_unnormalized` | works in raw Celsius |
| Drashta | `normalized_masked_autoencoder` | Asanskrita plus normalisation |
| Chakshu | `segformer_mae` | transformer, thermal |
| Indradhanu | `hyperspectral_segformer_mae` | transformer, 165 bands |

The first three share one class; only training differs.

Shipped checkpoints: Chakshu at 406,500 parameters, validation loss 0.2565 at epoch 495;
Indradhanu at 5,507,354 parameters, 0.0435 at epoch 200. Those figures will drift.

## L2 gave way to L1

The clearest trend across the roster.

L2 (squared error) punishes large errors quadratically, so training pushes hardest to fix the
worst-fitting pixels. But **the worst-fitting pixels are the anomalies** - L2 explicitly
teaches the model to reconstruct the things you want it to fail on.

L1 (absolute error) is robust to outliers. It cares much less about a few extreme pixels, so
the model learns ordinary terrain and leaves anomalies badly reconstructed. Later models are
all L1.

## Indradhanu's spectral bottleneck

165 bands is too many to attend over. Indradhanu compresses to 24 or 32 channels with a
learned 1x1 convolution before the encoder, and expands afterwards.

A 1x1 convolution across channels is a learned linear mixture of bands - conceptually a
trainable MNF (part 8), optimised end-to-end for reconstruction rather than for
signal-to-noise.

The asymmetry to leave alone: the **compressor has BatchNorm, the decompressor has none**.
The compressor's job is to hand the encoder a stable distribution; the decompressor's output
feeds denormalisation and must stay unconstrained. Same reasoning as the decoder's final
block.

## Common confusions

**"Is the bottleneck compression? It has more values than the input."**
Not in value count. The constraint is that those weights are shared across every patch, so
they must encode reusable structure rather than memorised pixels.

**"Why not a bigger model? It would reconstruct better."**
Better reconstruction of *everything*, including anomalies, means lower error everywhere and
nothing detected. Capacity is deliberately small - part 6.

**"Registered buffers - why does that matter?"**
Because normalisation statistics end up inside the `.pt` file. You cannot point a checkpoint
at a differently-calibrated sensor without either retraining or an override.

**"Three codenames for one class?"**
Yes. Pratibimba, Antardhana and Tirohita share `SpatialAutoencoder`, differing only in
training regime. Codename identifies a trained artifact, not an architecture.

## Check yourself

<details>
<summary>1. Compute the parameter count for SpatialEncoder(in_channels=1, base_channels=16, num_stages=2), channels 1 -> 16 -> 32.</summary>

```
stage 0: conv 1 * 16 * 16 = 256 + 16 = 272
         bn   2 * 16              =  32
                                    ---
                                    304

stage 1: conv 16 * 32 * 16 = 8192 + 32 = 8224
         bn   2 * 32                   =   64
                                        -----
                                        8288

total = 304 + 8288 = 8,592
```
</details>

<details>
<summary>2. Can this architecture accept a 100x100 patch with num_stages=3?</summary>

No. Each stage halves the spatial dimensions, so H and W must be divisible by `2^3 = 8`.
`100 / 8 = 12.5`. Use 96 or 104.
</details>

<details>
<summary>3. Why does the decoder's final block omit BatchNorm and GELU?</summary>

It must emit real physical values. BatchNorm would force zero mean and unit variance per
channel, making a scene at 300 K unrecoverable, and GELU would clip negative values that
genuine anomalies can have.
</details>

<details>
<summary>4. A model normalised for Celsius (mean 290, std 10) receives HotSat DN of 4,400. What normalised value, and why is that a problem?</summary>

```
(4400 - 290) / 10 = 4110 / 10 = 411.0
```

The network trained on inputs roughly in -3 to +3 receives 411. It reconstructs noise, so
every pixel shows large error and the score map is meaningless.
</details>

<details>
<summary>5. Why did the roster move from L2 to L1 loss?</summary>

L2 penalises large errors quadratically, so training concentrates on the worst-fitting
pixels - which are the anomalies. It teaches the model to reconstruct exactly what should
stay badly reconstructed. L1 is robust to outliers and leaves them alone.
</details>

---

Next: [part 11](11-masking.md) - why a pixel must never help reconstruct itself.
