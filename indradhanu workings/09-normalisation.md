# 09 · Per-band normalisation

> **The one thing this part teaches:** before anything else, every band is
> rescaled to "how many standard deviations from typical" — and those rescaling
> numbers live *inside* the saved model file.

**Source:**
[`app/foundation_models/components/pixel_normalization.py`](../app/foundation_models/components/pixel_normalization.py).
At 41 lines it is the smallest module in the model. Open it; you can read the
whole thing in a minute.

---

## Why normalise at all

Neural networks learn badly when their inputs are tiny, or huge, or when
different inputs are on wildly different scales.

Reflectance is tiny — around 0.04. And the bands are not on the same scale as
each other: vegetation is dark in blue wavelengths and bright in near-infrared,
so band 3 and band 140 have genuinely different typical values and different
spreads.

If you feed that in raw, the network spends its early training just learning
"band 140 is generally bigger than band 3", which is a fact we already know and
could have simply told it.

So we tell it. That is what normalisation is.

---

## The operation

For each band separately:

```
z = (value - typical value for this band) / typical spread for this band
```

Written with the standard names:

```
z_c(i, j) = ( x_c(i, j) - mean_c ) / std_c
```

- `mean_c` — the average value of band `c` across the entire training corpus.
- `std_c` — the standard deviation of band `c`, i.e. how much it typically
  varies.

The result, `z`, is a **z-score**: "how many standard deviations from typical is
this?" Most values land between -2 and +2 regardless of which band they came
from.

> **Analogy.** Exam marks from different subjects are not comparable — a 62 in a
> hard subject may be better than an 80 in an easy one. Converting each mark to
> "how many standard deviations above the class average" makes them comparable.
> Same idea, one per band.

---

## The code

```python
class PixelNormalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std
```

and its exact inverse, used at the very end of the model:

```python
class PixelDenormalize(nn.Module):
    def forward(self, x):
        return x * self.std + self.mean
```

### Unpacking `.view(1, -1, 1, 1)`

`view` reshapes without changing any numbers. The `-1` means "work this one
out for me". Starting from a flat list of 165 numbers:

```
(165,)  ->  view(1, -1, 1, 1)  ->  (1, 165, 1, 1)
```

Why that odd shape? So it lines up with the data for **broadcasting**
(part 07):

```
   (B, 165, H, W)      the cube
 - (1, 165, 1, 1)      the means
 = (B, 165, H, W)
```

The size-1 dimensions get stretched automatically. Band 7's mean is subtracted
from every pixel of band 7, in every patch of the batch, and from nothing else.

Without this trick you would need an explicit loop over 165 bands.

---

## Where the numbers come from

[`app/constants/hyperspectral.json`](../app/constants/hyperspectral.json) — a
file containing two lists of 165 numbers each, computed once over the whole
training corpus:

```json
{
  "mean": [0.04301138692474853, 0.043005908140582944, ...],
  "std":  [0.03523595904179349, 0.0351902150738695,   ...]
}
```

Training points at it via config:

```json
"pixel_stats_path": "/home/user57/constants/hyperspectral.json"
```

and inference finds the in-repo copy via the resolver:

```python
_HSI_STATS = "constants/hyperspectral.json"
```

---

## Worked example, using the real numbers

Band 0: `mean = 0.04301`, `std = 0.03524`.

**A typical pixel** reads 0.04301:

```
z = (0.04301 - 0.04301) / 0.03524
  = 0.00000 / 0.03524
  = 0.00
```

Exactly average. Makes sense.

**A bright pixel** reads 0.07825:

```
z = (0.07825 - 0.04301) / 0.03524
  = 0.03524 / 0.03524
  = 1.00
```

One standard deviation above typical.

**A dark pixel** reads 0.00777:

```
z = (0.00777 - 0.04301) / 0.03524
  = -0.03524 / 0.03524
  = -1.00
```

One standard deviation below.

**An invalid pixel** — this one is the interesting case. Invalid pixels were set
to exactly 0.0 before normalising:

```
z = (0.00000 - 0.04301) / 0.03524
  = -0.04301 / 0.03524
  = -1.22
```

---

## The order of operations, and why it matters

Look carefully at the model's `forward`:

```python
if mask is not None:
    x = x * mask           # step 1: invalid pixels become 0.0
if self.normalize is not None:
    x = self.normalize(x)  # step 2: everything becomes a z-score
```

Zeroing happens **before** normalising, not after. That is deliberate.

Because of the ordering, an invalid pixel does not end up at zero in the
network's input — it ends up at **-1.22**, well outside the range of typical
data. The network can recognise "this is not a measurement" as a distinct,
consistent signal.

If you normalised first and zeroed after, invalid pixels would sit at 0.0, which
in z-score space means "perfectly average" — indistinguishable from a completely
ordinary pixel. The network would have no way to tell missing data from typical
data.

> The thermal model's comment says the same thing about its own numbers:
> *"0°C -> ~ -2.3 in normalized space (clearly out-of-distribution)"*.

---

## The `register_buffer` gotcha — read this twice

There are three ways to attach a tensor to a PyTorch module:

| Method | Saved in the model file? | Trained? |
|---|---|---|
| plain attribute (`self.x = ...`) | no | no |
| `nn.Parameter` | yes | **yes** |
| **`register_buffer`** | **yes** | no |

The normalisation statistics use the third. So they:

- **travel inside the `.pt` checkpoint file**, all 660 of them
  (165 means + 165 stds, twice over for normalize and denormalize),
- move to the GPU automatically with `.to(device)`,
- never receive gradients — they are constants, not learnable.

### The consequence

**You cannot swap the normalisation statistics without rebuilding the model.**

When a checkpoint is loaded, PyTorch checks that every entry in the saved file
has a matching slot in the model you built, and it is strict about it. The
checkpoint contains `normalize.mean`, `normalize.std`, `denormalize.mean`,
`denormalize.std` — each a 165-long tensor.

So `build_model()` must construct the `PixelNormalize` layer **before** loading
weights, with the right length. If it does not, or if the length is wrong, you
get a loud shape error at load time.

Which is much better than the alternative. A silent mismatch would produce
plausible-looking but subtly wrong scores, and nobody would notice for months.

---

## PixelStatsOverride — the escape hatch

Some sensors do not ship reflectance at all.

HotSat-1 ships raw 14-bit digital numbers around **5000 ± 400**. A model whose
buffers were fitted on values around 0.04 would compute:

```
z = (5000 - 0.043) / 0.035  =  about 143,000
```

The network has never seen an input remotely like that. It would produce noise,
and the noise would drown any real anomaly.

`PixelStatsOverride`
([`app/models/training/inference_config.py`](../app/models/training/inference_config.py))
lets the caller substitute statistics computed from **this scene's own pixels**,
so the input comes out roughly centred and unit-spread as the model expects.

The scoring code builds it when the vendable's `units` string starts with
`"DN_"`:

```python
units = getattr(vendable, "units", None)
if isinstance(units, str) and units.startswith("DN_"):
    ...  # compute per-band mean/std over the kept pixels, in float64
    pixel_stats_override = PixelStatsOverride(
        mean=means, std=stds, source="per_scene_dn_zscore")
```

Two implementation details worth stealing for your own code:

**The moments are computed in float64.** Variance is normally computed as
`E[x^2] - E[x]^2`. With values around 5000, `E[x^2]` is about 25,000,000 and
`E[x]^2` is about 25,000,000 too — and the answer we want is the small
difference between them (about 160,000). In 32-bit floats, subtracting two large
nearly-equal numbers destroys most of the precision. This is called catastrophic
cancellation. Using 64-bit floats avoids it.

**A zero spread is clamped to 1.0.**

```python
if not math.isfinite(s) or s < 1e-6:
    s = 1.0
```

A scene where every kept pixel has the identical value would have `std = 0`, and
the normalisation layer would divide by zero. Rare, but it happens with test
data.

### The cost of using it

With an override, scores become **scene-relative**: comparable within that one
scene, but not against any other scene, because each was scaled by its own
statistics.

The Action records this honestly in its output, so the user interface can show a
warning:

```python
normalization_mode = "per_scene_dn_zscore"
```

### Does this apply to Indradhanu?

**No.** PRISMA, EnMAP and AVIRIS-NG all come out of `band_filter_apply` in
reflectance units that match the training distribution, so `normalization_mode`
stays `"baked"` and the checkpoint's own statistics are used.

The override path exists for HotSat-1 on the thermal side. It is covered here
because you will see the branch in the code and wonder what it is for.

---

## Common confusions

**"Is this the same as the `normalized_` in `normalized_hyperspectral_cube`?"**
No — genuine naming collision. That prefix means "cleaned up by the intake
pipeline" (units, layout, common grid). *This* normalisation is a z-score that
happens inside the model at run time.

**"Why do we denormalise at the end? Why not compare in z-score space?"**
Because the loss is defined in reflectance units, which makes it physically
interpretable (part 22). An L1 of 0.008 means "the average band is off by 0.008
reflectance", which is a statement about the world. An L1 in z-score space would
mean nothing to anyone.

**"Are `normalize` and `denormalize` sharing one copy of the numbers?"**
No — they are two separate modules, each with its own pair of buffers. That is
why the checkpoint holds 4 x 165 = 660 numbers, not 330. Part 26 uses that fact
to reconcile two different parameter counts.

---

## Check yourself

1. Write the z-score formula and its inverse.
2. Band 0 has mean 0.04301 and std 0.03524. What is the z-score of a pixel
   reading 0.11349? (Hint: it is a whole number.)
3. Why are invalid pixels zeroed *before* normalising rather than after?
4. What does `register_buffer` do that a plain attribute does not?
5. When does `PixelStatsOverride` fire, and what does it cost you?

<details>
<summary>Answers</summary>

1. `z = (x - mean) / std`, and `x = z * std + mean`.
2. `(0.11349 - 0.04301) / 0.03524 = 0.07048 / 0.03524 = 2.0`. Two standard
   deviations above typical.
3. So they land at `-mean/std` (about -1.22), which is clearly outside the
   normal range. Zeroing after normalisation would put them at 0.0, meaning
   "perfectly typical" — indistinguishable from real average data.
4. It saves the tensor inside the checkpoint and moves it with `.to(device)`,
   while keeping it untrainable.
5. When the vendable's `units` starts with `"DN_"` (currently HotSat-1). The
   cost is that scores become comparable only within that one scene.

</details>

---

**Next:** the module that makes 165 bands affordable, in
[10-spectral-compressor.md](10-spectral-compressor.md)
