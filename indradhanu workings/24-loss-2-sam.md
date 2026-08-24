# 24 · Loss, part 2: Spectral Angle Mapper

> **The one thing this part teaches:** SAM measures whether the predicted
> spectrum has the right *shape*, and it does not care at all about brightness.

**Source:**
[`app/foundation_models/components/sam_loss.py`](../app/foundation_models/components/sam_loss.py)
— 79 lines.

---

## Why a second measurement at all

L1 measures how wrong each band is, in reflectance units. That sounds complete.
It is not, and here is the problem.

L1 is dominated by **bright** pixels.

- A bright pixel at reflectance 0.30, off by 10%, costs `0.03`.
- A dark pixel at reflectance 0.03, off by 10%, costs `0.003`.

Same proportional error, ten times the penalty. So gradient descent spends ten
times more effort on the bright pixel. Shadowed areas, dark water and dark soil
get comparatively neglected.

But the diagnostic content of hyperspectral data is the **shape** of the
spectrum, not its brightness (part 02). Shadowed grass and sunlit grass are the
same material — same shape, different brightness. A gas plume changes the shape
at a few wavelengths without changing overall brightness much.

So we add a second term that measures shape and ignores brightness entirely.

---

## The definition

Treat each pixel's 165 numbers as a vector. The Spectral Angle is the angle
between the observed vector and the predicted one (part 08, section 3):

```
SAM(a, b) = arccos( (a . b) / (||a|| * ||b||) )        in radians
```

| Value | Meaning |
|---|---|
| 0 | identical shape, whatever the brightness |
| 0.01 to 0.10 rad | 0.6 to 5.7 degrees — a good reconstruction here |
| pi/2 ≈ 1.571 | at right angles, unrelated |
| pi ≈ 3.142 | exactly opposite |

> **Where the name comes from.** "Spectral Angle Mapper" is a standard technique
> in remote sensing, normally used to match an unknown spectrum against a library
> of known materials. Here it is repurposed as a training loss, comparing the
> observed spectrum against the reconstructed one.

---

## Worked example, in full

Same pixel as part 23, so you can compare:

```
x     = [0.100, 0.200, 0.300]
x_hat = [0.120, 0.180, 0.330]
```

### Step 1 — dot product

```
0.100 * 0.120 = 0.0120
0.200 * 0.180 = 0.0360
0.300 * 0.330 = 0.0990
                ------
       total  = 0.1470
```

### Step 2 — the two norms

```
||x||     = sqrt(0.100^2 + 0.200^2 + 0.300^2)
          = sqrt(0.0100 + 0.0400 + 0.0900)
          = sqrt(0.1400)
          = 0.374166

||x_hat|| = sqrt(0.120^2 + 0.180^2 + 0.330^2)
          = sqrt(0.0144 + 0.0324 + 0.1089)
          = sqrt(0.1557)
          = 0.394588
```

### Step 3 — the cosine

```
product of norms = 0.374166 * 0.394588 = 0.147644

cos(theta) = 0.1470 / 0.147644 = 0.995637
```

### Step 4 — the angle

```
theta = arccos(0.995637)
      = 0.0935 radians
```

In degrees:

```
0.0935 * 180 / 3.14159 = 5.35 degrees
```

### Two numbers, two meanings

For that single pixel:

```
L1  = 0.02333      the brightness is off by 0.023 reflectance on average
SAM = 0.0935 rad   the shape is off by about 5.4 degrees
```

Genuinely different aspects of the same error. Neither implies the other.

---

## The contrast case — do this one too

Now suppose the model had predicted:

```
x_hat = [0.200, 0.400, 0.600]        exactly twice the truth, band by band
```

**L1:**

```
|0.200 - 0.100| = 0.100
|0.400 - 0.200| = 0.200
|0.600 - 0.300| = 0.300
                  -----
mean            = 0.600 / 3 = 0.200
```

An L1 of 0.200. Enormous — twenty-five times worse than our earlier example.

**SAM:**

```
dot   = 0.100*0.200 + 0.200*0.400 + 0.300*0.600 = 0.02 + 0.08 + 0.18 = 0.28
||x|| = 0.374166
||x_hat|| = sqrt(0.04 + 0.16 + 0.36) = sqrt(0.56) = 0.748331
product   = 0.374166 * 0.748331 = 0.28

cos(theta) = 0.28 / 0.28 = 1.0
theta      = arccos(1.0) = 0
```

**Zero.** SAM reports a perfect match.

### What this proves

The reconstruction is twice as bright as reality in every single band, and SAM
does not notice at all. Because the shape is perfect — every band scaled by the
same factor is the same curve, just taller.

This is exactly why the model needs **both** terms. Neither alone is sufficient:

| Kind of error | caught by L1 | caught by SAM |
|---|---|---|
| brightness only (right shape, wrong scale) | yes | **no** |
| shape only (right brightness, wrong colour) | weakly | **yes** |
| both | yes | yes |

---

## Numerical stability: why the code avoids `arccos`

The formula above says `arccos`. The code does not use it. Here is why, and it
is a genuinely instructive piece of engineering.

### The problem

The derivative of arccos is:

```
d/du arccos(u) = -1 / sqrt(1 - u^2)
```

Look at what happens as `u` approaches 1:

| `u` | `sqrt(1 - u^2)` | derivative |
|---|---|---|
| 0.9 | 0.436 | -2.3 |
| 0.99 | 0.141 | -7.1 |
| 0.999 | 0.045 | -22.4 |
| 0.9999 | 0.014 | -70.7 |
| 1.0 | 0 | **infinite** |

Now recall where a *good* model lives: cosine similarity of 0.999 or better.

**So the naive implementation becomes numerically explosive precisely as
training succeeds.** You get NaNs late in a long run — the worst possible time
for it to happen.

### The fix

Rewrite the angle using `atan2` instead. From basic trigonometry:

```
||a|| ||b|| cos(theta) = a . b                                  (the dot product)
||a|| ||b|| sin(theta) = sqrt( (||a|| ||b||)^2 - (a . b)^2 )    (from sin^2 + cos^2 = 1)

theta = atan2( sin_term , cos_term )
```

`atan2` takes the two components and returns the angle, with a finite,
well-behaved derivative everywhere.

### The code, with all three guards

```python
cross_norm  = (norm_hat * norm_x).clamp(min=self.eps)
cos_term    = dot.clamp(-cross_norm, cross_norm)
sin_term_sq = (cross_norm ** 2 - cos_term ** 2).clamp(min=0)
sin_term    = (sin_term_sq + self.eps).sqrt()
angles      = torch.atan2(sin_term, cos_term)     # always in [0, pi]
```

Each guard earns its place:

**`clamp(min=eps)` on `cross_norm`.** An invalid pixel has been zeroed, so both
norms are 0 and their product is 0. Without this you divide by zero.

**`cos_term.clamp(-cross_norm, cross_norm)`.** Mathematically the dot product
can never exceed the product of the norms. In floating-point arithmetic it can,
by a hair. If it does, `sin_term_sq` goes negative and the square root produces
NaN.

**`+ eps` inside the square root.** `d/dx sqrt(x) = 1 / (2 sqrt(x))`, which is
infinite at `x = 0`. And `sin_term_sq` is exactly 0 for a perfect
reconstruction. This is the same disease as arccos, one level down — and it is
fixed the same way, by never quite reaching zero.

### Verify the fix on our example

```
cross_norm  = 0.147644
cos_term    = 0.147000

sin_term_sq = 0.147644^2 - 0.147000^2
            = 0.021799 - 0.021609
            = 0.000190

sin_term    = sqrt(0.000190) = 0.013784

theta       = atan2(0.013784, 0.147000) = 0.0935 rad
```

**0.0935** — identical to the `arccos` answer, with no gradient cliff. Same
value, safe derivative.

---

## Masking and averaging

```python
x_m  = x * mask
xh_m = x_hat * mask
...
spatial_mask = mask[:, 0:1, :, :]
num_valid    = spatial_mask.sum().clamp(min=1)
sam_loss     = (angles * spatial_mask).sum() / num_valid
```

The mask is `(B, 1, H, W)` and broadcasts across all 165 bands, so masked-out
pixels become zero vectors and contribute nothing.

`clamp(min=1)` on the denominator prevents division by zero when a batch happens
to have nothing graded — rare, but it would crash a whole training run.

Note this term **cannot** use the boolean-indexing trick that L1 uses (part 23),
because the angles keep their spatial shape. Hence the explicit
`sum() / count`.

---

## The same maths appears twice more

Two other places compute the spectral angle, and both use plain `arccos`:

```python
cos_sim = (dot / (norm_o * norm_r + 1e-8)).clamp(-1.0, 1.0)
sam = torch.acos(cos_sim).squeeze(1)
```

- `HyperspectralSegFormerMAEInferencer.compute_anomaly_scores`
- `_sam()` in
  [`app/utils/anomaly_detection/scoring.py`](../app/utils/anomaly_detection/scoring.py)
  — the numpy version the Action actually calls.

**Why is `arccos` acceptable there?** Because those run at inference, where no
gradients are computed. The infinite derivative is irrelevant if you never take
a derivative. The `clamp(-1.0, 1.0)` is still needed to keep `arccos` inside its
valid domain.

This is a good instinct to develop: **numerical stability requirements are
different in the forward and backward passes.**

---

## Common confusions

**"Is SAM a distance?"**
Not in the usual sense — it is an angle, in radians, bounded between 0 and pi.
Two spectra can be enormously far apart in magnitude and still have a SAM of
zero.

**"Would normalising each spectrum before L1 give the same effect?"**
Close, but not identical, and you would then lose the brightness information
entirely. Keeping two separate terms lets you weight them independently
(part 25).

**"Why radians and not degrees?"**
Radians are what the trigonometric functions produce. The trainer converts to
degrees only when it logs, for human readability.

**"What SAM value should I expect?"**
Early in training, 0.3 to 0.4 rad (20 to 25 degrees). Well trained, 0.02 to 0.08
rad (1 to 5 degrees). The training configuration lists these baselines
explicitly.

---

## Check yourself

1. Compute the spectral angle between `[1, 0]` and `[0, 1]`, in radians and
   degrees.
2. Why does a 2x-brighter reconstruction have a SAM of exactly zero?
3. What is wrong with the derivative of `arccos` near 1, and why does that
   matter *more* as training succeeds?
4. Name the three numerical guards in `SAMLoss` and what each prevents.
5. Why is plain `arccos` acceptable in the inference code but not in the loss?

<details>
<summary>Answers</summary>

1. `dot = 0`; both norms are 1; `cos = 0`; `arccos(0) = pi/2 = 1.571` rad =
   90 degrees.
2. Because scaling every band by the same factor produces a vector pointing in
   the same direction; the division by both norms cancels the scale entirely.
3. It is `-1/sqrt(1-u^2)`, which goes to infinity at `u = 1`. A well-trained
   model produces cosine similarities of 0.999+, so the better it gets, the
   closer it sits to the singularity — NaNs appear late in long runs.
4. `clamp(min=eps)` on the norm product prevents division by zero at invalid
   pixels; clamping `cos_term` to the norm product stops floating-point overshoot
   making `sin_term_sq` negative; `+ eps` inside the sqrt prevents the infinite
   derivative of sqrt at zero.
5. Because inference takes no gradients, so the infinite derivative never
   arises. Only the domain clamp is still needed.

</details>

---

**Next:** adding the two terms together, carefully, in
[25-loss-3-combined-and-ramp.md](25-loss-3-combined-and-ramp.md)
