# Part 8 - RX detection

> **The one thing this part teaches:** RX scores a pixel by its distance from the scene's
> average spectrum, corrected for how bands vary together - and doing this naively on 165
> bands produces numbers so wrong they are obvious.

## Start in one dimension

A thermal scene. One number per pixel. Scene mean 24.58 C, standard deviation 13.57 - these
are the real shipped values in `app/constants/thermal_pixel_consts.json`.

How unusual is a pixel at 71 C?

```
z = (71 - 24.58) / 13.57
  = 46.42 / 13.57
  = 3.42
```

Three and a half standard deviations above the mean. Unusual. That is the whole idea, and
RX is its generalisation to many bands.

## Why many bands needs more than a z-score

You could compute a z-score per band and add them up. That fails, because **bands are not
independent**.

Vegetation is dark in red and bright in near-infrared. Those two bands move together across
the scene. A pixel that is slightly bright in both is ordinary vegetation. A pixel bright in
red and *dark* in near-infrared is bizarre - even if neither value alone is extreme.

Per-band z-scores cannot see that. They treat each band separately and miss the relationship
entirely.

## Covariance, in three bands

The **covariance matrix** records how every pair of bands moves together. For three bands it
is 3x3: diagonal entries are each band's variance, off-diagonal entries say whether two bands
rise and fall together.

Build one by hand. Three bands, four pixels:

```
pixel   b1    b2    b3
  1    0.10  0.20  0.30
  2    0.12  0.24  0.28
  3    0.08  0.16  0.32
  4    0.10  0.20  0.30
```

Means:

```
b1: (0.10 + 0.12 + 0.08 + 0.10) / 4 = 0.40 / 4 = 0.100
b2: (0.20 + 0.24 + 0.16 + 0.20) / 4 = 0.80 / 4 = 0.200
b3: (0.30 + 0.28 + 0.32 + 0.30) / 4 = 1.20 / 4 = 0.300
```

Deviations from the mean:

```
pixel   d1      d2      d3
  1    0.00    0.00    0.00
  2   +0.02   +0.04   -0.02
  3   -0.02   -0.04   +0.02
  4    0.00    0.00    0.00
```

Look at pixels 2 and 3. When b1 goes up, b2 goes up and b3 goes **down**. Bands 1 and 2 are
positively correlated; both are negatively correlated with band 3. That structure is what
the covariance matrix stores, and what a per-band z-score throws away.

## The RX statistic

```
score(x) = (x - mu)^T  Sigma^-1  (x - mu)
```

- `x` is the pixel's spectrum
- `mu` is the scene's mean spectrum
- `Sigma` is the covariance matrix
- `Sigma^-1` is its inverse

Read it as: **distance from the mean, measured in units of how much the data actually varies
in that direction.** Ten units along a direction the scene varies a lot is unremarkable; ten
units along a direction it barely varies at all is extraordinary.

This is the **Mahalanobis distance**. In one dimension it collapses to the squared z-score -
`3.42^2 = 11.7` for the pixel above. RX is the multi-band version of exactly the calculation
you did at the start.

## Where it breaks, and why one technique is banned

`Sigma^-1` is the problem.

On 165 bands, `Sigma` is 165x165 - 27,225 entries estimated from the scene's valid pixels.
Neighbouring bands are highly correlated, so the matrix is close to **singular**: some
directions have almost no variance. Inverting it divides by numbers near zero, and the
resulting distances explode.

This is not theoretical. From `backend/allotrope/foundation_models/resolver.py`, plain RX on
hyperspectral was **removed on 2026-05-11** because distances reached about **1e11**. Not
merely inaccurate - meaningless. The ranking is then driven by numerical noise.

**Plain RX on hyperspectral data is banned in this codebase. Do not reintroduce it.**

## MNF, the replacement

The fix is to score in fewer, better-behaved dimensions - but choosing which dimensions
matters.

The obvious move is PCA: keep the directions of greatest variance. That is wrong here,
because the noisiest bands often have the greatest variance, so PCA cheerfully keeps the
noise.

**MNF** - Minimum Noise Fraction - orders directions by **signal-to-noise** instead:

1. Estimate the *noise* covariance from differences between neighbouring pixels. Adjacent
   pixels usually show the same ground, so their difference is mostly noise.
2. Whiten by that noise estimate, so noise becomes equal in all directions.
3. Take the directions of greatest variance *in that whitened space*. Now greatest variance
   really does mean greatest signal.

Keep the top 10 (`n_components=10`), and run RX there. 10x10 covariance from millions of
pixels is comfortably estimated, and the inverse is stable.

**`MNFCompressionDetector` is the one to use on hyperspectral data.**

## The detector roster

All in `app/detectors/`, all returning an `(H, W)` score map with `NaN` at invalid pixels.

| detector | background | notes |
|---|---|---|
| `GlobalRXDetector` | whole scene | fine on thermal, banned on raw hyperspectral |
| `ThermalGRXDetector` | whole scene | one band, so RX collapses to a squared z-score |
| `LocalRXDetector` | a ring per pixel | part 9 |
| `MNFCompressionDetector` | whole scene, in MNF space | **the hyperspectral choice** |
| `MNFCompressionLRXDetector` | ring, in MNF space | |
| `StatisticalEnsembler` | - | combines global and local; not registered, notebooks only |

## The eigenvalue check

`MNFCompressionRXResult` carries `mnf_eigenvalues`, and they are a diagnostic worth reading.

The spectrum should **fall away sharply** - a few components carrying the structure. Flat
eigenvalues mean the compression found no dominant signal, and the scores that follow deserve
suspicion. That is why `visualize()` plots them beside the score map instead of hiding them.

## Common confusions

**"Is RX machine learning?"**
No. It is descriptive statistics computed from the scene in front of it. No training, no
checkpoint, no labels. Hand it a new sensor and it works immediately.

**"Why not just regularise the covariance instead of doing MNF?"**
A ridge term is used, in the local variants (part 9). It keeps the solve from failing but
does not restore meaning to 165 near-degenerate dimensions. MNF addresses the cause.

**"MNF is just PCA, isn't it?"**
No, and the difference is the point. PCA maximises variance; MNF maximises signal-to-noise.
Where the noisiest bands are also the highest-variance, PCA keeps exactly the wrong
directions.

**"Estimating noise from neighbouring-pixel differences seems crude."**
It is an assumption - that adjacent pixels usually show similar ground, so their difference
is mostly noise. It fails on hard edges like a coastline. It holds well enough across a
scene that the estimate is useful, and it needs no calibration data.

## Check yourself

<details>
<summary>1. A thermal pixel reads 61 C. Scene mean 24.58, std 13.57. Compute the z-score and the equivalent one-band RX score.</summary>

```
z   = (61 - 24.58) / 13.57
    = 36.42 / 13.57
    = 2.684

RX  = z^2
    = 2.684 * 2.684
    = 7.20
```

One-band RX is the squared z-score, which is why `ThermalGRXDetector` short-circuits rather
than calling a matrix routine.
</details>

<details>
<summary>2. Why can per-band z-scores not detect a pixel bright in red and dark in near-infrared?</summary>

Neither value is individually extreme, so each band's z-score is unremarkable. What is
extreme is the *combination*, because those bands normally move together. Only the
covariance's off-diagonal terms encode that.
</details>

<details>
<summary>3. Why was plain RX banned on hyperspectral data, and what replaced it?</summary>

A 165x165 covariance from correlated bands is near-singular; inverting it divides by
near-zero and distances reached about 1e11. MNF-RX replaced it, running RX in about 10
noise-whitened dimensions.
</details>

<details>
<summary>4. From the four-pixel table, compute the covariance between bands 1 and 2 using the (n-1) convention.</summary>

```
products of deviations:
  pixel 1:  0.00 *  0.00 =  0.0000
  pixel 2: +0.02 * +0.04 = +0.0008
  pixel 3: -0.02 * -0.04 = +0.0008
  pixel 4:  0.00 *  0.00 =  0.0000
                           --------
  sum                     =  0.0016

cov(b1,b2) = 0.0016 / (4 - 1)
           = 0.0016 / 3
           = 0.000533
```

Positive, so the two bands rise and fall together - as the deviation table showed.
</details>

<details>
<summary>5. A colleague reports MNF eigenvalues that are nearly flat. What does that mean?</summary>

The compression found no dominant signal directions, so the top 10 components are not
meaningfully better than any others. Scores from that run should be treated with suspicion -
the assumption that a few components carry the structure did not hold.
</details>

---

Next: [part 9](09-local-rx.md) - scoring a pixel against its neighbours.
