# Part 12 - From residual to score

> **The one thing this part teaches:** magnitude error and shape error find different
> anomalies, and no fixed threshold works across scenes.

## Where we are

Part 10 gave a reconstruction. Subtracting it from the input gives a **residual cube** -
same shape as the input, one error value per band per pixel.

That is `(165, H, W)` for hyperspectral. A decision needs one number per pixel. Collapsing
165 into 1 is what `app/utils/anomaly_detection/scoring.py` does, and how you collapse
determines what you find.

## Two ways to be wrong

Take a pixel's true spectrum and the model's reconstruction. They can differ in two
independent ways.

**Magnitude.** Same shape, different height. The model got the brightness wrong.

**Shape.** Same overall brightness, different pattern of peaks and dips. The model got the
*material* wrong.

Three bands, worked through:

```
band          1      2      3
true       0.20   0.50   0.30
recon A    0.30   0.60   0.40      brighter by 0.10 everywhere
recon B    0.30   0.40   0.30      same total, different pattern
```

Reconstruction A is off by exactly 0.10 in every band. Its *shape* is perfect.
Reconstruction B has the same sum as the truth (1.00) but the wrong pattern - band 1 up,
band 2 down.

A magnitude metric flags A strongly and B weakly. A shape metric does the reverse.

## L1: magnitude

Mean absolute residual across bands.

```
recon A:
  |0.20 - 0.30| = 0.10
  |0.50 - 0.60| = 0.10
  |0.30 - 0.40| = 0.10
  sum           = 0.30
  L1            = 0.30 / 3 = 0.100

recon B:
  |0.20 - 0.30| = 0.10
  |0.50 - 0.40| = 0.10
  |0.30 - 0.30| = 0.00
  sum           = 0.20
  L1            = 0.20 / 3 = 0.0667
```

L1 scores A **higher** (0.100 vs 0.0667), even though A's shape is perfect. A cloud, which is
bright everywhere, scores high on L1 while being spectrally uninteresting.

## SAM: shape

**Spectral Angle Mapper** treats each spectrum as a vector and measures the angle between
them. Scaling a vector does not change its direction, so SAM is **invariant to brightness**.

That invariance is the point. Illumination, slope and shadow change a spectrum's magnitude;
they do not change its shape. SAM ignores exactly the variation you do not care about.

Compute it for reconstruction A:

```
dot(true, A)  = 0.20*0.30 + 0.50*0.60 + 0.30*0.40
              = 0.06 + 0.30 + 0.12
              = 0.48

|true| = sqrt(0.20^2 + 0.50^2 + 0.30^2)
       = sqrt(0.04 + 0.25 + 0.09)
       = sqrt(0.38)
       = 0.6164

|A|    = sqrt(0.30^2 + 0.60^2 + 0.40^2)
       = sqrt(0.09 + 0.36 + 0.16)
       = sqrt(0.61)
       = 0.7810

cos    = 0.48 / (0.6164 * 0.7810)
       = 0.48 / 0.4814
       = 0.9971

angle  = arccos(0.9971) = 0.0762 rad = 4.36 degrees
```

Now reconstruction B:

```
dot(true, B)  = 0.20*0.30 + 0.50*0.40 + 0.30*0.30
              = 0.06 + 0.20 + 0.09
              = 0.35

|B|    = sqrt(0.09 + 0.16 + 0.09) = sqrt(0.34) = 0.5831

cos    = 0.35 / (0.6164 * 0.5831)
       = 0.35 / 0.3594
       = 0.9738

angle  = arccos(0.9738) = 0.2290 rad = 13.12 degrees
```

**SAM reverses the ranking: 4.36 degrees for A, 13.12 for B.** L1 said A was worse; SAM says B
is, by a factor of three. **Do this arithmetic yourself** - it is the clearest demonstration
in the course that the choice of metric decides what you find.

## Why both are offered

| method | measures | catches | misses |
|---|---|---|---|
| `L1` | magnitude | anything unusually bright or dark | odd materials at normal brightness |
| `MSE` | magnitude, squared | as L1, more sensitive to extremes | as L1 |
| `SAM` | shape | unusual materials | anything with a normal spectrum shape |
| `combined` | both | `w * L1_norm + (1-w) * SAM_norm`, `w = 0.5` | - |

A bright cloud: large L1, small SAM. A chemically odd material at ordinary brightness: small
L1, large SAM. **Reporting both lets the operator choose which kind of anomaly matters
today.**

Each is normalised by its maximum over **valid pixels only** before combining, so the two
scales are comparable. Invalid pixels are then zeroed.

## Thresholding, and why it is a human's job

You have a score map. Which pixels are anomalies?

A fixed cut cannot work. Typical residuals differ by an order of magnitude between a calm
lake scene and a fire-affected one:

```
calm scene:      typical score 0.01,  threshold 0.05  ->  almost nothing flagged
burned scene:    typical score 0.10,  threshold 0.05  ->  most of the scene flagged
```

**The same number is both far too strict and far too loose, depending on the scene.**

So the convention is a **percentile**: "the top 1% of valid pixels in this scene". That
adapts to the scene's own distribution, which is what a human would do by eye.

Work an example. 1,000,000 valid pixels, top 0.5%:

```
flagged = 1,000,000 * 0.005
        = 5,000 pixels
```

Five thousand candidates from a million. Whatever the absolute scores were.

This is why `anomaly_detection_prep` parks at `needs_threshold` (part 3): the percentile is a
judgement call about how many candidates an analyst can review, not a property of the data.

## Measuring quality when ground truth exists

Occasionally an analyst attaches a ground-truth annotation. Then `compute_roc` sweeps
thresholds and computes true and false positive rates.

One detail matters: it sweeps **percentile-spaced** thresholds, not linear ones. Score
distributions are heavy-tailed - almost every pixel sits near zero and a handful sit far out.
Linear spacing would put nearly every sample in the empty upper range and almost none where
the decision actually happens.

## Common confusions

**"SAM in degrees or radians?"**
`_sam` returns radians. The spectral-match code converts to degrees for display, with a
default confidence threshold of 15 degrees. Check units before comparing values.

**"Why normalise by the max rather than standardising?"**
Max-normalisation puts both metrics on [0, 1] so a weighted sum is meaningful. Note this is
*different* from the rank-CDF normalisation used by `StatisticalEnsembler`, which needs it
because Mahalanobis distances from two detectors are not commensurable at all.

**"Is 'combined' always best?"**
No. It is the default for Indradhanu because it matches that model's training loss. If you
are hunting bright hot targets, plain L1 is more direct.

**"Can I use accuracy to evaluate this?"**
No - part 1 showed why. With 40 anomalies in 20 million pixels, always answering "normal"
scores 99.9998%. ROC and AUC, computed over valid pixels only.

## Check yourself

<details>
<summary>1. True spectrum (0.40, 0.20, 0.40), reconstruction (0.50, 0.30, 0.50). Compute L1.</summary>

```
|0.40 - 0.50| = 0.10
|0.20 - 0.30| = 0.10
|0.40 - 0.50| = 0.10
sum           = 0.30
L1            = 0.30 / 3 = 0.100
```
</details>

<details>
<summary>2. For the same pair, compute SAM in degrees. Comment on the result.</summary>

```
dot   = 0.40*0.50 + 0.20*0.30 + 0.40*0.50
      = 0.20 + 0.06 + 0.20 = 0.46

|t|   = sqrt(0.16 + 0.04 + 0.16) = sqrt(0.36) = 0.6000
|r|   = sqrt(0.25 + 0.09 + 0.25) = sqrt(0.59) = 0.7681

cos   = 0.46 / (0.6000 * 0.7681) = 0.46 / 0.4608 = 0.9983
angle = arccos(0.9983) = 0.0583 rad = 3.34 degrees
```

Small angle despite L1 of 0.100 - the reconstruction is uniformly brighter but the shape is
nearly right. A magnitude error, not a material error.
</details>

<details>
<summary>3. Why can a fixed absolute threshold not be used across scenes?</summary>

Typical residuals differ by an order of magnitude between scenes. A cut of 0.05 flags almost
nothing in a calm scene where typical scores are 0.01, and most of a burned scene where they
are 0.10.
</details>

<details>
<summary>4. A scene has 2,400,000 valid pixels. The analyst wants the top 0.25%. How many candidates?</summary>

```
2,400,000 * 0.0025 = 6,000 pixels
```
</details>

<details>
<summary>5. Why does compute_roc use percentile-spaced thresholds?</summary>

Score distributions are heavy-tailed - nearly every pixel sits near zero and a few sit far
out. Linear spacing would place almost every sample in the empty upper range and almost none
in the region where the decision is actually made.
</details>

---

Next: [part 13](13-end-to-end.md) - one scene, one anomaly, start to finish.
