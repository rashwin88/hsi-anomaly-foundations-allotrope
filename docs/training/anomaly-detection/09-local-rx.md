# Part 9 - Local RX

> **The one thing this part teaches:** comparing a pixel to its immediate neighbours instead
> of to the whole scene finds different anomalies - and costs a million matrix inversions.

## What global RX misses

Global RX compares every pixel to one scene-wide mean. That has a blind spot.

A scene is half cool forest, half warm bare ground. A hot patch of machinery sits in the
bare ground. Its temperature is high - but the scene-wide mean already sits between forest
and bare ground, and the standard deviation is inflated by that split. The machinery may be
only one or two standard deviations from the global mean.

Compare it instead to the twenty metres around it, all bare ground at a consistent
temperature, and it stands out immediately.

**Global RX asks "is this pixel unusual for the scene?". Local RX asks "is this pixel
unusual for its surroundings?".** Both are real questions with different answers.

## The annulus

For each pixel, the background is a ring - a square outer window with a square inner window
excluded.

```
outer_window = 25          inner_window = 5
+-------------------------+
|  b b b b b b b b b b b  |   b = background
|  b b b b b b b b b b b  |   . = guard, excluded
|  b b b . . . . . b b b  |   T = the pixel under test
|  b b b . . . . . b b b  |
|  b b b . . T . . b b b  |
|  b b b . . . . . b b b  |
|  b b b . . . . . b b b  |
|  b b b b b b b b b b b  |
|  b b b b b b b b b b b  |
+-------------------------+
```

Count the background pixels:

```
outer square = (2 * 25 + 1)^2 = 51 * 51 = 2601
inner guard  = (2 *  5 + 1)^2 = 11 * 11 =  121
                                          ----
background   = 2601 - 121             =  2480
```

**2,480 background pixels per scored pixel.** Do that arithmetic - it explains the cost
discussed below.

## Why the guard window exists

The inner window is the subtle part, and skipping it quietly ruins the detector.

If a target is larger than one pixel - and real targets usually are - then without a guard,
its neighbours are *also* target. Those neighbours go into the background statistics. The
mean shifts toward the target and the covariance inflates along exactly the direction that
makes the target unusual.

**The target partially whitens itself out of detection.** It contaminates the very statistics
meant to describe what surrounds it.

The guard excludes a margin so the background is genuinely background. Default 5, meaning an
11x11 region around the target is skipped.

## The cost, and why it is batched

Global RX computes one covariance and one inverse for the whole scene. Local RX needs one
**per pixel**.

A PRISMA scene is about 1,210 x 1,219:

```
pixels = 1210 * 1219 = 1,474,990
```

Roughly 1.47 million covariance matrices, 1.47 million inverses, 1.47 million quadratic
forms. Done one at a time in numpy, Python's per-call overhead dominates and it is hopeless.

The answer is batching: stack thousands of pixels into one `(N, B, B)` tensor and let torch
do them together. `app/detectors/_local_background.py`.

## Ragged backgrounds and the padding trick

Not every pixel has a full annulus. Near a scene edge, or beside a cloud-masked region, some
of the ring is missing.

Tensors need rectangular shapes. So every pixel's background is padded to the widest case,
and a companion array records how much of each row is real.

Work a two-pixel example with `max_bg = 4`, where pixel 0 has 4 real neighbours and pixel 1
has 2:

```
n = [4, 2]

mask = arange(4) < n[:, None]

pixel 0:  [0,1,2,3] < 4  ->  [True,  True,  True,  True ]
pixel 1:  [0,1,2,3] < 2  ->  [True,  True,  False, False]
```

Now the essential detail. The mean divides by the **real** count, not by `max_bg`:

```
counts = mask.sum(axis=1)  ->  [4, 2]
mu     = (X * mask).sum(axis=1) / counts
```

Suppose pixel 1's two real values are 0.30 and 0.34, padded with two zeros:

```
correct:  (0.30 + 0.34 + 0 + 0) / 2 = 0.64 / 2 = 0.320
wrong:    (0.30 + 0.34 + 0 + 0) / 4 = 0.64 / 4 = 0.160
```

Dividing by `max_bg` halves it. **Every edge pixel would get a corrupted background mean, and
edge pixels are exactly where you least want spurious detections.** The centred data is
masked again for the same reason, so padding contributes nothing to the covariance either.

## The ridge term

```python
cov += reg * torch.eye(B, device=device, dtype=dtype)
```

`DEFAULT_REGULARIZATION = 1e-4` in `local_rx_detector.py`.

A local background is small. With 2,480 samples and a 10-dimensional MNF cube, fine. But near
an edge the real count can fall to a few dozen, and a covariance estimated from fewer samples
than it has dimensions is singular. The ridge keeps the solve well posed.

**Note:** a since-deleted document claimed `1e-3`. The code says `1e-4`. When source and prose
disagree, the source wins.

## Two masks, and the one you must reduce over

`MNFCompressionLRXResult` carries both:

- `spatial_mask` - pixels **eligible** for scoring: valid, in-swath, kept
- `computed_mask` - pixels that actually **got** a score

They differ because a local background can fail on a perfectly good pixel. If the annulus
holds fewer valid neighbours than `min_bg_pixels`, the covariance is unusable and the pixel
goes unscored.

**Always reduce over `computed_mask`.** Treating an unscored pixel as a zero score silently
biases every statistic - and since unscored pixels cluster near edges and cloud, the bias is
systematic rather than random.

## Precision differs by hardware

```python
dtype = torch.float32 if device.type != "cpu" else torch.float64
```

float32 on GPU for speed, float64 on CPU for precision. **The same scene can therefore produce
slightly different scores on different hardware.** Far below any threshold anyone sets, but it
does mean a test asserting exact equality would pass on a laptop and fail on the GPU box.

## Common confusions

**"Why not always use local RX, if it is more sensitive?"**
It is far more expensive, and it misses anomalies that are large relative to the window. If
the target fills the whole annulus, it becomes its own background and vanishes. Global and
local fail differently, which is why `StatisticalEnsembler` fuses them.

**"Is the annulus square or circular?"**
Square. Two nested squares, the inner one excluded. Circular would be marginally more
principled and considerably slower to index.

**"What if a target is bigger than the guard window?"**
Then it contaminates its own background and detection degrades. The guard is a compromise,
not a guarantee - which is an argument for running global RX as well.

**"min_bg_pixels defaults to what?"**
Number of bands plus one - the minimum for a non-degenerate covariance. On a 10-component
MNF cube that is 11.

## Check yourself

<details>
<summary>1. Compute the background pixel count for outer_window=15, inner_window=3.</summary>

```
outer = (2*15 + 1)^2 = 31 * 31 = 961
inner = (2*3  + 1)^2 =  7 *  7 =  49
                                ----
background                     = 912
```
</details>

<details>
<summary>2. Explain what goes wrong without a guard window, in terms of the statistics.</summary>

A target larger than one pixel puts its own neighbours into the background. The background
mean shifts toward the target and the covariance inflates along the direction that makes it
unusual, so the Mahalanobis distance shrinks. The target partially whitens itself out.
</details>

<details>
<summary>3. A pixel has 3 real background neighbours padded to 6, with values 0.20, 0.22, 0.24. Compute the correct mean and the mean you would get by dividing by 6.</summary>

```
sum      = 0.20 + 0.22 + 0.24 = 0.66

correct  = 0.66 / 3 = 0.220
wrong    = 0.66 / 6 = 0.110
```

Exactly half, because half the slots are padding. The masked count is what prevents this.
</details>

<details>
<summary>4. Why must statistics be reduced over computed_mask rather than spatial_mask?</summary>

`spatial_mask` marks pixels eligible for scoring; `computed_mask` marks those that actually
got one. A pixel with too few valid neighbours is eligible but unscored. Reducing over the
wrong mask treats unscored pixels as zero, and since they cluster near edges and cloud the
resulting bias is systematic.
</details>

<details>
<summary>5. A test asserts local RX scores are exactly equal to a stored fixture. Why is that fragile?</summary>

The compute dtype depends on the device - float32 on GPU, float64 on CPU - so results differ
in the last bits across hardware. The test would pass where it was written and fail on the
GPU box. Assert a tolerance, or a ranking.
</details>

---

Next: [part 10](10-reconstruction-models.md) - teaching a network what normal looks like.
