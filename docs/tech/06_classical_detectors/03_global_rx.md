# 6.3 Global RX — `GlobalRXDetector`

GRX is the simplest classical detector. It computes one $\hat\mu$ and
one $\hat\Sigma$ from every valid pixel in the scene, then scores each
pixel by its squared Mahalanobis distance to that single background
distribution. It is also the canonical implementation that all the
other RX-family detectors borrow `fit()` logic from, so reading it once
saves you reading three more files.

## 6.3.1 What the code does

The whole detector lives in
[global_rx_detector.py](../../app/detectors/global_rx_detector.py) and
is around 250 lines.

- `fit()` ([global_rx_detector.py:52](../../app/detectors/global_rx_detector.py#L52))
  runs the two-stage band filter (section 6.2), builds the spatial
  mask, and stores `self._good_indices`, `self._good_wavelengths`,
  `self._spatial_mask`. No covariance is computed yet — `fit()` only
  decides *which pixels and which bands are eligible*.
- `detect()` ([global_rx_detector.py:152](../../app/detectors/global_rx_detector.py#L152))
  - Slices the cube to `_good_indices`.
  - Inside the spatial mask, band-mean-fills missing entries.
  - Flattens the result to an `(N_valid, 1, B_good)` pixel matrix.
  - Hands it to `spectral.rx`
    ([global_rx_detector.py:227](../../app/detectors/global_rx_detector.py#L227)),
    which internally computes
    $\hat\mu = \tfrac{1}{N}\sum_i x_i$,
    $\hat\Sigma = \tfrac{1}{N-1}\sum_i (x_i-\hat\mu)(x_i-\hat\mu)^{\top}$,
    and then $D(x_i) = (x_i-\hat\mu)^{\top}\hat\Sigma^{-1}(x_i-\hat\mu)$
    for every pixel against the *whole-scene* statistics.
  - The vector of scores is unflattened back into an `(H, W)` array
    with `NaN` outside the mask, and the result is wrapped in
    `GlobalRXResult`.

Under the hood, `spectral.rx` from the SPy library solves the linear
system $\hat\Sigma S = X_c^{\top}$ (with $X_c$ the centred pixel matrix)
via an LU factorisation rather than forming $\hat\Sigma^{-1}$ explicitly
— this is both faster and more numerically stable. The scores are then
the row sums of $X_c \odot S^{\top}$.

## 6.3.2 Pipeline diagram

```mermaid
flowchart TD
    A[VendableDataset] --> B[fit: band filter + spatial mask]
    B --> C[detect: slice to good bands]
    C --> D[band-mean fill inside mask]
    D --> E[flatten to N_valid x B_good]
    E --> F[spectral.rx: mu_hat, Sigma_hat]
    F --> G[solve Sigma_hat S = X_c^T]
    G --> H[D_i = row_i(X_c) dot col_i(S)]
    H --> I[unflatten to H x W, NaN outside mask]
    I --> J[GlobalRXResult]
```

## 6.3.3 Theory in plain language

GRX answers a single question: *is this pixel weird compared to the
average pixel in this scene?* The implicit model is that the scene is
mostly one big homogeneous Gaussian background. That assumption is
usually wrong — real scenes contain forest, water, roads, fields,
clouds — so $\hat\Sigma$ is a single Gaussian fit to a multi-modal
cloud of points.

In practice GRX still works, because **outliers in the literal
multi-modal sense** are exactly what we want to flag: a small bright
roof inside a field, a smoke plume inside forest, a hot pixel inside an
otherwise cool scene. The cost is that GRX over-flags rare-but-legitimate
land-cover classes too — small lakes inside forest, bright urban
fragments inside fields. The Local RX detector (section 6.4) fixes
that by replacing the global Gaussian with a *local* one.

### Failure modes worth memorising

- **Background is unimodal Gaussian** — false in any real scene;
  mitigated by MNF and especially by LRX.
- **$\hat\Sigma$ is well-conditioned** — requires $N \gtrsim 10B$
  valid pixels; the warning at
  [global_rx_detector.py:145](../../app/detectors/global_rx_detector.py#L145)
  exists for this.
- **No spatial correlation** — wrong but harmless for the score map;
  spatial correlation only affects the false-alarm-rate calibration,
  not the relative ranking of pixels.
- **Stripes flagged as anomalies** — fix with the destriper in the
  ensembler, or use an MNF variant which discards noisy components.

## 6.3.4 Worked example — the classic 3-pixel scene

Take a tiny 3-pixel, 2-band scene.

```
pixels (one row per pixel, columns are bands):
x1 = [2, 4]
x2 = [4, 6]
x3 = [3, 5]
```

Mean:

$$
\hat\mu \;=\; \tfrac{1}{3}([2,4] + [4,6] + [3,5]) \;=\; [3, 5].
$$

Centred matrix $X_c$ (rows = pixels):

```
[[-1, -1],
 [ 1,  1],
 [ 0,  0]]
```

Sample covariance with $(n-1)=2$ in the denominator:

$$
\hat\Sigma \;=\; \tfrac{1}{2}\,X_c^{\top} X_c \;=\; \tfrac{1}{2}\begin{bmatrix}2 & 2\\2 & 2\end{bmatrix} \;=\; \begin{bmatrix}1 & 1\\1 & 1\end{bmatrix}.
$$

This matrix is **singular** ($\det = 0$) — the two bands are perfectly
correlated. RX would blow up; in real code `spectral.rx` would either
return $\infty$ or rely on numerical pivoting to recover. Add a small
ridge $\lambda = 0.1$:

$$
\Sigma_\lambda = \begin{bmatrix}1.1 & 1.0\\1.0 & 1.1\end{bmatrix},\quad \det = 1.21 - 1.0 = 0.21,
$$

$$
\Sigma_\lambda^{-1} \;=\; \tfrac{1}{0.21}\begin{bmatrix}1.1 & -1.0\\-1.0 & 1.1\end{bmatrix} \;\approx\; \begin{bmatrix}5.24 & -4.76\\-4.76 & 5.24\end{bmatrix}.
$$

Now introduce a real test pixel $x^* = [5, 4]$ that was not in the
training set. Centre: $x^* - \hat\mu = [2, -1]$.

$$
\Sigma_\lambda^{-1}(x^* - \hat\mu) \;=\; \begin{bmatrix}5.24\cdot 2 + (-4.76)\cdot(-1)\\ -4.76\cdot 2 + 5.24\cdot(-1)\end{bmatrix} \;=\; \begin{bmatrix}15.24\\-14.24\end{bmatrix}.
$$

Final dot product:

$$
D(x^*) \;=\; 2\cdot 15.24 + (-1)\cdot(-14.24) \;=\; 30.48 + 14.24 \;\approx\; 44.7.
$$

The score is large because $x^*$ moves *across* the principal direction
of background variation $(1,1)$ — exactly the direction with tiny
inverse-covariance-eigenvalue. Pixels that vary *along* $(1,1)$ would
score near zero. This is the whole point of Mahalanobis: it punishes
deviation in directions the background never varies.

## 6.3.5 Second worked example — a 4-pixel scene with a real outlier

Now take four 2-band pixels where three are background and one is a
genuine outlier:

```
x1 = [10, 20]
x2 = [11, 21]
x3 = [12, 22]
x_out = [30, 5]   <- the suspect
```

We deliberately compute $\hat\mu, \hat\Sigma$ using **all four pixels**
(GRX cannot know in advance which are anomalous) and then check whether
the outlier's $D$ exceeds the $\chi^2_2$ 99th percentile, which is
$\approx 9.21$.

Step 1 — mean:

$$
\hat\mu = \tfrac{1}{4}([10,20]+[11,21]+[12,22]+[30,5]) = \tfrac{1}{4}[63, 68] = [15.75, 17.0].
$$

Step 2 — centred pixels:

```
x1 - mu = [-5.75,  3.0]
x2 - mu = [-4.75,  4.0]
x3 - mu = [-3.75,  5.0]
xo - mu = [14.25, -12.0]
```

Step 3 — sample covariance ($n-1 = 3$):

$$
\hat\Sigma_{11} = \tfrac{1}{3}((-5.75)^2 + (-4.75)^2 + (-3.75)^2 + 14.25^2) = \tfrac{1}{3}(33.06 + 22.56 + 14.06 + 203.06) = \tfrac{272.75}{3} \approx 90.92.
$$

$$
\hat\Sigma_{22} = \tfrac{1}{3}(3^2 + 4^2 + 5^2 + (-12)^2) = \tfrac{1}{3}(9 + 16 + 25 + 144) = \tfrac{194}{3} \approx 64.67.
$$

$$
\hat\Sigma_{12} = \tfrac{1}{3}\big((-5.75)(3) + (-4.75)(4) + (-3.75)(5) + (14.25)(-12)\big) = \tfrac{1}{3}(-17.25 - 19 - 18.75 - 171) = \tfrac{-226}{3} \approx -75.33.
$$

So

$$
\hat\Sigma \approx \begin{bmatrix}90.92 & -75.33\\ -75.33 & 64.67\end{bmatrix},\qquad
\det \hat\Sigma \approx 90.92\cdot 64.67 - 75.33^2 \approx 5879.9 - 5674.6 = 205.3.
$$

$$
\hat\Sigma^{-1} \approx \tfrac{1}{205.3}\begin{bmatrix}64.67 & 75.33\\ 75.33 & 90.92\end{bmatrix} \approx \begin{bmatrix}0.315 & 0.367\\ 0.367 & 0.443\end{bmatrix}.
$$

Step 4 — score the outlier with $\delta = x_{\text{out}} - \hat\mu =
[14.25, -12.0]$:

$$
\hat\Sigma^{-1}\delta \approx \begin{bmatrix}0.315\cdot 14.25 + 0.367\cdot(-12.0)\\ 0.367\cdot 14.25 + 0.443\cdot(-12.0)\end{bmatrix} = \begin{bmatrix}4.49 - 4.40\\ 5.23 - 5.32\end{bmatrix} = \begin{bmatrix}0.09\\ -0.09\end{bmatrix}.
$$

$$
D(x_{\text{out}}) \;\approx\; 14.25\cdot 0.09 + (-12.0)\cdot(-0.09) \;\approx\; 1.28 + 1.08 \;=\; 2.36.
$$

That is the famous **masking effect**: the outlier inflated the
covariance so much in its own direction that, ironically, it now scores
*below* the chi-squared threshold of 9.21. GRX has the smallest
robust-statistics breakdown point of any detector in this chapter, and
this example is the canonical illustration.

Compare: scoring one of the background pixels, say $x_1$ with
$\delta = [-5.75, 3.0]$:

$$
\hat\Sigma^{-1}\delta \approx \begin{bmatrix}0.315\cdot(-5.75)+0.367\cdot 3.0\\ 0.367\cdot(-5.75)+0.443\cdot 3.0\end{bmatrix} = \begin{bmatrix}-1.81 + 1.10\\ -2.11 + 1.33\end{bmatrix} = \begin{bmatrix}-0.71\\ -0.78\end{bmatrix}.
$$

$$
D(x_1) \approx (-5.75)(-0.71) + 3.0(-0.78) = 4.08 - 2.34 = 1.74.
$$

The outlier scores higher than $x_1$ (2.36 vs 1.74), so it *does* end up
flagged first in a ranked score map — but the absolute calibration via
$\chi^2_2$ is broken. The fix in practice is either
(a) more pixels (background outvotes the outlier), or
(b) LRX so the suspect pixel's annulus contains no other suspects, or
(c) MNF so the noise dimensions are removed.

## 6.3.6 What you should remember

GRX is the cheapest, most interpretable detector in the kit. It is the
right first thing to run on a scene because the resulting score map
tells you, at a glance, whether the scene is dominated by a single
homogeneous background (in which case GRX works well) or by multiple
land-cover classes (in which case you should switch to LRX or
MNF-LRX). Looking at the GRX map and looking at an RGB composite of
the scene side by side is the single most efficient triage you can
do before reaching for anything more sophisticated.
