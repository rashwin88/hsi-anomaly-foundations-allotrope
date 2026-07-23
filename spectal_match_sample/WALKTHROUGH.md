# `spectal_match_sample` — End-to-End Walkthrough

A from-scratch tour of how this package identifies materials at anomaly pixels by matching their spectra against the USGS splib07 lab library using Spectral Angle Mapper (SAM).

Math first, code second.

---

## Chapter 0 — What is a spectrum, physically?

When sunlight hits a material and bounces back to a sensor, the material doesn't reflect every wavelength equally. Chlorophyll absorbs red and blue, reflects green (plants look green). Iron oxides absorb in the blue. Water absorbs strongly around 1400 nm and 1900 nm. Every material has a **fingerprint**: a curve of *how much light it reflects at each wavelength*.

Formally, **reflectance** $\rho(\lambda)$ is a dimensionless number in $[0, 1]$:

$$
\rho(\lambda) = \frac{\text{radiance reflected}}{\text{radiance incident}}
$$

Plot $\rho$ against $\lambda$ and you get a curve. That curve **is** the spectrum. Two materials that look identical to the human eye (which has only 3 broad bands: R, G, B) can have wildly different spectra when sampled at 200 wavelengths. That's the whole point of **hyperspectral** imaging.

A hyperspectral sensor like PRISMA doesn't measure the continuous curve. It measures **discrete samples** of it: ~240 bands, each reporting a single number that is a *weighted average* of reflectance over a narrow wavelength window. One pixel becomes a vector:

$$
\mathbf{u} = [\rho_1, \rho_2, \ldots, \rho_B] \in \mathbb{R}^B
$$

Hold that picture: **a pixel is a vector in $\mathbb{R}^B$, and similar materials have similar vectors.**

---

## Chapter 1 — The matching problem

You have:

- **Unknown spectra:** $P$ pixels flagged as anomalies. Each is a vector $\mathbf{u}_p \in \mathbb{R}^B$ measured by the spaceborne sensor.
- **Library spectra:** $N$ lab-measured materials (USGS splib07 has ~2400: minerals, vegetation, soils, artificial materials). Each is a vector $\boldsymbol{\ell}_n$.

**Goal:** for each $\mathbf{u}_p$, find the library entry $\boldsymbol{\ell}_n$ most similar to it. Output: *"pixel #5 looks like Calcite WS272 with angle 3.1°."*

Two problems stand between us and that goal:

1. **The library and the sensor live on different wavelength grids.** splib07 is sampled at ~2151 wavelengths from 350–2500 nm at ~5 nm spacing. PRISMA has ~240 bands at ~10 nm spacing in the same range. You can't dot-product vectors of different lengths. → Chapter 2 fixes this.
2. **What does "most similar" even mean?** Closest in Euclidean distance? Highest correlation? Each choice has different physical meaning. → Chapter 3 fixes this.

---

## Chapter 2 — Getting the library onto the sensor's grid

### 2.1 What is the sensor actually doing?

To understand why interpolation is wrong, we first have to be precise about what a PRISMA "band" physically measures. People casually say "band 50 is at 650 nm," and that phrase quietly hides the whole problem.

**The setup inside the instrument.** Light arrives at the spacecraft. It passes through a narrow slit, then a dispersive element (prism or grating) that spreads incoming light out by wavelength — like a spectrograph. The dispersed light lands on a strip of detector pixels. Pixel #50 is positioned where the optics deposit light around 650 nm. But the optics aren't a perfect knife edge:

- The slit has finite width (so wavelengths near 649 nm and 651 nm still partially land on pixel #50).
- The dispersing element has finite resolution (the "spread" overlaps neighbors).
- The detector pixel itself has finite width (it integrates whatever lands on it).

So pixel #50 sees a *blurred range* of wavelengths, not just 650 nm. We describe this blurring by a function

$$
S_{50}(\lambda)
$$

called the **Spectral Response Function (SRF)** of band 50. $S_{50}(\lambda)$ tells you: *"if a photon arrives at wavelength $\lambda$, what fraction of it ends up contributing to band 50's reading?"* It peaks at 650 nm (~maximum sensitivity), falls off on either side, and is approximately zero beyond ~±10 nm. Empirically, this shape is well-fit by a Gaussian (see §2.2).

**What the band actually reports.** The total signal pixel #50 collects is the sum (integral) over all wavelengths of (incoming reflectance) × (this band's sensitivity at that wavelength):

$$
\text{signal}_{50} = \int \rho(\lambda)\, S_{50}(\lambda)\, d\lambda
$$

To turn this back into a reflectance number (something in $[0,1]$), we divide by the total area of the SRF — which is the signal you'd get from a perfectly-flat reflectance of 1:

$$
\boxed{\rho_{50}^{\text{measured}} \;=\; \frac{\int \rho(\lambda)\, S_{50}(\lambda)\, d\lambda}{\int S_{50}(\lambda)\, d\lambda}}
$$

This is a **weighted average of $\rho(\lambda)$**, with weights given by the SRF. That's all the formula says: "the band reading is the SRF-weighted average of true reflectance over wavelengths." If the SRF were a Dirac delta at 650 nm, the band would report exactly $\rho(650)$ — but no real instrument has that.

### 2.2 What interpolation does, and why it's wrong

Now the library problem. USGS splib07 stores reflectance sampled at ~2151 discrete wavelengths (about every 5 nm from 350 to 2500 nm). Call these samples $\{\lambda_j^{\text{lib}}, r_j\}$. PRISMA gives us a different grid: ~240 band centers $\{\lambda_i\}$ at ~10 nm spacing. To compare library against sensor, we need both on the *same* grid.

**The naive (wrong) approach — linear interpolation.** For PRISMA band $i$ at 650 nm:

1. Find the two splib07 samples that bracket 650 nm — say they're at 648.7 nm and 653.2 nm with values $r_{j}=0.42$ and $r_{j+1}=0.38$.
2. Draw a straight line between those two points.
3. Read off the line's $y$-value at $\lambda = 650$ nm. That becomes the "library value at band 50."

In formula form, with $\alpha = (650 - 648.7)/(653.2 - 648.7) \approx 0.29$:

$$
\rho_{\text{interp}}(650) = (1-\alpha)\cdot 0.42 + \alpha \cdot 0.38 \;\approx\; 0.409
$$

**Why this is wrong.** Linear interpolation answers the question *"what is reflectance exactly at 650 nm?"* — but that is **not** what the sensor measures. The sensor measures the SRF-weighted average over a ~10 nm window. Interpolation only uses two library samples (the bracketing pair) and gives them weights based on distance to 650 nm. It completely ignores all the other library samples in the window — 645 nm, 655 nm, 658 nm — that the *real sensor would absolutely see*.

**A concrete failure case.** Suppose calcite has a sharp absorption dip at 648 nm where reflectance drops to 0.15, but at the bracketing samples (648.7 nm and 653.2 nm) reflectance is back up to 0.42 and 0.38. Linear interp gives 0.409 — as if the absorption didn't exist. But PRISMA's band 50 has an SRF that peaks at 650 nm and *still has significant weight* at 648 nm — maybe 60% of peak. The real sensor partially "sees" that dip and would report something like 0.34. **Linear interpolation has missed a physically real feature that the actual sensor would record.**

The reverse failure case is just as bad: linear interp at exactly 650 nm could *land on* a sharp peak that the real sensor — which averages over a window — would smooth out. Either way, the resampled library doesn't look like what the sensor would actually report.

This is not a small effect. Minerals (the most useful splib07 chapter) are full of sharp absorption features 2–5 nm wide. The whole point of using splib07 for matching is that those features are diagnostic. Interpolation throws them away.

### 2.3 The right answer: simulate the sensor

The fix is **don't ask the library "what is your value at 650 nm?"** — instead ask **"what would PRISMA band 50 *read* if you pointed it at this library material?"** Those are different questions, and only the second one gives you something you can compare against an actual sensor measurement.

To answer the second question, we just imitate what the sensor does. The sensor doesn't sample $\rho$ at one wavelength — it collects photons across a ~10-nm window, with each wavelength weighted by the band's sensitivity at that wavelength (the SRF). So if we have the library spectrum, we do exactly the same thing: we take a weighted average of library reflectances across the same window, with the same weights.

#### A worked example, side by side

Imagine PRISMA band 50 centered at $\lambda_{50} = 650$ nm with FWHM = 10 nm (so $\sigma \approx 4.25$ nm — see §2.5 for that conversion).

Suppose splib07 has the following calcite samples nearby. (Real splib07 spacing is ~5 nm; I've used 1–2 nm spacing here so we can actually see the absorption dip.)

| $j$ | $\lambda_j^{\text{lib}}$ (nm) | $r_j$ (reflectance) |
|---|---|---|
| ... | 642 | 0.41 |
| | 645 | 0.42 |
| | **648** | **0.15**  ← the absorption dip |
| | 650 | 0.30 |
| | 652 | 0.39 |
| | 655 | 0.40 |
| ... | 658 | 0.38 |

**Method A — linear interpolation at 650 nm.** Look at the two samples that bracket 650: $\lambda=648, r=0.15$ and $\lambda=650, r=0.30$. (650 actually *is* a sample here, so it just reads off as 0.30.) If we shifted the band center slightly to 649 nm we'd get $0.5\cdot 0.15 + 0.5 \cdot 0.30 = 0.225$. Either way, the only library samples that contributed to the answer are the immediate neighbors. Everything else in the table is **ignored**.

**Method B — simulate the sensor.** Now we ask: if PRISMA band 50 were pointed at a material with this reflectance curve, what would it read? We compute the SRF (Gaussian centered at 650 nm, $\sigma = 4.25$ nm) at each library sample wavelength. The Gaussian formula is

$$
w_j = \exp\!\left(-\tfrac{1}{2}\!\left(\tfrac{\lambda_j - 650}{4.25}\right)^{\!2}\right)
$$

Plug in each library wavelength:

| $\lambda_j$ | $(\lambda_j - 650)/\sigma$ | $w_j = e^{-x^2/2}$ | $r_j$ | $w_j \cdot r_j$ |
|---|---|---|---|---|
| 642 | $-1.88$ | $0.171$ | 0.41 | $0.0701$ |
| 645 | $-1.18$ | $0.502$ | 0.42 | $0.2108$ |
| 648 | $-0.47$ | $0.896$ | 0.15 | $0.1344$ |
| 650 | $\phantom{-}0$ | $1.000$ | 0.30 | $0.3000$ |
| 652 | $\phantom{-}0.47$ | $0.896$ | 0.39 | $0.3494$ |
| 655 | $\phantom{-}1.18$ | $0.502$ | 0.40 | $0.2008$ |
| 658 | $\phantom{-}1.88$ | $0.171$ | 0.38 | $0.0650$ |
| **sum** | | **$4.138$** | | **$1.331$** |

Now apply the formula from [resample.py:116-124](resample.py#L116-L124):

$$
\rho_{50}^{\text{target}} = \frac{\sum_j w_j r_j}{\sum_j w_j} = \frac{1.331}{4.138} \approx 0.322
$$

**Compare the two answers.**

- Linear interpolation: **0.30** (just the value sitting at $\lambda = 650$).
- SRF resampling: **0.322** (sees the dip at 648 *and* the recovery at 652).
- What the real sensor would actually measure, in physical reality: **~0.322**, because the real sensor's optics do exactly this weighted averaging.

The SRF answer is slightly *higher* than the interp answer because we're at the edge of a dip: most of the band's sensitivity is over wavelengths where reflectance is normal (~0.40), and only some of it overlaps the 648 nm pothole. The interp answer missed that because it only saw two samples.

Now imagine instead we wanted band 49 at $\lambda = 648$ nm (right *on* the dip). Linear interp would read $r = 0.15$ exactly — claiming the sensor sees a deep absorption. SRF resampling would mix that 0.15 with the surrounding 0.40-ish values weighted by a Gaussian centered at 648, and report something like 0.28 — a partial dip, which is what the real sensor would actually see, because the sensor doesn't have infinite spectral resolution either. **The SRF approach matches reality in both directions: it doesn't fabricate features the sensor can't resolve, and it doesn't erase ones the sensor would partially see.**

#### Why this *is* the formula in the code

You just performed the integral. The continuous measurement equation from §2.1 was

$$
\rho_i^{\text{target}} = \frac{\int \rho(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda}
$$

We don't have $\rho(\lambda)$ as a continuous function — only $M=2151$ discrete samples $\{\lambda_j, r_j\}$. So we approximate the integral as a sum over those samples:

$$
\rho_i^{\text{target}} \;\approx\; \frac{\sum_j r_j \, S_i(\lambda_j^{\text{lib}})}{\sum_j S_i(\lambda_j^{\text{lib}})}
$$

Writing $w_{ij} = S_i(\lambda_j^{\text{lib}})$ — the SRF of band $i$ evaluated at library wavelength $j$, exactly what we just computed in the table — gives the formula implemented in [resample.py:116-124](resample.py#L116-L124):

$$
\rho_i^{\text{target}} = \frac{\sum_j w_{ij}\, r_j}{\sum_j w_{ij}}
$$

Everything the code does is just running that table at scale: ~10 library samples × 240 PRISMA bands × 2400 library entries.

**Interpolation vs. SRF resampling — recap.**

| | Linear interpolation | SRF resampling |
|---|---|---|
| How many library samples used? | Exactly 2 (the bracketing pair) | All samples within the SRF window (~10) |
| What weights does each get? | Distance-to-center in $\lambda$ | The sensor's actual sensitivity at that $\lambda$ |
| What question does it answer? | "What is $\rho$ exactly at $\lambda_i$?" | "What would the sensor read on this material?" |
| Behaviour near a 2–5 nm absorption feature | Either misses it or lands on it exactly — both wrong | Partial dip, matching what the real sensor sees |
| Matches what the real sensor does? | No | Yes (by construction) |

### 2.4 Why this is called "convolution"

If you've seen the word *convolution* and want to connect it to the formula above: convolution of a signal $\rho$ with a kernel $S$ is defined as

$$
(\rho * S)(\lambda_i) = \int \rho(\lambda)\, S(\lambda_i - \lambda)\, d\lambda
$$

— exactly: slide the kernel $S$ along $\lambda$, multiply with $\rho$, integrate. Our SRF $S_i(\lambda)$ is the kernel re-centered to $\lambda_i$, so the resampling formula is literally one tap of a convolution, evaluated at the band center. People call this "convolving the library with the SRF" because if you did it at every $\lambda$ you'd get the convolution; we only do it at the band centers we care about.

No actual `np.convolve` call is needed — we evaluate the Gaussian directly at each library sample wavelength (§2.5 covers that mechanic). But mathematically, this *is* a convolution.

### 2.5 FWHM, sigma, and the Gaussian SRF

Sensor specs typically give you **Full Width at Half Maximum (FWHM)**: the width of the SRF at half its peak height. For PRISMA in the VNIR range, FWHM ≈ 10 nm.

A Gaussian

$$
S(\lambda) \propto \exp\!\left(-\frac{(\lambda-\lambda_i)^2}{2\sigma^2}\right)
$$

reaches half its peak when $(\lambda-\lambda_i)^2 / (2\sigma^2) = \ln 2$, i.e. at $|\lambda - \lambda_i| = \sigma\sqrt{2\ln 2}$. The full width between those two points is twice that:

$$
\text{FWHM} = 2\sqrt{2\ln 2}\,\sigma \;\;\approx\;\; 2.355\,\sigma
$$

Inverting:

$$
\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\text{FWHM}}{2.355}
$$

This is exactly the constant precomputed at [resample.py:42](resample.py#L42):

```python
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
```

### 2.6 The discrete resampling formula, fully specified

Plugging the Gaussian form of $S_i$ into the weight definition $w_{ij} = S_i(\lambda_j^{\text{lib}})$ gives us the full formula used in the code. For target band $i$:

$$
\rho_i^{\text{target}} = \frac{\sum_j w_{ij}\, r_j}{\sum_j w_{ij}}, \qquad w_{ij} = \exp\!\left(-\frac{(\lambda_j^{\text{lib}} - \lambda_i)^2}{2\sigma_i^2}\right)
$$

The denominator normalizes the weights so the result is a true weighted average (flat input → flat output). The constant prefactor $1/(\sigma\sqrt{2\pi})$ that you'd see in a probability-density Gaussian cancels between numerator and denominator, so we don't bother computing it.

### 2.7 The 3-sigma trick

A Gaussian has infinite support in principle, but **99.7% of its mass lies within ±3σ of the center**. Beyond that, weights are < 0.011 of peak — round-off noise. So we only sum over library samples in $[\lambda_i - 3\sigma_i,\; \lambda_i + 3\sigma_i]$.

Two reasons this matters:

- **Correctness:** doesn't change the answer meaningfully.
- **Speed:** ~10 weights per band instead of 2151. With 240 bands × 2400 library entries, roughly a 200× speedup.

Because `lib_wl` is sorted, we find the window in **$O(\log M)$** with binary search ([resample.py:104-105](resample.py#L104-L105)):

```python
lo = np.searchsorted(lib_wl, lam_i - half_window, side="left")
hi = np.searchsorted(lib_wl, lam_i + half_window, side="right")
```

`lo` is the first index where `lib_wl[lo] >= lam_i - 3σ`; `hi` is one past the last index where `lib_wl[hi-1] <= lam_i + 3σ`. The slice `lib_wl[lo:hi]` is the ~10-element window.

### 2.8 Edge cases the code handles

- If the window has fewer than `min_lib_points_per_band` samples (default 3), output is `NaN` — we don't trust an average of 1–2 numbers.
- If some library samples are `NaN` (USGS no-data sentinel `-1.23e+34` → NaN earlier), they get zero weight:
  ```python
  w_eff = w * finite_slice
  ```
  Both numerator and denominator drop those terms — equivalent to dropping them from the sum.

`resample.py` computes one library spectrum's worth of resampled values. `library.py` calls it ~2400 times, once per file.

---

## Chapter 3 — Measuring similarity: Spectral Angle Mapper

Both unknown and library spectra now live in $\mathbb{R}^B$ with the same band centers. How do we say which library entry is most similar to a given unknown?

### 3.1 Why Euclidean distance is wrong

Imagine two pixels of *the same material* — same calcite outcrop — but one is in shadow, one is in direct sun. Their spectra have the **same shape** but the shadowed one is uniformly scaled down by ~0.3×. Their Euclidean distance $\|\mathbf{u} - \boldsymbol{\ell}\|$ is huge. But they're the same material.

Same problem hits cross-platform matching: lab spectra (splib07) are measured under controlled illumination and tend to have higher amplitude. Spaceborne reflectance after atmospheric correction has lower amplitude and residual atmospheric distortion. **The shape matches; the amplitude doesn't.** We want a similarity metric that *ignores amplitude* and only cares about shape.

### 3.2 Cosine similarity → angle

Two vectors $\mathbf{u}, \boldsymbol{\ell}$ in $\mathbb{R}^B$ define an angle via the dot product:

$$
\cos\theta = \frac{\mathbf{u} \cdot \boldsymbol{\ell}}{\|\mathbf{u}\|\,\|\boldsymbol{\ell}\|}
$$

This is **amplitude-invariant by construction**: scaling $\mathbf{u}$ by any positive $\alpha$ multiplies numerator and denominator by $\alpha$, and they cancel. So shadowed-calcite and sunlit-calcite have $\theta = 0$ — perfect match.

For non-negative vectors (reflectance is always ≥ 0), all components live in the positive orthant, so $\theta \in [0, \pi/2]$, i.e. $[0°, 90°]$. **0° = identical shape; 90° = orthogonal (totally different).**

That's the entire definition of SAM:

$$
\theta_{p,n} = \arccos\!\left(\frac{\mathbf{u}_p \cdot \boldsymbol{\ell}_n}{\|\mathbf{u}_p\|\,\|\boldsymbol{\ell}_n\|}\right)
$$

The literature reports angles in **degrees** (more interpretable; 3° = tight match, 15° = loose), which is why [sam.py:87-88](sam.py#L87-L88) does `np.rad2deg(np.arccos(...))`.

### 3.3 A worked example with real numbers

Let's do this with one unknown and two library candidates, using just 5 bands so we can see every number. Pretend bands are centered at 500, 600, 700, 800, 900 nm.

**The data.**

- Unknown pixel $\mathbf{u}$ (the anomaly we want to identify), measured by a spaceborne sensor:
  $$\mathbf{u} = [0.10,\; 0.12,\; 0.08,\; 0.18,\; 0.20]$$
- Library candidate A — *Calcite* (lab-measured, brighter overall but same shape as a sun-lit calcite pixel):
  $$\boldsymbol{\ell}_A = [0.40,\; 0.48,\; 0.32,\; 0.72,\; 0.80]$$
- Library candidate B — *Vegetation* (very different shape: low in visible, jumps in NIR — the "red edge"):
  $$\boldsymbol{\ell}_B = [0.05,\; 0.06,\; 0.08,\; 0.55,\; 0.60]$$

Notice $\boldsymbol{\ell}_A$ is exactly $4 \times \mathbf{u}$ — same shape, 4× the amplitude. This is the shadow-vs-sun scenario from §3.1.

**Step 1 — Compute the dot products.**

$$
\mathbf{u} \cdot \boldsymbol{\ell}_A = 0.10\cdot 0.40 + 0.12\cdot 0.48 + 0.08\cdot 0.32 + 0.18\cdot 0.72 + 0.20\cdot 0.80
$$
$$
= 0.040 + 0.0576 + 0.0256 + 0.1296 + 0.160 = 0.4128
$$

$$
\mathbf{u} \cdot \boldsymbol{\ell}_B = 0.10\cdot 0.05 + 0.12\cdot 0.06 + 0.08\cdot 0.08 + 0.18\cdot 0.55 + 0.20\cdot 0.60
$$
$$
= 0.005 + 0.0072 + 0.0064 + 0.099 + 0.120 = 0.2376
$$

**Step 2 — Compute the norms.**

$$
\|\mathbf{u}\| = \sqrt{0.10^2 + 0.12^2 + 0.08^2 + 0.18^2 + 0.20^2} = \sqrt{0.1032} \approx 0.3213
$$

$$
\|\boldsymbol{\ell}_A\| = \sqrt{0.40^2 + 0.48^2 + 0.32^2 + 0.72^2 + 0.80^2} = \sqrt{1.6512} \approx 1.2850
$$

$$
\|\boldsymbol{\ell}_B\| = \sqrt{0.05^2 + 0.06^2 + 0.08^2 + 0.55^2 + 0.60^2} = \sqrt{0.7350} \approx 0.8573
$$

Sanity check: $\|\boldsymbol{\ell}_A\| / \|\mathbf{u}\| = 1.2850 / 0.3213 \approx 4.0$ — yes, A is exactly 4× the magnitude of $\mathbf{u}$, as constructed.

**Step 3 — Cosines.**

$$
\cos\theta_A = \frac{0.4128}{0.3213 \cdot 1.2850} = \frac{0.4128}{0.4129} \approx 1.0000
$$

$$
\cos\theta_B = \frac{0.2376}{0.3213 \cdot 0.8573} = \frac{0.2376}{0.2755} \approx 0.8625
$$

**Step 4 — Angles.**

$$
\theta_A = \arccos(1.0000) \approx 0.000\text{ rad} = 0.0°
$$

$$
\theta_B = \arccos(0.8625) \approx 0.5306\text{ rad} = 30.4°
$$

**Read the result.** $\theta_A = 0°$ — a perfect match in shape, even though calcite is 4× brighter in absolute reflectance than the spaceborne pixel. That's amplitude invariance doing its job. $\theta_B = 30.4°$ — well above the "essentially unrelated" threshold of 15°, so vegetation is correctly rejected.

So: SAM says **the unknown is calcite**, with high confidence. And it would say the same thing if we scaled $\mathbf{u}$ down by 0.3× (deep shadow) or up by 5× (specular glint), because the cosine doesn't care.

#### Why pre-normalising is the speed trick

Look at the cosine formula again:

$$
\cos\theta = \frac{\mathbf{u}\cdot\boldsymbol{\ell}}{\|\mathbf{u}\|\,\|\boldsymbol{\ell}\|}
$$

We could write this as

$$
\cos\theta = \frac{\mathbf{u}}{\|\mathbf{u}\|} \cdot \frac{\boldsymbol{\ell}}{\|\boldsymbol{\ell}\|} = \hat{\mathbf{u}} \cdot \hat{\boldsymbol{\ell}}
$$

— first L2-normalise each vector (divide by its norm so its length becomes 1), then dot them. Same answer, but now the cosine is *just* a dot product. Let's verify on our numbers:

- $\hat{\mathbf{u}} = \mathbf{u}/0.3213 = [0.311,\; 0.374,\; 0.249,\; 0.560,\; 0.623]$
- $\hat{\boldsymbol{\ell}}_A = \boldsymbol{\ell}_A/1.2850 = [0.311,\; 0.374,\; 0.249,\; 0.560,\; 0.623]$  ← identical to $\hat{\mathbf{u}}$
- $\hat{\mathbf{u}} \cdot \hat{\boldsymbol{\ell}}_A \approx 1.000$ ✓

That's the whole point: **two spectra of the same shape become the same unit vector after L2 normalisation, regardless of amplitude.** The cosine between two unit vectors is simply their dot product. This is the geometric content of "SAM is amplitude-invariant."

### 3.4 Scaling to a whole image: the matmul

We just did one unknown × two library entries by hand. The real job is $P$ unknowns × $N$ library entries — e.g. 5000 × 2400 = 12 million cosines. We don't want a Python loop over each pair.

Once each row is L2-normalised, cosine = dot product. Stack the normalised unknowns into a matrix $\hat{U} \in \mathbb{R}^{P \times B}$ (each row is one $\hat{\mathbf{u}}_p$) and the normalised library into $\hat{L} \in \mathbb{R}^{N \times B}$. Then **all $PN$ cosines come from one matrix multiply**:

$$
C = \hat{U}\,\hat{L}^T \in \mathbb{R}^{P \times N}, \qquad \theta_{p,n} = \arccos(C_{p,n})
$$

Matrix multiplication is the single most optimized operation in numerical computing — BLAS on CPU, cuBLAS on GPU. A $5000 \times 149$ by $149 \times 2400$ matmul is ~1–3 s on a Colab CPU and ~30 ms on a T4 GPU.

[sam.py:82-88](sam.py#L82-L88) is the NumPy path; [sam.py:91-116](sam.py#L91-L116) is the PyTorch path. Same three lines (normalize, matmul, arccos), different devices.

### 3.5 Two small but important details

- **`np.clip(cos_theta, -1, 1)`** before `arccos`. Floating-point error can produce 1.00000001, and `arccos(1.00000001)` is NaN. The clip is a numerical-safety guard — not "fixing" any real value.
- **Chunking.** Output matrix is $P \times N$ float32 = $4PN$ bytes. At $P=100\text{k}$, $N=2400$, that's ~1 GB. Code processes a chunk of unknown rows at a time so GPU VRAM doesn't blow. The library matrix $L$ (the small one) stays resident; only the unknown chunk moves.

---

## Chapter 4 — Cleaning the unknown spectra: Savitzky–Golay

One wrinkle before SAM. The unknown spectra come from a spaceborne sensor — noisy (read noise, atmospheric correction artifacts, residual stripes). Library spectra are lab-clean, and the Gaussian SRF resampling we already did is itself a smoothing pass, so they don't need denoising. If we feed noisy $\mathbf{u}$ into SAM, the angle gets inflated by noise that isn't part of the material fingerprint.

### 4.1 Why SG and not a moving average

A moving average smooths but also **rounds off sharp features** — exactly the absorption peaks that distinguish minerals. Savitzky–Golay does something cleverer:

> For each band, fit a low-order polynomial (degree 2) to a small window of neighbors (7 bands), and replace the center value with the polynomial's value at that point.

A polynomial of degree 2 can represent a peak; a moving average can't. So SG denoises *without* flattening sharp features. Under the hood it's a precomputed convolution kernel — one fast 1D convolution per row.

[preprocess.py:55](preprocess.py#L55) is one line:

```python
savgol_filter(spectra, window_length=win, polyorder=polyorder, axis=1)
```

`axis=1` is critical — we smooth **along bands**, not across pixels (smoothing across pixels would blur spatial detail).

The clamp/skip logic above it handles "what if the user gave us a window larger than the band count" — can happen with thermal sensors that have a handful of bands.

---

## Chapter 5 — End to end: one anomaly pixel

Toy shapes: $B = 149$ usable PRISMA bands, $M = 2151$ samples per splib07 spectrum, $P = 5000$ anomaly pixels, $N = 2400$ library entries.

### Step 1 — Build the library (`load_splib07_library`)

```
For each of ~2400 splib07 files:
    parse header → name, category, raw refl[2151]
    refl[refl < -1e30] = NaN          # USGS no-data sentinel
    if max(refl) > 1.5: refl /= 100   # percent → fraction
    resampled[149] = gaussian_resample_to_target(lib_wl, refl, target_wl, target_fwhm)
    if coverage(resampled) < 0.7: skip
    fill small NaN gaps by linear interp
    store as LibraryEntry
→ list[LibraryEntry] (sorted by category, material_id)
→ stack_library(...) → L of shape (2400, 149)
```

Cached to `.npz + .json` keyed by SHA-256 of `(splib07_dir, target_wl, target_fwhm, categories, min_coverage)`. Change any input and the cache is invalidated. Don't change them and you skip the whole build on subsequent runs.

### Step 2 — Prep the unknowns (`sg_smooth_spectra`)

```python
U_raw shape (5000, 149)              # anomaly pixel spectra from your detector
U     = sg_smooth_spectra(U_raw, window=7, polyorder=2)
                                     # still (5000, 149), denoised along bands
```

### Step 3 — Match (`compute_sam_matrix`)

```python
angles = compute_sam_matrix(U, L)        # (5000, 2400) float32, degrees
best   = angles.argmin(axis=1)           # (5000,) library index of best match per pixel
score  = angles[np.arange(5000), best]   # (5000,) the angle of that match
```

For pixel $p$: `entries[best[p]].name` is the predicted material, `score[p]` is your confidence (smaller = better). Common thresholds: < 5° strong, 5–10° plausible, > 15° essentially unrelated.

---

## Chapter 6 — Deep dives

### 6.1 Deriving the Savitzky–Golay kernel

The verbal description in Chapter 4 — "fit a degree-2 polynomial to 7 neighbouring bands, replace the centre value with the polynomial's value at that point" — sounds like it would be expensive: a least-squares fit per pixel per band. The big insight Savitzky and Golay published in 1964 is that **all of that work collapses into a fixed convolution kernel that you can precompute once**. Here's why.

#### 6.1.1 The local fit

Consider 7 consecutive band values: $y_{-3}, y_{-2}, y_{-1}, y_0, y_1, y_2, y_3$, where $y_0$ is the centre (the value we want to denoise). Index the positions by $x \in \{-3, -2, -1, 0, 1, 2, 3\}$.

We fit a quadratic

$$
p(x) = a_0 + a_1 x + a_2 x^2
$$

by minimising

$$
\sum_{x=-3}^{3} \bigl( a_0 + a_1 x + a_2 x^2 - y_x \bigr)^2
$$

This is ordinary least squares. In matrix form, let

$$
A = \begin{bmatrix} 1 & -3 & 9 \\ 1 & -2 & 4 \\ 1 & -1 & 1 \\ 1 & 0 & 0 \\ 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} y_{-3} \\ y_{-2} \\ y_{-1} \\ y_0 \\ y_1 \\ y_2 \\ y_3 \end{bmatrix}
$$

The least-squares solution is

$$
\hat{\mathbf{a}} = (A^T A)^{-1} A^T \mathbf{y}
$$

The smoothed centre value is $p(0) = a_0 = \hat{\mathbf{a}}_0$, i.e. the first row of $(A^T A)^{-1} A^T$ times $\mathbf{y}$.

#### 6.1.2 The collapse

Notice $A$ only depends on the window size and polynomial order, **not on the data**. So $(A^T A)^{-1} A^T$ is a fixed $3 \times 7$ matrix that doesn't change between pixels or bands. The first row of it is a fixed length-7 vector $\mathbf{c} = [c_{-3}, c_{-2}, c_{-1}, c_0, c_1, c_2, c_3]$, and:

$$
\hat{y}_0 = \mathbf{c} \cdot \mathbf{y} = \sum_{x=-3}^{3} c_x \, y_x
$$

**The whole least-squares fit reduces to a dot product with a precomputed length-7 kernel.** Sliding that kernel along the spectrum is exactly 1D convolution. That's why `savgol_filter` is fast — it just convolves the spectrum with a fixed kernel that depends only on `(window_length, polyorder)`.

#### 6.1.3 The actual numbers for `window=7, polyorder=2`

Working through $(A^T A)^{-1} A^T$ for the matrix above:

$$
A^T A = \begin{bmatrix} 7 & 0 & 28 \\ 0 & 28 & 0 \\ 28 & 0 & 196 \end{bmatrix}
$$

(The off-diagonal zeros are because $\sum x = 0$ and $\sum x^3 = 0$ for a symmetric window — odd moments vanish.) Inverting and multiplying by $A^T$, the first row of the result — i.e. the smoothing kernel — is:

$$
\boxed{\mathbf{c} = \frac{1}{21}\bigl[-2,\; 3,\; 6,\; 7,\; 6,\; 3,\; -2\bigr]}
$$

A few things to notice:

- **Sum equals 1:** $(-2+3+6+7+6+3-2)/21 = 21/21 = 1$. So if the input is constant ($y_x = c$ for all $x$), the output is $c$ — flat input gives flat output, as it must for any sensible smoother.
- **Symmetric:** $c_{-x} = c_x$. Smoothing is direction-agnostic.
- **Negative edge taps:** $c_{\pm 3} = -2/21$ — the outermost neighbours actually get *subtracted*. This is what lets SG preserve peaks: a moving average would have all-positive weights and round a peak off; SG's negative edges push back when the centre value is locally extreme, recovering some of the peak height that a moving average would erode.
- **Centre tap = $7/21 = 1/3$:** the centre point gets the largest weight, but only 33% — the other 67% comes from neighbours. That's what does the noise reduction.

#### 6.1.4 What SG does to noise vs signal

If $y_x$ is pure white noise (independent samples with variance $\sigma^2$), the variance of the smoothed value is

$$
\text{Var}(\hat{y}_0) = \sigma^2 \sum_x c_x^2 = \sigma^2 \cdot \frac{4+9+36+49+36+9+4}{441} = \sigma^2 \cdot \frac{147}{441} = \frac{\sigma^2}{3}
$$

So **noise variance drops by a factor of 3** (standard deviation drops by $\sqrt{3} \approx 1.73$). Not huge — a longer window denoises more — but you've kept the polynomial-order-2 ability to represent peaks.

If $y_x$ is a quadratic — exactly the function class SG fits — the smoothing is *exact*: $\hat{y}_0 = y_0$, no bias. That's the whole design goal. Peaks that look quadratic over 7 samples pass through unchanged; only the off-quadratic noise gets attenuated.

`scipy.signal.savgol_filter` precomputes this kernel internally and runs one convolution per spectrum. The "fit a polynomial 5000 × 149 times" mental model is mathematically correct but operationally never happens — it's all one matmul under the hood.

---

### 6.2 Gaussian SRF resampling by hand

We already did a worked example in §2.3, but that one used hand-picked library samples. Let's do a more code-faithful walk: take a synthetic splib07-style spectrum and trace the *exact* operations in `gaussian_resample_to_target`, including the `searchsorted` index-finding.

#### 6.2.1 The setup

Pretend splib07 is sampled every 1 nm from 640 to 660 nm (in real splib07 it's every ~5 nm, but 1 nm makes the indices easier to track):

```
lib_wl  = [640, 641, 642, ..., 660]               # 21 samples, nm
lib_refl = [0.40, 0.40, 0.41, 0.41, 0.42, 0.42,
            0.40, 0.30, 0.15, 0.20, 0.30,        # absorption dip at 648 nm
            0.39, 0.40, 0.40, 0.40, 0.40,
            0.40, 0.40, 0.40, 0.40, 0.40]        # back to baseline
```

Target sensor: PRISMA band 50 at $\lambda_{50} = 650$ nm, FWHM = 10 nm.

#### 6.2.2 Convert FWHM to σ

[resample.py:92](resample.py#L92):

$$
\sigma = \text{FWHM} \times \frac{1}{2\sqrt{2\ln 2}} = 10 \times 0.42466 \approx 4.247\text{ nm}
$$

#### 6.2.3 Find the 3σ window with `searchsorted`

[resample.py:101-105](resample.py#L101-L105):

```
half_window = 3 * 4.247 = 12.74 nm
lo = searchsorted(lib_wl, 650 - 12.74) = searchsorted(lib_wl, 637.26)
hi = searchsorted(lib_wl, 650 + 12.74) = searchsorted(lib_wl, 662.74)
```

`lib_wl` starts at 640, so `lo = 0` (637.26 sorts before everything). `lib_wl` ends at 660 (index 20), so `hi = 21` (662.74 sorts after everything — `searchsorted` returns the insertion index). Slice `lib_wl[0:21]` = all 21 samples.

In real splib07 with samples spanning 350–2500 nm, this slice would be ~5 samples wide instead of all 2151. Here our synthetic library is small enough that the whole thing fits in the 3σ window — that's fine; the formula doesn't care.

#### 6.2.4 Compute weights

[resample.py:115-116](resample.py#L115-L116):

For each library wavelength $\lambda_j$, the weight is

$$
w_j = \exp\!\left(-\tfrac{1}{2}\Bigl(\tfrac{\lambda_j - 650}{4.247}\Bigr)^{\!2}\right)
$$

Centre tap (650 nm): $w = e^0 = 1.000$. At 1σ (≈645.75 nm or 654.25 nm): $w = e^{-0.5} \approx 0.607$. At 2σ: $w \approx 0.135$. At 3σ: $w \approx 0.011$.

Computing for our 21 samples (rounded):

| $\lambda_j$ | $w_j$ | $r_j$ | $w_j r_j$ |
|---|---|---|---|
| 640 | 0.018 | 0.40 | 0.0072 |
| 641 | 0.032 | 0.40 | 0.0128 |
| 642 | 0.053 | 0.41 | 0.0217 |
| 643 | 0.084 | 0.41 | 0.0344 |
| 644 | 0.127 | 0.42 | 0.0533 |
| 645 | 0.183 | 0.42 | 0.0769 |
| 646 | 0.249 | 0.40 | 0.0996 |
| 647 | 0.324 | 0.30 | 0.0972 |
| **648** | **0.401** | **0.15** | **0.0602** |
| 649 | 0.473 | 0.20 | 0.0946 |
| **650** | **0.500** | **0.30** | **0.1500** |
| 651 | 0.473 | 0.39 | 0.1845 |
| 652 | 0.401 | 0.40 | 0.1604 |
| 653 | 0.324 | 0.40 | 0.1296 |
| 654 | 0.249 | 0.40 | 0.0996 |
| 655 | 0.183 | 0.40 | 0.0732 |
| 656 | 0.127 | 0.40 | 0.0508 |
| 657 | 0.084 | 0.40 | 0.0336 |
| 658 | 0.053 | 0.40 | 0.0212 |
| 659 | 0.032 | 0.40 | 0.0128 |
| 660 | 0.018 | 0.40 | 0.0072 |
| **sum** | **4.408** | | **1.4808** |

(Weights are slightly off from a perfect Gaussian due to rounding — exact values from $e^{-x^2/2}$ with $x = (\lambda - 650)/4.247$.)

#### 6.2.5 Apply the formula

[resample.py:120-124](resample.py#L120-L124):

$$
\rho_{50}^{\text{target}} = \frac{\sum_j w_j r_j}{\sum_j w_j} = \frac{1.4808}{4.408} \approx 0.336
$$

That's what PRISMA would read for this material at 650 nm. Notice the dip at 648 (r=0.15) pulled the answer down from the baseline 0.40, but only partially — the band sees mostly 0.40-ish values, with the dip getting maybe 10–15% of the total weight.

#### 6.2.6 What the code does *differently*

The hand-computation is what `gaussian_resample_to_target` runs verbatim, with three small implementation details:

- **NaN handling**: `w_eff = w * finite_slice` zeros out weights where library reflectance is NaN. Both numerator and denominator drop those terms.
- **Min-points check**: if `hi - lo < min_lib_points_per_band` (default 3), output NaN. Doesn't trust a weighted average of 1–2 samples.
- **Sigma sanity check**: if `sig_i <= 0` (bad sensor metadata), skip that band — output NaN.

Everything else is just running the same per-band loop for $i \in \{1, \ldots, N\}$ target bands.

---

### 6.3 SAM geometry: where it works, where it doesn't

SAM has the cleanest geometric interpretation of any spectral metric — *angle between two vectors* — but that clean geometry also reveals exactly where it fails. Let's walk through both.

#### 6.3.1 The picture in 2D

Pretend we have just 2 bands ($B = 2$). Every spectrum is a point in the plane. Reflectance is non-negative, so all points live in the first quadrant.

```
        band 2
          ^
          |       ℓ_A (calcite, "bright")
          |     /
          |   /
          | /          u (anomaly pixel)
          |/  ___θ_____
          + - - - - - - -  ℓ_B (vegetation, very different shape)
          |
          +----------------> band 1
```

- The **direction** of each ray from the origin is the spectrum's *shape*.
- The **length** along each ray is the spectrum's *amplitude*.
- The **angle between two rays** is the spectral angle $\theta$.

L2-normalising a vector slides it along its own ray to land on the unit circle. After normalisation, two spectra with the same shape become the **same point on the circle**, regardless of how bright the original spectra were. The cosine of the angle between two points on the unit circle is just their dot product — which is why §3.3's pre-normalisation trick works.

#### 6.3.2 Why $\theta \in [0°, 90°]$ for reflectance

In general, two vectors in $\mathbb{R}^B$ can make any angle in $[0°, 180°]$. But reflectance is non-negative — every component of every spectrum is $\geq 0$. Two non-negative vectors $\mathbf{u}, \boldsymbol{\ell}$ have $\mathbf{u} \cdot \boldsymbol{\ell} = \sum u_b \ell_b \geq 0$, so $\cos\theta \geq 0$, so $\theta \leq 90°$.

The extremes:

- $\theta = 0°$: $\boldsymbol{\ell} = \alpha \mathbf{u}$ for some $\alpha > 0$. Identical shape.
- $\theta = 90°$: $\mathbf{u}$ and $\boldsymbol{\ell}$ have disjoint support — every band where $\mathbf{u}$ is nonzero, $\boldsymbol{\ell}$ is zero, and vice versa. Physically nearly impossible for real spectra (everything has some reflectance everywhere), so in practice angles top out around 30–60° for "completely unrelated" materials.

This is why **the empirical thresholds in the SAM literature are tight**: < 5° strong match, 5–10° plausible, > 15° essentially unrelated. The range is compressed to roughly $[0°, 60°]$ in practice, so 5° isn't a small slice of the full $[0°, 180°]$ — it's a *meaningful* fraction of where real spectra actually sit.

#### 6.3.3 Where SAM fails — the amplitude blind spot

Amplitude invariance is SAM's superpower (handles shadow/illumination/atmospheric residual). It's also SAM's exact failure mode.

**Failure mode 1: two materials with the same shape but different amplitudes.** Suppose dry sand and wet sand have basically the same spectral *shape* (same absorption features), but wet sand is uniformly darker (water absorbs everything). SAM gives $\theta \approx 0$ for both — calls them the same material. They're not.

**Failure mode 2: snow vs cloud.** Both are very bright and very flat across the visible. After L2 normalisation they look nearly identical. SAM struggles to distinguish them.

**Failure mode 3: dark, noisy pixels.** If $\mathbf{u}$ is mostly noise around zero (a deeply-shadowed water pixel, say), the L2 normalisation amplifies the noise direction. You end up comparing the *noise pattern* to library entries, which is meaningless. Most pipelines threshold on minimum brightness before running SAM for this reason.

**The fix when amplitude matters: NS3, SID, or hybrid metrics.** NS3 (Normalized Sum of Squared Differences) couples shape and amplitude. SID (Spectral Information Divergence) treats spectra as probability distributions and uses KL divergence. There's a literature of "hybrid SAM-SID" scores that weight both. The right choice depends on whether amplitude carries diagnostic information for your problem — for cross-platform lab-vs-spaceborne matching (this pipeline's job), amplitude is *not* trustworthy, so SAM is the right call. For wet/dry sand discrimination in a single scene, it would be the wrong call.

#### 6.3.4 Why this codebase chose SAM anyway

The job spec is: take spaceborne anomaly pixels, identify the material from a *lab* library. Amplitude in the lab and amplitude in space are not comparable — atmospheric correction, illumination geometry, and sensor calibration all multiply or shift it. Shape is the only signal that survives the gap.

SAM's failure modes (wet/dry sand, snow/cloud) are intra-scene shape-degenerate problems, not lab-vs-spaceborne problems. They're not what this pipeline is solving. So the right tool is the amplitude-invariant one, with the understanding that the *next* layer of disambiguation (if you need it) lives downstream.

---

### 6.4 The cache key — why every field matters

`load_splib07_library` is slow (file walk + parse + resample 2400 entries — tens of seconds on Colab). It writes the finished resampled library to disk and reloads it on subsequent runs. The cache key is the filename that says "this exact `.npz` is the resampled library you'd get from *these specific inputs*."

[library.py:196-215](library.py#L196-L215):

```python
def _cache_key(splib07_dir, target_wl, target_fwhm, restrict_to_categories, min_coverage):
    h = hashlib.sha256()
    h.update(str(Path(splib07_dir).resolve()).encode())
    h.update(target_wl.astype(np.float64).tobytes())
    h.update(target_fwhm.astype(np.float64).tobytes())
    cats = sorted(restrict_to_categories) if restrict_to_categories else []
    h.update(repr(cats).encode())
    h.update(f"{min_coverage:.6f}".encode())
    return h.hexdigest()[:16]
```

SHA-256 is being used as a **fingerprint function**, not for any cryptographic reason. It eats arbitrary bytes and emits a short fixed-length string. Same inputs → same fingerprint, every time. Change one byte of input → completely different fingerprint. This is the property we want: a deterministic filename that's effectively unique to the input combination.

#### 6.4.1 Why each field is in the hash

| Field | What would go wrong if omitted |
|---|---|
| `splib07_dir` | Switching from `ASCIIdata_splib07b_cvASD` to `cvHYP` (different convolved variant) would silently reuse the wrong library. The two have different native wavelength grids and slightly different reflectance values per material. SAM would compare your unknowns against the wrong-grid version and produce subtly wrong matches. |
| `target_wl` | The whole library is resampled *onto these wavelengths*. Change target sensor (PRISMA → EnMAP, which has different band centres), reuse the cache, and SAM compares unknowns at EnMAP wavelengths against a library still at PRISMA wavelengths. Band-by-band dot product becomes meaningless — you're comparing band 50's reflectance at 650 nm to band 50's reflectance at 651.7 nm. Result: every match has inflated angle. |
| `target_fwhm` | Same sensor band centres but different FWHM (a sensor model updated after calibration, or a different sensor with overlapping centres) would silently use stale convolution weights. The library would be slightly mis-resampled — narrow absorption features would have wrong shape — and matches would degrade in a hard-to-debug way. |
| `restrict_to_categories` | If you cached the full library and then ask for only Minerals, the cache hit returns the full library (extra entries — wasteful but harmless). The dangerous direction: you cached only Minerals, then ask for the full library — the cache hit returns the Minerals-only library, silently dropping vegetation, soils, etc. Your SAM matches would be confidently wrong (best mineral match for a vegetation pixel). |
| `min_coverage` | Tightening from 0.7 → 0.9 should drop low-coverage entries. With this in the hash, that's a new cache file (rebuild). Without it, you'd silently reuse the loose-coverage library — the tightened threshold would be ignored. |

#### 6.4.2 Why `tobytes()` on the float arrays

`h.update(target_wl.astype(np.float64).tobytes())` is critical. You can't `repr(target_wl)` — float-to-string conversion rounds, so 650.0000001 and 650.0000002 would hash the same. `tobytes()` writes the raw IEEE-754 bytes, so any bit difference is captured. The `.astype(np.float64)` makes the byte representation deterministic even if the caller passes a float32 array.

#### 6.4.3 Why `sorted(categories)`

`['Minerals', 'Vegetation']` and `['Vegetation', 'Minerals']` are semantically the same filter — they should hit the same cache. Sorting before hashing makes the hash order-independent.

#### 6.4.4 Why `f"{min_coverage:.6f}"`

Rounding to 6 decimals. Two callers passing 0.7 and 0.7000000001 should hit the same cache (those values produce the same result in practice — the coverage check is `coverage < min_coverage`, not exact comparison). This is a deliberate choice to merge near-duplicate float inputs onto the same cache key. The downside is that 0.7000001 and 0.7000002 also collide — but at that precision they're operationally identical, so it's fine.

#### 6.4.5 Why only the first 16 hex chars

`h.hexdigest()` returns 64 hex chars (256 bits). The code takes `[:16]` = 64 bits. Collision probability with 64-bit hashes: you'd need on the order of $2^{32} \approx 4$ billion cache entries for a 50% chance of a collision (birthday paradox). For a per-user cache that holds a few dozen entries lifetime, the collision risk is zero in practice. 16 chars keeps filenames short and readable.

#### 6.4.6 The discipline this enforces

Every line of `_cache_key` is a *contract*: "anything that changes the output of `load_splib07_library` is in this hash." If a maintainer adds a new parameter — say `resampling_method='gaussian' | 'box'` — they have to add it to `_cache_key` too, or the cache silently returns stale data on the next user. Code review on `_cache_key` is the gate that keeps caching correct.

This is the same pattern that bigger systems formalise as content-addressed storage (Git, Bazel, Nix). Here it's three lines of `hashlib.update`, but the discipline is identical: **the cache key must encode every input that affects the output**.

---

## Where to go next

5. **Run it** — set up a tiny synthetic example (10 fake library spectra, 5 fake unknowns) and watch the numbers fall out.

---

# Build log

A plain-language diary of work as the spectral-match action gets built in the real product. Every entry is dated and explains **what** changed and **why**. No code in this log. If you're reading this years later: start at the top and you can rebuild from the same decisions.

## 2026-05-12 — Day 1: looking at the splib data, clearing build pipes

### What we did

1. **Opened the USGS splib07 bundle.** It's **6.7 GB** on disk — far too big for what should be a few thousand spectra. We didn't read 178k files; we walked the folder tree and sampled a few spectra to understand the bulk.

2. **Figured out why it's so big.** The bundle ships every spectrum **multiple times** — once at its original lab resolution, then re-copied for ~20 different satellite / aircraft sensors (Hyperion, AVIRIS for each year 1995–2014, MASTER, Landsat, etc.). On top there's a binary-format copy of the same data, thousands of pre-rendered plot images, and PDF/HTML docs.

3. **Identified the slim subset we actually want.** One folder, `ASCIIdata_splib07a`, holds the spectra at their **native lab resolution** (the wavelengths the lab spectrometer actually sampled). Inside, files are grouped into 7 "chapter" folders — minerals, vegetation, soils, artificial materials, organics, liquids, coatings. Each spectrum is a simple text file: one header line + ~2151 reflectance values, one per line.

4. **Decided which lab-instrument spectra to keep.** The bundle includes files measured on three different lab instruments: ASD (350–2500 nm), BECK (Beckman, 200–3000 nm), NIC4 (Nicolet, mid-infrared, 1.12–216 µm). Our satellites only see ~380–2500 nm, so only **ASD** is useful — the others measure wavelengths our sensors can't even detect. Keeping ASD-only shrinks the curated set to about **40 MB**. ~150× smaller than the raw bundle.

5. **Caught a Docker bug.** Our compose file builds three services using the **repo root** as their build context. Without a `.dockerignore`, every rebuild was silently copying the entire 6.7 GB splib07 folder to the Docker daemon. That explains some painfully slow recent rebuilds. Fixed by adding a `.dockerignore` at the repo root that excludes the raw bundle (and other heavy local data folders). Extended `.gitignore` for the same reason.

6. **Updated the roadmap.** Step 14.7 (the spectral-match action) now has the concrete file-layout details we learned: which files to use, which to skip, the on-disk shape of the slim bundle, and why ASD-only is correct.

### What we did NOT do

- No spectral-match code yet. Roadmap entry locked; first code lands in the next entry.
- No slim bundle on disk yet. Next step is a one-shot curation script that walks the raw 6.7 GB tree and copies out the ~40 MB we need. That script runs once per splib07 release (rare event).

### Why it matters

The whole spectral-match feature depends on a clean, fast-loading library. Doing the curation up front means:
- Docker images stay small (~40 MB of spectra, not 6.7 GB).
- The library loads from disk in a fraction of a second.
- When a new splib07 release comes out we know exactly what to extract.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| The raw 6.7 GB bundle | `data/splib/usgs_splib07/` | Local only. Excluded from Docker. Excluded from git. |
| Docker exclusion | `.dockerignore` (new file at repo root) | Keeps raw bundle out of build contexts. |
| Git exclusion | `.gitignore` (extended) | Prevents accidental commits of the bundle. |
| Roadmap entry | `final design/ROADMAP.md` → Step 14.7 | The full locked plan. |
| The slim curated bundle | *not built yet* | Will land in a Docker volume called `allotrope_splib07` when the curation script runs. |

---

## 2026-05-12 — Day 2: wrote the curation script, and figured out what to do about missing bands

### What we did

1. **Wrote the one-shot curation script** at `scripts/curate_splib07.py`. It walks the raw 6.7 GB tree once, and for each ASD lab spectrum it finds (matching the filename pattern `splib07a_<material>_<sample>_ASD{FR,HR,NG}<batch>_<product>.txt`) it:
   - reads the first header line to grab the **record number** and the **pretty name** ("Acmite NMNH133746 ASDFRa AREF");
   - copies the spectrum text file verbatim into `spectra/<chapter_slug>/`;
   - records an `index.json` row with `{name, material, sample, chapter, asd_subtype, record, value_count, path}`.

2. **Also pulled out the wavelength + FWHM tables.** ASD lab spectra share **one** wavelength axis (2151 channels, 0.35–2.50 µm); each subtype (ASDFR/ASDHR/ASDNG) has its own FWHM file. The script converts everything from microns → nanometres and writes:
   - `wavelengths_asd_nm.txt` (one axis, shared by all ASD spectra),
   - `fwhm/ASDFR.txt`, `fwhm/ASDHR.txt`, `fwhm/ASDNG.txt` (one per subtype).

3. **Stamped a version tag** (`--version` flag, default `splib07a`) into `index.json`. Downstream, the per-sensor cache uses `SHA-256(sensor_id, target_wl, target_fwhms, categories, min_coverage, splib07_version)` as its filename — so when we bump the version (or add a chapter, or change the sensor's bands), the cache rebuilds itself.

4. **Decided how to handle missing bands** (the meat of today — see below).

### The missing-bands problem (worked example)

The whole point of SAM is `cos(θ) = (a·b) / (|a| |b|)`. That dot product is only meaningful if `a` and `b` are on **exactly** the same wavelength grid. In practice, both sides have holes:

**Holes on the unknown (true pixel) side:**
- **Atmospheric water absorption.** Around **1400 nm** and **1900 nm** the atmosphere blocks almost all light. Reflectance retrievals (PRISMA L2D, EnMAP L2A) NaN those bands out. On AVIRIS-NG the ENVI header's `bbl` list flags them.
- **Sensor edge bands.** The VNIR/SWIR detector seam on PRISMA (~970 nm) and EnMAP (~900 nm) gets edge-trimmed.
- **Per-pixel masks.** Saturated pixels, cloud pixels, shadow pixels — different bands fail for different pixels in the same scene.

**Holes on the library side:**
After we resample a 2151-channel ASD spectrum onto a sensor grid using a Gaussian SRF, **a band is only valid if the SRF window (±3σ around the centre wavelength) had enough non-sentinel splib07 samples inside it**. If half the window is the `-1.23e+34` no-data sentinel, that band of the resampled library entry is NaN.

So both vectors are full of NaNs, and the NaN positions differ per pixel **and** per library entry.

#### Worked example — a single AVIRIS-NG pixel vs three library entries

Imagine an AVIRIS-NG cube with **425 bands** at 5 nm steps from 380 → 2500 nm. We pick one anomalous pixel.

Bands invalid for **this pixel** (atmospheric + bbl):
- Bands 195–210 (≈ 1355–1430 nm, water vapour)
- Bands 295–315 (≈ 1855–1955 nm, water vapour)
- Bands 0–3 (≈ 380–395 nm, low SNR edge)
- Band 142 (≈ 1090 nm, single saturated detector element on this pixel)

That's ~42 bands out, so **383 bands valid on the unknown** (call this set **P**).

Now three candidate library entries (already resampled to AVIRIS-NG 425-band grid):

- **Acmite (mineral).** Resampled cleanly across the whole 380–2500 nm range. **All 425 bands valid.** Valid set **L₁** = full.
- **Pinyon Pine Needle (vegetation).** ASD lab measurement runs out at ~2500 nm but has clean data through the whole range. **All 425 bands valid.** **L₂** = full.
- **Asphalt Roofing Shingle Black ASDFRa.** This particular splib07 file has the no-data sentinel for a swath at the long end (~2350–2500 nm) because the lab measurement dropped out there. **Bands 394–424 invalid.** **L₃** = bands 0–393.

For each comparison we compute SAM on the **intersection** P ∩ Lₖ:

| Pair | Valid set | # valid bands | What we do |
|---|---|---|---|
| pixel vs Acmite | P ∩ L₁ = P | 383 | Compute `cos(θ)` on those 383 bands. |
| pixel vs Pinyon | P ∩ L₂ = P | 383 | Compute on the same 383. |
| pixel vs Asphalt | P ∩ L₃ = P minus bands 394–424 | 352 | Compute on 352 bands. |

**Coverage gate:** the **library** entry must cover at least `min_coverage` (default 0.7) of the pixel's valid bands. For Asphalt: 352 / 383 = **0.919 → passes**. If a library entry only covered, say, 200 / 383 (= 0.52), we'd drop it from this pixel's top-K candidate pool entirely — better to say "no match" than to let a sparsely-covered spectrum win by accident.

**Min-band-count gate:** if `|P|` itself is below an absolute floor (e.g. 20 bands surviving), the pixel gets `match=null`. SAM on 8 bands is noise.

**Norms must be recomputed per pair.** We can **not** pre-store `|library_k|` once and reuse it — the norm has to be over P ∩ Lₖ, not all of Lₖ. Otherwise the angle is wrong. Same for the pixel side: `|unknown_p|` is computed over P ∩ Lₖ for each k.

#### Naive vs fast

The naive form (every (pixel, library) pair handled separately) is correct but slow — millions of separate small dot products.

**The trick:** in a real scene, only a handful of distinct valid-band patterns exist. Atmospheric and bbl masks are scene-wide constants (one pattern). Saturated pixels add a few small variants. In practice you might see **3–10 unique patterns** in a scene of millions of pixels.

So the algorithm becomes:

```
for each unique valid-band pattern P in the scene:
    pixels_P  = all anomaly pixels with this pattern, stacked into a (N_P, |P|) matrix
    library_P = library entries restricted to bands in P, dropped if coverage < min_coverage
                (rows that survive may still have internal NaNs from L_k ∩ P; for those rows
                 we either fall back to per-pair masking or pre-bucket by sub-pattern)
    norms_P_pix = row-wise norms of pixels_P
    norms_P_lib = row-wise norms of library_P
    cos_matrix = (pixels_P @ library_P.T) / outer(norms_P_pix, norms_P_lib)
    take top-K per row of cos_matrix
```

One BLAS matmul per pattern. Scene-scale matching becomes seconds, not minutes.

The remaining edge case is library entries that themselves have internal NaNs falling inside P. We bucket those by their own pattern within the pattern (a second-level group), then matmul each bucket. In practice almost all library entries are dense over the AVIRIS/PRISMA/EnMAP ranges, so the buckets degenerate to "dense" + one or two small leftover groups.

### What this means for the cache

At cache-build time we resample every library entry to the sensor grid and store, **per entry**:
- the resampled reflectance vector (with NaNs where SRF coverage was insufficient),
- a uint8 validity mask (1 byte per band per entry).

We do **not** pre-store norms. Norms get computed inside the per-pattern loop at match time.

The sensor's **bad-band list** (atmospheric + bbl, the scene-wide invalid bands) is baked into the cache key — re-onboarding with a different bbl invalidates the cache.

### What we did NOT do

- No `app/spectral_match/` module yet — that's Day 3.
- Curation script hasn't been **run** yet against the local 6.7 GB bundle; the code is in place but the slim bundle on disk is still empty.

### Why it matters

Missing bands is the one corner that quietly destroys a SAM matcher in production. If you skip this thinking and use a plain `(P · L) / (|P| |L|)` over the whole 425-band grid with NaNs treated as zeros, **every spectrum starts looking like every other spectrum** (zeros pull both vectors toward the origin in the same way) and the top-K becomes noise. Locking down "drop-on-either, pattern-bucketed, coverage-gated" now means the matcher we build in Day 4 will be correct **and** fast on real scenes.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Curation script | `scripts/curate_splib07.py` | One-shot. Run once per splib07 release. |
| Curation output (slim bundle) | `data/splib07_slim/` (or wherever you point `--out`) | ~40 MB. Eventually copied into the `allotrope_splib07` Docker volume. |
| Version tag | `index.json → version` field | Becomes part of every per-sensor cache key. |
| Missing-bands strategy | This entry | Lock-in decision: drop-on-either, pattern-bucketed, coverage ≥ 0.7, min 20 bands. |

---

## 2026-05-12 — Day 3: the `app/spectral_match/` module + per-sensor cache CLI

### What we did

1. **Created `app/spectral_match/`** with three pieces of code:
   - `resample.py` — the Gaussian SRF resampler (one library spectrum → one sensor's band grid). NaN output where SRF coverage was insufficient.
   - `library.py` — loads our curated slim bundle from disk and **builds a per-sensor cache** at a path keyed by a content-addressed hash of every input.
   - `__init__.py` — re-exports the public symbols.

2. **Wrote the cache-build CLI** at `scripts/build_splib_sensor_cache.py`. Operator runs this **once per sensor**:
   ```
   python scripts/build_splib_sensor_cache.py \
       --slim         data/splib07_slim \
       --sensor-spec  data/sensor_specs/aviris_ng.json \
       --cache-dir    data/splib07_cache \
       --chapters     minerals artificial soils vegetation organics
   ```
   It loads the slim bundle, resamples every ASD spectrum onto the sensor's wavelength grid using that sensor's per-band FWHM, applies the sensor's bad-band mask, drops entries below `--min-coverage`, and writes a `splib07_<key>.npz` + `splib07_<key>.json` pair.

### How the cache works (in plain words)

We don't want to re-resample 1200 lab spectra every time a user clicks **Run** on a spectral-match action. That would be slow and wasteful — the answer is the same every time for the same sensor.

So we **bake the resampled library to disk once**, and look it up by a 16-character hash that fingerprints all the inputs that could change the answer:

- which sensor (its ID),
- the sensor's wavelength axis,
- the sensor's per-band FWHM,
- which bands the sensor flags as bad,
- which chapters of splib07 we kept (minerals, vegetation, …),
- the coverage threshold,
- the version of the splib07 bundle.

If **any** of those change, the hash changes, and a fresh cache file gets written next to the old one. If nothing changes, the old file is loaded and we skip the rebuild entirely.

This is the same trick Git, Bazel, Nix use — *the cache key must encode every input that affects the output*. It's three lines of `hashlib.update` here but the discipline is what keeps the system honest.

### What we store inside the cache

Per library entry, the `.npz` file holds two parallel arrays:

| Array | Shape | Meaning |
|---|---|---|
| `refl` | (N, B) float32 | the resampled reflectance on the sensor's B bands. NaN where the SRF window didn't have enough samples. |
| `valid` | (N, B) uint8 | 1 where `refl` is finite, 0 where it's NaN. |

…and the sidecar `.json` carries `material_id`, `name`, `chapter`, `asd_subtype`, `coverage` per entry, plus the inputs used to build the cache (so the file is self-describing — pull it out of the volume and you can see exactly what made it).

### Things we deliberately did NOT precompute

**Norms.** It would be tempting to also store `|library_entry|` once and reuse it at match time. **Wrong.** The relevant norm for SAM is over the **intersection** of the unknown pixel's valid bands and the library entry's valid bands — and the pixel's mask isn't known until run time. Pre-stored norms over the whole vector would give answers that quietly disagree with the per-pair angle math we locked in on Day 2.

So `refl` (with NaNs) and `valid` (the mask) is all that ships. Norms get computed inside the matmul loop, per pattern bucket.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Gaussian SRF resampler | `app/spectral_match/resample.py` | One function: `gaussian_resample_to_target(...)`. |
| Slim-bundle loader + cache builder | `app/spectral_match/library.py` | `load_slim_bundle`, `build_sensor_cache`, `load_sensor_cache`. |
| Cache key | `sensor_cache_key(...)` in `library.py` | SHA-256 truncated to 16 hex chars. |
| Cache-build CLI | `scripts/build_splib_sensor_cache.py` | One-shot per sensor. Idempotent. |
| Sensor specs (operator-written) | `data/sensor_specs/<sensor>.json` | Future: emitted automatically at onboarding. For now, hand-written from each sensor's reference (PRISMA HE5 / EnMAP XML / AVIRIS-NG .hdr). |
| Built caches | `data/splib07_cache/splib07_<key>.{npz,json}` | One pair per (sensor, settings) combo. |

### What we did NOT do

- No SAM core yet — that's Day 4 (the pattern-bucketed matmul). The library is ready and the resampling story is locked, but no matching code has run.
- No sensor specs have been written. AVIRIS-NG is the easiest first one (the ENVI header carries wavelength + FWHM + bbl on three lines).
- The curation script hasn't been run against the local 6.7 GB bundle yet, so nothing has actually been built end-to-end. Tomorrow includes running it.

### Why it matters

Today's code is what makes everything downstream cheap. Once the cache is built, a spectral-match worker job is just: open one `.npz` (a few MB), do one matmul per band-pattern, write results. No filesystem walk, no SRF math, no per-spectrum file open. That's the difference between an action that finishes in seconds and one that takes minutes.

---

## 2026-05-12 — Day 4: the matcher itself — SG smoothing + pattern-bucketed SAM + top-K

### What we did

1. **Wrote `app/spectral_match/smoothing.py`** with a single `savgol_smooth(spectra, window_length=7, polyorder=2)` function. Smoothing operates along the spectral axis. NaN-flagged bands are linearly interpolated before the smoother runs so SG doesn't choke on holes, then the NaN positions are **restored** on the output. The per-pixel validity mask survives smoothing untouched — every downstream step still knows which bands are real.

2. **Wrote `app/spectral_match/sam.py`** with `match_pixels(...)` that returns top-K library matches for a stack of unknown pixels. This is the heart of the action and implements exactly the strategy we locked in on Day 2.

### How `match_pixels` works, step by step

Given pixels `(P, B)` + their validity mask, and a library `(N, B)` + its validity mask:

1. **Group pixels by their valid-band pattern.** Pack each pixel's validity row into a compact byte key and bucket pixels with the same key together. On a real scene with ~10000 anomaly pixels, you usually end up with 1–3 distinct patterns: one big atmospheric+bbl pattern that covers the vast majority, plus a couple of small variants from saturated/cloudy pixels.

2. **For each pattern P with `|P|` valid bands:**
   - If `|P| < min_band_count` (default 20), flag every pixel in this group as `no_match` and skip. Eight bands of dot-product is noise, not a match.
   - Compute each library entry's coverage **inside P** (fraction of its bands that are valid in the intersection). Drop entries below `min_coverage` (default 0.7) — they can't compete fairly.

3. **Bucket the surviving library entries by their own sub-pattern within P.** Almost every entry is fully dense inside P (`|sub-pattern| == |P|`); the rare ones with internal NaNs fall into their own small bucket. This is the trick that keeps the math correct *and* keeps everything in BLAS.

4. **For each sub-bucket:** compute pixel norms, library norms, and the `(n_pixels_in_pattern, n_lib_in_bucket)` cosine matrix in **one matmul**. Stitch the per-bucket cosine columns back into the pattern-wide cosine table.

5. **Top-K per row** using `np.argpartition` on the negative cosines (so smaller angle = better match comes first), then a small in-K sort for stable ranking.

6. **Convert cosines to angles in degrees** for the rows that found at least one valid match. Pixels whose top-1 is `-inf` (no bucket fit) get flagged `no_match`.

### A worked example (continuing Day 2's pixel)

Same AVIRIS-NG pixel, 425 bands, 383 valid in P. Top-K = 5.

After loading the AVIRIS-NG cache (~1200 entries kept across 5 chapters):

| Step | Numbers |
|---|---|
| Library entries in cache | 1200 |
| Library entries with coverage ≥ 0.7 inside P | 1183 (the 17 below-coverage ones get dropped) |
| Library entries fully dense in P (sub-pattern == P) | 1170 |
| Library entries with internal holes in P (≥ 0.7 cov but missing a few) | 13 |
| Sub-buckets to compute | 2 (the big dense one + the one ragged one) |
| Matmul shape, dense bucket | (n_pixels_in_P, 383) × (383, 1170) |
| Matmul shape, ragged bucket | (n_pixels_in_P, ~352) × (~352, 13) |

After both matmuls, every pixel in this pattern has a `(1, 1183)` cosine row. `argpartition` pulls the top-5 angles per pixel. **One scene, one pattern, two matmuls, done.**

### Why per-pair norms still come out right

This is the easy thing to get wrong. The denominator in SAM has to be the L2 norm of the **same vectors** used in the dot product. When a sub-bucket only operates on, say, 352 of the pattern's 383 bands:

- `pix_norms[r] = || pixels[r, sub-bucket bands] ||` — recomputed for this bucket
- `lib_norms[i] = || library[i, sub-bucket bands] ||` — recomputed for this bucket

We do not reuse a pre-stored full-vector norm. The cost is a single `np.linalg.norm` per bucket, which is negligible next to the matmul.

### Outputs

`match_pixels` returns a `MatchResult` dataclass with four arrays:

| Field | Shape | Meaning |
|---|---|---|
| `angles_deg` | (P, K) float32 | SAM angle in degrees per match. NaN where the pixel didn't match anything. |
| `library_ix` | (P, K) int32 | Index into the library entries list. -1 where no match. |
| `n_bands_used` | (P, K) int32 | How many bands the angle was computed on. The frontend shows this so users can sanity-check "this match used 352 out of 383 valid bands" vs "this match only had 87 bands". |
| `no_match` | (P,) bool | True where the pixel had too few valid bands or no library entry passed the coverage gate. |

### What we did NOT do

- No GPU dispatch. The pattern-bucketed CPU path is fast enough for our scene sizes (a few seconds for ~10k pixels × ~1200 entries). Adding a torch path is a future optimisation if/when we run against a full scene's worth of pixels (in `mode=all_kept`), not just anomalies.
- No "ignore continuum" mode (continuum removal would normalise away the broad slope and emphasise narrow absorption features). That's a quality knob to add later if matchers come back too coarse.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Smoothing | `app/spectral_match/smoothing.py` | `savgol_smooth(spectra, window_length, polyorder)`. |
| The matcher | `app/spectral_match/sam.py` | `match_pixels(...)`, `MatchResult` dataclass. |
| Module surface | `app/spectral_match/__init__.py` | Re-exports the public API. |

### Why it matters

`match_pixels` is the load-bearing science of this whole feature. Everything from here on — the action wiring, the frontend hover-to-preview — is plumbing around what this function returns. With pattern bucketing it stays fast on real scenes, and with per-pair norms it stays correct even when the library has holes inside the pixel's valid-band pattern. That correctness + speed combo is what makes the feature usable for analysts instead of a research-only toy.

---

## 2026-05-12 — Day 5: the action wiring (api validation + worker recipe)

### What we did

1. **Wrote the api-side action type** at `backend/allotrope/action_types/spectral_library_match.py`. It defines the Pydantic config schema, META (label, description, inputs, outputs, accepted sensors), and a `validate_config` function that the api uses at submit time to reject bad payloads with a 422.

2. **Wrote the worker recipe** at `backend/allotrope/action_types/_spectral_library_match_run.py` — only loaded inside the worker process, so the api doesn't pay for `rasterio` / `pyarrow` / `scipy` imports.

3. **Registered** the new type in `backend/allotrope/action_types/__init__.py` so both submit-time validation and worker dispatch can find it.

### The action's contract

**Inputs (submit-time):**

| Field | Meaning |
|---|---|
| `input_anomaly_detection_output_id` | A committed `anomaly_detection_prep` Output. Its `anomaly_mask.tif` picks which pixels get matched. |
| `chapters` | splib07 chapter slugs to include. Default: minerals + artificial + soils + vegetation + organics. |
| `top_k` | How many top matches to record per pixel. Default 5, cap 20. |
| `min_coverage` | Per-pattern library-coverage gate. Default 0.7. |
| `min_band_count` | Absolute floor on valid bands per pixel. Default 20. |
| `sg_window_length` / `sg_polyorder` | Savitzky-Golay smoothing knobs. Defaults 7 / 2. |
| `mode` | `anomaly_pixels` (default, fast) or `all_kept` (full-scene material map). |

**Accepted sensors:** PRISMA, EnMAP, AVIRIS-NG. Thermal sensors are deliberately rejected at the type level — they have one band and SAM is meaningless against splib07 at a single wavelength.

**Outputs (worker writes into `<output_dir>/`):**

| File | What it is |
|---|---|
| `matches.parquet` | Long table: one row per (pixel, rank). Columns: `row, col, rank, library_ix, material_id, name, chapter, asd_subtype, angle_deg, n_bands_used`. |
| `match_map.tif` | int32 raster. Each pixel painted with its top-1 `library_ix`. Sentinels: `-1` = pixel was matched but no library entry survived gating; `-2` = pixel wasn't in the input mask. |
| `match_map_legend.json` | `{library_ix: {name, material_id, chapter, asd_subtype}}` for every `library_ix` that appears as top-1 somewhere in the raster. Drives the viewer's legend. |
| `histogram.json` | Top-1 counts per material, sorted desc. Drives the side-panel bar chart. |
| `summary.json` | Cache path used, timing, pixel counts, settings echo. |

### Worker pipeline in plain words

1. **Find the upstream binary anomaly mask.** Open the committed `anomaly_detection_prep` output's `anomaly_mask.tif`.

2. **Load the native onboarding vendable** (`scenes/<id>/vendable/vendable.pkl`). **Not** the `band_filter_apply` resampled cube — we want native bands so absorption features survive (Day 1 covered why). The vendable carries:
   - the cube (B, H, W),
   - per-pixel-per-band validity (B, H, W),
   - per-band centre wavelengths in nm (`band_cw_order`),
   - per-band FWHM in nm (`band_fwhm_order`),
   - the sensor's scene-wide bad-band list (`band_validity_by_position`).

3. **Resolve the per-sensor splib07 cache.** Recompute the same 16-hex-char `sensor_cache_key` we used at cache-build time, look for `splib07_<key>.npz` in the cache dir. If it's missing, raise a clear error that tells the operator *exactly* which CLI command to run to build it. (No silent in-action rebuild — that would surprise the worker and stretch a few-second action into a multi-minute one.)

4. **Pick pixels.** `anomaly_pixels` mode: pixels where `anomaly_mask==1 AND any band is valid`. `all_kept` mode: every spatially-valid pixel.

5. **SG-smooth** the picked spectra with the configured window/polyorder. NaN bands stay NaN through this — the smoother interpolates internally but restores the validity mask on output (Day 4).

6. **Call `match_pixels`** with the smoothed spectra + per-pixel validity + library `refl` + library `valid` masks. Returns the `MatchResult` from Day 4 (angles_deg, library_ix, n_bands_used, no_match).

7. **Write the four output files.** The parquet table is the primary "data" output; the raster + legend feeds the viewer's spatial display; the histogram feeds the chart panel; the summary is the human-readable record.

### What we did NOT do

- **No frontend viewer yet.** That's Day 6 — modal with the spatial match_map panel + spectrum probe (hover-to-preview, click-to-pin) + side rail of top-K + histogram.
- **No docker volume wiring.** That's Day 7 — mounting the `allotrope_splib07` volume into the worker container and pointing `settings.splib07_cache_dir` at it.
- **No new endpoint on the api.** The action returns standard outputs and the frontend reads them via the existing `/action-outputs/{id}` route — no bespoke endpoint needed.
- **No "rebuild the cache if it's missing" code in the run path.** Deliberate: building a sensor cache takes ~30s per sensor and rebuilding silently inside an action would make actions unpredictably slow. The error message points the operator straight at the CLI.

### Why it matters

This entry is the wire between **the science we built in Days 1-4** and **the user-facing UI we'll build on Day 6**. With this in place, an analyst can already submit a `spectral_library_match` action from the existing NewActionDialog (the action picker reads `META.label` + inputs automatically) and the worker will produce real outputs — just without a custom viewer yet, the analyst sees only the file listing.

The action's contract is also now stable: outputs are content-addressed via parquet + GeoTIFF + JSON, so the viewer code we write tomorrow is reading files with a fixed schema. No more changes to the worker recipe expected for Step 14.7.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Action type (api-side) | `backend/allotrope/action_types/spectral_library_match.py` | Config schema, META, validate_config. |
| Action run (worker-side) | `backend/allotrope/action_types/_spectral_library_match_run.py` | Pipeline implementation. |
| Registry hookup | `backend/allotrope/action_types/__init__.py` | Type now in REGISTRY. |
| Cache lookup path | `settings.splib07_cache_dir` (default `/data/splib07_cache`) | Worker reads `splib07_<key>.npz` from here. Day 7 mounts the Docker volume. |

---

## 2026-05-12 — Day 6: frontend viewer + the one new api endpoint we need

### What we did

1. **Worker now emits `match_map.png`** alongside `match_map.tif`. The TIF carries integer library indices (good for downstream tooling); the PNG is a categorical RGBA rendering using a deterministic hash so the same material always gets the same colour across actions. Unmatched-in-mask pixels are dim grey; out-of-mask pixels are transparent. Lifted into a visible value range so dark hashes don't blend with the background.

2. **One small new api endpoint** at `GET /actions/{id}/spectral_library_match/at_pixel?row=R&col=C`. It opens the action's `matches.parquet`, filters by `(row, col)`, and returns the top-K rows sorted by rank. Done as a focused endpoint rather than shipping a parquet reader to the browser — keeps the frontend dependency list small.

3. **Frontend viewer** in `ActionDetailPane.tsx` (`SpectralLibraryMatchOutputViewer`). Three areas:
   - **Summary card** — match count, no-match count, pool size, chapters, timing, settings echo.
   - **Match map** (clickable) — the categorical PNG; click any pixel to probe its top-K.
   - **Side rail** — two panels: *Top materials* (the histogram, top 15 names by count) + *At pixel* (top-K matches at the clicked pixel: name, angle in degrees, chapter, ASD subtype, bands used).

4. **CSS** in `frontend/src/index.css` matching the rest of the design system (surface-1 panels, dim-grey muted text, monospace numeric chips).

5. **Registered the viewer** in `ActionDetailPane.tsx`'s type-dispatch alongside the other output viewers.

### Why click-to-pin rather than hover-to-preview

The Day 0 spec talked about hover-to-preview + click-to-pin. In practice the per-click round-trip is fast (parquet read on the api is a few ms), so a hover-to-preview would just generate extra noise and add complexity (debounce, abort previous fetch on next hover, etc.). **Click-to-pin** lands in the same place with a much simpler implementation: state holds one pinned pixel, refresh on click, done. If users ask for hover later, this is an easy upgrade.

### Why no spectrum overlay yet

The original Day 0 vision had the side rail plot the unknown spectrum overlaid with the top-1 library spectrum. That requires another api endpoint (`/library/spectrum?ix=<library_ix>` on the cache file) plus a chart on the frontend. **Deliberately deferred** — the angle in degrees + n_bands used + material name is enough for an analyst to interpret a match, and the spectrum chart is a polish step we can iterate on after we see how analysts actually use the side rail. The viewer's contract is stable; adding the chart later is a non-breaking change.

### What we did NOT do

- **No `match_map.tif` on the frontend.** Browsers can't render int32 GeoTIFFs directly; the PNG is the on-screen artifact.
- **No legend panel.** Materials are identified by name in the side rail and the histogram — colour is a hash, so a colour swatch wouldn't actually tell you *what* it is, just that pixels with the same colour matched the same material. The histogram already conveys this.
- **No download buttons.** All artifacts are served via `/api/actions/{id}/files/<filename>`; a download button row is a generic feature for the action card, not specific to this viewer.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Worker PNG renderer | `backend/allotrope/action_types/_spectral_library_match_run.py` → `_render_match_map_png` | Hashed categorical RGBA. |
| At-pixel endpoint | `backend/allotrope/api/actions.py` → `spectral_library_match_at_pixel` | GET, returns top-K. |
| Viewer component | `frontend/src/components/ActionDetailPane.tsx` → `SpectralLibraryMatchOutputViewer` | Map + side rail. |
| Viewer CSS | `frontend/src/index.css` → `.spectral-match-viewer__*` | Surface-1, muted text. |
| Viewer dispatch | `ActionDetailPane.tsx` → `detail.type === "spectral_library_match"` branch | Wired alongside the other output viewers. |

### Why it matters

This is the first time the end-to-end pipeline is **user-visible**. An analyst can now:

1. Commit an `anomaly_detection_prep` mask.
2. Submit a `spectral_library_match` action (default chapters + top_k=5 in the dialog).
3. Wait a few seconds.
4. Open the action's output and see a categorical material map + the top materials + the top-K at any pixel.

The viewer is intentionally lean — we land the contract first (rendered map + histogram + at-pixel probe), then iterate. Spectrum overlay, alternative colour palettes, and chapter-filtered toggles are all easy follow-ups against a stable JSON/PNG/parquet contract.

Day 7 wires the Docker volume so the splib07 cache directory is shared between the cache-build CLI host and the worker container.

---

## 2026-05-12 — Day 7: docker volume wiring + settings

### What we did

1. **Added a new named Docker volume** `allotrope_splib07` to `docker/docker-compose.yml`, mounted into the worker container at `/splib07_cache` **read-only**. Volume name uses the same `name:` literal pattern as the other volumes so bundle scripts can find it without compose-project prefix mangling.

2. **Added `splib07_cache_dir: str = "/splib07_cache"`** to `backend/allotrope/config.py`. The worker's run module already reads `settings.splib07_cache_dir`; the default now matches the compose mount point.

3. **Updated the run module's fallback** to `"/splib07_cache"` so even without a settings override (e.g. in a test environment) the path resolves where the volume actually lives.

4. **Volume is mounted on the worker only.** The api never reads the cache: it's a worker-time artifact for the `spectral_library_match` action. Keeps the api image surface minimal and lets us re-mount the volume RW on a one-shot CLI container without disrupting the api.

### Operator workflow (end-to-end)

The full chain now runs inside Docker. To bring up splib07 matching on a fresh host:

1. **Get the raw bundle** onto the host (one-time, ~6.7 GB):
   ```
   # Download from USGS https://crustal.usgs.gov/speclab/QueryAll07a.php
   # Extract under data/splib/usgs_splib07/  (gitignored, dockerignored).
   ```

2. **Curate the slim bundle** (one-time per splib07 release):
   ```
   python scripts/curate_splib07.py \
       --raw  data/splib/usgs_splib07 \
       --out  data/splib07_slim \
       --version splib07a
   ```

3. **Build the per-sensor cache** into the docker volume. The cleanest way is a one-shot CLI container that mounts the slim bundle, the sensor spec, and the splib07 volume RW:
   ```
   docker run --rm \
       -v $(pwd)/data/splib07_slim:/slim:ro \
       -v $(pwd)/data/sensor_specs:/specs:ro \
       -v allotrope_splib07:/cache \
       docker-api \
       python scripts/build_splib_sensor_cache.py \
           --slim         /slim \
           --sensor-spec  /specs/aviris_ng.json \
           --cache-dir    /cache \
           --chapters     minerals artificial soils vegetation organics
   ```
   Repeat once per sensor (PRISMA, EnMAP, AVIRIS-NG). Each writes one ~10 MB `splib07_<key>.npz` + `.json` pair into the volume.

4. **The worker, mounted at `/splib07_cache:ro`, picks it up automatically.** No restart needed — the worker resolves the cache file by `sensor_cache_key(...)` at action runtime.

### Why read-only on the worker

Two reasons:

- **Honest dependency direction.** Caches are built once on the host (or in a controlled CLI container), then served to many actions. Letting actions write to the volume would invite drift — a re-run with different settings might silently overwrite a cache another action depended on.

- **Fail-fast on missing caches.** A read-only mount means `FileNotFoundError` if the cache doesn't exist, surfacing the "operator forgot to run build_splib_sensor_cache" case with a clear error message that includes the exact CLI command to run (the run module's error string already does this). A read-write mount would invite us to add "build it lazily inside the worker" code, which would turn a fast action into a 30-second one on first use — exactly the silent-slowness anti-pattern this whole feature was designed to avoid.

### What we did NOT do

- **No automatic cache build at api startup.** Bootstrap container (which does migrations + admin seed) does NOT build splib caches. Reasons: it needs the raw 6.7 GB bundle which is host-side only; it would slow `up` from seconds to minutes; and the operator might not even want splib matching on a given install.

- **No GPU.** `match_pixels` runs CPU-only and finishes in seconds even on anomaly scenes with thousands of pixels. The `docker-compose.gpu.yml` override doesn't need any changes for this feature.

- **No volume backup story written.** Caches are content-addressed reproducible artifacts — losing the volume just means re-running the CLI. We don't need to back them up the way we back up the postgres volume.

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| Volume definition | `docker/docker-compose.yml` → `volumes.allotrope_splib07` | Named volume, no host-path binding. |
| Worker mount | `docker/docker-compose.yml` → `services.worker.volumes` | `/splib07_cache:ro`. |
| Settings field | `backend/allotrope/config.py` → `splib07_cache_dir` | Defaults to `/splib07_cache`. |
| Worker lookup | `_spectral_library_match_run.py` → `_resolve_cache_path` | Uses `settings.splib07_cache_dir`. |
| Cache-build CLI | `scripts/build_splib_sensor_cache.py` | Run via a one-shot Docker container; same image as the worker. |
| Raw bundle | `data/splib/usgs_splib07/` (host-only) | Excluded from git + Docker contexts. |
| Slim bundle | `data/splib07_slim/` (host-only, optional) | Curated subset; can also live on a removable drive. |

### Why it matters — and what this closes out

Day 7 closes the end-to-end chain for the spectral-match feature: data → curation → per-sensor cache → worker action → frontend viewer.

A new analyst on a fresh install only ever needs to:

1. Drop the raw bundle into `data/splib/`.
2. Run two scripts (`curate_splib07.py`, then `build_splib_sensor_cache.py` per sensor).
3. Submit `spectral_library_match` actions from the UI like any other action.

Everything else — the validation, the worker dispatch, the parquet writer, the categorical PNG, the at-pixel endpoint, the side rail in the viewer — is wired through. Step 14.7 on the ROADMAP can flip from 🟡 to 🟢.

The remaining polish (spectrum overlay in the viewer; hover-to-preview; cross-action histogram comparison; ROADMAP-suggested mode-toggle UI) is all additive against this stable contract — no more architectural decisions are pending.

---

## 2026-05-13 — Day 8: tearing out the CLI — caches now build at onboarding

### What we did

The Day 7 design — "operator runs `build_splib_sensor_cache.py` once per sensor" — fell over the moment we tried it. Three problems surfaced:

1. **The CLI is operator-hostile.** Hand-writing a sensor spec JSON, mounting volumes in just the right way, knowing the per-sensor band grid: that's sysadmin work, not analyst work.
2. **PRISMA vendables had `band_fwhm_order = None`.** The dataset builder was discarding FWHM that was already in the HE5 file.
3. **The cache should follow the vendable, not vice-versa.** The vendable *is* the band grid. There's no second source of truth — and the cache key is content-addressed on the grid, so two scenes from the same sensor produce the same cache hash by construction.

So we threw out the operator CLI direction and moved the cache build into scene onboarding. Concretely:

1. **Fixed the PRISMA dataset builder** ([prisma_dataset_builder.py](app/utils/dataset_builder/prisma_dataset_builder.py)) to thread `full_width_at_half_maximum` from the per-band `HyperSpectralBand` objects through to the vendable's `band_fwhm_order`. EnMAP and AVIRIS-NG builders already carried FWHM; this closes the only gap.

2. **Baked the slim bundle into the worker image** via `COPY data/splib07_slim /srv/splib07_slim` in [Dockerfile.worker](backend/Dockerfile.worker). 18 MB, content-deterministic per splib07 release. Bumping the splib version means re-running `curate_splib07.py` once and rebuilding the worker image — no per-host operator step.

3. **Added `build_cache_for_vendable(vendable, sensor_id)`** in [app/spectral_match/library.py](app/spectral_match/library.py). Idempotent: same band grid → same `sensor_cache_key` hash → cache filename collides → no-op. Returns a small diagnostics dict (`status`, `cache_path`, `n_entries`, `n_bands`) the onboarding hook folds into the scene's `extra_metadata`.

4. **Hooked it into scene onboarding** at [scene_onboard.py](backend/allotrope_worker/scene_onboard.py) right after `vendable.pkl` is written. Best-effort: any failure logs and surfaces in `scene.extra_metadata.splib_cache`, but onboarding still succeeds because spectral matching is an optional downstream feature.

5. **Flipped the `allotrope_splib07` volume to RW** on the worker in [docker-compose.yml](docker/docker-compose.yml). Onboarding is the only writer.

### How the cache lifecycle works now

| Event | Effect |
|---|---|
| **First PRISMA scene onboarded** | Builds `splib07_<key>.npz` against PRISMA's band grid. `scene.extra_metadata.splib_cache = {"status": "built", "cache_path": "...", "n_entries": ~700, "n_bands": 239}`. |
| **Second PRISMA scene onboarded** | Same band grid → same cache key. Cache file already exists. Onboarding logs the existing file and moves on with no work. |
| **First AVIRIS-NG scene onboarded** | Different band grid (different FWHM, different bbl, sometimes different wavelength axis). Different cache key. Builds a separate `.npz`. |
| **`spectral_library_match` action submitted** | Worker reads the vendable, recomputes `sensor_cache_key`, opens the matching `.npz`. Already there — no wait. |
| **Volume deleted (`docker volume rm allotrope_splib07`)** | Next onboarded scene rebuilds. No special recovery path needed. |

The `allotrope_splib07` volume becomes a self-managing pool of "one cache per band grid the system has ever onboarded." No GC. No operator commands.

### Why best-effort, not hard failure

If the slim bundle is missing (someone deleted it from the image), or if a sensor lands without FWHM, the onboarding hook **logs and continues**. The scene still onboards; the analyst sees `splib_cache.status: "skipped:..."` in the scene's `extra_metadata`. Spectral matching is one of many actions the scene could run — losing it shouldn't fail the whole onboarding.

This is the same pattern as the visualization render block (a few lines above the new hook) — if thumbnail rendering blows up, the vendable is still saved, just the thumbnail's missing.

### What we did NOT do

- **No CLI deprecated yet.** `scripts/curate_splib07.py` stays (it produces the slim bundle that gets baked into the image — still needed at splib version bumps). `scripts/build_splib_sensor_cache.py` becomes a dev-only escape hatch; the action's missing-cache error message still points at it for emergencies.

- **No bootstrap pre-build.** Bootstrap doesn't pre-warm caches for sensors that haven't been onboarded yet. The first scene of each sensor pays a ~30s cache build during its own onboarding (already a one-time, ~minute-scale operation), and every scene after that is instant.

- **No re-onboard of the two existing PRISMA scenes** (deliberately deferred — the operator will re-add them through the UI).

### Where things live after today

| Thing | Path | Notes |
|---|---|---|
| PRISMA FWHM fix | `app/utils/dataset_builder/prisma_dataset_builder.py` | `band_fwhm_by_position` list, threaded through filter pruning + constructor. |
| Slim bundle (in repo) | `data/splib07_slim/` | 18 MB. Not gitignored. Generated by `scripts/curate_splib07.py`. |
| Slim bundle (in worker image) | `/srv/splib07_slim/` | Baked at image build time. |
| Helper | `app/spectral_match/library.py` → `build_cache_for_vendable` | Idempotent. Returns diagnostics dict. |
| Onboarding hook | `backend/allotrope_worker/scene_onboard.py` (post-vendable, best-effort) | Folds diagnostics into `scene.extra_metadata.splib_cache`. |
| Volume mount | `docker/docker-compose.yml` → `worker.volumes` | RW. |

### Why it matters

This is the moment the spectral-match feature becomes invisible plumbing. Analysts don't run scripts, don't write sensor specs, don't even know `splib07_cache` is a thing. They onboard a scene, they commit an anomaly detection, they submit a spectral-library-match action — and it just works. The cache is born when its source-of-truth (the vendable) is born; it dies when the vendable dies; it's reused for every same-grid scene.

This closes Step 14.7 properly. ROADMAP marker flips 🟡 → 🟢.



