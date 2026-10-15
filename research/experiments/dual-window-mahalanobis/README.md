# Indradhanu + Mahalanobis: scoring the reconstruction residual with RX

**Status:** research result, for review before any product change. This branch carries the
experiment code and results only. It does **not** change `compute_score` or the
`anomaly_scoring` action. That would be a follow-up, designed separately.

**One-line result:** take Indradhanu's residual (cube − reconstruction), project it onto its
top-30 principal components, and score each pixel by its Mahalanobis distance. Nothing else
changes: same model, same reconstruction, no retraining. The scoring step alone lifts mean
ROC-AUC on the four labelled PRISMA scenes from **0.662** (the product default, `combined`)
to **0.839**. On HAD100 it lifts AUC from **0.848** (Indradhanu L1) to **0.973**.

The full chronological log (window sweeps, MUUFL, ABU/HYDICE, land-cover rings) is in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). This README is the reviewer's summary.

---

## 1. The problem with the current score

Indradhanu is a masked autoencoder. It learns what normal ground looks like and reconstructs
each pixel's spectrum, x̂, from its context. The product then scores the **size** of the
residual r = x − x̂ (165 bands):

- L1: `mean_b |r_b|`
- SAM: the angle between x and x̂
- combined: `0.5·L1_norm + 0.5·SAM_norm` (the current default)

All three treat every direction in residual space as equally surprising. L1 is essentially a
Euclidean length: a residual of 0.01 in band 40 counts the same as 0.01 in band 140, and
independent errors count the same as errors that move together.

Real background residuals are neither isotropic nor independent:

- **Errors are correlated across bands.** When the model under- or over-shoots a pixel's
  brightness, the residual moves in the same direction in every band at once. Neighbouring
  bands share physics (water vapour edges, vegetation red edge), so their errors move
  together.
- **Error variance differs a lot between directions.** Along a few directions (overall
  brightness scale, the shape of vegetation or soil continua, sensor striping), *normal*
  pixels already have large residuals. Along most other directions, normal residuals are
  tiny.

So L1 makes two mistakes. It flags bright-but-normal pixels whose large residual lies along
a common error direction. It also misses real anomalies whose residual is small in absolute
terms but points in a direction background pixels almost never take.

## 2. The fix: Mahalanobis distance on the residual

Estimate the mean μ and covariance Σ of the residual over the scene's valid pixels. Then
score each pixel by

```
D²(r) = (r − μ)ᵀ Σ⁻¹ (r − μ)
```

Diagonalise Σ = V Λ Vᵀ, with eigenvalues λ₁ ≥ λ₂ ≥ … and eigenvectors v₁, v₂, …. Then

```
D²(r) = Σ_k  ( v_kᵀ (r − μ) )² / λ_k
```

Each principal direction's contribution is divided by that direction's own background
variance. A residual of a given size counts for little along a direction where normal pixels
vary a lot (large λ_k), and for a great deal along a direction where they barely vary
(small λ_k). This is "whitening": after dividing, every direction has unit background
variance, and D² is just the squared Euclidean length in that whitened space.

### Worked example (2 dimensions)

Say background residuals have standard deviation 10 along a "brightness" direction and 1
along a "shape" direction, i.e. Σ = diag(100, 1).

| pixel | residual r | Euclidean ‖r‖ (≈ what L1 sees) | D² = r₁²/100 + r₂²/1 |
|---|---|---|---|
| A: bright roof, normal material | (10, 0) | **10** | 1 |
| B: small unusual material | (0, 3) | 3 | **9** |

L1 ranks A above B. Mahalanobis ranks B, a 3σ outlier, nine times above A, a 1σ wobble.
That reversal is the whole effect, repeated across 30 dimensions and a million pixels.

### Why this is the right statistic

This is exactly the RX detector (Reed & Yu, 1990), applied to the residual instead of the
raw cube. Suppose background residuals are roughly Gaussian, N(μ, Σ), and an anomaly is any
pixel drawn from something else with unknown signature. Then:

- The generalised likelihood ratio test for "this pixel is background" reduces to D².
- Under the background hypothesis, D² ~ χ²_d. That gives a principled, scene-independent
  false-alarm calibration if one is ever needed.
- Among detectors that must not assume the target's spectrum, RX is the constant-false-alarm,
  invariant optimum under that Gaussian model.

L1 and SAM have no such justification. They are the right statistic only if Σ ∝ I.

### Why apply RX to the residual, not the cube?

RX on the raw cube (PCA30-RX here) needs the *cube* to be roughly Gaussian. A satellite
scene is a mixture of land covers (water, crops, urban, bare soil), so the cube's
distribution is multimodal. One global Gaussian fits it badly, and the dominant land covers
inflate Σ.

The reconstruction model removes most of that structured, explainable variation. What it
leaves behind is closer to a single, centred, unimodal cloud, which is the setting where RX's
assumptions hold. The division of labour:

- Indradhanu explains away "what land cover is this pixel".
- RX on the leftover asks "is what's left unusual in an unusual direction".

The data support this, but the gain over plain RX is **small**; see §5.

### Why the top 30 principal components

Plain RX on all 165 bands was dropped from the product on 2026-05-11. Adjacent bands are so
correlated that Σ is near-singular, its smallest eigenvalues are noise, and dividing by them
sent distances to ~1e11 (see CLAUDE.md). Projecting onto the top d = 30 PCs keeps the
directions that carry real variance and discards the noise floor. A 30 × 30 covariance has
465 free parameters, and we estimate it from ~10⁶ valid pixels, so it is very well
conditioned. Implementation:
[step1b_compare_scores.py](../indradhanu-fp-filter/step1b_compare_scores.py)
`pca_mahalanobis`, and [dw_core.py](dw_core.py) `whitened_pca`.

### Variant: local mean, global covariance

The best PRISMA number comes from a small refinement. It subtracts the mean of a local ring
(inner 21 px guard, outer 61 px), but keeps the **global** covariance:

```
D²_local(r) = (r − μ_ring)ᵀ Σ_global⁻¹ (r − μ_ring)
```

The model has a per-land-cover bias: its residual over water is not centred where its
residual over crops is. The local mean removes that offset. The covariance is still estimated
from the whole scene.

The fully local version (ring covariance too) is worse everywhere on PRISMA. A ring of
~800 pixels cannot estimate 465 covariance parameters reliably. It is noisy, and it absorbs
neighbouring targets. EXPERIMENT_LOG.md steps 1 and 4 have the 38-window sweep.

## 3. Results on the four labelled PRISMA scenes

Model: production Indradhanu, `hyperspectral_segformer_mae_v0.2.0_epoch200.pt`, run through
the product path (band_filter_apply → full-scene inferencer). Ground truth: the Stage-II
Set-1..4 masks, 886 labelled objects. Pixel ROC-AUC, over spatially valid pixels.

| scene | Indradhanu combined (**product default**) | Indradhanu L1 | Indradhanu SAM | MNF-RX (product classical) | PCA30-RX (cube) | **Indradhanu + Mahalanobis** | Indradhanu + local-mean Mahalanobis 21/61 |
|---|---|---|---|---|---|---|---|
| Set 1 | 0.7088 | 0.8958 | 0.4737 | 0.8125 | 0.8950 | **0.9335** | 0.9419 |
| Set 2 | 0.6890 | 0.7045 | 0.5783 | 0.7617 | 0.8197 | **0.8479** | 0.8462 |
| Set 3 | 0.7267 | 0.6613 | 0.7294 | 0.6577 | 0.7662 | **0.7088** | 0.7492 |
| Set 4 | 0.5233 | 0.5473 | 0.4970 | 0.7767 | 0.8448 | **0.8660** | 0.8768 |
| **mean** | 0.662 | 0.702 | 0.570 | 0.752 | 0.831 | **0.839** | **0.854** |

Head-to-head for Indradhanu + Mahalanobis:

- vs product default (combined): wins 3/4, +0.177 mean
- vs L1: wins 4/4, +0.137
- vs MNF-RX: wins 4/4, +0.087
- vs PCA30-RX: wins 3/4, +0.008

**Set 3 is the exception.** There, PCA30-RX on the raw cube (0.766) beats residual-Mahalanobis
(0.709). The plain combined score (0.727) also edges it.

**Objects found at a fixed false-alarm budget** (step 5 counting, 2 px neutral zone, out of
886). This is the analyst's view, because a fixed top-k% cut rewards detectors that flag more.

| detector | N=25 spots | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| Indradhanu combined (product default) | 0 | 42 | 70 | 110 |
| Indradhanu L1 | 0 | 61 | 100 | 135 |
| MNF-RX (product classical) | 67 | **111** | **149** | 200 |
| PCA30-RX (cube) | 10 | 22 | 136 | 211 |
| **Indradhanu + Mahalanobis** | **85** | 106 | 133 | **237** |
| Indradhanu + local-mean Mahalanobis 21/61 | **85** | 102 | 134 | **237** |

At the tightest budget (25 false-alarm spots per scene), the current default finds nothing.
Residual-Mahalanobis finds 85 objects.

## 4. Results on HAD100 (public AVIRIS-NG benchmark)

Indradhanu has never seen AVIRIS-NG, radiance, or 2–5 m pixels. The radiance was mapped onto
the model's training statistics; the caveats are in the step3 docstring. So this tests
transfer, not tuned performance.

### 4a. 94 full tiles, 466 objects (step 3)

| detector | mean AUC | Indradhanu + Mahalanobis wins / losses | Wilcoxon p |
|---|---|---|---|
| **Indradhanu + Mahalanobis** | **0.973** | – | – |
| Indradhanu + local-mean Mahalanobis 5/15 | 0.974 | – | – |
| PCA30-RX (cube) | 0.968 | 51 / 43 | 0.057 |
| MNF-RX (product classical) | 0.888 | 79 / 15 | 2.1e-13 |
| Indradhanu L1 | 0.848 | 90 / 4 | 8.5e-17 |

### 4b. Official HAD100 protocol, against the published leaderboard (step 6)

Protocol rebuilt from HAD100's `main.py`: 100 crops of 64 × 64, the 276 good bands, mAUC =
mean per-image AUC. Published rows are from Li et al., "You Only Train Once", IEEE TGRS 2023,
Table V.

**Calibration check:** our GRX reproduces the paper's GRX to four decimals at all three band
settings, so the harness matches the published one.

| method | 50 bands | 100 bands | 200 bands |
|---|---|---|---|
| AETNet-KCSR (paper) | 0.9948 | 0.9897 | 0.9869 |
| KCSR (paper) | 0.9939 | 0.9865 | 0.9860 |
| CR (paper) | 0.9929 | 0.9859 | 0.9849 |
| AETNet (paper) | 0.9925 | 0.9875 | 0.9818 |
| **Indradhanu + local-mean Mahalanobis** (full spectrum) | 0.9748 | 0.9748 | 0.9748 |
| **Indradhanu + Mahalanobis** (full spectrum) | 0.9740 | 0.9740 | 0.9740 |
| LRX (paper) | 0.9775 | 0.9692 | 0.9720 |
| GRX (paper / ours) | 0.9799 / 0.9799 | 0.9714 / 0.9714 | 0.9649 / 0.9649 |
| AutoAD (paper) | 0.9688 | 0.9661 | 0.9642 |
| LREN (paper) | 0.9583 | 0.9286 | 0.9309 |
| RGAE (paper) | 0.9003 | 0.8876 | 0.8823 |
| Indradhanu L1 (full spectrum) | 0.8445 | 0.8445 | 0.8445 |

Indradhanu needs its 165-band grid and cannot take a band subset, so one full-spectrum score
fills every column. The fairest column is 200 bands. There Indradhanu + Mahalanobis ranks 5th
of 11: above LRX, GRX, AutoAD, LREN and RGAE, and 0.013 below AETNet-KCSR. AETNet is trained
on HAD100's own anomaly-free AVIRIS-NG images. The CR/KCSR leaders are local
collaborative-representation detectors, which never estimate a local covariance.

## 5. RX scoring check

Residual-Mahalanobis *is* RX, so the question is: does running it on the Indradhanu residual
beat running it on the cube? Every RX variant measured:

| RX variant | input | PRISMA mean AUC (4 scenes) | HAD100 94 tiles | HAD100 official (mAUC) |
|---|---|---|---|---|
| GRX, all bands (pseudo-inverse) | cube | not run (unstable at 165 bands; dropped from the product) | – | 0.9634 (276 b) · 0.9649 (200) · 0.9714 (100) · 0.9799 (50) |
| PCA30-RX | cube | 0.831 | 0.968 | 0.9648 (276 b) · 0.9636 (200) · 0.9603 (100) · 0.9797 (50) |
| MNF-RX, 10 comps (**product classical**) | cube | 0.752 | 0.888 | – |
| LRX, dual window | cube | 0.743 (11/31) | – | 0.9566 (7/11, 200 b) |
| **RX on Indradhanu residual (PCA30)** | residual | **0.839** | **0.973** | **0.9740** (full spectrum) |
| RX on residual, local mean + global cov | residual | **0.854** | **0.974** | **0.9748** |

Reading:

1. **Against the product's current RX (MNF-RX, 10 components), residual RX wins clearly.**
   It gains +0.087 on PRISMA (4/4 scenes) and +0.085 on HAD100 (79/94 tiles, p ≈ 2e-13).
   PCA30-RX on the cube also beats MNF-RX on both datasets, which suggests MNF-RX's 10
   components are too few. That is a separate, untested hypothesis.
2. **Against a well-built cube RX (PCA30), residual RX is ahead but only just.** The margins
   are +0.008 on PRISMA (3/4, with a loss on Set 3) and +0.005 on HAD100 (51/94, p = 0.057,
   not significant).
3. **On the official HAD100 protocol, GRX at 50 bands (0.9799) beats residual RX (0.9740).**
   Fewer bands help classical RX on these small 64 × 64 crops, and Indradhanu cannot use a
   band subset. At 200 bands or the full 276, residual RX is ahead (0.9740 vs 0.9649 / 0.9634).
4. **Most of the gain comes from whitening, not from the model.** L1 → Mahalanobis on the
   same residual is worth +0.137 (PRISMA) and +0.125 (HAD100). Cube RX → residual RX is worth
   +0.005 to +0.008.

   The honest framing: **Indradhanu's residual is a good input to RX, and RX is a far better
   score than L1.** The residual adds a small, consistent edge on top of a strong classical
   baseline. It does not add a large one.

## 6. Caveats

- **Four PRISMA scenes cannot give significance.** The smallest possible Wilcoxon p with
  n = 4 is 0.125. The PRISMA rows rest on direction and the size of the gap. Validate on more
  PRISMA/EnMAP scenes before switching the product default.
- **AUC and objects-at-budget disagree in places.** MNF-RX finds the most objects at 50–100
  spots per scene on PRISMA despite a lower AUC.
- **HAD100 input is radiance mapped onto reflectance training statistics.** The model was
  never trained on it, so the numbers measure transfer.
- **The ground truth may be offset by ~1 px** on the PRISMA sets. That affects all detectors
  equally.

## 7. Recommendation, and what review is being asked for

1. Make **residual Mahalanobis (global, PCA30)** the Indradhanu score, replacing `combined`.
   The local-mean variant is the better number on PRISMA, but it adds a window parameter;
   start with the global version and offer local-mean as an option.
2. Do **not** ship dual-window with a local covariance for satellite scenes.
3. Separately, revisit MNF-RX's component count.

Please review:

- the scoring maths in `step1b_compare_scores.py::pca_mahalanobis` and `dw_core.py`;
- the evaluation protocol: valid-pixel AUC, the step-5 object counting, the HAD100 harness
  calibration;
- whether the evidence is enough to open the product change (a new `compute_score` method
  plus an `anomaly_scoring` default), which would go through the normal design → chunks flow.

## Reproducing

Data lives outside the repo:

- PRISMA: `C:\Users\FITT\Documents\stage1_datasets`
- public sets: `C:\Users\FITT\Documents\public_had_datasets\bench`

Run on Windows with `python -X utf8`.

```
python ../indradhanu-fp-filter/step1_candidates.py      # Indradhanu L1/SAM/combined maps, GT
python ../indradhanu-fp-filter/step1b_compare_scores.py # residual-Mahalanobis and PCA30-RX
python step1_prisma.py                                  # PRISMA window sweep
python step3_public.py                                  # ABU / San Diego / HYDICE / River / HAD100
python step4_segmented.py                               # local-mean, land-cover rings, MNF-RX
python step5_localmean_public.py
python step6_had100_protocol.py                         # official HAD100 leaderboard protocol
python compare_auc.py <results.json> ...                # paired wins and Wilcoxon p
python summarise_results.py                             # writes results/auc_summary.json
```

[results/auc_summary.json](results/auc_summary.json) holds every per-scene AUC quoted above.
It was generated from the step outputs on 2026-10-15.
