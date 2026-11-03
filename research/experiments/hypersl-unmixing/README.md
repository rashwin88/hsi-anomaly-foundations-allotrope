# HyperSL, unmixing and sub-pixel matching: a fresh, separate study

This is a research side-track, run 2026-11-01. It changes no production code: nothing under `backend/`, `frontend/`
or `docker/`, and nothing in `app/` (`app/spectral_match/namer/` is only imported). Nothing is committed.

HyperSL is treated as a separate experiment, not plugged into our namer. **NH001** (HyperSL's encoder used as the
namer's tower) was dropped before anything was built, so no files exist for it. The Argon analysis was not audited or
reproduced. Two things from it were reused as plumbing: the HyperSL loading recipe and the dataset band-drop lists.

## Question

**Main question:** does a frozen spectral foundation model (HyperSL) help with unmixing, or with naming the material
of a sub-pixel target? And how much does a target's neighbourhood matter, compared with the representation used?

**What HyperSL is:** a per-pixel spectral masked autoencoder (MAE).
- Configuration: 256-d, 8 encoder blocks, the HF `10_base_mask95` checkpoint.
- Training data: EnMAP and DESIS L2A, each band min-max scaled per scene.

**Setup used throughout this study:**
- The embedding z is the upstream `aggregate`: the global token plus the mean of the band tokens. A check against
  upstream `encoder_forward` agrees to within 3e-6.
- Every spectrum is normalised in its own scene's frame: per-band min-max over that scene's pixels.

**Comparator:** N002, our namer with textile/fabric held out. It was run inference-only from its epoch-40
checkpoint (`s3://…/spectral-namer/N002/last.pt`), through `load_namer` / `forward_pixel` / `class_scores`.

## Hypotheses

| | Hypothesis |
|---|---|
| **H5** (headline) | The neighbourhood decides sub-pixel naming. A better background estimate b̂ (none, then scene mean, then local ring) beats a better representation. The ranking of the levels is set by the size of the background error, ‖δ⊥‖. |
| H1 | Within one scene's frame, the encoder keeps linear mixing linear in latent space. |
| H2 | FCLS in latent space is as good as FCLS in reflectance, given the same endmembers. |
| H3 | A scene-adaptive mixture bank matched in latent space beats one matched in reflectance. This is folded into H5 as `bank_lat` vs `bank_refl`. |
| H4 | N002 on the same public data. Covered in "Real targets" and "N002 on the unmixing datasets". |

## Data and sample sizes

**Public unmixing sets:** DC1-3 (HySUPP), EGU-Net synthetic, Samson, Jasper, Urban4.
- Source: G:/My Drive/PS_11/claude_experiments/eval_datasets, copied read-only to `C:/es/hsu/data`.
- Pixel order and scale are recovered by least-squares reconstruction E·A ≈ Y (`common.load`).
- **Samson's ground truth does not reconstruct its Y** (mean |Y − EA| = 0.24, against 0.03-0.05 for Jasper and
  Urban4), in both copies on G:. Samson is therefore left out of H1, H2 and H5.

**Planted backgrounds (H5):**
- Jasper (AVIRIS, ÷1e4) and Urban4 (HYDICE, ÷1e3): 1,000 samples each.
- 8 PRISMA test tiles of 128 × 128 from fine-tune shard 0, one tile per scene: 150 samples each, 1,200 in total. Bands
  dead at more than 1% of in-swath pixels were dropped.

**Library:** `data/family_library` through `build_bank`, which gives the namer's v1 classes. At 380 entries this is
160 man-made spectra: oil 40, plastic 46, textile 14, wood 12, paint 4, metal 10, concrete/asphalt 34.

**Real targets:**
- Gudalur AVIRIS-NG at 4 m: 5 sheets of 16 px each, so 80 target px.
- The same coarsened to 32 m: 4 grid phases, 27 target px from 19 target-phase pairs. **This is anecdotal.**
- DLR HySU: 236 target px, 22 targets (5 materials × 5 sizes).

**Intervals:** the H5 bootstrap resamples library spectra (clusters) 300 times, as in the namer evaluation.

## H5: does the neighbourhood matter? (headline)

**Setup:**
- Each target is planted as x = (1−α)b + αt + n into a real background pixel b.
  - t is a random man-made library spectrum.
  - n ~ N(0, σ_band), with σ estimated from neighbour differences.
  - α is taken from {0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}.
- **Task:** name t's class from 8 options (7 man-made classes + natural), with t's own spectrum removed from the
  library. Being called "natural" counts as wrong.

**Background estimates b̂:**
- none: b̂ = 0
- scene: b̂ = the scene mean μ. The whitened variant is ACE, using the scene covariance Σ.
- local: b̂ = the 3-7 px Chebyshev ring mean, the same ring the namer uses.

**Reflectance arms:**
- `ls_*`: per-candidate least squares. For each library spectrum t_j:
  - α_j = (t_j−b̂)ᵀ(x−b̂)/‖t_j−b̂‖², clipped to [0,1]
  - r_j = ‖x − b̂ − α_j(t_j − b̂)‖
  - the answer is the class of the t_j with the smallest r_j.
- `fix_*_k`: a fixed estimate α̂ = kα (k = 1 is the oracle; also k = 0.85). Then t̂ = (x−(1−α̂)b̂)/α̂ is matched by SAM.
- `ace`: ACE, whitened with the scene statistics.
- `bank_refl`: a mixture bank. 12 scene k-means prototypes × the library × α ∈ {.05, .1, .2, .35, .5, .75, 1}, matched to
  the nearest mixture by Euclidean distance.

**Latent arms:**
- `lat_none`: cos(z(x), z(t_j))
- `lat_mean` / `lat_ring`: t̂ is built from the matching `ls_*` arm's best α, then encoded and matched
- `lat_ring_or`: as `lat_ring`, but with the oracle α
- `bank_lat`: the same mixture bank, matched by cosine in latent space

**N002:** `namer_ring` uses the ring as supplied in production. `namer_self` sets the ring to the pixel itself, which
removes the neighbourhood.

### Accuracy vs fill (8-way, own spectrum left out; 95% cluster-bootstrap interval)

| arm | Jasper α=0.1 | α=0.2 | α=0.5 | α=1 | Urban4 α=0.1 | α=0.2 | α=0.5 | α=1 | PRISMA α=0.1 | α=0.2 | α=0.5 | α=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_none | .10 [.07,.12] | .17 | .40 | .56 | .09 [.07,.11] | .12 | .33 | .55 | .06 [.04,.08] | .09 | .25 | .58 |
| ls_mean | .08 [.06,.10] | .14 | .41 | .51 | .13 [.10,.17] | .19 | .47 | .56 | .30 [.27,.34] | .41 | .59 | .59 |
| **ls_ring** | .19 [.15,.23] | .32 | .52 | .55 | .15 [.13,.18] | .26 | .47 | .56 | .36 [.33,.41] | .47 | .60 | .59 |
| fix_ring_1.0 (oracle α) | .15 | .28 | .50 | .62 | .13 | .21 | .45 | .61 | .33 | .45 | .60 | .64 |
| fix_ring_0.85 | .15 | .28 | .49 | .58 | .12 | .20 | .43 | .57 | .31 | .44 | .56 | .57 |
| ace (scene, whitened) | .17 [.13,.21] | .25 | .39 | .56 | .09 [.07,.12] | .12 | .22 | .37 | .35 [.30,.40] | .48 | .63 | .68 |
| **bank_refl** | .30 [.25,.35] | .41 | .60 | .59 | .17 [.15,.20] | .29 | .53 | .58 | .44 [.40,.48] | .54 | .62 | .58 |
| lat_none | .09 [.06,.12] | .13 | .33 | .49 | .09 [.07,.12] | .10 | .25 | .43 | .09 [.07,.11] | .12 | .27 | .54 |
| lat_mean | .10 | .13 | .33 | .41 | .11 | .15 | .24 | .38 | .28 | .39 | .54 | .54 |
| lat_ring | .14 [.11,.17] | .19 | .35 | .43 | .06 [.05,.08] | .08 | .22 | .39 | .30 [.27,.33] | .42 | .55 | .54 |
| lat_ring_or | .13 | .19 | .38 | .49 | .02 | .03 | .22 | .43 | .32 | .43 | .52 | .54 |
| bank_lat | .17 [.13,.21] | .27 | .38 | .46 | .05 [.04,.07] | .09 | .18 | .33 | .27 [.24,.31] | .39 | .52 | .54 |
| **namer_ring (N002)** | **.39 [.33,.44]** | **.51** | **.65** | **.65** | .13 [.11,.16] | .17 | .27 | .31 | **.53 [.48,.58]** | **.61** | **.68** | **.68** |
| namer_self (ring := pixel) | .07 [.04,.09] | .07 | .24 | .35 | .11 [.08,.15] | .08 | .07 | .17 | .01 [.01,.02] | .04 | .21 | .33 |

Sample sizes: n = 1,000 / 1,000 / 1,200, from 159 / 159 / 201 distinct spectra, across 1 / 1 / 8 scenes. The full
curves are in `figures/h5_accuracy_vs_fill.png`. Accuracy-coverage curves at α = 0.2 are in
`figures/h5_accuracy_coverage_fill0.2.png`; the numbers behind both are in `results/h5_summary.json`.

### Fill at which accuracy reaches half its α = 1 plateau (α½)

| | ls_none | ls_mean | ls_ring | ace | bank_refl | lat_none | lat_ring | bank_lat | namer_ring | namer_self |
|---|---|---|---|---|---|---|---|---|---|---|
| Jasper | .34 | .33 | .16 | .25 | .10 | .37 | .23 | .16 | **.08** | .40 |
| Urban4 | .42 | .29 | .23 | .41 | .20 | .45 | .45 | .46 | **.17** | .59 |
| PRISMA | .55 | .10 | .07 | .10 | .05 | .50 | .09 | .10 | **.05** | .38 |

### Theory check (reflectance `ls_*` arms)

The model predicts a correct answer when α·d > (1−α)‖δ⊥‖ + ‖n⊥‖. Here:
- δ = b − b̂, the error in the background estimate;
- the perpendicular parts (⊥) are taken against (t − b̂);
- d is the class margin of the pure, noise-free target.

Median values per level:

| | ‖δ⊥‖ none / mean / ring | ‖n⊥‖ | d (median) | share with d ≤ 0 | per-sample agreement, none / mean / ring |
|---|---|---|---|---|---|
| Jasper | 0.72 / 0.75 / **0.27** | 0.27 | 0.01-0.03 | 0.41-0.49 | .79 / .80 / .70 |
| Urban4 | 1.03 / 0.69 / **0.43** | 0.47 | 0.02-0.04 | 0.40-0.44 | .79 / .72 / .70 |
| PRISMA | 1.50 / 0.17 / **0.13** | 0.17 | 0.03-0.06 | 0.41 | .86 / .69 / .66 |

**Findings on H5:**

1. **The ranking of the levels follows ‖δ⊥‖, as predicted.**
   - On all three backgrounds the ring has the smallest ‖δ⊥‖ and the earliest α½ among the `ls_*` arms.
   - The ring's lead over the scene mean is large where the mean is a poor background: Jasper and Urban4, where scene
     ‖δ⊥‖ is 0.7-1.0.
   - It is small on the homogeneous PRISMA tiles, where the mean ‖δ⊥‖ is 0.17 against the ring's 0.13.
   - At α = 1 every level converges to the same plateau, about 0.55-0.6. Context only matters below about α = 0.5.
2. **The closed-form bound is too pessimistic.**
   - It predicts α½ of 0.5-0.85 where the observed value is 0.07-0.42.
   - It adds ‖n⊥‖ linearly, but noise inflates every candidate's residual about equally.
   - It gets the *ordering* right and the *threshold* wrong. Per-sample agreement is 0.66-0.86.
3. **About 40% of targets cannot be named even when pure** (d ≤ 0). Their nearest other-class spectrum is closer than
   their nearest same-class spectrum. That sets the ~0.6 plateau. This is a limit of the library's class separability,
   not of context.
4. **Heterogeneous backgrounds cut the ring's advantage.** Compare the bottom half of ring bias (‖b−b̂‖/‖b‖) with the
   top quarter, at α = 0.1:
   - `ls_ring`: 0.28 → 0.10 on Jasper, 0.20 → 0.09 on Urban4, 0.53 → 0.12 on PRISMA.
   - In the top quarter, the scene-mean and global arms catch up or win: ACE scores 0.30 on heterogeneous PRISMA.
   - This is the ((1−α)/α)(b−b̂) amplification predicted in advance.
5. **Fixing α̂ adds little.** The oracle α (`fix_ring_1.0`) is *worse* than per-candidate least squares at α ≤ 0.2.
   Estimating α per candidate is itself discriminative. α̂ = 0.85α costs 0.01-0.07, mostly at α = 1.
6. **Ring contamination is negligible for point targets.** Under a Gaussian PSF with σ = 1 px, the ring mean carries
   0.1% (1 px target), 0.7% (3 px) or 3.8% (5 px) of the centre pixel's target signal. Extended real targets are a
   different matter: see the real-pixel results below.
7. **The representation matters less than the background estimate, and HyperSL's latent space is worse than
   reflectance at every knowledge level:**
   - `lat_ring` < `ls_ring`
   - `bank_lat` < `bank_refl`, by 0.1-0.2 at α ≤ 0.2
   - `lat_none` ≈ `ls_none`
   - Its best showing is PRISMA α = 0.1: `lat_ring` 0.30 against `ls_ring` 0.36.
8. **N002 is a neighbourhood model.**
   - With the ring it is the best arm on Jasper and PRISMA at every α: α½ = 0.05-0.08, and 0.53 at α = 0.1 on PRISMA,
     its training sensor.
   - With ring := pixel it collapses to 0.01-0.07 at α ≤ 0.2. Its level-1 "man-made" call is essentially "differs from
     its ring".
   - On Urban4 (HYDICE, a sensor it never saw) it is weak (0.31 at α = 1).
   - The reflectance mixture bank (`bank_refl`) is the best training-free arm everywhere. It comes within 0.1 of N002
     at α ≥ 0.2 on PRISMA, and beats it on Urban4.

## Real targets (measured, inference only)

**Setup:**
- Positives are the ROI pixels. Negatives are 3,000 pixels at least 10 px from any target (3 px for HySU). They are
  assumed natural, but at Gudalur they can include roads and buildings, which hurts every arm's level 1.
- The library is `family_library`. The secondary "+field" library adds the set's own field spectra: the Gudalur
  point spectra, or the HySU SVC means.
- N002 gets the production 3-7 px ring.

**Material mapping:**

| target | strict class | lenient (also accepted) |
|---|---|---|
| Gudalur green / red / yellow / black nylon sheet | plastic | textile_fabric |
| Gudalur white cotton sheet | textile_fabric | - |
| HySU bitumen | concrete_asphalt | oil_fuel |
| HySU red painted metal | paint_coatings | metal |
| HySU blue / red / green synthetic fabric | textile_fabric | plastic |

### Results with the family library

Columns:
- L1: AUC, and recall at 5% background false alarms.
- L2: lenient accuracy at 100% / 50% coverage.
- majority: targets whose majority vote is right.

| arm | Gudalur 4 m (80 px, 5 tgt) L1 | L2 | maj. | Gudalur 32 m (27 px, anecdotal) L1 | L2 | maj. | HySU VNIR (236 px, 22 tgt) L1 | L2 | maj. |
|---|---|---|---|---|---|---|---|---|---|
| sam | .69 / .19 | .40 / .45 | 2/5 | .39 / .00 | .15 / .00 | 2/19 | .65 / .37 | .38 / .44 | 7/22 |
| ls_ring | .56 / .23 | .49 / .50 | 3/5 | .27 / .00 | .22 / .08 | 6/19 | .55 / .31 | .44 / .53 | 11/22 |
| ls_mean | .54 / .23 | .39 / .45 | 2/5 | .27 / .00 | .26 / .08 | 7/19 | .61 / .30 | .43 / .53 | 10/22 |
| **ace** | .58 / **.49** | **.94 / .97** | **5/5** | .58 / **.37** | **.96 / 1.0** | **18/19** | .63 / .38 | .44 / .54 | 10/22 |
| bank_refl | .32 / .12 | .33 / .42 | 3/5 | .18 / .04 | .26 / .23 | 7/19 | .60 / .41 | .54 / .66 | 13/22 |
| lat_none | .62 / .26 | .33 / .30 | 1/5 | .57 / .00 | .11 / .15 | 3/19 | **.72** / .31 | .51 / .61 | 12/22 |
| bank_lat | .51 / .35 | .46 / .75 | 3/5 | .15 / .00 | .18 / .15 | 3/19 | .57 / **.48** | **.63 / .73** | **16/22** |
| namer_ring (N002) | **.15** / .00 | .47 / .28 | 3/5 | .17 / .00 | .18 / .23 | 1/19 | .58 / .29 | .48 / .70 | 12/22 |
| namer_self | .58 / .30 | .25 / .10 | 1/5 | .60 / .07 | .00 / .00 | 0/19 | .60 / .46 | .50 / .47 | 11/22 |

**With +field:**
- ACE improves to L1 0.89 / 0.79 at 4 m and 0.86 / 0.74 at 32 m, with L2 strict 0.81 / 0.70.
- On HySU every training-free arm improves to L1 AUC 0.73-0.81 and L2 0.58-0.77. `bank_lat` is best there: L2 0.77,
  18/22 targets.
- N002 barely changes: HySU L2 0.47, Gudalur 32 m 0.41. Its library tower does not exploit new entries the way
  direct matching does.

**What this shows:**
- **N002 fails level 1 on the real Gudalur sheets.** AUC is 0.15-0.18, inverted: background pixels look more
  "man-made" to it than the sheets do.
- HySU is VNIR-only (417-903 nm), outside N002's training range. It runs, but it is no better than SAM.
- On these sets ACE, with the scene covariance, is the most reliable namer of the material.
- HyperSL's `bank_lat` is the best arm on the VNIR HySU set, but weak at Gudalur.
- All 32 m numbers rest on 27 pixels.

## N002 on the unmixing datasets

N002 is a matcher, not an unmixer, so it has no abundance aRMSE. It is scored on naming.

**Endmember naming (GT endmembers; the ring is set to the spectrum itself):**
- **DC1-3 (USGS minerals, all natural):** N002 calls 3 of 5 (DC1), 4 of 9 (DC2) and 2 of 6 (DC3) mineral endmembers
  man-made, at P(man-made) 0.64-0.93, mostly as plastic, metal or concrete. VCA endmembers give the same calls.
- **EGU (assumed natural):** 0 of 5 called man-made.
- **Samson (VNIR only, 401-889 nm, outside N002's range):** soil and water are called man-made (0.57 / 0.64).
- **Jasper:** the road endmember is called natural (P(man-made) 0.24), though its best class is the right one
  (concrete_asphalt, 0.55).
- **Urban4:** asphalt and roof are called natural (0.18 / 0.25), while grass and tree are called man-made
  (0.68 / 0.78). The class for asphalt is right (concrete_asphalt, 0.67).

**Per-pixel naming** uses the dominant GT abundance: "pure" is at least 0.9, "mixed" is 0.4-0.9. Up to 6,000 interior
pixels per set; the full table is `results/h7_unmixing_sets_n002.json`.

The table shows the share of natural pixels called man-made. Lower is better; level 1 is trivial on these sets, since
everything is natural.

| | N pure / mixed | namer_ring | namer_self | sam | ls_ring | ace | bank_refl | lat_none |
|---|---|---|---|---|---|---|---|---|
| DC1 | 63 / 3,385 | .38 / .01 | .62 / .02 | .62 / .02 | .00 / .08 | .24 / .00 | .38 / .00 | .00 / .00 |
| DC2 | 1,703 / 4,297 | .04 / .21 | .11 / .25 | .46 / .46 | .08 / .11 | .09 / .19 | .36 / .30 | .10 / .11 |
| DC3 | 0 / 695 | - / .12 | - / .15 | - / .45 | - / .05 | - / .00 | - / .26 | - / .23 |
| EGU | 185 / 5,815 | .07 / .11 | .00 / .13 | .74 / .46 | .48 / .49 | .34 / .37 | .45 / .45 | .14 / .15 |
| Samson (out of range) | 2,266 / 3,720 | .80 / .56 | .94 / .85 | .34 / .20 | .22 / .23 | .38 / .43 | .37 / .37 | .39 / .23 |

**Jasper and Urban4** have man-made pixels (road; asphalt and roof): pure / mixed values below.

| | metric | namer_ring | namer_self | sam | ls_mean | ls_ring | ace | bank_refl | lat_none |
|---|---|---|---|---|---|---|---|---|---|
| Jasper (road 114 pure / 286 mixed) | L1 AUC | .78 / .47 | .03 / .11 | **.97 / .86** | **.995** / .74 | .71 / .56 | .34 / .44 | .44 / .49 | .76 / .69 |
| | L2 concrete acc. | .92 / **.84** | .98 / .82 | **1.0** / .41 | .61 / .73 | .63 / .55 | .11 / .08 | .69 / .41 | .54 / .15 |
| Urban4 (asphalt 374 / 1,565; roof L1 only) | L1 AUC | .16 / .39 | .00 / .01 | .45 / .59 | .59 / .46 | .39 / .46 | .52 / .49 | .45 / .49 | **.69** / .55 |
| | L2 concrete acc. | **.86** / .47 | 1.0 / .80 | .78 / .55 | .86 / .51 | .71 / .40 | .28 / .17 | .79 / .36 | .82 / .47 |

**N002's level 2 is good on real pavement** (0.84-0.92 on Jasper road, 0.86 on pure Urban asphalt).

**N002's level 1 fails on extended man-made surfaces.** It is inverted on Urban4 (AUC 0.16), and setting ring := pixel
drives it to 0. A road or a roof is its own ring, so "differs from the neighbourhood" is false and N002 says natural.

**The pasted Argon table** ("VCA+FCLS vs HyperSL-latent FCLS, mSAD / aRMSE") is an *unmixing* table. N002 cannot be
scored on it. For context only (not an audit): our own `refl` VCA+FCLS aRMSE matches its VCA column. We get DC3 .087,
EGU .084, Jasper .318 and Urban4 .414; it reports .0871, .0840, .3184 and .4138. Our pure-latent FCLS on the same VCA
endmembers is below.

## H1: is latent mixing linear? (`h1_linearity.py`, `figures/h1_linearity.png`)

**Setup:** all pairs of GT endmembers per set (6-36 pairs) and up to 20 triples, mixed in the scene frame.

| | dev_max (median / p90) | t_err (median) | monotone share (min) | tangent-residual slope | \|z(α)−z0\|/(α\|g\|) at α = .05 / .2 / .5 | triples: barycentric error, latent FCLS (frame: 0) |
|---|---|---|---|---|---|---|
| DC1 | .67 / .87 | .15 | .90 | 1.2 | .36 / .12 / .06 | .14 |
| DC2 | .55 / .72 | .16 | .90 | 1.8 | .92 / .66 / .42 | .15 |
| DC3 | .74 / 1.10 | .24 | .90 | 1.9 | 1.13 / .90 / .55 | .21 |
| EGU | .45 / .62 | .06 | .95 | 1.7 | .84 / .60 / .36 | .08 |
| Jasper | .50 / .57 | .20 | 1.0 | 1.8 | .90 / .70 / .50 | .13 |
| Urban4 | .39 / .45 | .09 | 1.0 | 1.9 | .94 / .70 / .44 | .12 |

**H1 is rejected.**
- The latent mixing path bows off the chord by 0.4-0.7 of the chord's length.
- Its *projection* on the chord is nearly monotone and close to α: t_err is 0.06-0.24.
- The departure is smooth, second order (slope about 1.8). The embedding moves fastest at small α, then saturates:
  |z(α)−z0| / α|g| falls to about 0.4-0.5 by α = 0.5.
- Barycentric weights recovered by latent FCLS from known corners are off by 0.08-0.21 on average. In the frame
  (reflectance) they are exact.

## H2: latent FCLS vs physical FCLS, same endmembers (`h2_unmixing.py`, `h2b_scaleinv.py`)

Each cell gives aRMSE / AAD (°). `refl_nnls_norm` is NNLS without the sum-to-one constraint, then normalised. It is
the scale-invariant physical baseline.

| set (N, K) | endmembers | refl FCLS | refl NNLS-norm | frame FCLS | latent FCLS | per-token FCLS |
|---|---|---|---|---|---|---|
| DC1 (5,625, 5) | GT = VCA | .000 / 0.0 | .000 | .000 | .253 / 50.8 | .195 / 41.7 |
| DC2 (10,000, 9) | GT = VCA | .000 | .000 | .000 | .113 / 17.1 | .110 / 16.9 |
| DC3 (11,025, 6) | GT | .000 | .000 | .000 | .261 / 56.8 | .219 / 51.8 |
| | VCA (mSAD 4.2°) | .087 / 24.9 | .094 | .087 | .176 / 44.9 | .160 / 42.1 |
| EGU (40,000, 5) | GT | .058 / 7.8 | **.018 / 2.5** | .056 | .142 / 22.6 | .140 / 22.3 |
| | VCA | .084 | **.023** | .083 | .164 | .161 |
| Jasper (10,000, 4) | GT | .078 / 6.7 | **.050 / 3.6** | .076 | .106 / 10.6 | .098 / 9.6 |
| | VCA (mSAD 22.5°) | .318 | **.203** | .324 | .339 | .339 |
| Urban4 (94,249, 4) | GT | .214 / 25.6 | **.059 / 4.4** | .217 | .170 / 20.2 | .165 / 19.7 |
| | VCA (mSAD 25.1°) | .414 | **.256** | .431 | .401 | .394 |

**H2 is rejected.**
- Latent FCLS, pooled or per-token, is worse than a physical baseline on every set and endmember source.
- Its only "win" is over plain reflectance FCLS on Urban4 (.170 against .214). That gap is a scale effect: Urban4's
  GT endmembers are in different units from Y. A scale-invariant physical method removes it, at .059.

## Verdict: where HyperSL shows real promise

- **For unmixing: not shown.**
  - The latent mixing path is curved (H1).
  - Latent FCLS is clearly worse than physical unmixing with the same endmembers (H2).
  - Per-token FCLS helps only slightly over pooled.
- **For sub-pixel naming: no gain over reflectance.**
  - Matched in latent space, it trails the same pipeline in reflectance at every background level (H5).
  - Its mixture bank is the best arm only on the VNIR-only DLR HySU set (L2 0.63, or 0.77 with field spectra). That is
    the one case where it beats our namer and the reflectance arms, and it rests on 22 targets.
- **What matters most is the background estimate,** as the δ⊥ analysis predicts:
  - local ring better than scene mean, better than none;
  - a scene-adaptive mixture bank in reflectance is the strongest training-free arm.
- **N002:**
  - It is the best namer of planted point targets on its own sensors (PRISMA, AVIRIS) at every α.
  - Its level 1 is a neighbourhood-contrast detector. It fails on extended man-made surfaces (Urban4 roads, the
    Gudalur sheets) and on sensors outside its range (HYDICE, VNIR-only).
  - Its level-2 class call on real pavement is good (0.84-0.92).

## Caveats

- **Sample sizes:**
  - H5 uses 1-8 scenes per group and about 160-200 distinct target spectra.
  - 40% of targets cannot be named even when pure (class-level, own spectrum left out), so absolute accuracies cap
    near 0.6.
  - Gudalur 32 m is 27 px and HySU 22 targets: treat both as anecdotal.
- **Synthetic vs real:**
  - H5 targets are planted single pixels with no PSF; real extended targets behave differently (see N002 above).
  - The H5 noise σ includes scene texture, so it overestimates sensor noise, especially on PRISMA.
- **Grade-D ground truth:** Jasper, Urban4 and Samson abundances are themselves FCLS-derived. Samson's ground truth
  is inconsistent and was dropped from H1, H2 and H5.
- **Scales:**
  - Urban4 ÷1e3 and Jasper ÷1e4 are assumed to give reflectance; N002's poor Urban4 numbers could partly be a scale or
    sensor shift.
  - EGU endmembers are assumed natural.
- **Normalisation choices:**
  - HyperSL sees each scene's own per-band frame.
  - Library spectra are not clipped to it.
  - t̂ for the latent arms is clipped to [0, 1.5] reflectance.
  - Jasper, Urban4 and PRISMA bands are non-contiguous after water-band removal, while HyperSL's tokenizer convolves
    over the band index.
- **Scope:** the "real target" backgrounds at Gudalur are not guaranteed natural.

## Files

- `common.py`: loaders, scene frame, HyperSL encoder, batched FCLS
- `sanity.py`: loader and encoder checks
- `h1_linearity.py`
- `h2_unmixing.py`, `h2b_scaleinv.py`
- `h4_real_pixels.py`: first real-pixel pass on Jasper / Urban4 (thresholds 0.8 / 0.5), superseded by h7
- `h5_neighbourhood.py`, `h5_summarise.py`
- `h6_real_targets.py`
- `h7_unmixing_sets_n002.py`
- `prisma_tiles.py`
- `results/*.json`, `figures/*.png`, `logs/*.log`

**Run:** in WSL, `bash /mnt/c/es/hsu/run.sh <script>`, using the venv `~/hypersl_eval/.venv` on the local RTX 4000.

**External data and checkpoints (not in the repo):**
- `C:/es/hsu/data`: the copied public sets, about 560 MB on C:
- `~/hsu/gud`: Gudalur, extracted, in WSL
- `~/hsu/prisma_tiles.npz` and `~/hsu/N002/last.pt`
