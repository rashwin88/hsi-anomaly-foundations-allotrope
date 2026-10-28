# A foundation model for spectral matching

**Status:** proposed, for agreement (2026-10-28). Nothing here is built or run. A detour from the
Indradhanu Heavy (F002-heavy) work; it needs none of the H100 block.
**Builds on:** family matching (`docs/lld/spectral-family-match.md`), the wavelength-aware spectral stem
(`docs/lld/spectral-stem.md`), the implant bench (`docs/lld/implant-bench.md`), HyperSL's evaluation
(`research/experiments/hypersl-validation/summary.md`).


> **Prior evidence, added 2026-10-28 (after the agent's draft): this idea has been gated once already.** The
> `unmixing_next` experiment (Colab, 2026-10-14/15; team Drive `G:/My Drive/PS_11/claude_experiments/unmixing_next/README.md`)
> pre-registered a "Phase 6 - learned retrieval" step and did **not** run it: *"Its gate is closed: library-like
> signal on real targets in only 2 of 4 D1 scenes."* The same experiment found nothing beating the baselines on
> held-out data (Simplex ACE with per-family calibration the lead but not adopted; no fill estimator works on real
> deployed materials). So the honest status is **not yet**: E0 is worth running only as an explicit re-test of that
> gate - e.g. with the H5 depth threshold and Simplex ACE's calibration now in place, or new real targets with
> library-like signal - and it should be framed and pre-registered that way. The Gudalur / oil data (D3 / D4) are in
> that Drive folder (`runs/baseline_data`; 455 MB in all), so the data blocker in Decisions item 3 is resolved.

## The answer in short

- **Can we?** Yes, and cheaply. A per-pixel spectral encoder is small (~0.5 M parameters), we already own a
  stem that reads any band list in absolute reflectance, and the training pairs can be synthesised from
  library spectra and real pixels we already have on S3.
- **Should we?** Only as a gated experiment against Simplex ACE, not as a planned replacement. The limit is
  not compute or architecture: it is that we have **411 curated library spectra (176 man-made in 10
  families, one of them a single glass spectrum) and no real material-labelled 30 m targets** apart from 5
  Gudalur sheets and the Gulf oil scenes. A learned matcher can beat ACE on synthetic implants and still
  fail on real materials - the absorption-feature check did exactly that (62 % right planted, 0 % on the
  real sheets; LLD section 3).
- **Cheapest first experiment:** a frozen-stem probe (E0, below) on the laptop GPU, a few hours, scored on
  the same planted set, Gudalur and oil as Simplex ACE. If E0 cannot match ACE, stop.

## 1. What "a foundation model for spectral matching" should mean here

A **per-pixel spectral encoder** `f(spectrum, wavelengths, local background) -> embedding` such that the
embedding of an image pixel lands near the embeddings of library spectra of the same material family,
and stays there when the following change:

| Nuisance | Why it matters here | Measured or stated where |
|---|---|---|
| Band set and response | PRISMA 234, EnMAP 224, AVIRIS-NG 372, SpectralEarth 202 native bands; library at 1 nm (ASD axis, 2,151 channels) | shard manifests; `data/family_library/sources.md` |
| Sub-pixel fill | the object is usually under 30 % of a 30 m pixel; whole-pixel SAM names the right man-made family for 6 % of pixels at 30 % fill | LLD section 1 |
| Illumination / atmosphere residuals | L2 reflectance carries correction error; ISOFIT vs PACO biases on EnMAP were measured | memory note on the EnMAP ISOFIT pipeline |
| Noise | EnMAP stated SNR 354 VNIR, 260 SWIR | LLD section 4 |
| Nodata / saturated bands | the EnMAP product scene had 5-18 invalid bands per anomaly pixel (29 % not scored) | LLD section 7 |

The output is **a score per family**, so it slots into the existing calibration (per-family percentile
against the scene's ordinary pixels) and decision (p\* = 1 - alpha / F). The foundation model replaces
one box of the family matcher - "score each family" - not the calibration or the decision, which are
what keep false names at or below alpha on ordinary pixels.

**Contrast with what exists.**

| | Input scale | Spectral context | Background | Library use | Fill |
|---|---|---|---|---|---|
| SAM (`app/spectral_match/sam.py`) | any (angle ignores brightness) | none, fixed bands | none | nearest single spectrum | ignored |
| Simplex ACE (`family_match.py`) | 0-1 reflectance | none | 13 x 13 ring mean, scene whitening | non-negative mix of a family's spectra | implicit, not reported |
| HyperSL | **per-scene, per-band min-max** | conv over band index + attention | none (no spatial input) | not designed for it; library spectra have no defined input | ignored |
| Proposed | 0-1 reflectance, global `WavelengthStats` z-score | wavelength-tagged bins | ring mean as a second input | embeddings of resampled library spectra | augmentation range, optional head |

HyperSL's embeddings also depend on band sampling as much as on material: an ECOSTRESS entry fed at its
lab sampling found its own PRISMA-resampled copy as nearest neighbour only 28 % of the time (memory note,
measured 2026-09-23). That is the property a matching model must not have, and the reason not to start
from HyperSL's checkpoint.

## 2. Training objective

### Options

| Option | What it learns | For matching |
|---|---|---|
| A. Masked spectral modelling (HyperSL-like) | to fill in hidden bands from the rest | Learns what spectra look like, not which differences matter. Nothing pulls a 20 % roof pixel toward a roof library spectrum. Good initialisation only. |
| B. Contrastive / metric learning with library spectra | embedding distance = material family | Directly the task. Needs synthetic pairs, so it learns the augmentation model; risk of fitting it rather than reality. |
| C. Hybrid: A-pretrained stem, B fine-tune | both | **Recommended.** The stem from SpectralEarth pretraining has seen ~2.5 billion real EnMAP spectra (below); the contrastive stage teaches identity. |

### Recommended: C, with these pieces

**Anchor.** A real background pixel `x` (from a shard) and its ring mean `m`, with a library spectrum `t`
mixed in: `x' = (1 - f) x + f * t`, the same linear-mixing implant `app/evaluation/implant.py` uses. The
encoder sees `(x', m)` - pixel and its surroundings - because the LLD's central finding is that matching
must compare against the surroundings, not the whole library. Without `m` a learned model reproduces SAM's
failure (the pixel is mostly grass).

**Positives / negatives.** Supervised contrastive (SupCon) over **families**, with the library spectrum's
own embedding as a prototype: pull `f(x', m)` toward `g(t)` and other same-family entries, push from other
families. Entry-level identity is a secondary, lower-weighted term (families are the product answer;
entries are a hint). Add a **"none" class**: `f = 0` (untouched pixel against its ring) must land near no
man-made prototype, so the model learns to abstain on ordinary pixels.

**Augmentations**, each with its source:
- **Band set:** library resampled with the product's Gaussian resampler (`app/spectral_match/resample.py`)
  to the anchor pixel's real sensor bands and FWHM, plus `sample_band_subset` (min 40 bands) and
  `band_jitter_nm` from the stem trainer.
- **Fill:** f drawn from 0.05-0.5 (the planted set's range), log-uniform so low fill is not starved.
- **Non-linear mixing:** a small share with a bilinear term (`f (1 - f) t * x`) for the
  multiple-scattering case.
- **Brightness / illumination:** a scale on `t` of 0.6-1.4 and a small additive offset, so brightness
  invariance is learned to the extent it should be, instead of imposed by normalisation.
- **Atmosphere residual:** a smooth multiplicative ripple (low-order spline over wavelength, few %), and
  extra noise near the water-vapour edges.
- **Noise:** EnMAP SNR model (354 / 260); PRISMA and AVIRIS-NG at their own noise if known, else EnMAP's.
- **Invalid bands:** random dropouts in the shapes seen in the product (scattered saturation).

**Why per-scene min-max must be avoided.** It maps the same material to different numbers in every scene
(it depends on that scene's brightest and darkest pixel per band), it stretches a near-constant band into
the full 0-1 range, and a library spectrum has no scene of its own, so it has no defined input at all.
HyperSL needed every library spectrum forced into the scene's per-band min/max frame first (memory note).
**`WavelengthStats` is the opposite:** one fixed mean/std curve over wavelength, measured once on
SpectralEarth's train split and stored in the checkpoint, read at any wavelength by interpolation. It is
a per-wavelength affine map, identical for an image pixel and a library spectrum, so absolute reflectance,
band depth and brightness all survive. Its only precondition is that the input is 0-1 surface
reflectance - which is the open radiometric-scale gap (section 7).

## 3. Architecture

- **Spectral stem:** `SpectralReducedCompressor` (reduced attention over 40 nm wavelength bins from
  400 nm, 53 bins; per-band weights computed from wavelength; dead bands drop out exactly). It already
  takes `(B, N, H, W)` with any `(N,)` wavelength list, so a library spectrum is just a `(1, N, 1, 1)`
  input at the same wavelengths as the pixel. Today it compresses to D = 32 channels for the SegFormer
  encoder; here it would keep the bin tokens instead.
- **Encoder:** 4 transformer layers over the 53 bin tokens, width 64, plus a 2-token background input
  (the ring mean `m` encoded by the same stem, and `x' - m` encoded by it too). Estimated
  ~0.3-0.5 M parameters (HyperSL is 9.16 M with 256 x 8 layers; we need far less because there is no
  reconstruction decoder and the vocabulary is ~17 families).
- **Heads:** a projection to a 128-d unit-norm embedding; optional fill regression (training signal only,
  not shown in the product: the LLD says fill is not reportable without an in-scene spectrum).
- **Library side:** `g(t) = f(t, m = t)` - a library spectrum is "a pure pixel with itself as
  background" - or a separate small library tower sharing the stem. Start with the shared form (fewer
  parameters, one space by construction).
- **Initialisation:** the reduced-attention stem weights from the SpectralEarth MAE runs
  (`research/experiments/indradhanu-se-pretrain/`, E004 / the FT04 lineage). Which checkpoint is the
  best stem donor is not established; E0 checks it.
- **Local context beyond the ring:** not in v1. The ring mean is the context that the classical results
  justify; spatial patches would turn this back into a scene model.

## 4. Data

| Source | Count | Use |
|---|---|---|
| SpectralEarth `se-v1z` (EnMAP L2A, 202 bands) | 150,000 train / 3,000 test patches of 128 x 128 = **~2.46 B / 49 M pixel spectra** (the full `se-v1` split is 512,849 patches, ~8.4 B); 533 GB on S3 | background pixels and rings for implants; MAE stem pretraining (already done) |
| EnMAP `native-v1z` (224) | 14,542 / 950 patches (~238 M / 16 M pixels, upper bound: tiles need only 10 % valid) | background, sensor 1 |
| PRISMA `native-v1z` (234) | 14,802 / 1,570 patches (~243 M / 26 M); **benchmark sites held out within 30 km** | background, sensor 2 |
| AVIRIS-NG `native-v1` (372) | 22,377 / 2,270 patches (~367 M / 37 M), 79 flight lines; benchmark sites held out | background, sensor 3 |
| Family library (`data/family_library`) | **411 spectra, 17 families: 176 man-made in 10, 235 natural in 7** (glass 1, paint 8, wood/paper 8, building 11, metal 13) | positives and prototypes |
| Raw pools behind it (local zips in `Downloads/`) | USGS splib07b cvASD: 2,457 spectra (360 are lab organics, excluded); ECOSTRESS: ~6,100 spectra of which 72 man-made, many minerals thermal-only | natural-family expansion (uncapped ~1,900 minerals); man-made barely grows |
| `data/splib07_slim` | 519 (3 usable man-made) | not used for training |

The background pool is effectively unlimited: a 1 % pixel sample of `se-v1z` train alone is ~25 M
spectra, ~5 GB as int16, which fits on the laptop. **The library is the scarce side**: every man-made
positive is one of 176 spectra, re-mixed. That is why the design leans on family-level identity and
heavy augmentation, and why the held-out tests below matter more than the training loss.

**Held out:**
- **Library entries:** ~20 % of entries per family (with near-copies, by the same rule the planted set
  used), never seen in training, so "named roofing" means "generalised to an unseen roof".
- **Families:** one man-made family left out entirely in a second run (e.g. textile), which must come
  out as "man-made, type unconfirmed", not as a wrong name.
- **Sensor:** train backgrounds from SpectralEarth + EnMAP + AVIRIS-NG, test implants on PRISMA, and the
  reverse once.
- **Scenes:** the 4 PRISMA bench scenes, Gudalur, and Gulf oil are never used for training or tuning.

## 5. Evaluation that would convince us

All on the protocol the family-matching LLD already uses (same coverage, 10 seeds, thresholds from the
development set only), with Simplex ACE as the baseline in every row.

1. **Planted retrieval (development).** The LLD's planted set (1,600 real PRISMA pixels, library spectrum
   blended at 5-50 % fill, planted entry and near-copies removed from the library) plus the same on EnMAP
   and AVIRIS-NG pixels. Report top-1 / top-5 family accuracy by fill bin (5-10, 10-20, 20-30, 30-50 %),
   coverage at p\*, and false-name rate on untouched pixels. Built from `app/evaluation/implant.py`'s
   material mode.
2. **Gudalur sheets (held out).** AVIRIS-NG 4 m simulated to 30 m, library only and library + field
   spectra. Simplex ACE: 35 / 27 / 2.1 (answered / right / false names) and ~47 / ~78 / ~2.5. This is the
   test the feature check failed, so it is the gate.
3. **Gulf oil (held out).** Simplex ACE: 11 / 26 / 1.1.
4. **4 PRISMA bench scenes, product-style.** Top 2 % candidates, per scene: targets named man-made, objects
   with a man-made name (Simplex: 51/99, 29/85, 13/59, 32/152), false alarms named man-made (9-24 %).
   **What this cannot show:** the GT is a binary anomaly mask, not material. It measures whether names
   land more on targets than on false alarms; it cannot say a name is right. (The bench counts 909
   objects in `docs/lld/evaluation.md`; the family check's denominators sum to 395, the objects reaching
   the top 2 %.)
5. **Intervals.** Paired bootstrap over objects per scene (`app/evaluation/significance.py`
   `paired_bootstrap`, `hits_*` = object got a man-made name) for item 4; seed spread for items 1-3. Per-
   scene differences under ~8 points are noise (LLD rule).

**Pass rule (proposed):** at alpha = 5 %, not worse than Simplex ACE on Gudalur and oil in "right" at
equal coverage, and better on planted low fill (5-20 %) by more than the seed spread. Failing Gudalur
with a planted gain is a fail.

## 6. Compute and schedule

Arithmetic, for the proposed encoder: ~25 MFLOP per spectrum forward (4 layers x 53 tokens x width 64,
plus the stem), ~75 MFLOP forward + backward, x2 for pixel and ring, ~150 MFLOP per training pair. 100 M
pairs = 1.5e16 FLOP: **well under one GPU-hour** on the laptop's RTX 4000 Ada (20 GB) at any realistic
utilisation. For scale, the reduced-attention stem plus the full SegFormer ran at 80 patches/s
(~1.3 M pixels/s) on the L40S (`docs/lld/spectral-stem.md`).

Wall clock is data and evaluation, not the model: budget **5-10 laptop-hours per training run**, plus a
few CPU-hours per evaluation sweep (Simplex ACE itself takes ~45 s per PRISMA scene on CPU).

- **H100 block 2 (31 Oct - 2 Nov):** not needed, and not worth taking from F002-heavy.
- **Laptop GPU:** fits comfortably. One caution from the HyperSL work: concurrent CUDA jobs in WSL died
  with `CUDA_ERROR_UNKNOWN` and needed a `wsl --shutdown`, which also stops the Docker stack. Run one job
  at a time, and ask before a shutdown.

**Minimum viable experiments, in order (each stops the line if it fails):**

- **E0 - frozen-stem probe (a few hours).** Freeze the SE-pretrained reduced stem; train only a small
  head on `(x', m)` with the family objective, on SpectralEarth + EnMAP backgrounds. Score items 1-3 of
  section 5 against Simplex ACE. Tells us whether pretrained spectral features carry identity at all and
  which checkpoint is the stem donor.
- **E1 - full contrastive fine-tune (5-10 h).** Unfreeze, all augmentations, entry hold-out. Full
  section 5.
- **E2 - ablations.** No ring input; random-init stem; family hold-out; sensor hold-out. Only if E1
  passes.

## 7. Risks, and what would make the answer "not yet"

- **Library-to-image domain gap.** Lab spectra are clean, nadir, single-material; deployed materials are
  weathered, dusty, shadowed and bidirectional. The Gudalur sheets' own field spectra lifted Simplex from
  27 % to ~78 % right, which says the gap is large. A learned model trained only on library positives can
  only learn the gap we simulate.
- **No material-labelled real ground truth.** 5 sheets and one oil campaign are the whole real test. A
  result that hinges on them is directional, as the LLD already says of Simplex. **If Gudalur and oil are
  not obtainable from the team Drive in a usable form, the answer is "not yet"**: we would have only
  synthetic evidence, which has already misled once.
- **Few man-made positives.** 176 spectra; glass has 1. Families under ~10 entries should be merged or
  left to "man-made, type unconfirmed".
- **Radiometric scale unverified (blocking).** Both this model and Simplex need 0-1 surface reflectance.
  The open gap in `docs/09-known-issues.md` stands: a PRISMA band-filtered vendable had medians of
  0.001-0.09, and the EnMAP product scene had a median valid reflectance of 0.96. Training on shards that
  are reflectance and serving on vendables that may not be would make the model meaningless in the
  product. Resolve this first; it also matters for Simplex ACE today.
- **It still will not filter false alarms.** False alarms outnumber targets ~50 : 1 and are unusual by
  construction; a better namer does not change that. The product wording stays "likely material".
- **Learning the augmentation.** Linear mixing at known fill is also how the planted set is made, so the
  planted score flatters any model trained on it. Hence the pass rule weights Gudalur and oil.

## 8. Plugging into the product

Additive, the way family matching was added beside SAM:

- **Science in `app/`:** a new `app/spectral_match/embed_match.py` (torch, no DB / FastAPI) with
  `score_families_embedded(cube, validity, ring_mean, rows, cols, entries, model) -> per-family scores`,
  returning the same shape the Simplex scorer feeds into calibration. The model class under
  `app/foundation_models/components/` beside the stem it reuses; a codename if it ships.
- **Worker:** `_spectral_library_match_run._family_match` gains a branch on a new config field
  `family_scorer: "simplex_ace" | "<codename>"` (default `simplex_ace`, so stored configs are unchanged).
  The import of `embed_match` and torch stays inside that function - the api's import graph is untouched,
  and `spectral_library_match.py` adds only the pydantic field. Calibration, p\*, the `family_*` files and
  the viewer are reused as they are.
- **Library embeddings:** computed per sensor cache, content-addressed by adding the model checkpoint's
  sha256 to the existing cache key, so they never collide with Simplex's cache.
- **Export:** bundle schema stays `allotrope-export-hyper/2`; the manifest and `family_legend.json` record
  which scorer and checkpoint produced the names.
- **Cost in the worker:** ~50 MFLOP per pixel (pixel + ring) on CPU is 500 GFLOP per 10,000 pixels,
  ~2-20 s at 25-250 GFLOP/s effective; the calibration null (~10,000 pixels) is the same again. A
  25,000-pixel PRISMA action lands near the current 5-85 s; a 200,000-pixel AVIRIS-NG one could take a few
  minutes. To be measured.
- **Check after wiring:** `docker compose ps -a` - the worker must say `running` (CLAUDE.md, known
  breakage 1).

## Decisions for you

1. **Go / no-go on E0** (laptop GPU, a few hours, no AWS)? I recommend go, after item 2.
2. **Radiometric scale first?** I recommend verifying `normalized_hyperspectral_cube` is 0-1 reflectance
   per sensor before any training, since it gates Simplex ACE too.
3. **Gudalur and Gulf oil data:** confirm they can be pulled from `PS_11/claude_experiments/unmixing_next/`
   on the team Drive. Without them this stays "not yet".
4. **Ring input (pixel + surroundings) or pixel only?** I recommend the ring; pixel-only only as an E2
   ablation.
5. **Library expansion:** worth curating more man-made spectra (e.g. field spectra from our own sites)
   before E1? It is the single biggest lever, and no model choice substitutes for it.

## References

- Kong et al., 2025, HyperSL, *IEEE TGRS*, doi 10.1109/TGRS.2025.3566205 (behaviour verified from source,
  see the memory note and `research/experiments/hypersl-validation/`).
- Hong et al., SpectralGPT (*IEEE TPAMI*; 3D masked pretraining on ~1 M multispectral images, up to ~632 M
  parameters), arXiv 2311.07113.
- Wang et al., HyperSIGMA (ViT HSI foundation model, >1 B parameters, HyperGlobal-450K), arXiv 2406.11519.
- Siamese / metric learning for hyperspectral target detection with library spectra and synthetic linear
  mixing: e.g. Zhang et al., 2022, *Remote Sensing* (Siamese ensembles with pseudo data); Wang et al.,
  2025, *Infrared Physics & Technology* (SN-HTD). Abstracts only were read; not verified in detail.
- Ziemann & Theiler, 2017, Simplex ACE, *Optical Engineering* 56(8) (the baseline).
