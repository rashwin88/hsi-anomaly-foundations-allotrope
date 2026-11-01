# Spectral namer - natural first, then the material class (design)

**Status:** agreed 2026-10-31 (user: "Yes. Please on all three"); built the same day - see **As built** at the end.
**Product-wired 2026-11-01** (N001b epoch 40, in `spectral_library_match`; local stack, not committed or deployed) -
see docs/lld/spectral-namer-in-product.md. Training on H100 block 2 (`cr-0dccba00b34dc29e1`, 2026-10-31 11:30 ->
2026-11-02 11:30 UTC; launch approved by the user). Decisions taken with the user on 2026-10-31: two levels, the class
list below, a matching (not closed-classifier) model so a new library needs no retraining. Evidence:
research/experiments/spectral-namer/README.md (separability check). Maths: research/math/spectral-matching/.

## What it answers
For an anomalous pixel (or a detected object's pixels):
1. **Level 1** - natural or man-made, with P(man-made).
2. **Level 2** - if man-made: which class, with P(class | man-made), or "class unknown".

P(class) = P(man-made) x P(class | man-made). Outputs: natural / man-made, class unknown / man-made + class. Thresholds
are per scene (calibrated on the scene's own background, as everywhere in this project).

**Level-2 classes (v1):** oil/fuel, plastic, textile/fabric, wood/paper, paint/coatings, metal, concrete/asphalt.
glass/PV once spectra exist; camouflage and military coatings when field spectra arrive. Mapping from today's family
library: building + pavement + asphalt roofing -> concrete/asphalt; roofing dissolved by material (asphalt shingle /
tar paper -> concrete/asphalt, wood shake -> wood/paper, metal roofing -> metal, clear roof -> plastic); the rest 1:1.

## Model - a dual encoder
```
pixel side:   x (pixel), r (ring mean 2-7 px), d = x - r   --stem--> tokens --MLP--> z in R^128
                                                                      +--> level-1 logit (natural / man-made)
                                                                      +--> fill estimate a_hat in [0, 1]
library side: t (library spectrum)                         --stem--> --MLP--> e(t) in R^128
match:        s_j = cos(z, e(t_j)) / tau;   class score = logsumexp of s_j over the class's spectra
              P(class | man-made) = softmax over classes (+ an "unknown" score when the best match is weak)
```
- **Stem:** the wavelength-aware reduced-attention stem (`app/foundation_models/components/spectral_reduced.py`),
  initialised from Indradhanu Heavy's `spectral_in`. It reads any band list, so one model serves PRISMA, EnMAP and
  AVIRIS-NG, and it keeps absolute reflectance (wavelength normalisation, never per-scene min-max).
- **Library embeddings** e(t_j) are computed once per library and sensor. **A new library or class = encode its spectra;
  no retraining.** Level 1 does not use the library at all.
- Size: small (~1-3 M parameters). Per-pixel, so inference is cheap next to the detector.

## Training data - a generator, on the GPU
Each sample is built on the fly from real pixels:
- **Background:** a real pixel b and its ring mean r from training tiles of the native EnMAP / PRISMA / AVIRIS-NG shards
  (`patches/<sensor>/native-v1(z)`, staged to the H100's NVMe), each with its tile's band list and its shift-difference
  noise sigma.
- **Target** (by sample type):
  - **man-made:** a library man-made spectrum t (train split) of class c; fill a log-uniform in [0.05, 1];
  - **natural-mix:** a natural library spectrum (soil, mineral, rock, npv, vegetation, water, snow) mixed the same way;
  - **plain:** a = 0, the real pixel itself.
  The last two are level-1 "natural".
- **Mixing:** x = g (a t + (1 - a) b) + n and r' = g r, with g a smooth multiplicative gain curve (within +-10%; the
  atmospheric residual), n Gaussian at the tile's noise. The library is resampled to the tile's band list
  (`app/spectral_match/library.build_sensor_cache`; caches precomputed per band list). Band-list jitter as in
  pretraining (`band_augment_prob`).
- The mixing is a few tensor operations: done on the GPU in batches of ~4-8k samples; the loaders only feed tiles.

## Losses
- **Level 1:** binary cross-entropy (man-made vs natural).
- **Level 2 matching:** InfoNCE over the batch's library spectra. The positive is the planted spectrum; negatives are
  the other spectra; temperature tau learned. Applied to man-made samples only.
- **Fill:** L1 on a_hat, man-made samples only.
- Total: L1-BCE + L-InfoNCE + 0.5 L-fill.

## Splits and evaluation
- **Held-out spectra:** per class, ~20% of the library spectra never seen in training (spectrum-level split; the
  separability check's leave-one-out made precise).
- **Held-out class (open set):** one man-made class (e.g. paint/coatings) left out entirely. Pass if level 1 still calls
  it man-made and level 2 says "unknown" (not a confident wrong class).
- **Baseline:** the classical ACE numbers from the separability check, same sensors and fills - the namer must beat
  them, mainly at fill 0.1-0.25 where ACE is weak (level 1 0.48-0.61 man-made at fill 0.1).
- **Real targets:** the Gudalur sheets and Gulf oil scenes (team Drive, `PS_11/claude_experiments/unmixing_next/`)
  are the only real material-labelled targets - reported, too few for a verdict; the OSM benchmark
  (docs/proposals/spectral-matching-fresh-start.md) is the follow-up.
- Paired bootstrap intervals (app/evaluation/significance.py style) over planted samples / objects.

**Pass criteria (fixed now), on held-out spectra, per sensor, fill 0.25:** level-1 man-made recall at least +0.10 over
ACE at the same natural false-alarm rate; level-2 class accuracy at least +0.10 over ACE; interval above zero. Open
set: held-out class called man-made >= 70% and assigned a known class with confidence > 0.5 in <= 20%.

## Files
- `app/spectral_match/namer/` - model (dual encoder), generator, class map from family-library tags, inference API.
- `scripts/train_namer.py` + `configs/namer/*.json`; experiment folder `research/experiments/spectral-namer/`.
- Product wiring (`spectral_library_match` option) only after it passes.

## Risks
- **Generator gap:** linear mixing, smooth gain and Gaussian noise are simplifications; real adjacency, BRDF and
  correction artefacts are not modelled. Real-target checks are the guard.
- **Library size:** ~140 man-made spectra; classes of 8-40 spectra. Expand from USGS cvASD / ECOSTRESS raw before or
  during the block if time allows.
- **Level 1 vs natural look-alikes:** concrete/asphalt is mineral-like (the check's weakest class).
- **Time:** 48 h block; generator + model + trainer built on the laptop before 11:30, smoke-tested on CPU.

## As built (2026-10-31)
Where the code departs from the text above, the code is what runs:
- **Level-2 loss is class-level, not per spectrum:** cross-entropy over the class scores (logsumexp of cos / tau over
  each class's train-split library embeddings), with each sample's own planted spectrum left out - the separability
  check's leave-one-out, applied in training as well, so the model is rewarded for naming the class rather than
  recognising one spectrum.
- **Fill L1 on every sample** (natural mixes and plain pixels too, plain target 0), not man-made only.
- **Start point:** the released V2 (FTH3 epoch 5), not Heavy's pretraining checkpoint - same architecture, stem already
  tuned on the three sensors. The post-stem BatchNorm is carried over with the stem.
- **Standalone loop** (`app/spectral_match/namer/train.py`, `scripts/train_namer.py`, plain JSON configs in
  `configs/namer/`), not a FinetuneTrainer subclass registered in the config union: the reconstruction trainer's
  epoch, metrics and checkpoint machinery assume a reconstruction output. The shard pipeline is the fine-tuner's.
- **Bands:** per batch only bands live across the batch's tiles are used (water bands drop out); the library is
  subset to the same bands.
- **Natural spectra** are planted in both splits (the held-out split is man-made only), so level-1 natural recall is
  not an unseen-spectrum test.
- **Level 1 by matching** (2026-10-31, after N001): natural library spectra are one more class in the scores and
  P(man-made) = sigmoid(logsumexp over the man-made classes - natural score + bias); the linear level-1 head sat at
  chance (research/experiments/spectral-namer/N001/README.md). Runs N001b, N002.
- **Open-set class:** textile/fabric, 14 spectra (run N002) - paint has 4 after de-duplication (user, 2026-10-31).
- **Ring:** valid pixels at Chebyshev distance 3-7 (200), not "2-7 px". Not built: band-list jitter, an explicit
  "unknown" output, bootstrap intervals. 347,685 parameters, not ~1-3 M.
- Runs and results: research/experiments/spectral-namer/N001/, N002/.
