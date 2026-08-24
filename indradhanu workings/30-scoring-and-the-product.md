# 30 · From residual to answer — and everything else

> **The one thing this part teaches:** the model produces a reconstruction; a
> separate, simple piece of code turns that into a score map; and a human turns
> the score map into a decision.

**Sources:**
[`app/utils/anomaly_detection/scoring.py`](../app/utils/anomaly_detection/scoring.py)
and
[`backend/allotrope/action_types/_anomaly_scoring_run.py`](../backend/allotrope/action_types/_anomaly_scoring_run.py)

---

## Where we are

Part 29 ended with a full-scene reconstruction: a `(165, H, W)` array, the
model's best guess at what the scene should look like, with every pixel
predicted without seeing itself.

Now we subtract.

---

## The four scoring methods

```python
compute_score(original, reconstruction, validity, method, combined_weight)
```

Returns one `(H, W)` float32 map. Higher = more anomalous. Zero wherever the
pixel is invalid.

| Method | What it computes | Offered for Indradhanu? |
|---|---|---|
| `L1` | mean absolute difference across bands | yes |
| `MSE` | mean squared difference across bands | not offered |
| `SAM` | spectral angle between the two spectra | yes |
| `combined` | a weighted blend of L1 and SAM | yes — **the default** |

The resolver decides what the interface offers:

```python
scoring_methods=("L1", "SAM", "combined"),
default_scoring_method="combined",
```

Why is `combined` the default? Because the model was **trained** on
`L1 + lambda * SAM` (part 25). Scoring with the same pair of measurements keeps
the score aligned with what the model was optimised for.

---

## How `combined` works

```python
l1   = np.mean(np.abs(original - reconstruction), axis=0)
sam  = _sam(original, reconstruction)
l1n  = _normalise(l1,  spatial)          # divide by the max over valid pixels
samn = _normalise(sam, spatial)
score = combined_weight * l1n + (1.0 - combined_weight) * samn
```

The problem it has to solve: **L1 is in reflectance units and SAM is in
radians.** You cannot add them directly — it would be like adding metres to
kilograms.

So each is first scaled to the range 0 to 1 by dividing by its own maximum
across the scene. Then they are comparable, and a weighted average makes sense.

### Worked example

Suppose across the whole scene:

```
max L1  = 0.05
max SAM = 0.40 rad
```

And one particular pixel has:

```
L1  = 0.025
SAM = 0.10 rad
```

**Normalise each:**

```
l1n  = 0.025 / 0.05 = 0.50      "half as bad as the worst pixel, by brightness"
samn = 0.10  / 0.40 = 0.25      "a quarter as bad as the worst, by shape"
```

**Blend, with the default weight of 0.5:**

```
score = 0.5 * 0.50 + 0.5 * 0.25
      = 0.250 + 0.125
      = 0.375
```

The weight is exposed per-Action as `sam_l1_alpha`, defaulting to 0.5. Raise it
to weight brightness errors more; lower it to weight spectral shape more.

### Gotcha: max-normalisation is fragile

Read `_normalise` carefully:

```python
hi = float(vals.max())
out = (arr / hi)
```

Everything is divided by the single largest value in the scene.

So **one extreme pixel** — a sensor spike, an unmasked sliver of cloud, a dead
detector — sets the denominator and compresses everything else toward zero.

**The symptom:** a score map that looks uniformly dark with one bright dot.

**Before blaming the model**, check `stats.json` in the output, which records
score percentiles. If the 99.9th percentile is tiny and the maximum is huge, you
have one outlier dominating the normalisation, not a model failure.

**A second consequence:** because each scene is normalised by its own maximum,
`combined` scores are **not comparable across scenes**. This is the same rule as
part 01's percentile thresholds, appearing again in a different guise.

---

## What the Action writes to disk

For each model, in its own subfolder `models/<Codename>/`:

| File | What it is |
|---|---|
| `anomaly_score.tif` | `(H, W)` float32 — the score map |
| `anomaly_score.png` | a rendered preview for the interface |
| `reconstruction.tif` | `(165, H, W)` float32 — the model's redrawing |
| `reconstruction.png` | a rendered preview |
| `stats.json` | percentiles, timings, diagnostics |

Two details worth knowing.

**The PNG stretch differs by model family:**

```python
stretch="sqrt" if is_classical else "linear"
```

Classical detectors (MNF-RX) produce squared Mahalanobis distances, which have a
very heavy right tail — a few pixels enormously larger than the rest. A square
root pulls the bulk of the values into the middle of the colour scale where you
can actually see contrast. Foundation-model residuals are roughly unimodal, so a
linear stretch works.

**And remember part 04:** these GeoTIFFs carry an **identity transform**. They
have no real geography. Coordinates are reattached only at export time by
re-reading the original raw file.

---

## ROC and AUC, when ground truth exists

If an analyst attached an annotation raster — a hand-drawn map of where the
anomalies really are — the Action can measure how good the score map is.

```python
ths = np.percentile(s, np.linspace(0, 100, n_pts))
ths = np.unique(ths)[::-1]           # descending
for t in ths:
    pred = s >= t
    tpr = TP / n_pos
    fpr = FP / n_neg
auc = np.trapezoid(tpr_arr, fpr_arr)
```

### What ROC means, from scratch

Pick a threshold. Every pixel scoring above it is "flagged". Now count four
things:

| | truly anomalous | truly normal |
|---|---|---|
| **flagged** | TP (true positive) | FP (false positive) |
| **not flagged** | FN (false negative) | TN (true negative) |

Two rates:

```
TPR (true positive rate)  = TP / (TP + FN)    what fraction of real anomalies did we catch?
FPR (false positive rate) = FP / (FP + TN)    what fraction of normal pixels did we wrongly flag?
```

Sweep every possible threshold and plot TPR against FPR. That curve is the ROC
curve.

**AUC** is the area under it: 1.0 is perfect, 0.5 is a coin flip.

### Worked micro-example

Ten valid pixels, two of them truly anomalous. At some threshold the model flags
three pixels, and both real anomalies are among them:

```
TP = 2      both anomalies caught
FN = 0      none missed
FP = 1      one normal pixel wrongly flagged
TN = 7      seven normal pixels correctly ignored

TPR = 2 / (2 + 0) = 2/2 = 1.000
FPR = 1 / (1 + 7) = 1/8 = 0.125
```

One point on the curve, at `(0.125, 1.00)` — caught everything, at the cost of
one false alarm. Sweeping all thresholds traces the whole curve.

### Note the thresholds are percentiles

```python
ths = np.percentile(s, np.linspace(0, 100, n_pts))
```

Not evenly spaced score values — evenly spaced **percentiles** of the score
distribution. Consistent with the project-wide rule from part 01.

Degenerate cases (no anomalies at all, or nothing but anomalies) return a
straight diagonal with `"degenerate": true` rather than crashing.

---

## How a score becomes a delivered answer

```
anomaly_scoring
    one score map per selected model
        |
        v
anomaly_detection_prep
    combine the models into a composite
    A HUMAN picks a percentile threshold
    -> composite_score.tif + anomaly_mask.tif
        |
        v
spectral_library_match                              [hyperspectral only]
    take each flagged pixel's spectrum
    match it against the USGS splib07 library of known materials
    -> matches.parquet + match_map.tif
        |
        v
export
    re-read the raw file for the real coordinates
    bundle everything into a zip
```

**The human threshold step is deliberate.** The model *ranks* pixels; a person
decides where to cut. That decision depends on how many candidates the team can
investigate, how costly a miss is, and what the scene is.

`spectral_library_match` is what turns *"pixel 412,908 is unusual"* into *"pixel
412,908 looks like alunite"* — a statement somebody can act on. Its algorithm is
specified in `spectal_match_sample/WALKTHROUGH.md`.

---

## Frequently asked questions

**Does Indradhanu know what an anomaly is?**
No. It has never seen a label. It only knows how to redraw ordinary imagery.
Everything else is subtraction and human judgement.

**Can it run on a thermal scene?**
No. It is built for `in_channels = 165`. The resolver routes thermal scenes to
Chakshu and the autoencoder family via `sensor_family()`.

**Why does the same scene sometimes score slightly differently?**
If `masking_strategy = "random"`, a fresh mask is drawn each run. Checkerboard
is deterministic. Either way, averaging sixteen overlapping tiles damps the
difference.

**Why does the reconstruction look blurry?**
It should. It is a prediction made with most of the evidence removed. A crisp,
perfect reconstruction would mean the model found a way to copy — which is the
failure the whole masking design exists to prevent (part 06).

**Why are large anomalies harder to find than small ones?**
A single anomalous pixel sits inside a completely ordinary neighbourhood, so the
model confidently predicts "ordinary" and the residual is large. A 200-pixel
anomalous region provides *its own context* in the surviving tokens, so the
model can partially reconstruct it. This is a genuine, known limitation of
reconstruction-based detection — not a bug, and worth stating to users.

**The score map has a faint grid pattern.**
Checkerboard artefacts (part 29). Try `masking_strategy = "random"`, or a
smaller stride.

**The score map is bright along every cloud edge.**
Erosion (part 20). Raise `erosion_kernel_size` (inference default 15), and/or
set `keep_mask_erosion_kernel_size` on the Action. Both must be odd.

**The score map is black except one bright dot.**
Max-normalisation, above. Look at the percentiles in `stats.json` before
concluding anything about the model.

**Can I swap the normalisation statistics without retraining?**
Only via `PixelStatsOverride` at inference (part 09). The baked buffers live
inside the checkpoint.

**How do I compare two models?**
Not by their training losses — those are in different units (part 25). Compare
score quality: ROC and AUC on annotated scenes, or a side-by-side look at the
score maps, which the Action supports by running several models at once.

---

## Glossary

| Term | Meaning |
|---|---|
| **Action** | one step of analysis in the product; chains into a graph |
| **AUC** | area under the ROC curve; 1.0 perfect, 0.5 chance |
| **band** | one wavelength layer of a cube |
| **cube** | a `(C, H, W)` image array |
| **D** | the compressed spectral dimension, 32 |
| **ESA** | Efficient Self-Attention — attention with summarised keys and values |
| **GELU** | the smooth activation function used throughout |
| **Indradhanu** | codename of `hyperspectral_segformer_mae`; Hindi for rainbow |
| **keep_mask** | token mask; 1 = show this token to the encoder |
| **L1** | mean absolute error |
| **LayerNorm** | normalisation across one token's own features |
| **MAE** | Masked AutoEncoder — hide input, predict it, grade only there |
| **Mix-FFN** | feed-forward network with a depthwise convolution inside |
| **MNF** | Minimum Noise Fraction, a classical spectral compression |
| **OPE** | OverlapPatchEmbedding |
| **PixelShuffle** | rearranging channels into spatial positions |
| **pred_mask** | token mask; 1 = hide this token and grade it |
| **reflectance** | fraction of light returned; unitless, typically 0.02–0.30 |
| **residual** | the difference between observation and reconstruction |
| **RX** | a classical Mahalanobis-distance detector |
| **SAM** | Spectral Angle Mapper; the angle between two spectra, in radians |
| **spectrum** | one pixel's values across all bands — the material fingerprint |
| **token** | a 4x4 pixel block summarised as a vector |
| **vendable** | the standardised cube + validity object everything consumes |

---

## What to read next, in order

1. [`hyperspectral_seg_former_mae.py`](../app/foundation_models/components/hyperspectral_seg_former_mae.py),
   top to bottom. It should now read like a summary of parts 09 to 22.
2. `research/model_break_down/05_hyperspectral_segformer_mae.md` — diagrams and
   the full `torchinfo` table.
3. `docs/04-models.md` and `docs/05-detectors.md` — where Indradhanu sits among
   its siblings.
4. `final design/diagrams/model-runs/indradhanu.md` — sequence diagrams of one
   real scoring run, from the product's point of view.
5. `spectal_match_sample/WALKTHROUGH.md` — when you want to follow the chain
   past the score map.

---

## Exercises to prove you have it

1. Derive 165 from the grid parameters and the five exclusion ranges (part 03).
2. Reproduce the parameter table and land on 5,506,629 (part 26).
3. Do the same for `D = 24` and land on 5,205,538 including buffers.
4. Hand-compute L1 and SAM for a three-band pair, then the combined score given
   the scene maxima and `alpha = 0.5`.
5. For a 1200x900 scene with `ps = 128, stride = 32`, compute the tile count and
   the per-pixel coverage.
6. Explain to somebody else, without notes, why the model must never see the
   pixel it is predicting — and name the three places that rule is enforced.

<details>
<summary>Answer to 6 (the one that matters most)</summary>

Because a model that can see the pixel will simply copy it, giving zero residual
everywhere including at anomalies — which detects nothing.

Enforced in three places:

1. **Token removal** — hidden tokens are physically deleted from the encoder's
   input sequence (part 19).
2. **Two-pass inference** — the scene is reconstructed twice with complementary
   masks, and each pixel's value is taken from the pass in which it was hidden
   (part 29).
3. **The loss mask** — during training, only hidden positions are graded
   (part 23).

</details>

---

## Before you change anything in this repo

Two conventions, both non-negotiable.

**The workflow.** This project uses the `iterative-nano-chunking` skill as its
default coding process: restate the problem, propose a design and **stop for
agreement**, break the work into chunks of at most 20 changed lines, then execute
**one chunk per message** and stop for feedback. "These next few are small so
I'll batch them" is exactly the failure it exists to prevent.

**The layering.** Science in `app/` with no database or web framework;
orchestration in `backend/`; heavy imports inside functions, never at module
top level (part 05).

And the habit this whole course has been trying to build:

> **Verify claims against source.** You have now seen three places where the
> comments and the code disagree — the stage-1 kernel size, the inference
> validity threshold, and `warmup_epochs`. Documentation drifts. Code does not.

---

## You are done

Thirty parts. If you can now open
`hyperspectral_seg_former_mae.py` and read it without stopping, and explain to a
colleague why the model must never see the pixel it is predicting, you know this
model better than most people who have used it.

Go and read some code.
