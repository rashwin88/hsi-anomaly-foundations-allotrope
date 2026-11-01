# Part 31 - Anomaly detection with HyperSL: a model that completes the anomaly

> **The one thing this part teaches:** HyperSL rebuilds a pixel's hidden bands from that same pixel's visible bands, so it
> learns to complete whatever spectrum it is shown, anomalies included; measured on the 909-object PRISMA bench with the
> evaluator's own measures (2026-10-31), its residual found 130 objects in its top 1% (random scores find about 93), and
> even scored with our own residual Mahalanobis its reconstruction made the score worse than using no reconstruction at
> all (pAUC 0.083 against 0.128).

## Where we are

Part 11 set the rule detection rests on: **a pixel must never help reconstruct itself.** Indradhanu hides whole pixels and
rebuilds them from neighbours. HyperSL (part 29) hides 142 of a pixel's 149 bands and rebuilds them from the other 7 of the
same pixel. This part works out why that is the opposite of what a detector needs, then measures it.

| file | what it holds |
|---|---|
| `research/experiments/hypersl-anomaly-bench/README.md` | the bench run: config, commands, checks, results, limitations |
| `research/experiments/hypersl-anomaly-bench/step1_hypersl_scores.py` | HyperSL's masked reconstruction of every valid pixel (WSL, GPU) |
| `research/experiments/hypersl-anomaly-bench/step2_evaluate.py` | the evaluator's measures, baselines, paired bootstrap |
| `research/experiments/hypersl-anomaly-bench/results.json` | every number, per scene and tier |
| `C:/Users/FITT/Documents/hypersl_match_data/results/summary.json` | the 2026-09-23 reconstruction-angle measurement |
| `C:/Users/FITT/Documents/stage1_datasets/indradhanu_fp_filter/step3/results.json` | HyperSL as a false-alarm filter (2026-09-26) |

> **Analogy.** A cloze test: fill in the missing words of a sentence. A fluent reader completes *any* sentence, however
> odd - "the giraffe ordered a ___ of tea" as easily as "the man ordered a cup of tea" - because the rest of the sentence
> says what is missing. To find odd sentences you ask whether a sentence fits its paragraph, not whether it can be
> completed. HyperSL is the fluent cloze reader of spectra; the paragraph is the pixel's neighbourhood.

## Step 1 - two ways to predict a pixel

**Spatial completion** predicts a pixel from other pixels (Indradhanu). **Spectral self-completion** predicts some of a
pixel's bands from its other bands (HyperSL). A reconstruction detector needs its predictor to *fail* on anomalies;
neighbours carry no information about an anomaly, but its own bands carry a great deal.

Three-band toy. Smooth background `b = (0.20, 0.30, 0.40)`; smooth anomaly `t = (0.60, 0.50, 0.40)` (a bright roof); anomaly
with a dip `d = (0.60, 0.30, 0.60)`.

**Self-completer:** hide band 2, predict from bands 1 and 3. Trained by least squares on smooth backgrounds, the best
predictor is the average - on 2,000 noisy smooth backgrounds the fit gives `w1 = 0.4981, w3 = 0.5011, c = -0.00018`.

```
background:      (0.20 + 0.40) / 2 = 0.30,  residual 0
smooth anomaly:  (0.60 + 0.40) / 2 = 0.50,  residual 0       <- invisible
dip:             (0.60 + 0.60) / 2 = 0.60,  residual -0.30   <- only the narrow feature shows
```

**Spatial completer:** predict each band as the mean of 4 background neighbours.

```
background:      residual (0, 0, 0)                          squared 0
smooth anomaly:  (0.40, 0.20, 0)   squared 0.16 + 0.04 + 0 = 0.20
dip:             (0.40, 0, 0.20)   squared 0.16 + 0 + 0.04 = 0.20
```

**A self-completer flags only what it cannot infer from the anomaly itself: a feature narrower than the spacing of the
bands it sees.** HyperSL sees 7 random bands; they carry the anomaly's level and shape, and the other 142 are completed
to match. Its loss also covers visible bands (the identity trap of part 10), and min-max pins an anomaly to the edge of
[0, 1] where the sigmoid is comfortable.

**First evidence (2026-09-23)**, HyperSL's reconstruction angle at mask 0, per-curve scaling:

| scene | anomaly median | background median | AUC (anomaly above background) |
|---|---|---|---|
| set1 | 3.18 deg | 6.17 deg | 0.094 |
| set2 | 4.02 deg | 5.16 deg | 0.220 |
| set3 | 3.26 deg | 4.27 deg | 0.242 |
| set4 | 2.74 deg | 3.99 deg | 0.104 |

AUC below 0.5: anomalies reconstructed **better** than background on every scene (mean 0.165). That run used mask 0 and
per-curve scaling, HyperSL's weakest settings. The next run fixes both.

## Step 2 - the measurement (2026-10-31)

- **Data:** the 4 PRISMA scenes, 909 objects, masks exported from the evaluator's `PrismaBench`.
- **HyperSL as trained:** 149-band axis, per-band min-max over the scene, mask 0.95, 4 draws (seeds 1000-1003), each band
  averaged over the draws in which it was hidden.
- **Scores:** **L2** (mean squared residual), **loss** (its own MSE + (1 - cos)), **RM** (our residual Mahalanobis on the
  residual in reflectance).
- **Baselines:** FTH3 epoch 5 (cached maps of the 2026-10-29 evaluation), PCA-RX (residual Mahalanobis with a **zero**
  reconstruction), MNF-RX (product detector, 10 components). All also with FTH3's thick-cloud pixels scored 0.

**Checks:** FTH3 reproduces its report exactly (0.1749 / 0.1892); PCA-RX gives the stage-3 0.128. Never-hidden (pixel,
band) pairs: one draw keeps a band visible with probability 7/149, four draws `(7/149)^4 = 0.046980^4 = 4.8713e-6`; set1
`992,708 x 149 = 147,913,492` pairs `x 4.8713e-6 = 720.5` expected, 706 logged.

| method | caught top 1% | top 0.1% | pAUC | ROC-AUC | TPR at 0.001 |
|---|---|---|---|---|---|
| *chance* | 10.2% | | | | |
| FTH3 ep 5, RM | 422 (46.4%) | 13.9% | 0.1749 | 0.866 | 0.058 |
| PCA-RX (no model) | 361 (39.7%) | 8.1% | 0.1281 | 0.839 | 0.029 |
| **HyperSL RM** | 350 (38.5%) | 5.4% | 0.0832 | 0.757 | 0.020 |
| MNF-RX | 249 (27.4%) | 3.4% | 0.0750 | 0.749 | 0.019 |
| **HyperSL L2** | 130 (14.3%) | 4.0% | 0.0175 | 0.624 | 0.007 |
| **HyperSL loss** | 63 (6.9%) | 0.9% | 0.0039 | 0.501 | 0.000 |
| *cloud excluded:* FTH3 ep 5 | 449 (49.4%) | 16.0% | 0.1892 | 0.869 | 0.065 |
| PCA-RX | 377 (41.5%) | 8.5% | 0.1390 | 0.842 | 0.031 |
| **HyperSL RM** | 369 (40.6%) | 6.7% | 0.0947 | 0.761 | 0.024 |
| MNF-RX | 280 (30.8%) | 5.2% | 0.0966 | 0.753 | 0.024 |

1. **HyperSL's own residual is near chance.** Its loss catches 63 objects, *fewer* than random (~93): it fits anomalies
   better than background.
2. **Our scorer rescues part of it** - 350 objects - but most of that is the scorer's work:
3. **The decisive comparison.** PCA-RX *is* residual Mahalanobis with zero reconstruction, so "HyperSL RM - PCA-RX"
   isolates what subtracting HyperSL's reconstruction does:

| difference (B - A) | pAUC [95% CI] | caught top 1% [95% CI] |
|---|---|---|
| HyperSL RM - PCA-RX | **-0.0450 [-0.0589, -0.0318]** | -1.2 [-4.0, +1.5] pts |
| PCA-RX - FTH3 ep 5 | -0.0467 [-0.0604, -0.0327] | -6.7 [-8.8, -4.5] pts |
| HyperSL RM - FTH3 ep 5, cloud excl. | -0.0944 [-0.1094, -0.0800] | -8.8 [-11.7, -6.2] pts |

Indradhanu's residual **adds** `+0.0467` over PCA-RX; HyperSL's **removes** `0.0450`. `0.0467 + (-0.0450) = 0.0017`: almost
exactly opposite, both resolved. A spatial completer leaves the anomaly in the residual; a self-completer takes it out.

Per scene, HyperSL RM (0.0875 / 0.0966 / 0.0286 / 0.1199) is below PCA-RX and FTH3 on every scene. By tier it is furthest
behind on the subtle one (29.7% against PCA-RX's 35.0%). Cost: 428-462 s of laptop GPU per scene, 438 us per pixel at 4
draws (Indradhanu Heavy: 454 s per scene on 16 CPU threads).

**Source check (what was not tried):** the 149-band axis, not the evaluator's 186 native bands; no water bands; mask and
draws untuned. None was chosen on the bench, and none can remove self-completion, which is in the masking.

## Step 3 - HyperSL as a false-alarm filter (2026-09-26)

Keep Indradhanu as detector; classify its top-2% candidates with HyperSL embeddings of pixel and ring; train on 3 scenes,
test on the 4th. Headline: false-alarm pixels removed at 90% object recall.

| features | set1 | set2 | set3 | set4 | mean |
|---|---|---|---|---|---|
| *oracle (cut on test labels)* none | 0.339 | 0.128 | 0.198 | 0.262 | 0.232 |
| detector scores only | 0.059 | 0.219 | 0.223 | 0.249 | 0.188 |
| HyperSL embeddings | 0.615 | 0.464 | 0.188 | 0.318 | 0.396 |
| *honest (cut on training scenes)* HyperSL: recall / removed | 0.72 / 0.77 | 0.42 / 0.87 | 0.45 / 0.50 | 0.69 / 0.59 | |

HyperSL's best showing in the whole series: with hindsight it removed 40% of false alarms against 23%. But the honest cut
kept only 42-72% of the objects instead of 90%. Not adopted.

## Common confusions

**"HyperSL hides 95% of bands, so it cannot copy."** The 7 it sees carry the anomaly's level and shape; hiding bands is not
hiding the pixel.

**"350 objects with RM shows HyperSL detects anomalies."** The same scorer with no model catches 361 at a higher pAUC.

**"Below chance is just noise."** Below chance means it ranks anomalies lower than background - it fits them better.

**"Indradhanu would have the same problem."** It would with band masking, which is why spectral self-completion is listed
as rejected in `docs/lld/spectral-stem.md`.

**"The filter result shows HyperSL helps."** Only with a cut chosen on the test labels.

## Check yourself

1. Anomaly `(0.50, 0.45, 0.40)`: self-completion residual in band 2, and the spatial residual and its squared length.
2. Set2 has 1,050,895 valid pixels. How many (pixel, band) pairs should never be hidden in 4 draws? (Logged: 736.)
3. An object covers 5 pixels after dilation. Its chance of being caught in the top 1% by random scores?
4. Does the bootstrap show HyperSL RM catching fewer objects than PCA-RX? A lower pAUC?
5. How can a score catch fewer objects than chance, and what does it say about HyperSL?
6. At 438 us per pixel with 4 draws, how long for a 1,000 x 1,000 scene, and roughly with 1 draw?

<details>
<summary>Answers</summary>

1. `(0.50 + 0.40) / 2 = 0.45`, residual 0. Spatial `(0.30, 0.15, 0)`, squared `0.09 + 0.0225 = 0.1125`.
2. `1,050,895 x 149 = 156,583,355`; `x 4.8713e-6 = 762.8`, about 763; 736 is one standard deviation (about 27.6) below.
3. `1 - 0.99^5 = 1 - 0.95099 = 0.049`, 4.9%.
4. Objects: interval `[-4.0, +1.5]` contains 0, so no. pAUC: `[-0.0589, -0.0318]` is all below 0, so yes.
5. Chance is what unrelated scores catch; fewer means the objects score systematically lower than background, i.e. HyperSL
   reconstructs anomalies better - self-completion.
6. `10^6 x 438 us = 438 s` (7.3 min); one draw roughly a quarter, ~110 s (assuming time per draw).

</details>

---

Next: part 32, unmixing and sub-pixel targets with HyperSL.
