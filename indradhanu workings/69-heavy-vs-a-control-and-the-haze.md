# 69 - The fair test: Heavy against a same-recipe control, and the haze in the thin-cloud head

> **The one thing this part teaches:** to learn whether the bigger model, not the training recipe,
> made the difference, Heavy was fine-tuned beside a control - F001 given exactly the same recipe -
> and judged at three neighbouring checkpoints; once its thin-cloud output (which fires on haze) was
> left out of cloud exclusion, Heavy beat the control by about +0.023 pAUC and shipped FT04 by +0.015
> to +0.020, every interval above zero, while its reconstruction was no better than the control's.

## Where we are

**Status (2026-10-29): H100 block 1 done at 10:39 UTC. Release candidate: FTH3 epoch 5, real-bench
pAUC 0.189 against shipped FT04's 0.170, resolved; caught at the top 0.1% a tie. Not shipped: the
product still runs Indradhanu V2 (FT04) opt-in and v0.2.0 by default, and shipping Heavy is the user's
call.**

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-finetune/FTH1-heavy-joint/README.md` | FTH1 / FTB1 setup, detection, the mask diagnosis |
| `research/experiments/indradhanu-finetune/FTH2-heavy-joint-more/README.md`, `bootstrap.txt` | FTH2 / FTB2, the thick-cloud rule's first use |
| `research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/README.md`, `bootstrap.txt` | the plateau and the release candidate |
| each folder's `eval_report.md`, `finetune_metrics.csv`, `config.json` | the evaluator's tables, validation per epoch, the run settings |
| `FTB1-base-joint/`, `FTB2-base-joint-more/` | the control's configs and validation |
| `research/experiments/indradhanu-finetune/h100_chain.sh` | the script that ran FTH1 and FTB1 after pretraining |
| `app/evaluation/config.py`, `app/evaluation/models.py` | `cloud_rule: "thick"` |
| `app/foundation_models/inferencers/spectral_stem_inferencer.py` | `segment_full_scene`: cloud = cloud or thin cloud |

> **Analogy.** You want to know whether a new engine makes a car faster. Last year's car won with a
> driver who practised the track for a month. If you put the new engine in a car, hand it to a driver
> who has had one afternoon, and it loses, you have learned nothing about the engine. The fair test is
> two cars, old engine and new, given the same driver, the same afternoon of practice, the same tyres.
> Here the engine is the pretrained model and the afternoon is the fine-tuning recipe.

## Why a control is needed

FT04, the shipped model, is F001 plus a **four-stage curriculum** (parts 49 to 60): a probe (FT01),
adapt runs (FT02 to FT02c), joint (FT03), then the final joint run with pseudo-labels (FT04). Block 1
had about 12 hours after pretraining - not enough to repeat four stages for Heavy.

So Heavy got a shorter recipe: FT04's *final* recipe in one joint run. If that model beat FT04, you
could not say whether the bigger model won or whether something else in the change did; if it lost, it
might be the short recipe. Two causes are mixed into one comparison. That mixing has a name:
**confounding**.

The fix is a **control**: a run identical to the experiment in every way except the thing under test.
`FTB1-base-joint/config.json` says it directly: "F001 (FT04's size, SegFormer decoder) fine-tuned with
exactly FTH1's one-run recipe, so Heavy vs FTB1 isolates size + decoder + pretraining from the recipe."

Read that sentence carefully. The control separates **the model from the recipe**. It does not
separate size alone: Heavy also differs in decoder (progressive against SegFormer), in its pretraining
run (5-epoch rotation, 80 epochs, part 68), and in its segmentation head's width (256 against 128, in
the configs' `seg_decoder_dim`). "Size" in the READMEs means that whole bundle.

The runs, from the configs:

| run | starts from | joint epochs | lr (cosine) | total joint epochs |
|---|---|---|---|---|
| FTH1 | Heavy (F002-heavy `best.pt`, epoch 72) | 10 | 1e-4 | 10 |
| FTB1 (control) | F001 epoch 109 | 10 | 1e-4 | 10 |
| FTH2 | FTH1 | 8 | 5e-5 | 18 |
| FTB2 (control) | FTB1 | 8 | 5e-5 | 18 |
| FTH3 | FTH2 | 6 | 3e-5 | 24 (no control) |

The recipe, shared by all five: mixed batches (part 52), EnMAP labels plus PRISMA and AVIRIS-NG
pseudo-labels from FT04 (part 60), rare-class weights `[1, 1, 1, 3, 3]`, `seg_weight` 0.1.

## Three neighbouring checkpoints

Part 67 ended on a lesson: FT09b's gain cleared the object bootstrap and then vanished five epochs
later. The bootstrap resamples objects; it cannot see how much a score moves from one epoch's weights
to the next. So `h100_chain.sh`'s runs kept a checkpoint after every epoch (the commit says "per-epoch
checkpoints kept for checkpoint-robust evaluation"), and every comparison here is made at **three
neighbouring checkpoints**. A gain counts if it holds at all three.

## FTH1 and FTB1: validation

Epoch 10, from each `finetune_metrics.csv`:

| | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (four-stage curriculum) | 3.88 / **4.11** / 5.46 | **0.839 / 0.810** |
| FTB1 (control) | 3.90 / 4.48 / **5.34** | 0.782 / 0.721 |
| FTH1 (Heavy) | 3.94 / 4.74 / 5.60 | 0.795 / 0.717 |

Both one-run models fall short of FT04's curriculum on PRISMA reconstruction and on masks, and Heavy
reconstructs **worse** than the control on all three sensors. Shadow and snow IoU are 0 for both. On
these numbers alone Heavy looked like a disappointment; the fine-tuning README's log entry of that
morning says "Heavy not ahead of the control".

## FTH1 and FTB1: detection

`configs/eval/fth1_ftb1.json`, real PRISMA bench, residual Mahalanobis, cloud excluded on **FT04's**
masks for both, so the masks are the same and only the detector differs (`eval_report.md`; intervals
from the README):

| checkpoint | FTB1 pAUC | FTH1 pAUC | FTH1 - FTB1 [95% interval] | caught top 1% FTB1 -> FTH1 |
|---|---|---|---|---|
| epoch 8 | 0.162 | **0.175** | **+0.0132 [+0.0022, +0.0243]** | 45.9 -> 47.9% |
| epoch 9 | 0.162 | **0.171** | +0.0093 [-0.0013, +0.0195], P 0.96 | 46.4 -> 47.1% |
| epoch 10 | 0.161 | **0.175** | **+0.0145 [+0.0045, +0.0246]** | 46.3 -> 47.2% |

Heavy ahead at all three, resolved at two. **Worse reconstruction, better detection** - the first sign
of this part's lesson.

Against shipped FT04 (0.170), FTH1 needs masks. With its own, it collapses (next section). With the
control's masks, FTH1 epochs 8 / 9 / 10 score 0.182 / 0.178 / 0.182 in the README (the
`eval_report.md` row for epoch 8 reads 0.183; a rounding difference), +0.0130 [+0.0009, +0.0242] at
epoch 8 and +0.0123 [+0.0011, +0.0224] at epoch 10, epoch 9 unresolved. Caught at the top 0.1%:
15.4 / 15.3 / 15.6% against 15.8%, a tie.

## The mask failure: haze in the thin-cloud head

Excluding cloud with FTH1's own head wrecked every detector it was applied to. From `eval_report.md`:

| detector | cloud excluded on | pAUC | ROC-AUC | caught top 1% |
|---|---|---|---|---|
| FT04 epoch 10 | FT04's masks | 0.170 | 0.852 | 46.0% |
| FT04 epoch 10 | **FTH1's masks** | **0.127** | **0.731** | **32.3%** |
| FTH1 epoch 10 | FTB1's masks | 0.182 | 0.867 | 48.1% |
| FTH1 epoch 10 | **FTH1's masks** | **0.133** | **0.742** | **32.8%** |

The head is the problem, not the detector: FT04 suffers the same drop. Yet FTH1's EnMAP validation
cloud IoU, 0.717, is close to FTB1's 0.721.

### What the head calls cloud

The product's class map comes from `segment_full_scene` in `spectral_stem_inferencer.py`. The head
predicts five independent probabilities per pixel: water, cloud, thin cloud, shadow, snow (part 47).
The class map then sets priorities, cloud last so it wins:

```python
cls[prob[0] > 0.5] = 1                              # water
cls[prob[3] > 0.5] = 3                              # shadow
cls[prob[4] > 0.5] = 4                              # snow
cls[np.maximum(prob[1], prob[2]) > 0.5] = 2         # cloud: cloud OR thin cloud
```

Its docstring: "classes by priority cloud (cloud or thin cloud) > snow > shadow > water > land". So a
pixel is excluded as cloud if **either** cloud output passes 0.5.

The README's diagnosis, share of each bench scene called cloud / targets inside it:

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| FT04 | 0.0% / 0 | 0.0% / 0 | 0.8% / 0 | 11.8% / 2 |
| FTB1 | 0.0% / 0 | 0.0% / 0 | 1.7% / 0 | 0.3% / 0 |
| FTH1 | 0.4% / 17 | 0.0% / 0 | 2.1% / 0 | **75.4% / 209 of 230** |

"All of it is the thin-cloud output": on set4, FTH1's thick-cloud channel covers 0.4% (no targets),
its thin-cloud channel 75.4% (209 targets). That is the haze over set4 (the 2026-10-18 correction noted in part 55). FT04's own 11.8% on
set4 is thin cloud too. Thin cloud was always the weak label - part 50 reported a thin-cloud IoU of
0.09 for the probe, and FTH1's is 0.07 (`finetune_metrics.csv`) - and the validation IoU above measures the **cloud** channel, which is why it did
not warn anyone.

### By hand: why zeroing targets wrecks pAUC

Cloud exclusion sets the score of every excluded pixel to 0 after scoring (`scorers.py`, kind
`exclude`; `fit_on_kept` is false in `fth1_ftb1.json`, so nothing is refitted). Recall part 64's
`c(p)`: a target pixel's credit is `max(0, 0.01 - FPR(p)) / 0.01`, where `FPR(p)` is the share of
background pixels scoring at or above it.

A target pixel scored 0 has every background pixel at or above it: `FPR = 1`, so
`c = max(0, 0.01 - 1) / 0.01 = 0`. It contributes nothing, however anomalous it was.

Do a tiny scene first, on paper. Ten one-pixel targets, each with `c = 0.5`:

```
before   (10 x 0.5) / 10           = 5.0 / 10 = 0.50
zero 9   (1 x 0.5 + 9 x 0) / 10    = 0.5 / 10 = 0.05
```

Now set4, as an estimate. FTH1 epoch 10 with thick-cloud-only exclusion (below) scores set4 at 0.222.
Suppose the 209 excluded targets were typical of set4's 230, so the 21 kept ones carry their share:

```
kept share        21 / 230             = 0.0913
set4 pAUC         0.222 x 0.0913       = 0.020
bench, thick      (0.253 + 0.162 + 0.086 + 0.222) / 4 = 0.723 / 4 = 0.181
bench, estimate   (0.253 + 0.162 + 0.086 + 0.020) / 4 = 0.521 / 4 = 0.130
measured                                                              0.133
```

The estimate ignores set1's 17 excluded targets and assumes the targets are alike, so do not read the
third decimal. But it shows the whole collapse is one scene's targets being switched off. **A mask
head that calls haze "cloud" deletes the targets under the haze.**

### The fix: thick cloud only

The evaluator gained a switch, `ModelSpec.cloud_rule` in `app/evaluation/config.py`:
`"cloud_or_thin"` (the default, `segment_full_scene`'s rule) or `"thick"`, described as "thin cloud
fires on haze". `models.py` applies it after `segment_full_scene`:

```python
if self.spec.cloud_rule == "thick":            # thin-cloud-only pixels fall to the next class by priority
    thin_only = (cls == 2) & (prob[1] <= 0.5)
    nxt = np.select([prob[4] > 0.5, prob[3] > 0.5, prob[0] > 0.5], [4, 3, 1], default=0)
    cls = np.where(thin_only, nxt, cls)
```

A pixel that was cloud only because of thin cloud (`prob[1]`, thick cloud, at or below 0.5) drops to
whatever it would have been next: snow, shadow, water, or land.

With FTH1's own head and the thick rule, the README reports: epoch 10 pAUC **0.181** (set1-4 0.253 /
0.162 / 0.086 / 0.222), caught top 1% 48.0%, top 0.1% 15.5%; epoch 8, 0.181 / 48.5% / 15.4%. "The same
as with FTB1's masks. **Heavy stands alone at pAUC 0.181 vs shipped FT04's 0.170.**"

Does the thick rule hurt FT04? `FTH2-heavy-joint-more/bootstrap.txt`, FT04 shipped -> FT04 thick:
**-0.0004 [-0.0015, +0.0015]**, a tie. The README: "switching the product to thick cloud costs V2
nothing". The product has not switched; `segment_full_scene` still uses cloud or thin cloud.

## FTH2 and FTB2: 18 joint epochs

Both continued 8 epochs at lr 5e-5. From here every model excludes cloud with **its own thick-cloud
masks**. Validation at epoch 8 (18 joint epochs):

| | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / **4.11** / 5.46 | **0.839 / 0.810** |
| FTB2 (control) | **3.85** / 4.36 / **5.33** | 0.806 / 0.729 |
| FTH2 (Heavy) | 3.88 / 4.48 / 5.39 | 0.818 / 0.736 |

Detection (`configs/eval/fth2_ftb2.json`, `bootstrap.txt`, 1000 draws):

| | pAUC epoch 6 / 7 / 8 | caught top 1% | caught top 0.1% |
|---|---|---|---|
| FT04 shipped (cloud or thin) | 0.170 | 46.0% | **15.8%** |
| FTB2 (own thick) | 0.160 / 0.163 / 0.162 | 46.1 / 46.4 / 46.8% | 15.1 / 15.1 / 15.1% |
| **FTH2 (own thick)** | **0.185 / 0.184 / 0.186** | **48.2 / 49.0 / 49.0%** | 15.3 / 15.2 / 15.7% |

| comparison | epoch 6 | epoch 7 | epoch 8 |
|---|---|---|---|
| FTH2 - FTB2, pAUC | +0.0245 [+0.0140, +0.0352] | +0.0209 [+0.0102, +0.0321] | +0.0244 [+0.0135, +0.0362] |
| FTH2 - FTB2, caught top 1% | +2.1 [+0.2, +4.1] | +2.5 [+0.6, +4.3] | +2.2 [+0.3, +4.0] |
| FTH2 - FT04, pAUC | +0.0151 [+0.0036, +0.0257] | +0.0143 [+0.0032, +0.0247] | +0.0168 [+0.0053, +0.0275] |
| FTH2 - FT04, caught top 0.1% | -0.6 [-1.8, +0.8] | -0.7 [-2.1, +0.8] | -0.1 [-1.3, +1.2] |

Every pAUC interval clears zero, at every checkpoint, against both. The average gap over the control:

```
(0.0245 + 0.0209 + 0.0244) / 3 = 0.0698 / 3 = 0.0233
```

The README calls this "about twice what cloud exclusion bought". Part 64's figure for cloud exclusion
is +0.0074, and `0.0233 / 0.0074 = 3.1`: about three times, not two. The README's conclusion stands
either way.

The gap **grew** with fine-tuning: about +0.013 at 10 epochs (FTH1 - FTB1), about +0.023 at 18. And
reconstruction still did not separate them: FTH2's PRISMA SAM (4.48) is worse than the control's
(4.36).

## FTH3: the plateau, and the release candidate

Six more joint epochs at lr 3e-5, Heavy only (no control; the block was ending). From `FTH3`'s README
and `bootstrap.txt`:

| | pAUC | caught top 1% | caught top 0.1% |
|---|---|---|---|
| FT04 shipped | 0.170 | 46.0% | 15.8% |
| FTH2 epoch 8 | 0.186 | 49.0% | 15.7% |
| FTH3 epoch 4 / **5** / 6 | 0.184 / **0.189** / 0.187 | 49.2 / **49.4** / 48.8% | 15.4 / **16.0** / 15.5% |

| comparison | epoch 4 | epoch 5 | epoch 6 |
|---|---|---|---|
| FTH3 - FTH2 ep 8, pAUC | -0.0021 [-0.0043, -0.0000] | +0.0028 [-0.0002, +0.0058] | +0.0011 [-0.0009, +0.0030] |
| FTH3 - FT04, pAUC | **+0.0148 [+0.0029, +0.0259]** | **+0.0196 [+0.0072, +0.0316]** | **+0.0179 [+0.0055, +0.0292]** |
| FTH3 - FT04, caught top 0.1% | -0.4 [-1.9, +0.9] | +0.1 [-1.2, +1.5] | -0.3 [-1.5, +1.0] |

Against FTH2 the three checkpoints go -0.002, +0.003, +0.001: one barely below zero, one barely above,
one a tie. That is a **plateau** - more epochs at this recipe no longer move detection. Against FT04
all three stay resolved.

The README's verdict: "**Release candidate: FTH3 epoch 5** (`ckpt_all/spectral_stem_finetune_v0.5.0_epoch5.pt`;
neighbours 4 and 6 within 0.005)". Caught at the top 1% by tier, FT04 -> FTH3 epoch 5: easy 58.2 ->
62.2%, medium 39.3 -> 43.9%, subtle 40.3 -> 42.0%. Heavy's thick masks hardly change across epochs 4-6
(within 0.001 pAUC).

Why choose epoch 5 when its gain over FTH2 is unresolved? This course's reading: not because it is better than its
neighbours - the plateau says they are alike - but because it is the best of a stable set, and the
only one whose top-0.1% count is not below FT04's. Pick it as the representative of the plateau, not
as a discovery.

**What stands between it and the product**, from the FTH3 README: "the thick-cloud rule, a manifest /
resolver path for the progressive-decoder architecture, and a worker CPU-cost check (34 M parameters)".
And the caveats every part since 64 has carried: four scenes, and a bootstrap that resamples objects,
not scenes.

## The lesson: reconstruction did not separate the models, detection did

Put the reconstruction and detection results side by side:

| | PRISMA SAM (deg) | pAUC |
|---|---|---|
| FTB1 (control, 10 epochs) | **4.48** | 0.161-0.162 (FT04 masks) |
| FTH1 (Heavy, 10 epochs) | 4.74 | **0.171-0.175** (FT04 masks) |
| FTB2 (control, 18 epochs) | **4.36** | 0.160-0.163 (own thick masks) |
| FTH2 (Heavy, 18 epochs) | 4.48 | **0.184-0.186** (own thick masks) |

**The model that reconstructs worse is the better detector, at every stage.** Had the decision been
made on validation SAM - the number every pretraining and fine-tuning run logs each epoch - Heavy
would have been dropped after FTH1. The FTH2 README: "Better detection without better reconstruction."

Parts 66 and 67 already measured *contrast* - hidden-target error against hidden-background error -
rather than average error, because detection depends on whether targets stand out from their
background, not on how well the background is redrawn on average. This part is the
strongest evidence for it in the course. Judge detectors on detection.

The cumulative table from part 67, with this part's rows (each compared with FT04 on the real bench):

| change tried | taught in | pAUC change vs FT04 [95% interval] | verdict |
|---|---|---|---|
| cloud exclusion | parts 57, 61, 64 | **+0.0074 [+0.0047, +0.0101]** | real gain, adopted |
| FT09b progressive decoder, joint | part 67 | +0.0060 [+0.0002, +0.0118] | did not hold (FT09c) |
| FTH2 epoch 8, Heavy, own thick masks | this part | **+0.0168 [+0.0053, +0.0275]** | real gain, three checkpoints |
| **FTH3 epoch 5, Heavy, own thick masks** | this part | **+0.0196 [+0.0072, +0.0316]** | real gain, release candidate |

(The other rows of parts 65 and 67 are unchanged.)

## Common confusions

**"FTH1 - FTB1 measures the effect of size."**
It measures the effect of the model: size, decoder, pretraining run and segmentation-head width
together. The recipe is what it holds fixed. Separating size from decoder would need another control.

**"FTH1's masks were bad, so Heavy's segmentation head is broken."**
Its thick-cloud channel works - thick-only exclusion with its own head matches the control's masks.
The thin-cloud channel fires on haze, and the product's rule takes either channel. FT04's head does it
too, on a smaller scale (11.8% of set4).

**"The thick rule was chosen because it made Heavy look good."**
It was chosen after the diagnosis showed thin cloud covering 75% of a hazy scene, and it was checked on
FT04, where it changes nothing (-0.0004 [-0.0015, +0.0015]). Still, it was adopted after seeing results,
the situation part 67 warned about; that is one more reason the release needs more than this bench.

**"FTH3 epoch 5 beat FTH2, so the extra epochs helped."**
Its gain over FTH2 epoch 8 is +0.0028 [-0.0002, +0.0058], unresolved, and its neighbour epoch 4 is
slightly below. The extra epochs found a plateau.

**"The release candidate catches more of the top targets."**
At the top 0.1% it ties FT04 (16.0% against 15.8%, interval [-1.2, +1.5]). The gains are in pAUC and
the top 1%.

**"Better validation SAM means a better product."**
Not for this product. See the lesson table.

## Check yourself

1. Why could FTH1 losing to FT04 not have shown that Heavy is a worse model? What does FTB1 fix, and
   what does it not fix?
2. A pixel's thick-cloud probability is 0.3, thin-cloud 0.8, water 0.6, and nothing else passes 0.5.
   What class does `segment_full_scene` give it, and what does the thick rule change it to?
3. A scene has 8 one-pixel targets with `c` = 1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.0, 0.0. Compute its pAUC.
   Then exclusion zeros the first three. Compute it again.
4. From the FTH2 - FT04 row, compute the mean pAUC gain over the three checkpoints. Is any checkpoint's
   interval touching zero?
5. FTH3's three checkpoints against FTH2 epoch 8 are -0.0021, +0.0028 and +0.0011. Why is this called a
   plateau rather than a gain at epoch 5?
6. Name the three things the FTH3 README lists before the release candidate could reach the product.

<details>
<summary>Answers</summary>

1. FT04 had a four-stage curriculum and FTH1 one short run, so a loss could be the recipe's fault (the
   comparison is confounded). FTB1 gives F001 exactly FTH1's recipe, so FTH1 - FTB1 holds the recipe
   fixed. It does not separate size from the decoder, the pretraining run or the head width.
2. `max(0.3, 0.8) = 0.8 > 0.5`, so class 2, cloud (cloud beats water by priority). Under the thick rule
   `prob[1] = 0.3 <= 0.5`, so it is thin-only and falls to the next class: snow and shadow do not pass,
   water does (0.6), so class 1, water.
3. Sum `1.0 + 0.8 + 0.8 + 0.6 + 0.4 + 0.4 + 0.0 + 0.0 = 4.0`; `4.0 / 8 = 0.50`. After zeroing the first
   three: `0.6 + 0.4 + 0.4 = 1.4`; `1.4 / 8 = 0.175`. The best targets went first, so the drop is
   larger than the 3 of 8 you might expect.
4. `0.0151 + 0.0143 + 0.0168 = 0.0462`; `0.0462 / 3 = 0.0154`. The lowest lower bound is +0.0032
   (epoch 7), so none touches zero.
5. One is barely below zero (its interval ends at -0.0000), one's interval includes zero, one is a tie.
   No checkpoint is resolved better, and they straddle zero: more epochs no longer move detection.
   Epoch 5 is picked as the plateau's representative.
6. The thick-cloud rule in the product's exclusion, a manifest / resolver path for the
   progressive-decoder architecture, and a worker CPU-cost check for 34 M parameters.

</details>

---

That is the course so far. Indradhanu V2 (FT04, part 61) is still what ships; Indradhanu Heavy is the
base model, and FTH3 epoch 5 is the release candidate waiting on the product work above.
