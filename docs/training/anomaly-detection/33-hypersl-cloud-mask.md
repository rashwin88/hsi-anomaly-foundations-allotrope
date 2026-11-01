# Part 33 - Cloud masking with HyperSL

> **The one thing this part teaches:** a cloud classifier given HyperSL's 256-number embedding plus seven raw bands found
> 28.9% of the cirrus that EnMAP's own masks label, against 67.9% for the same classifier given the 148 raw bands alone;
> HyperSL was pretrained on scenes chosen to be nearly cloud-free, and its per-scene scaling turns the same thin cirrus
> into a different input in every scene.

## Where we are

Part 21 set the scope: cloud is excluded from detections. In September 2026 cloud masks came from thresholds and from
EnMAP's quality layers, which are wrong over the high plateau, and PRISMA L2D has no cloud mask at all. A foundation model's
embedding plus a small classifier is the textbook way to build a mask with few labels. The experiment
`research/experiments/cloud-mask-hypersl/` (2026-10-01) tried it.

| file | what it holds |
|---|---|
| `git show fa8c1ab:research/experiments/cloud-mask-hypersl/step1_training_set.py` | the training pixels: 97 EnMAP scenes on PRISMA bands, labels and their corrections |
| `git show b452e8d:research/experiments/cloud-mask-hypersl/step2_embed.py` | HyperSL embeddings, per-band scene scaling, 148 bands |
| `git show b452e8d:research/experiments/cloud-mask-hypersl/step3_classifier.py` | the classifier and the grouped folds |
| `git show b452e8d:research/experiments/cloud-mask-hypersl/step3_report.txt` | both confusion tables |

> **Analogy.** A wine taster trained only on still wines, each poured at its own cellar's temperature, asked to detect a
> faint sparkle. The sparkle was never in the training glasses, and how a sparkle feels depends on the cellar. A second
> taster with a thermometer and the bare glass does better with no training on wine at all.

## Step 1 - the experiment

- **Pixels.** 97 EnMAP L2A scenes, each spectrum resampled to PRISMA's bands with a Gaussian of PRISMA's band width
  (HyperSL's embedding moves with band sampling, part 30). Classes clear / cloud / cirrus / snow from EnMAP's
  `QL_QUALITY_*` masks, corrected: no cloud or cirrus labels above 2,500 m (EnMAP calls 51-57% of clear plateau cirrus);
  no snow below 1,000 m (salt flats); 2-pixel mask edges dropped. Cirrus = `QL_QUALITY_CIRRUS >= 2`.
- **No 1.38 um band.** EnMAP L2A blanks 1343-1390 nm, the classic cirrus band; PRISMA's 149 axis drops 1325-1495 nm. Both
  variants lack it, so the comparison is fair.
- **Embeddings.** HyperSL at mask 0 on 148 bands, each pixel scaled by its own scene's per-band min and max.
- **Classifier.** `HistGradientBoostingClassifier` (300 rounds, balanced classes, early stopping), two feature sets:
  **C** ("the chosen design") = 256 HyperSL numbers + raw reflectance at 490, 560, 665, 865, 1240, 1610, 2200 nm;
  **B** ("a yardstick only") = the 148 raw bands.
- **Grouped cross-validation:** 5 folds, an EnMAP datatake (one overpass) never split between train and test. 784,120
  pixels, 47 datatakes.

## Step 2 - the results, in full

Variant C (rows EnMAP's label, columns the classifier's out-of-fold call):

| label | clear | cloud | cirrus | snow | row total |
|---|---|---|---|---|---|
| clear | **341,395** | 36,215 | 4,955 | 909 | 383,474 |
| cloud | 22,708 | **244,246** | 11,452 | 2,042 | 280,448 |
| cirrus | 21,979 | 28,341 | **20,472** | 1 | 70,793 |
| snow | 1,619 | 5,025 | 36 | **42,725** | 49,405 |

Variant B:

| label | clear | cloud | cirrus | snow | row total |
|---|---|---|---|---|---|
| clear | **342,688** | 25,158 | 12,901 | 2,727 | 383,474 |
| cloud | 27,357 | **225,595** | 25,510 | 1,986 | 280,448 |
| cirrus | 6,861 | 15,888 | **48,037** | 7 | 70,793 |
| snow | 406 | 1,135 | 16 | **47,848** | 49,405 |

**Recall** = diagonal / row total ("found x% of EnMAP's"); **precision** = diagonal / column total ("right y% of the time").

Worked, cirrus, variant C: row `21,979 + 28,341 + 20,472 + 1 = 70,793`; recall `20,472 / 70,793 = 28.9%`; column
`4,955 + 11,452 + 20,472 + 36 = 36,915`; precision `20,472 / 36,915 = 55.5%`. Variant B: recall `48,037 / 70,793 = 67.9%`;
column `12,901 + 25,510 + 48,037 + 16 = 86,464`; precision `48,037 / 86,464 = 55.6%`. Same precision, 2.35x the cirrus found.

| class | recall C | recall B | precision C | precision B |
|---|---|---|---|---|
| clear | 89.0% | 89.4% | 88.1% | 90.8% |
| cloud | **87.1%** | 80.4% | 77.8% | **84.2%** |
| cirrus | 28.9% | **67.9%** | 55.5% | 55.6% |
| snow | 86.5% | **96.8%** | 93.5% | 91.0% |

Overall accuracy C 82.75%, B 84.70%. High-plateau clear pixels called cloud or cirrus: **C 4.34%**, B 6.62%.

**Being fair:** C finds more thick cloud and makes fewer plateau false calls. And all of this is agreement with EnMAP's
masks, not truth.

**Source check:** the run never tested the 7 raw bands alone, or HyperSL plus all 148 bands, so it cannot split B's advantage
into "more raw bands" and "no HyperSL". It does show that HyperSL's 256 numbers on top of 7 bands are worth far less, on
cirrus, than the other 141 bands.

## Step 3 - why thin cloud is hard for this model

**It was never shown clouds.** The paper picked pretraining scenes with "cloud cover near zero" (part 29).

**The same cirrus is a different input in every scene.** Cirrus roughly adds `delta` (an assumption). Under min-max:

```
x'_cirrus - x'_clear = ((x + delta) - m) / (M - m) - (x - m) / (M - m) = delta / (M - m)
```

The step depends on the scene's span. `delta = 0.03` at 865 nm: in a dark scene with span 0.20, `0.03 / 0.20 = 0.15`; in a
bright desert with span 0.60, `0.03 / 0.60 = 0.05`. Three times different for one physical signal; with raw reflectance it is
0.03 in both. And the brightest pixels of a cloudy scene are often the cloud, which then sets the frame for everyone (part 30).

**What the product does:** cloud is a class of Indradhanu's own segmentation head, trained on labelled masks from
reflectance; the release candidate (FTH3) excludes thick cloud only (Indradhanu part 69). The HyperSL classifier was not
adopted.

## Common confusions

**"HyperSL is worse at every cloud class."** Better on thick cloud recall and plateau false calls; worse on cirrus, snow, and
overall by two points.

**"Cirrus failed because 1.38 um was missing."** Missing for both variants.

**"Recall and precision are the same thing twice."** Row versus column; C and B share cirrus precision but not recall.

**"These numbers say how often the mask is right."** How often it agrees with EnMAP's masks.

## Check yourself

1. From variant C, compute snow recall and precision.
2. From variant B, compute cloud recall and precision.
3. Compute both overall accuracies (diagonal / 784,120).
4. A scene has span 0.40 at 865 nm. What step does `delta = 0.03` make? What span makes it 0.10?
5. What share of C's missed cirrus was called cloud? Why is that less harmful to the product than "clear"?

<details>
<summary>Answers</summary>

1. `42,725 / 49,405 = 86.5%`; column `909 + 2,042 + 1 + 42,725 = 45,677`; `42,725 / 45,677 = 93.5%`.
2. `225,595 / 280,448 = 80.4%`; column `25,158 + 225,595 + 15,888 + 1,135 = 267,776`; `225,595 / 267,776 = 84.2%`.
3. C `341,395 + 244,246 + 20,472 + 42,725 = 648,838`, `/ 784,120 = 82.75%`; B `342,688 + 225,595 + 48,037 + 47,848 =
   664,168`, `/ 784,120 = 84.70%`.
4. `0.03 / 0.40 = 0.075`; span `0.03 / 0.10 = 0.30`.
5. Missed `70,793 - 20,472 = 50,321`; `28,341 / 50,321 = 56.3%`. Cloud is excluded too, so those pixels still leave the
   detections; cirrus called clear stays, and any false alarm on it is reported.

</details>

---

Next: part 34, the verdict - what HyperSL would need to become useful here.
