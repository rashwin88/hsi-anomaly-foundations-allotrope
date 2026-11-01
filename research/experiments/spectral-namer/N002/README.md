# N002 - spectral namer, open set (textile/fabric held out)

N001b's recipe (level 1 by matching) with one man-made class, **textile/fabric (14 spectra)**, removed from training
and from the training class scores entirely; it replaced paint/coatings (4 spectra after de-duplication), user
2026-10-31. Config: [config.json](config.json) (= `configs/namer/N002.json`). Bundle `namer3`; ran 2026-10-31 21:12 ->
2026-11-01 06:25 UTC, 40 epochs, EXIT 0, on the block-2 H100. Outputs on S3 under
experiments/indradhanu-finetune/spectral-namer/N002/.

**Pass criteria (fixed before the run):** the held-out class is called man-made >= 70%, and is given a confident
(> 0.5) known class <= 20%.

## Result: fails, and abstention does not rescue it

`scripts/namer_eval.py` on epoch 40 (`eval_ep40/summary.json`): test shards, held-out spectra, own spectrum left
out, ~300k planted samples per sensor and fill. Fill 0.25, EnMAP / PRISMA / AVIRIS-NG:

| unseen textile | before abstention | abstain on best cosine (known keep 90%) | abstain on top P(class) (90%) |
|---|---|---|---|
| called man-made | 0.47 / 0.46 / 0.46 | - | - |
| confident wrong class | 0.46 / 0.45 / 0.45 | 0.45 / 0.43 / 0.45 | 0.46 / 0.42 / 0.45 |
| sent to "class unknown" | - | 0.01 / 0.02 / 0.01 | 0.01 / 0.04 / 0.01 |

- **About half of unseen textile is called natural.** Plant-fibre textiles are cellulose, spectrally close to dry
  vegetation.
- **When called man-made, it almost always gets a confident known class.** At a threshold that keeps 90% of the known
  classes' man-made calls, only 1-4% of textile is sent to "unknown". Neither the best library cosine nor the top class
  probability separates "a new material" from "an unseen spectrum of a known one". The failure is in the
  representation, not the threshold.
- **On the known classes** (textile excluded) class accuracy is 0.69 / 0.77 / 0.73 at fill 0.25. The abstention rule
  does work as a confidence flag: at 90% coverage the classes it keeps are 0.80 / 0.84 / 0.81 accurate.
- **What it means for the product:** treat level 2 as "closest known class" with a confidence flag, not a closed-world
  answer. The remedy for unseen materials is library coverage (more man-made spectra and classes).

**Intervals:** the first pass bootstrapped pixels (about +/-0.003), which is wrong: the independent unit is the
library spectrum. `namer_eval.py` now resamples spectra; the re-run is `eval_ep40_v2/`.

## Calibration of the product's "confident" flag

The product (`spectral_library_match`, `namer_confident_p`, default 0.55) marks a man-made object's closest known
class "confident" when its P(class | man-made) is at least that value, and "tentative" below it. 0.55 sits inside the
range of the P(class) thresholds that keep 90% of the known classes' man-made calls in `eval_ep40/summary.json`
(`abstain_pcls_90.threshold`):

| sensor | fill 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| EnMAP | 0.49 | 0.50 | 0.60 | 0.79 |
| PRISMA | 0.51 | 0.61 | 0.76 | 0.89 |
| AVIRIS-NG | 0.51 | 0.54 | 0.71 | 0.83 |

The threshold rises with fill (bigger fills give peakier class probabilities), so one value is a compromise aimed at
the small fills the detector mostly finds. The served model is N001b epoch 40; these thresholds come from N002's
model - close, not identical.
