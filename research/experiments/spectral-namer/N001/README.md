# N001 - spectral namer, first run (closed set) - STOPPED after epoch 1, superseded by [N001b](../N001b/README.md)

**Design:** [docs/proposals/spectral-namer.md](../../../../docs/proposals/spectral-namer.md). **Code:**
`app/spectral_match/namer/` (classes, bank, model, generator, train) + `scripts/train_namer.py`.
**Config:** [config.json](config.json) (= `configs/namer/N001.json`). **Box:** H100 capacity block 2
(`cr-0dccba00b34dc29e1`, 2026-10-31 11:30 -> 2026-11-02 11:30 UTC), driver [../namer_run.sh](../namer_run.sh),
code bundle `allotrope-f4f3abd-namer2` (working tree of f4f3abd + the namer; namer2 = BatchNorm frozen at V2 statistics, repeated library names dropped).

## What it trains

- **Start point:** stem, post-stem BatchNorm and the normalisation curve of the released Indradhanu V2 (FTH3
  epoch 5, Heavy fine-tuned on EnMAP / PRISMA / AVIRIS-NG). Never recomputed.
- **Data:** real tiles from the fine-tune shard sets of all three sensors (each on its own band list; per batch only
  the bands live across the tiles, so water bands drop out). Per step one batch of 16 tiles per sensor, 512 sampled
  pixels per tile.
- **Planting** (on the GPU): 50% man-made (train-split spectrum), 25% natural mix, 25% plain; fill log-uniform in
  [0.05, 1]; smooth gain within +/-10%; noise from the tile's own neighbour differences; ring mean 3-7 px.
- **Loss:** BCE (man-made) + class cross-entropy over logsumexp-pooled cosine scores against the train-split
  library, with each sample's own spectrum left out + 0.5 x fill L1.
- **Library:** the 411-entry family library -> 378 usable entries after dropping repeated names, 159 man-made in 7 classes, 34 man-made held out
  (hash of the name, ~20% overall, 17-30% per class) - never planted in training, never in the training class scores.
- 40 epochs x 500 steps, AdamW 3e-4 (stem 0.3x), 500 warmup steps, cosine.

## Evaluation (each epoch, test shards, held-out spectra, own spectrum left out)

Per sensor at fills 0.1 / 0.25 / 0.5 / 1.0: level-1 accuracy, level-2 class accuracy, and **man-made recall at the
classical ACE check's natural false-alarm rate** (from `../separability_<sensor>.json`), which is the pass test.
best.pt = best mean level-2 accuracy at fill 0.25.

**Pass (fixed before the run), fill 0.25, per sensor:** level-1 recall at ACE's false-alarm rate >= ACE + 0.10;
level-2 accuracy >= ACE + 0.10. Caveat: ACE's level-2 numbers were measured on the 17 old families. They must be
re-scored on the v1 classes (building + pavement + asphalt roofing merged, roofing dissolved) before comparing;
that re-score is the first job after launch.

## CPU smoke (laptop, 2026-10-31)

EnMAP + SpectralEarth smoke shards, 120 steps: class CE 1.81 -> 1.02, level-2 accuracy at fill 1.0 0.62 -> 0.65
(EnMAP); level-1 BCE still ~0.68 (too short to judge). Resume from last.pt verified. 8 unit tests in
`tests/test_spectral_match/test_namer.py`.

## Results

Started 2026-10-31 11:37 UTC (bundle namer2), stopped 11:59 after epoch 1 (802 s). Level 2 learned (class accuracy
0.60-0.68 at fill 0.25-1.0 on all three sensors) but **level 1 sat at chance**: accuracy 0.49-0.52 at every fill,
0.50 even at fill 1.0, where ACE gets 0.87; man-made recall at ACE's false-alarm rate 0.21-0.30 (random = the FA
rate, ~0.15-0.2). Train BCE 0.684.

Diagnosis (laptop, EnMAP smoke shards, 150 steps at fill ~1): the linear level-1 head reached BCE 0.64, accuracy 0.65,
recall at ACE's FA 0.41; deciding level 1 by matching (natural library spectra as one more class, P(man-made) =
sigmoid(logsumexp over man-made classes - natural score + bias)) reached BCE 0.43, accuracy 0.80, recall 0.74. The
class scores have library prototypes; the bare head had to learn all of nature in one linear layer. Fix:
`level1: "matching"` (train.py, model.level1_logit). Run again as N001b; metrics.csv / train.log / EXIT on S3 under
experiments/indradhanu-finetune/spectral-namer/N001/.
