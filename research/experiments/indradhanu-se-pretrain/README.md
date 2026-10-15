# Indradhanu: SpectralEarth pretraining

**Question:** does a cross-attention spectral stem (variant C) find anomalies better than
today's fixed 1x1 compressor (A) or a cheap per-pixel gate (B), when all three are pretrained on
SpectralEarth with the same spatial model and budget?

**Design and decision rule:** `docs/proposals/spectral-attention-bottleneck.md`. The rule was
fixed before any run; do not move it after seeing results.
**Study material:** `indradhanu workings/31-*.md` to `39-*.md`.

## Current verdict

Wiring verified (S01). C runs at ~5 patches/s and ~3.7 GiB per patch in bf16 on a 20 GB card,
78x slower than the linear stem. E003 (per-band C) ran 3 epochs on a kept g6e.2xlarge at 13.5
patches/s and was superseded by E004, the reduced-attention stem (80 patches/s, 2026-10-14);
E004 -> E001 -> E002 run in sequence on that instance, batch 8 x 4, validation at full SAM weight.

**Every run is capped at 20 epochs** (user decision, 2026-10-14, so the queue finishes before a
final pretraining run after a code sanity review). `/opt/cap20.sh` on the instance stops the
active run once its `metrics.csv` has 20 rows - after that epoch's checkpoint is in S3 - and the
run's `EXIT` reads 143 with a `CAPPED_AT_20` note beside it. The LR schedule is still the
200-epoch one, so all three stop at the same point on the same curve: equal budget, the decision
rule's condition, holds. None of these is a converged model.

## Runs

One row per run, newest last. A run that crashed or was abandoned still gets a row.

| id | date | variant | question | instance | hours | cost | recon SAM (deg) | PRISMA ROC-AUC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S01 | 2026-10-14 | A, B, C | wiring smoke (CPU) + GPU cost probe | laptop RTX 4000 Ada 20 GB | 0.2 | 0 | - | - | done: all wired; C ~5 patches/s, ~3.7 GiB/patch bf16 (78x slower than A) |
| E003 | 2026-10-14 | C (per-band) | per-band cross-attention stem | g6e.2xlarge (kept) | ~3 | ~$8 | 13.5 (ep 2) | - | superseded after epoch 3: 13.5 patches/s; replaced by E004 |
| E004 | 2026-10-14 | C' (reduced) | does the reduced-attention stem beat A and B? | g6e.2xlarge (kept) | 3.0 | ~$8 | 4.84 (ep 20) | 0.754 resampled / 0.613 native (ep 19) | best detection; native band lists broken (part 42) |
| E001 | 2026-10-14 | A | baseline | g6e.2xlarge (kept) | 2.9 | ~$8 | 5.57 (ep 20) | 0.663 resampled (ep 18) | = v0.2.0 (0.662) |
| E002 | 2026-10-14 | B | does per-pixel adaptivity help at all? | g6e.2xlarge (kept) | 2.8 | ~$8 | 4.54 (ep 20) | 0.739 resampled (ep 19) | best reconstruction; 0.015 behind C' on detection; 202 bands only |
| E005 | 2026-10-15 | C' + band fixes | do the part-42 fixes make native band lists read as well as resampled? | g6e.2xlarge (kept) | 1.5 | ~$4 | 5.85 (ep 10) | 0.720 resampled / 0.705 native (ep 10) | 3 of 4 criteria; set1 native gap 0.09 vs 0.05 |
| F001 | 2026-10-15 | C' + band fixes | final pretraining: 200 epochs, data rotated every 20 through 10 blocks of 30,000 patches | g6e.2xlarge (kept) | - | - | - | - | running |

## Layout

```
indradhanu-se-pretrain/
  README.md              this file: question, verdict, run table
  _TEMPLATE.md           copy to start a run
  E001-baseline-a/
    README.md            question -> setup -> results -> graphs -> observations -> next
    config.json          the exact config the run used
    metrics.csv          one row per epoch
    plots/               PNGs referenced from the run README
```

Rules:

- **Copy `_TEMPLATE.md` before launching**, and fill in the question and setup first. A run
  without a written question does not start.
- **Observations are written after every run**, failures included - what surprised you, what
  you would change.
- **Checkpoints never enter git** (`*.pt` is gitignored). They go to
  `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/<id>/`, and the
  run README records the path.
- **Every GPU launch is approved first**, with instance, expected hours and cost.
- Keep the plots small (PNG, ~150 dpi). Large rasters stay in S3.
