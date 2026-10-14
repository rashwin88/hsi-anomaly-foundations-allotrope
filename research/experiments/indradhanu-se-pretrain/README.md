# Indradhanu: SpectralEarth pretraining

**Question:** does a cross-attention spectral stem (variant C) find anomalies better than
today's fixed 1x1 compressor (A) or a cheap per-pixel gate (B), when all three are pretrained on
SpectralEarth with the same spatial model and budget?

**Design and decision rule:** `docs/proposals/spectral-attention-bottleneck.md`. The rule was
fixed before any run; do not move it after seeing results.
**Study material:** `indradhanu workings/31-*.md` to `39-*.md`.

## Current verdict

Wiring verified (S01). C runs at ~5 patches/s and ~3.7 GiB per patch in bf16 on a 20 GB card,
78x slower than the linear stem. Order decided 2026-10-14: E003 (C, no binning) first on one
g6e.2xlarge, stopped early when losses flatten; then E001 (A) and E002 (B) under the same rule.

## Runs

One row per run, newest last. A run that crashed or was abandoned still gets a row.

| id | date | variant | question | instance | hours | cost | recon SAM (deg) | PRISMA ROC-AUC | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S01 | 2026-10-14 | A, B, C | wiring smoke (CPU) + GPU cost probe | laptop RTX 4000 Ada 20 GB | 0.2 | 0 | - | - | done: all wired; C ~5 patches/s, ~3.7 GiB/patch bf16 (78x slower than A) |
| E000 | - | C | throughput and batch size that fit | g6e.2xlarge | - | - | - | - | planned |
| E001 | - | A | baseline | g6e.2xlarge | - | - | - | - | planned |
| E002 | - | B | does per-pixel adaptivity help at all? | g6e.2xlarge | - | - | - | - | planned |
| E003 | - | C | does the full design beat A and B? | g6e.2xlarge | - | - | - | - | planned |

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
