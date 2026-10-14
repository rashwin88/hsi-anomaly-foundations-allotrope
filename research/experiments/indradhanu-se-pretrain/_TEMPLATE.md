# E0NN - <short name>

**Date:** YYYY-MM-DD to YYYY-MM-DD
**Status:** planned | running | done | failed | abandoned

## Question

One sentence. What will this run tell us that we do not know?

**Expected outcome, written before launch:** what you think will happen, and why.

## Setup

| | |
|---|---|
| variant | A / B / C |
| git commit | `<sha>` |
| config | [config.json](config.json) |
| data | shard set, number of shards, patches seen per epoch |
| normalisation | path of `stats.json` used |
| instance | type, account, on-demand or spot |
| wall time | hours |
| cost | USD |
| checkpoints | `s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/E0NN/` |

What differs from the previous run, and nothing else:

-

## Results

| metric | value | vs baseline |
|---|---|---|
| recon L1 (se-v1 test, hidden pixels) | | |
| recon SAM, degrees | | |
| PRISMA ROC-AUC, mean of 4 scenes | | |
| PRISMA TPR at 1% FPR, mean | | |
| throughput, patches/s | | |
| peak GPU memory, GB | | |

Per-scene detection:

| scene | ROC-AUC | TPR @ 1% FPR |
|---|---|---|
| Set-1 | | |
| Set-2 | | |
| Set-3 | | |
| Set-4 | | |

## Graphs

![loss curves](plots/loss.png)

## Observations

What happened, what surprised you, what you would change. Write this even if the run failed.

## Next

The run this one argues for, and why.
