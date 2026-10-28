# Evaluation: F002-heavy pretrained vs F001 (same recipe, FT04 size) and FT04: real PRISMA bench

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| F001_ep109 | RM | 44.0% | 13.2% | 0.148 | 0.861 | 0.056 |
| F001_ep109 | RM, cloud excluded (FT04 masks) | 45.1% | 14.0% | 0.155 | 0.860 | 0.059 |
| F002_heavy | RM | 44.9% | 12.0% | 0.157 | 0.860 | 0.054 |
| F002_heavy | RM, cloud excluded (FT04 masks) | 47.4% | 12.4% | 0.166 | 0.859 | 0.058 |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded (FT04 masks) | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| F001_ep109 | RM | 54.6% (0.199) | 37.7% (0.127) | 39.7% (0.092) |
| F001_ep109 | RM, cloud excluded (FT04 masks) | 55.9% (0.209) | 38.7% (0.131) | 40.7% (0.096) |
| F002_heavy | RM | 55.6% (0.208) | 38.4% (0.115) | 40.7% (0.113) |
| F002_heavy | RM, cloud excluded (FT04 masks) | 58.2% (0.222) | 40.7% (0.122) | 43.3% (0.118) |
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded (FT04 masks) | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
