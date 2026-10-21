# Evaluation: FT08 gated + blocks vs FT04: real bench and bench-v2.1, cloud excluded on FT04's masks

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT08 | RM | 44.8% | 14.7% | 0.156 | 0.845 | 0.057 |
| FT08 | RM, cloud excluded | 46.4% | 16.1% | 0.164 | 0.844 | 0.062 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT08 | RM | 57.9% (0.213) | 37.0% (0.135) | 39.3% (0.087) |
| FT08 | RM, cloud excluded | 58.6% (0.225) | 39.3% (0.141) | 41.3% (0.092) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT04_ep10 | RM | 45.7% |
| FT04_ep10 | RM, cloud excluded | 46.0% |
| FT08 | RM | 44.8% |
| FT08 | RM, cloud excluded | 46.4% |

## implant

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.5% | 1.1% | - | - | - |
| FT04_ep10 | RM | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT04_ep10 | RM, cloud excluded | 36.2% | 13.2% | 0.172 | 0.800 | 0.085 |
| FT08 | RM | 35.3% | 13.0% | 0.166 | 0.798 | 0.084 |
| FT08 | RM, cloud excluded | 36.7% | 13.4% | 0.170 | 0.797 | 0.085 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT04_ep10 | RM, cloud excluded | 55.4% (0.334) | 29.4% (0.123) | 19.5% (0.054) |
| FT08 | RM | 54.4% (0.331) | 27.8% (0.113) | 19.7% (0.047) |
| FT08 | RM, cloud excluded | 56.4% (0.337) | 29.2% (0.119) | 20.5% (0.049) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | aviris_ng | enmap | prisma |
|---|---|---|---|---|
| FT04_ep10 | RM | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT04_ep10 | RM, cloud excluded | 44.1% (0.227) | 23.2% (0.093) | 40.8% (0.197) |
| FT08 | RM | 43.7% (0.216) | 20.9% (0.086) | 40.7% (0.195) |
| FT08 | RM, cloud excluded | 43.8% (0.216) | 24.6% (0.096) | 41.3% (0.197) |
