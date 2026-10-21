# Evaluation: FT07 gated vs FT04: real bench and bench-v2.1, cloud excluded on FT04's masks

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT07 | RM | 45.4% | 14.5% | 0.157 | 0.852 | 0.057 |
| FT07 | RM, cloud excluded | 46.1% | 15.4% | 0.165 | 0.851 | 0.062 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT07 | RM | 56.9% (0.219) | 39.3% (0.134) | 40.0% (0.085) |
| FT07 | RM, cloud excluded | 57.9% (0.230) | 39.7% (0.139) | 40.7% (0.090) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT04_ep10 | RM | 45.7% |
| FT04_ep10 | RM, cloud excluded | 46.0% |
| FT07 | RM | 45.4% |
| FT07 | RM, cloud excluded | 46.1% |

## implant

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.5% | 1.1% | - | - | - |
| FT04_ep10 | RM | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT04_ep10 | RM, cloud excluded | 36.2% | 13.2% | 0.172 | 0.800 | 0.085 |
| FT07 | RM | 35.2% | 12.6% | 0.169 | 0.802 | 0.084 |
| FT07 | RM, cloud excluded | 36.4% | 13.2% | 0.173 | 0.801 | 0.086 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT04_ep10 | RM, cloud excluded | 55.4% (0.334) | 29.4% (0.123) | 19.5% (0.054) |
| FT07 | RM | 54.2% (0.331) | 28.7% (0.117) | 18.4% (0.053) |
| FT07 | RM, cloud excluded | 56.0% (0.335) | 29.7% (0.123) | 19.2% (0.055) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | aviris_ng | enmap | prisma |
|---|---|---|---|---|
| FT04_ep10 | RM | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT04_ep10 | RM, cloud excluded | 44.1% (0.227) | 23.2% (0.093) | 40.8% (0.197) |
| FT07 | RM | 43.9% (0.227) | 20.6% (0.084) | 40.5% (0.195) |
| FT07 | RM, cloud excluded | 43.9% (0.227) | 24.1% (0.092) | 40.8% (0.198) |
