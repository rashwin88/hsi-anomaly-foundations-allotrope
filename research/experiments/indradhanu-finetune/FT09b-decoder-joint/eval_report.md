# Evaluation: FT09b progressive decoder vs FT04: real bench and bench-v2.1, cloud excluded on FT04's masks

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT09b | RM | 45.9% | 14.3% | 0.169 | 0.855 | 0.058 |
| FT09b | RM, cloud excluded | 47.1% | 15.4% | 0.176 | 0.854 | 0.063 |
| FT09a | RM | 45.1% | 14.2% | 0.162 | 0.857 | 0.055 |
| FT09a | RM, cloud excluded | 46.1% | 15.6% | 0.169 | 0.856 | 0.061 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT09b | RM | 55.3% (0.222) | 41.3% (0.154) | 41.0% (0.100) |
| FT09b | RM, cloud excluded | 57.2% (0.232) | 42.6% (0.158) | 41.3% (0.104) |
| FT09a | RM | 55.9% (0.217) | 40.3% (0.141) | 39.0% (0.095) |
| FT09a | RM, cloud excluded | 56.9% (0.228) | 41.3% (0.146) | 40.0% (0.099) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT04_ep10 | RM | 45.7% |
| FT04_ep10 | RM, cloud excluded | 46.0% |
| FT09b | RM | 45.9% |
| FT09b | RM, cloud excluded | 47.1% |
| FT09a | RM | 45.1% |
| FT09a | RM, cloud excluded | 46.1% |

## implant

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.5% | 1.1% | - | - | - |
| FT04_ep10 | RM | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT04_ep10 | RM, cloud excluded | 36.2% | 13.2% | 0.172 | 0.800 | 0.085 |
| FT09b | RM | 35.3% | 12.4% | 0.167 | 0.804 | 0.083 |
| FT09b | RM, cloud excluded | 36.1% | 13.0% | 0.170 | 0.803 | 0.085 |
| FT09a | RM | 34.8% | 12.2% | 0.166 | 0.803 | 0.081 |
| FT09a | RM, cloud excluded | 35.8% | 12.9% | 0.169 | 0.802 | 0.083 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT04_ep10 | RM, cloud excluded | 55.4% (0.334) | 29.4% (0.123) | 19.5% (0.054) |
| FT09b | RM | 53.5% (0.326) | 29.0% (0.114) | 19.5% (0.054) |
| FT09b | RM, cloud excluded | 54.5% (0.329) | 29.6% (0.119) | 20.1% (0.056) |
| FT09a | RM | 53.2% (0.324) | 28.4% (0.113) | 18.7% (0.055) |
| FT09a | RM, cloud excluded | 54.5% (0.327) | 29.5% (0.118) | 19.2% (0.057) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | aviris_ng | enmap | prisma |
|---|---|---|---|---|
| FT04_ep10 | RM | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT04_ep10 | RM, cloud excluded | 44.1% (0.227) | 23.2% (0.093) | 40.8% (0.197) |
| FT09b | RM | 45.4% (0.227) | 20.3% (0.083) | 40.1% (0.191) |
| FT09b | RM, cloud excluded | 45.4% (0.227) | 22.6% (0.089) | 40.2% (0.193) |
| FT09a | RM | 43.7% (0.226) | 20.4% (0.083) | 39.8% (0.190) |
| FT09a | RM, cloud excluded | 43.7% (0.226) | 23.2% (0.089) | 40.2% (0.192) |
