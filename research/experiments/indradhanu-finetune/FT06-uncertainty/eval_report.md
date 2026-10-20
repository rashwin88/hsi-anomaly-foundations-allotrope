# Evaluation: FT06 uncertainty head: RM on residual / b vs FT04 RM, with and without cloud exclusion

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT06 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT06 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT06 | scaled RM | 37.6% | 13.0% | 0.122 | 0.707 | 0.063 |
| FT06 | scaled RM, cloud excluded | 37.8% | 13.2% | 0.125 | 0.708 | 0.064 |
| FT06 | scaled RM sqrt | 44.3% | 16.4% | 0.158 | 0.806 | 0.071 |
| FT06 | scaled RM sqrt, cloud excluded | 44.8% | 16.6% | 0.163 | 0.805 | 0.072 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT06 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT06 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT06 | scaled RM | 50.7% (0.213) | 31.5% (0.088) | 30.7% (0.035) |
| FT06 | scaled RM, cloud excluded | 51.3% (0.217) | 31.8% (0.089) | 30.3% (0.037) |
| FT06 | scaled RM sqrt | 57.9% (0.242) | 37.7% (0.128) | 37.3% (0.072) |
| FT06 | scaled RM sqrt, cloud excluded | 58.2% (0.249) | 38.4% (0.129) | 37.7% (0.076) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT04_ep10 | RM | 45.7% |
| FT04_ep10 | RM, cloud excluded | 46.0% |
| FT06 | RM | 45.7% |
| FT06 | RM, cloud excluded | 46.0% |
| FT06 | scaled RM | 37.6% |
| FT06 | scaled RM, cloud excluded | 37.8% |
| FT06 | scaled RM sqrt | 44.3% |
| FT06 | scaled RM sqrt, cloud excluded | 44.8% |

## implant

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.5% | 1.1% | - | - | - |
| FT04_ep10 | RM | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT04_ep10 | RM, cloud excluded | 36.2% | 13.2% | 0.172 | 0.800 | 0.085 |
| FT06 | RM | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT06 | RM, cloud excluded | 36.2% | 13.2% | 0.172 | 0.800 | 0.085 |
| FT06 | scaled RM | 33.8% | 11.8% | 0.141 | 0.768 | 0.069 |
| FT06 | scaled RM, cloud excluded | 33.9% | 11.8% | 0.140 | 0.767 | 0.068 |
| FT06 | scaled RM sqrt | 35.8% | 12.8% | 0.165 | 0.797 | 0.082 |
| FT06 | scaled RM sqrt, cloud excluded | 36.2% | 12.8% | 0.166 | 0.797 | 0.082 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT04_ep10 | RM, cloud excluded | 55.4% (0.334) | 29.4% (0.123) | 19.5% (0.054) |
| FT06 | RM | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT06 | RM, cloud excluded | 55.4% (0.334) | 29.4% (0.123) | 19.5% (0.054) |
| FT06 | scaled RM | 53.2% (0.287) | 28.0% (0.090) | 15.5% (0.039) |
| FT06 | scaled RM, cloud excluded | 53.5% (0.284) | 28.2% (0.090) | 15.5% (0.039) |
| FT06 | scaled RM sqrt | 55.9% (0.328) | 28.8% (0.112) | 18.3% (0.051) |
| FT06 | scaled RM sqrt, cloud excluded | 56.3% (0.327) | 29.5% (0.114) | 18.4% (0.051) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | aviris_ng | enmap | prisma |
|---|---|---|---|---|
| FT04_ep10 | RM | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT04_ep10 | RM, cloud excluded | 44.1% (0.227) | 23.2% (0.093) | 40.8% (0.197) |
| FT06 | RM | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT06 | RM, cloud excluded | 44.1% (0.227) | 23.2% (0.093) | 40.8% (0.197) |
| FT06 | scaled RM | 39.4% (0.171) | 21.6% (0.074) | 39.2% (0.177) |
| FT06 | scaled RM, cloud excluded | 39.4% (0.172) | 21.8% (0.070) | 39.4% (0.177) |
| FT06 | scaled RM sqrt | 44.1% (0.219) | 21.7% (0.083) | 41.0% (0.195) |
| FT06 | scaled RM sqrt, cloud excluded | 43.9% (0.219) | 23.0% (0.082) | 41.2% (0.196) |
