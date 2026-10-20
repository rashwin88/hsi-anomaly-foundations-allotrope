# Evaluation: Scoring gains A1-A3 on FT04: spatial matched filter, RM + PCA-RX fusion, mixture of backgrounds

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM (baseline) | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | A1 RM blur 1.0 | 40.7% | 12.0% | 0.177 | 0.891 | 0.064 |
| FT04_ep10 | A1 RM blur 1.4 | 36.3% | 10.2% | 0.177 | 0.895 | 0.069 |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 43.8% | 14.0% | 0.162 | 0.851 | 0.057 |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 42.7% | 11.4% | 0.138 | 0.858 | 0.046 |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 43.0% | 12.8% | 0.180 | 0.887 | 0.064 |
| FT04_ep10 | A3 mixture k=6 | 39.4% | 11.4% | 0.111 | 0.785 | 0.045 |
| FT04_ep10 | A3 mixture k=10 | 34.4% | 8.1% | 0.096 | 0.755 | 0.034 |
| - | PCA-RX | 39.7% | 8.1% | 0.128 | 0.839 | 0.029 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM (baseline) | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | A1 RM blur 1.0 | 49.3% (0.218) | 36.7% (0.172) | 36.0% (0.119) |
| FT04_ep10 | A1 RM blur 1.4 | 40.8% (0.207) | 35.1% (0.179) | 33.0% (0.130) |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 53.0% (0.214) | 40.0% (0.147) | 38.3% (0.095) |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 53.0% (0.196) | 38.0% (0.114) | 37.0% (0.077) |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 52.6% (0.223) | 38.4% (0.174) | 38.0% (0.118) |
| FT04_ep10 | A3 mixture k=6 | 50.3% (0.182) | 34.1% (0.072) | 33.7% (0.053) |
| FT04_ep10 | A3 mixture k=10 | 45.4% (0.161) | 31.8% (0.065) | 26.0% (0.040) |
| - | PCA-RX | 48.7% (0.172) | 35.4% (0.112) | 35.0% (0.075) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT04_ep10 | RM (baseline) | 45.7% |
| FT04_ep10 | A1 RM blur 1.0 | 40.7% |
| FT04_ep10 | A1 RM blur 1.4 | 36.3% |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 43.8% |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 42.7% |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 43.0% |
| FT04_ep10 | A3 mixture k=6 | 39.4% |
| FT04_ep10 | A3 mixture k=10 | 34.4% |
| - | PCA-RX | 39.7% |

## implant

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.5% | 1.1% | - | - | - |
| FT04_ep10 | RM (baseline) | 35.1% | 12.7% | 0.169 | 0.801 | 0.083 |
| FT04_ep10 | A1 RM blur 1.0 | 33.4% | 12.2% | 0.199 | 0.830 | 0.104 |
| FT04_ep10 | A1 RM blur 1.4 | 29.4% | 10.2% | 0.187 | 0.824 | 0.096 |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 34.9% | 12.0% | 0.168 | 0.811 | 0.082 |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 34.8% | 11.3% | 0.166 | 0.813 | 0.079 |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 35.0% | 12.6% | 0.198 | 0.823 | 0.101 |
| FT04_ep10 | A3 mixture k=6 | 41.2% | 18.0% | 0.206 | 0.815 | 0.114 |
| FT04_ep10 | A3 mixture k=10 | 41.9% | 18.0% | 0.209 | 0.815 | 0.122 |
| - | PCA-RX | 33.2% | 9.8% | 0.154 | 0.811 | 0.069 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM (baseline) | 53.7% (0.330) | 28.6% (0.117) | 18.8% (0.052) |
| FT04_ep10 | A1 RM blur 1.0 | 52.4% (0.375) | 27.2% (0.138) | 16.2% (0.072) |
| FT04_ep10 | A1 RM blur 1.4 | 46.8% (0.355) | 23.2% (0.126) | 14.2% (0.067) |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 53.7% (0.325) | 28.0% (0.117) | 19.0% (0.057) |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 53.2% (0.328) | 28.6% (0.113) | 18.4% (0.052) |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 53.7% (0.375) | 28.7% (0.140) | 18.3% (0.067) |
| FT04_ep10 | A3 mixture k=6 | 59.1% (0.369) | 35.7% (0.165) | 24.8% (0.079) |
| FT04_ep10 | A3 mixture k=10 | 60.2% (0.379) | 37.0% (0.163) | 23.9% (0.076) |
| - | PCA-RX | 50.4% (0.302) | 26.9% (0.103) | 18.4% (0.054) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | aviris_ng | enmap | prisma |
|---|---|---|---|---|
| FT04_ep10 | RM (baseline) | 44.1% (0.227) | 20.3% (0.084) | 40.4% (0.195) |
| FT04_ep10 | A1 RM blur 1.0 | 41.8% (0.271) | 24.8% (0.142) | 34.4% (0.184) |
| FT04_ep10 | A1 RM blur 1.4 | 36.9% (0.252) | 22.9% (0.142) | 29.4% (0.167) |
| FT04_ep10 | A2 RM + PCA-RX, rank mean | 44.1% (0.227) | 20.6% (0.081) | 39.8% (0.195) |
| FT04_ep10 | A2 RM + PCA-RX, rank max | 44.5% (0.227) | 19.9% (0.082) | 39.7% (0.190) |
| FT04_ep10 | A1+ RM + RM blur 1.4, rank mean | 42.6% (0.262) | 25.0% (0.137) | 37.5% (0.196) |
| FT04_ep10 | A3 mixture k=6 | 55.9% (0.296) | 23.0% (0.093) | 45.5% (0.229) |
| FT04_ep10 | A3 mixture k=10 | 54.6% (0.300) | 23.0% (0.092) | 47.8% (0.234) |
| - | PCA-RX | 41.5% (0.213) | 19.8% (0.076) | 37.7% (0.174) |
