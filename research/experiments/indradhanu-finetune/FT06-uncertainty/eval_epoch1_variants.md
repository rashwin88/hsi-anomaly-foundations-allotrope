# Evaluation: FT06 epoch 1: softer scaling variants, real bench

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT06_ep1 | RM, cloud excl. | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT06_ep1 | scaled p1 f5, cloud excl. | 41.0% | 15.2% | 0.151 | 0.757 | 0.073 |
| FT06_ep1 | scaled p0.5 f5, cloud excl. | 46.1% | 17.3% | 0.176 | 0.822 | 0.079 |
| FT06_ep1 | scaled p1 f25, cloud excl. | 41.7% | 15.2% | 0.154 | 0.768 | 0.073 |
| FT06_ep1 | scaled p0.5 f25, cloud excl. | 46.3% | 17.3% | 0.177 | 0.826 | 0.079 |
| FT06_ep1 | rank mean RM + scaled, cloud excl. | 45.3% | 17.3% | 0.168 | 0.816 | 0.076 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT06_ep1 | RM, cloud excl. | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT06_ep1 | scaled p1 f5, cloud excl. | 54.9% (0.255) | 34.4% (0.107) | 33.7% (0.052) |
| FT06_ep1 | scaled p0.5 f5, cloud excl. | 59.2% (0.265) | 39.7% (0.142) | 39.3% (0.083) |
| FT06_ep1 | scaled p1 f25, cloud excl. | 55.6% (0.259) | 34.8% (0.109) | 34.7% (0.055) |
| FT06_ep1 | scaled p0.5 f25, cloud excl. | 59.2% (0.265) | 39.7% (0.143) | 40.0% (0.084) |
| FT06_ep1 | rank mean RM + scaled, cloud excl. | 58.2% (0.255) | 39.7% (0.134) | 38.0% (0.077) |

### by sensor (caught top 1%; pAUC where measured)

| model | scorer | prisma |
|---|---|---|
| FT06_ep1 | RM, cloud excl. | 46.0% |
| FT06_ep1 | scaled p1 f5, cloud excl. | 41.0% |
| FT06_ep1 | scaled p0.5 f5, cloud excl. | 46.1% |
| FT06_ep1 | scaled p1 f25, cloud excl. | 41.7% |
| FT06_ep1 | scaled p0.5 f25, cloud excl. | 46.3% |
| FT06_ep1 | rank mean RM + scaled, cloud excl. | 45.3% |
