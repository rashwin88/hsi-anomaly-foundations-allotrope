# Evaluation: Gated scene adaptation of FT04: N = 50 / 200 / 1000 gated, 200 ungated; real PRISMA bench

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excluded | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| adapt50 | RM | 46.0% | 13.9% | 0.153 | 0.851 | 0.055 |
| adapt50 | RM, cloud excluded | 46.8% | 15.2% | 0.160 | 0.850 | 0.059 |
| adapt200 | RM | 45.9% | 13.6% | 0.151 | 0.851 | 0.056 |
| adapt200 | RM, cloud excluded | 46.9% | 14.9% | 0.159 | 0.850 | 0.059 |
| adapt1000 | RM | 45.3% | 13.4% | 0.153 | 0.849 | 0.055 |
| adapt1000 | RM, cloud excluded | 46.5% | 15.0% | 0.161 | 0.848 | 0.057 |
| adapt200_ungated | RM | 45.8% | 13.5% | 0.151 | 0.852 | 0.056 |
| adapt200_ungated | RM, cloud excluded | 46.9% | 14.9% | 0.159 | 0.850 | 0.059 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excluded | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| adapt50 | RM | 57.9% (0.214) | 39.7% (0.130) | 40.3% (0.084) |
| adapt50 | RM, cloud excluded | 58.9% (0.224) | 40.3% (0.135) | 41.0% (0.089) |
| adapt200 | RM | 57.2% (0.211) | 39.3% (0.129) | 41.0% (0.083) |
| adapt200 | RM, cloud excluded | 58.9% (0.222) | 40.3% (0.135) | 41.3% (0.087) |
| adapt1000 | RM | 56.6% (0.210) | 39.7% (0.133) | 39.7% (0.084) |
| adapt1000 | RM, cloud excluded | 58.6% (0.222) | 41.0% (0.140) | 40.0% (0.089) |
| adapt200_ungated | RM | 56.6% (0.211) | 39.7% (0.128) | 41.0% (0.083) |
| adapt200_ungated | RM, cloud excluded | 58.2% (0.222) | 41.0% (0.134) | 41.3% (0.087) |
