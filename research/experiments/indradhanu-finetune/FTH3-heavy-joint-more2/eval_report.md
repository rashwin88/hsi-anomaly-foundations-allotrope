# Evaluation: FTH3 (Heavy, 24 epochs) epochs 4-6 vs FTH2 epoch 8 and FT04 shipped: real PRISMA bench, own thick-cloud masks

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_shipped | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_shipped | excl. FT04 masks (cloud or thin) | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT04_shipped | excl. FTH2 thick | 47.0% | 16.5% | 0.174 | 0.856 | 0.067 |
| FT04_shipped | excl. FTH3_ep4 thick | 46.9% | 16.5% | 0.173 | 0.856 | 0.067 |
| FT04_shipped | excl. FTH3_ep5 thick | 47.0% | 16.7% | 0.174 | 0.857 | 0.067 |
| FT04_shipped | excl. FTH3_ep6 thick | 46.9% | 16.8% | 0.174 | 0.856 | 0.067 |
| FTH2_ep8 | RM | 46.6% | 13.8% | 0.173 | 0.866 | 0.058 |
| FTH2_ep8 | excl. FT04 masks (cloud or thin) | 48.0% | 15.2% | 0.181 | 0.865 | 0.063 |
| FTH2_ep8 | excl. FTH2 thick | 49.0% | 15.7% | 0.186 | 0.870 | 0.066 |
| FTH2_ep8 | excl. FTH3_ep4 thick | 48.7% | 15.7% | 0.186 | 0.869 | 0.065 |
| FTH2_ep8 | excl. FTH3_ep5 thick | 49.0% | 15.8% | 0.187 | 0.870 | 0.066 |
| FTH2_ep8 | excl. FTH3_ep6 thick | 48.8% | 15.8% | 0.186 | 0.870 | 0.066 |
| FTH3_ep4 | RM | 46.2% | 13.6% | 0.171 | 0.866 | 0.056 |
| FTH3_ep4 | excl. FT04 masks (cloud or thin) | 48.3% | 15.1% | 0.180 | 0.865 | 0.060 |
| FTH3_ep4 | excl. FTH2 thick | 49.6% | 15.4% | 0.185 | 0.869 | 0.063 |
| FTH3_ep4 | excl. FTH3_ep4 thick | 49.2% | 15.4% | 0.184 | 0.869 | 0.063 |
| FTH3_ep4 | excl. FTH3_ep5 thick | 49.6% | 15.4% | 0.185 | 0.869 | 0.063 |
| FTH3_ep4 | excl. FTH3_ep6 thick | 49.4% | 15.4% | 0.185 | 0.869 | 0.063 |
| FTH3_ep5 | RM | 46.4% | 13.9% | 0.175 | 0.866 | 0.058 |
| FTH3_ep5 | excl. FT04 masks (cloud or thin) | 48.2% | 15.2% | 0.183 | 0.865 | 0.063 |
| FTH3_ep5 | excl. FTH2 thick | 49.4% | 16.0% | 0.189 | 0.869 | 0.065 |
| FTH3_ep5 | excl. FTH3_ep4 thick | 49.1% | 16.0% | 0.188 | 0.869 | 0.065 |
| FTH3_ep5 | excl. FTH3_ep5 thick | 49.4% | 16.0% | 0.189 | 0.869 | 0.065 |
| FTH3_ep5 | excl. FTH3_ep6 thick | 49.4% | 16.0% | 0.189 | 0.869 | 0.065 |
| FTH3_ep6 | RM | 46.1% | 13.6% | 0.174 | 0.867 | 0.057 |
| FTH3_ep6 | excl. FT04 masks (cloud or thin) | 48.1% | 15.3% | 0.182 | 0.866 | 0.060 |
| FTH3_ep6 | excl. FTH2 thick | 49.0% | 15.5% | 0.188 | 0.870 | 0.064 |
| FTH3_ep6 | excl. FTH3_ep4 thick | 48.6% | 15.5% | 0.187 | 0.870 | 0.063 |
| FTH3_ep6 | excl. FTH3_ep5 thick | 49.0% | 15.5% | 0.188 | 0.870 | 0.064 |
| FTH3_ep6 | excl. FTH3_ep6 thick | 48.8% | 15.5% | 0.187 | 0.870 | 0.064 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_shipped | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_shipped | excl. FT04 masks (cloud or thin) | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT04_shipped | excl. FTH2 thick | 58.9% (0.236) | 40.7% (0.154) | 41.3% (0.099) |
| FT04_shipped | excl. FTH3_ep4 thick | 58.6% (0.235) | 40.7% (0.154) | 41.3% (0.098) |
| FT04_shipped | excl. FTH3_ep5 thick | 58.9% (0.236) | 40.7% (0.154) | 41.3% (0.099) |
| FT04_shipped | excl. FTH3_ep6 thick | 58.6% (0.236) | 40.7% (0.154) | 41.3% (0.099) |
| FTH2_ep8 | RM | 59.5% (0.229) | 40.7% (0.139) | 39.7% (0.113) |
| FTH2_ep8 | excl. FT04 masks (cloud or thin) | 60.2% (0.242) | 42.3% (0.147) | 41.3% (0.114) |
| FTH2_ep8 | excl. FTH2 thick | 60.9% (0.247) | 43.9% (0.151) | 42.0% (0.119) |
| FTH2_ep8 | excl. FTH3_ep4 thick | 60.5% (0.246) | 43.6% (0.151) | 42.0% (0.119) |
| FTH2_ep8 | excl. FTH3_ep5 thick | 60.9% (0.247) | 43.9% (0.151) | 42.0% (0.119) |
| FTH2_ep8 | excl. FTH3_ep6 thick | 60.9% (0.247) | 43.6% (0.151) | 42.0% (0.119) |
| FTH3_ep4 | RM | 59.2% (0.230) | 40.0% (0.134) | 39.3% (0.109) |
| FTH3_ep4 | excl. FT04 masks (cloud or thin) | 60.9% (0.243) | 42.3% (0.142) | 41.7% (0.111) |
| FTH3_ep4 | excl. FTH2 thick | 62.2% (0.249) | 44.3% (0.147) | 42.3% (0.117) |
| FTH3_ep4 | excl. FTH3_ep4 thick | 61.5% (0.248) | 43.6% (0.146) | 42.3% (0.116) |
| FTH3_ep4 | excl. FTH3_ep5 thick | 62.2% (0.249) | 44.3% (0.147) | 42.3% (0.117) |
| FTH3_ep4 | excl. FTH3_ep6 thick | 61.8% (0.249) | 43.9% (0.147) | 42.3% (0.117) |
| FTH3_ep5 | RM | 59.2% (0.236) | 41.0% (0.138) | 39.0% (0.110) |
| FTH3_ep5 | excl. FT04 masks (cloud or thin) | 61.2% (0.249) | 42.3% (0.146) | 41.0% (0.111) |
| FTH3_ep5 | excl. FTH2 thick | 62.2% (0.255) | 43.9% (0.150) | 42.0% (0.118) |
| FTH3_ep5 | excl. FTH3_ep4 thick | 61.8% (0.254) | 43.6% (0.150) | 41.7% (0.117) |
| FTH3_ep5 | excl. FTH3_ep5 thick | 62.2% (0.255) | 43.9% (0.151) | 42.0% (0.118) |
| FTH3_ep5 | excl. FTH3_ep6 thick | 62.2% (0.255) | 43.9% (0.150) | 42.0% (0.118) |
| FTH3_ep6 | RM | 59.2% (0.232) | 40.7% (0.140) | 38.3% (0.112) |
| FTH3_ep6 | excl. FT04 masks (cloud or thin) | 60.9% (0.244) | 42.6% (0.148) | 40.7% (0.113) |
| FTH3_ep6 | excl. FTH2 thick | 61.8% (0.249) | 43.9% (0.152) | 41.0% (0.119) |
| FTH3_ep6 | excl. FTH3_ep4 thick | 61.2% (0.249) | 43.6% (0.151) | 41.0% (0.119) |
| FTH3_ep6 | excl. FTH3_ep5 thick | 61.8% (0.249) | 43.9% (0.152) | 41.0% (0.119) |
| FTH3_ep6 | excl. FTH3_ep6 thick | 61.5% (0.249) | 43.9% (0.152) | 41.0% (0.119) |
