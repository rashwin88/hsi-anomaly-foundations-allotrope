# Evaluation: FTH2 (Heavy, 18 epochs) vs FTB2 (control) epochs 6-8, thick-cloud exclusion, vs FT04 shipped and thick: real PRISMA bench

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_shipped | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_shipped | excl. FT04 masks (cloud or thin) | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT04_shipped | excl. FT04 thick | 46.1% | 15.8% | 0.169 | 0.855 | 0.065 |
| FT04_shipped | excl. FTH2 thick | 47.0% | 16.5% | 0.174 | 0.856 | 0.067 |
| FT04_shipped | excl. FTB2 thick | 46.2% | 15.8% | 0.170 | 0.855 | 0.065 |
| FT04_thick | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_thick | excl. FT04 masks (cloud or thin) | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT04_thick | excl. FT04 thick | 46.1% | 15.8% | 0.169 | 0.855 | 0.065 |
| FT04_thick | excl. FTH2 thick | 47.0% | 16.5% | 0.174 | 0.856 | 0.067 |
| FT04_thick | excl. FTB2 thick | 46.2% | 15.8% | 0.170 | 0.855 | 0.065 |
| FTH2_ep6 | RM | 45.7% | 13.6% | 0.171 | 0.866 | 0.059 |
| FTH2_ep6 | excl. FT04 masks (cloud or thin) | 47.6% | 14.7% | 0.180 | 0.865 | 0.064 |
| FTH2_ep6 | excl. FT04 thick | 47.1% | 14.7% | 0.179 | 0.867 | 0.063 |
| FTH2_ep6 | excl. FTH2 thick | 48.2% | 15.3% | 0.185 | 0.869 | 0.066 |
| FTH2_ep6 | excl. FTB2 thick | 47.3% | 14.7% | 0.179 | 0.867 | 0.064 |
| FTH2_ep7 | RM | 46.3% | 13.6% | 0.170 | 0.864 | 0.058 |
| FTH2_ep7 | excl. FT04 masks (cloud or thin) | 48.2% | 14.7% | 0.179 | 0.863 | 0.062 |
| FTH2_ep7 | excl. FT04 thick | 47.7% | 14.7% | 0.178 | 0.865 | 0.061 |
| FTH2_ep7 | excl. FTH2 thick | 49.0% | 15.2% | 0.184 | 0.867 | 0.063 |
| FTH2_ep7 | excl. FTB2 thick | 48.0% | 14.7% | 0.178 | 0.865 | 0.062 |
| FTH2_ep8 | RM | 46.6% | 13.8% | 0.173 | 0.866 | 0.058 |
| FTH2_ep8 | excl. FT04 masks (cloud or thin) | 48.0% | 15.2% | 0.181 | 0.865 | 0.063 |
| FTH2_ep8 | excl. FT04 thick | 47.7% | 15.2% | 0.181 | 0.868 | 0.062 |
| FTH2_ep8 | excl. FTH2 thick | 49.0% | 15.7% | 0.186 | 0.870 | 0.066 |
| FTH2_ep8 | excl. FTB2 thick | 48.0% | 15.2% | 0.181 | 0.868 | 0.063 |
| FTB2_ep6 | RM | 44.7% | 14.3% | 0.152 | 0.858 | 0.056 |
| FTB2_ep6 | excl. FT04 masks (cloud or thin) | 46.2% | 15.2% | 0.161 | 0.856 | 0.061 |
| FTB2_ep6 | excl. FT04 thick | 45.9% | 15.1% | 0.160 | 0.859 | 0.061 |
| FTB2_ep6 | excl. FTH2 thick | 46.8% | 15.4% | 0.165 | 0.861 | 0.063 |
| FTB2_ep6 | excl. FTB2 thick | 46.1% | 15.1% | 0.160 | 0.860 | 0.061 |
| FTB2_ep7 | RM | 44.8% | 14.4% | 0.155 | 0.860 | 0.056 |
| FTB2_ep7 | excl. FT04 masks (cloud or thin) | 46.4% | 15.1% | 0.163 | 0.858 | 0.061 |
| FTB2_ep7 | excl. FT04 thick | 46.3% | 15.0% | 0.162 | 0.861 | 0.060 |
| FTB2_ep7 | excl. FTH2 thick | 47.1% | 15.5% | 0.168 | 0.863 | 0.062 |
| FTB2_ep7 | excl. FTB2 thick | 46.4% | 15.1% | 0.163 | 0.861 | 0.061 |
| FTB2_ep8 | RM | 45.4% | 14.3% | 0.154 | 0.859 | 0.057 |
| FTB2_ep8 | excl. FT04 masks (cloud or thin) | 46.5% | 15.1% | 0.162 | 0.858 | 0.061 |
| FTB2_ep8 | excl. FT04 thick | 46.6% | 14.9% | 0.161 | 0.861 | 0.060 |
| FTB2_ep8 | excl. FTH2 thick | 47.4% | 15.5% | 0.167 | 0.862 | 0.062 |
| FTB2_ep8 | excl. FTB2 thick | 46.8% | 15.1% | 0.162 | 0.861 | 0.061 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_shipped | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_shipped | excl. FT04 masks (cloud or thin) | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT04_shipped | excl. FT04 thick | 58.2% (0.230) | 39.3% (0.150) | 40.7% (0.095) |
| FT04_shipped | excl. FTH2 thick | 58.9% (0.236) | 40.7% (0.154) | 41.3% (0.099) |
| FT04_shipped | excl. FTB2 thick | 58.2% (0.231) | 39.7% (0.150) | 40.7% (0.095) |
| FT04_thick | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_thick | excl. FT04 masks (cloud or thin) | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT04_thick | excl. FT04 thick | 58.2% (0.230) | 39.3% (0.150) | 40.7% (0.095) |
| FT04_thick | excl. FTH2 thick | 58.9% (0.236) | 40.7% (0.154) | 41.3% (0.099) |
| FT04_thick | excl. FTB2 thick | 58.2% (0.231) | 39.7% (0.150) | 40.7% (0.095) |
| FTH2_ep6 | RM | 57.6% (0.226) | 40.3% (0.138) | 39.0% (0.113) |
| FTH2_ep6 | excl. FT04 masks (cloud or thin) | 59.5% (0.239) | 42.0% (0.146) | 41.3% (0.115) |
| FTH2_ep6 | excl. FT04 thick | 58.9% (0.237) | 41.6% (0.144) | 40.7% (0.116) |
| FTH2_ep6 | excl. FTH2 thick | 59.9% (0.244) | 43.3% (0.150) | 41.3% (0.120) |
| FTH2_ep6 | excl. FTB2 thick | 59.2% (0.238) | 41.6% (0.144) | 41.0% (0.117) |
| FTH2_ep7 | RM | 58.9% (0.227) | 40.3% (0.136) | 39.7% (0.111) |
| FTH2_ep7 | excl. FT04 masks (cloud or thin) | 60.9% (0.239) | 42.6% (0.144) | 41.0% (0.113) |
| FTH2_ep7 | excl. FT04 thick | 60.2% (0.237) | 42.3% (0.142) | 40.7% (0.114) |
| FTH2_ep7 | excl. FTH2 thick | 61.2% (0.245) | 43.6% (0.148) | 42.0% (0.118) |
| FTH2_ep7 | excl. FTB2 thick | 60.2% (0.238) | 42.6% (0.142) | 41.0% (0.115) |
| FTH2_ep8 | RM | 59.5% (0.229) | 40.7% (0.139) | 39.7% (0.113) |
| FTH2_ep8 | excl. FT04 masks (cloud or thin) | 60.2% (0.242) | 42.3% (0.147) | 41.3% (0.114) |
| FTH2_ep8 | excl. FT04 thick | 59.5% (0.240) | 42.6% (0.145) | 41.0% (0.116) |
| FTH2_ep8 | excl. FTH2 thick | 60.9% (0.247) | 43.9% (0.151) | 42.0% (0.119) |
| FTH2_ep8 | excl. FTB2 thick | 59.5% (0.241) | 43.0% (0.146) | 41.3% (0.116) |
| FTB2_ep6 | RM | 57.6% (0.214) | 38.4% (0.125) | 38.0% (0.086) |
| FTB2_ep6 | excl. FT04 masks (cloud or thin) | 59.5% (0.226) | 40.0% (0.131) | 39.0% (0.091) |
| FTB2_ep6 | excl. FT04 thick | 59.2% (0.224) | 40.0% (0.131) | 38.3% (0.090) |
| FTB2_ep6 | excl. FTH2 thick | 59.9% (0.230) | 40.7% (0.135) | 39.7% (0.094) |
| FTB2_ep6 | excl. FTB2 thick | 59.2% (0.225) | 40.3% (0.131) | 38.7% (0.090) |
| FTB2_ep7 | RM | 58.2% (0.216) | 38.4% (0.128) | 37.7% (0.089) |
| FTB2_ep7 | excl. FT04 masks (cloud or thin) | 60.5% (0.227) | 40.0% (0.134) | 38.7% (0.094) |
| FTB2_ep7 | excl. FT04 thick | 60.2% (0.226) | 40.0% (0.134) | 38.7% (0.093) |
| FTB2_ep7 | excl. FTH2 thick | 60.9% (0.232) | 41.0% (0.139) | 39.3% (0.098) |
| FTB2_ep7 | excl. FTB2 thick | 60.5% (0.226) | 40.0% (0.135) | 38.7% (0.094) |
| FTB2_ep8 | RM | 58.9% (0.214) | 39.7% (0.128) | 37.7% (0.089) |
| FTB2_ep8 | excl. FT04 masks (cloud or thin) | 60.5% (0.226) | 40.7% (0.134) | 38.3% (0.094) |
| FTB2_ep8 | excl. FT04 thick | 60.5% (0.224) | 41.0% (0.134) | 38.3% (0.093) |
| FTB2_ep8 | excl. FTH2 thick | 60.9% (0.230) | 42.0% (0.139) | 39.3% (0.098) |
| FTB2_ep8 | excl. FTB2 thick | 60.5% (0.225) | 41.3% (0.135) | 38.3% (0.093) |
