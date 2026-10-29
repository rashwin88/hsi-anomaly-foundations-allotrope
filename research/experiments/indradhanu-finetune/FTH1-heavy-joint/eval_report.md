# Evaluation: FTH1 (Heavy fine-tuned) vs FTB1 (control), epochs 8-10, vs FT04: real PRISMA bench

Objects caught = share of objects with a pixel within 1 px of the top q% of scores; chance is the analytic level for random scores. Pixel measures are means over scenes.

## prisma_bench

| model | scorer | caught top 1% | caught top 0.1% | pauc_0.01 | roc_auc | tpr_at_0.001 |
|---|---|---|---|---|---|---|
| *chance* | | 10.2% | 1.1% | - | - | - |
| FT04_ep10 | RM | 45.7% | 14.9% | 0.162 | 0.853 | 0.061 |
| FT04_ep10 | RM, cloud excl. FT04 masks | 46.0% | 15.8% | 0.170 | 0.852 | 0.065 |
| FT04_ep10 | RM, cloud excl. FTH1 masks | 32.3% | 13.9% | 0.127 | 0.731 | 0.060 |
| FT04_ep10 | RM, cloud excl. FTB1 masks | 47.5% | 16.4% | 0.176 | 0.857 | 0.067 |
| FTH1_ep8 | RM | 45.9% | 13.6% | 0.167 | 0.866 | 0.058 |
| FTH1_ep8 | RM, cloud excl. FT04 masks | 47.9% | 14.4% | 0.175 | 0.865 | 0.062 |
| FTH1_ep8 | RM, cloud excl. FTH1 masks | 33.8% | 13.3% | 0.134 | 0.744 | 0.059 |
| FTH1_ep8 | RM, cloud excl. FTB1 masks | 48.5% | 15.4% | 0.183 | 0.869 | 0.065 |
| FTH1_ep9 | RM | 45.2% | 13.6% | 0.163 | 0.864 | 0.057 |
| FTH1_ep9 | RM, cloud excl. FT04 masks | 47.1% | 14.4% | 0.171 | 0.864 | 0.061 |
| FTH1_ep9 | RM, cloud excl. FTH1 masks | 33.3% | 13.1% | 0.130 | 0.743 | 0.058 |
| FTH1_ep9 | RM, cloud excl. FTB1 masks | 47.7% | 15.3% | 0.178 | 0.868 | 0.064 |
| FTH1_ep10 | RM | 45.3% | 13.4% | 0.167 | 0.864 | 0.058 |
| FTH1_ep10 | RM, cloud excl. FT04 masks | 47.2% | 14.5% | 0.175 | 0.863 | 0.063 |
| FTH1_ep10 | RM, cloud excl. FTH1 masks | 32.8% | 13.5% | 0.133 | 0.742 | 0.058 |
| FTH1_ep10 | RM, cloud excl. FTB1 masks | 48.1% | 15.6% | 0.182 | 0.867 | 0.065 |
| FTB1_ep8 | RM | 44.4% | 14.2% | 0.154 | 0.861 | 0.056 |
| FTB1_ep8 | RM, cloud excl. FT04 masks | 45.9% | 14.9% | 0.162 | 0.859 | 0.060 |
| FTB1_ep8 | RM, cloud excl. FTH1 masks | 31.6% | 12.5% | 0.121 | 0.741 | 0.057 |
| FTB1_ep8 | RM, cloud excl. FTB1 masks | 46.8% | 15.2% | 0.168 | 0.864 | 0.063 |
| FTB1_ep9 | RM | 45.1% | 14.1% | 0.153 | 0.860 | 0.057 |
| FTB1_ep9 | RM, cloud excl. FT04 masks | 46.4% | 14.6% | 0.161 | 0.858 | 0.061 |
| FTB1_ep9 | RM, cloud excl. FTH1 masks | 31.9% | 12.8% | 0.121 | 0.740 | 0.058 |
| FTB1_ep9 | RM, cloud excl. FTB1 masks | 47.4% | 15.2% | 0.168 | 0.863 | 0.063 |
| FTB1_ep10 | RM | 44.8% | 14.0% | 0.152 | 0.859 | 0.056 |
| FTB1_ep10 | RM, cloud excl. FT04 masks | 46.3% | 14.4% | 0.161 | 0.858 | 0.060 |
| FTB1_ep10 | RM, cloud excl. FTH1 masks | 31.6% | 12.4% | 0.120 | 0.739 | 0.057 |
| FTB1_ep10 | RM, cloud excl. FTB1 masks | 47.2% | 15.0% | 0.167 | 0.863 | 0.063 |

### by tier (caught top 1%; pAUC where measured)

| model | scorer | easy | medium | subtle |
|---|---|---|---|---|
| FT04_ep10 | RM | 57.6% (0.220) | 39.3% (0.145) | 40.0% (0.091) |
| FT04_ep10 | RM, cloud excl. FT04 masks | 58.2% (0.231) | 39.3% (0.148) | 40.3% (0.096) |
| FT04_ep10 | RM, cloud excl. FTH1 masks | 43.4% (0.189) | 28.9% (0.105) | 24.7% (0.050) |
| FT04_ep10 | RM, cloud excl. FTB1 masks | 59.9% (0.238) | 41.3% (0.156) | 41.3% (0.100) |
| FTH1_ep8 | RM | 57.6% (0.220) | 40.3% (0.132) | 39.7% (0.113) |
| FTH1_ep8 | RM, cloud excl. FT04 masks | 59.5% (0.234) | 42.6% (0.140) | 41.3% (0.114) |
| FTH1_ep8 | RM, cloud excl. FTH1 masks | 43.8% (0.193) | 30.8% (0.100) | 26.7% (0.068) |
| FTH1_ep8 | RM, cloud excl. FTB1 masks | 60.2% (0.241) | 43.6% (0.146) | 41.7% (0.122) |
| FTH1_ep9 | RM | 56.6% (0.218) | 39.7% (0.129) | 39.3% (0.106) |
| FTH1_ep9 | RM, cloud excl. FT04 masks | 58.2% (0.231) | 41.6% (0.136) | 41.3% (0.107) |
| FTH1_ep9 | RM, cloud excl. FTH1 masks | 43.8% (0.192) | 29.8% (0.096) | 26.3% (0.062) |
| FTH1_ep9 | RM, cloud excl. FTB1 masks | 58.6% (0.238) | 43.0% (0.142) | 41.7% (0.114) |
| FTH1_ep10 | RM | 56.9% (0.222) | 39.7% (0.132) | 39.3% (0.111) |
| FTH1_ep10 | RM, cloud excl. FT04 masks | 58.6% (0.235) | 42.0% (0.139) | 41.0% (0.111) |
| FTH1_ep10 | RM, cloud excl. FTH1 masks | 43.1% (0.194) | 29.2% (0.097) | 26.0% (0.067) |
| FTH1_ep10 | RM, cloud excl. FTB1 masks | 59.2% (0.243) | 43.6% (0.145) | 41.3% (0.119) |
| FTB1_ep8 | RM | 57.9% (0.215) | 38.0% (0.128) | 37.3% (0.086) |
| FTB1_ep8 | RM, cloud excl. FT04 masks | 59.5% (0.227) | 40.3% (0.135) | 37.7% (0.090) |
| FTB1_ep8 | RM, cloud excl. FTH1 masks | 44.4% (0.188) | 27.5% (0.093) | 22.7% (0.046) |
| FTB1_ep8 | RM, cloud excl. FTB1 masks | 60.2% (0.233) | 41.6% (0.140) | 38.3% (0.095) |
| FTB1_ep9 | RM | 58.9% (0.214) | 38.7% (0.127) | 37.7% (0.086) |
| FTB1_ep9 | RM, cloud excl. FT04 masks | 59.9% (0.226) | 41.0% (0.133) | 38.3% (0.091) |
| FTB1_ep9 | RM, cloud excl. FTH1 masks | 44.7% (0.188) | 27.9% (0.092) | 23.0% (0.045) |
| FTB1_ep9 | RM, cloud excl. FTB1 masks | 60.5% (0.233) | 42.0% (0.139) | 39.7% (0.095) |
| FTB1_ep10 | RM | 58.2% (0.213) | 38.4% (0.126) | 37.7% (0.085) |
| FTB1_ep10 | RM, cloud excl. FT04 masks | 59.9% (0.225) | 41.0% (0.133) | 38.0% (0.090) |
| FTB1_ep10 | RM, cloud excl. FTH1 masks | 44.4% (0.187) | 27.9% (0.091) | 22.3% (0.045) |
| FTB1_ep10 | RM, cloud excl. FTB1 masks | 60.5% (0.232) | 42.0% (0.139) | 39.0% (0.094) |
