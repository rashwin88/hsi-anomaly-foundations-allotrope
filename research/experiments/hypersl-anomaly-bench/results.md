| method | caught top 1% | caught top 0.1% | pAUC (FPR<=0.01) | ROC-AUC | TPR at 0.001 |
|---|---|---|---|---|---|
| FTH3_ep5 RM | 422 (46.4%) | 13.9% | 0.1749 | 0.866 | 0.058 |
| FTH3_ep5 RM, cloud excl. | 449 (49.4%) | 16.0% | 0.1892 | 0.869 | 0.065 |
| HyperSL L2 | 130 (14.3%) | 4.0% | 0.0175 | 0.624 | 0.007 |
| HyperSL loss | 63 (6.9%) | 0.9% | 0.0039 | 0.501 | 0.000 |
| HyperSL RM | 350 (38.5%) | 5.4% | 0.0832 | 0.757 | 0.020 |
| PCA-RX | 361 (39.7%) | 8.1% | 0.1281 | 0.839 | 0.029 |
| MNF-RX | 249 (27.4%) | 3.4% | 0.0750 | 0.749 | 0.019 |
| HyperSL L2, cloud excl. | 137 (15.1%) | 4.3% | 0.0182 | 0.628 | 0.007 |
| HyperSL loss, cloud excl. | 63 (6.9%) | 0.9% | 0.0039 | 0.503 | 0.000 |
| HyperSL RM, cloud excl. | 369 (40.6%) | 6.7% | 0.0947 | 0.761 | 0.024 |
| PCA-RX, cloud excl. | 377 (41.5%) | 8.5% | 0.1390 | 0.842 | 0.031 |
| MNF-RX, cloud excl. | 280 (30.8%) | 5.2% | 0.0966 | 0.753 | 0.024 |

| difference (B - A) | pAUC diff [95% CI] | caught top 1% diff [95% CI] |
|---|---|---|
| HyperSL L2 - FTH3_ep5 RM | -0.1574 [-0.1790, -0.1351] | -32.1 [-35.4, -28.9] pts |
| HyperSL loss - FTH3_ep5 RM | -0.1710 [-0.1936, -0.1479] | -39.5 [-42.9, -36.2] pts |
| HyperSL RM - FTH3_ep5 RM | -0.0917 [-0.1060, -0.0776] | -7.9 [-10.8, -5.2] pts |
| PCA-RX - FTH3_ep5 RM | -0.0467 [-0.0604, -0.0327] | -6.7 [-8.8, -4.5] pts |
| MNF-RX - FTH3_ep5 RM | -0.0999 [-0.1170, -0.0826] | -19.0 [-22.0, -16.3] pts |
| HyperSL L2, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.1709 [-0.1945, -0.1476] | -34.3 [-37.6, -31.0] pts |
| HyperSL loss, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.1852 [-0.2092, -0.1606] | -42.5 [-45.8, -39.2] pts |
| HyperSL RM, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.0944 [-0.1094, -0.0800] | -8.8 [-11.7, -6.2] pts |
| PCA-RX, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.0502 [-0.0638, -0.0364] | -7.9 [-10.2, -5.6] pts |
| MNF-RX, cloud excl. - FTH3_ep5 RM, cloud excl. | -0.0926 [-0.1097, -0.0753] | -18.6 [-21.5, -15.8] pts |
| HyperSL RM - PCA-RX | -0.0450 [-0.0589, -0.0318] | -1.2 [-4.0, +1.5] pts |
| HyperSL RM, cloud excl. - PCA-RX, cloud excl. | -0.0443 [-0.0577, -0.0320] | -0.9 [-3.5, +1.9] pts |
