# Thermal Benchmark Dataset Summary

## Overview

4 Landsat 8/9 Surface Temperature (Band 10) scenes with pixel-level binary anomaly ground truth labels. All scenes cover regions in South/Central Asia.

| Property | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
|---|---|---|---|---|
| **Scene ID** | LC09_L2SP_141045_20250604 | LC08_L2SP_138045_20250215 | LC09_L2SP_150044_20251009 | LC09_L2SP_147049_20251121 |
| **Satellite** | Landsat 9 | Landsat 8 | Landsat 9 | Landsat 9 |
| **Acquisition Date** | 2025-06-04 | 2025-02-15 | 2025-10-09 | 2025-11-21 |
| **CRS** | EPSG:32644 | EPSG:32645 | EPSG:32642 | EPSG:32643 |
| **Image Dimensions** | 7681 x 7521 | 7721 x 7571 | 7721 x 7581 | 7781 x 7641 |
| **Total Pixels** | 57,788,001 | 58,450,891 | 58,546,201 | 59,443,821 |
| **Valid Pixels** | 40,711,448 (70.5%) | 40,533,402 (69.3%) | 40,670,474 (69.5%) | 40,644,289 (68.4%) |

## Temperature Statistics (Celsius)

| Statistic | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
|---|---|---|---|---|
| **Mean** | 35.57 | 26.68 | 33.46 | 29.84 |
| **Min** | -21.19 | 21.27 | -2.62 | 16.21 |
| **Max** | 80.35 | 67.40 | 51.59 | 51.13 |
| **Range** | 101.54 | 46.13 | 54.21 | 34.92 |

## Anomaly Statistics

| Statistic | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
|---|---|---|---|---|
| **Anomaly Pixels** | 6,636 | 5,132 | 205 | 1,051 |
| **% of Valid Pixels** | 0.0163% | 0.0127% | 0.0005% | 0.0026% |
| **Anomaly Label Values** | {0, 1} | {0, 1} | {0, 1} | {0, 1} |

## Bounding Boxes (Geographic)

| Dataset | West | South | East | North |
|---|---|---|---|---|
| **1** | 684,885 m E | 2,283,585 m N | 910,515 m E | 2,514,015 m N |
| **2** | 540,285 m E | 2,280,885 m N | 767,415 m E | 2,512,515 m N |
| **3** | 518,385 m E | 2,440,185 m N | 745,815 m E | 2,671,815 m N |
| **4** | 201,885 m E | 1,641,585 m N | 431,115 m E | 1,875,015 m N |

## Key Observations

- **Anomaly sparsity**: Anomalies are extremely rare (0.0005% to 0.0163% of valid pixels), making this a highly imbalanced detection problem.
- **Dataset 3** has the fewest anomalies (205 pixels) — an order of magnitude fewer than the others.
- **Dataset 1** has the widest temperature range (-21 to 80 °C) and the most anomaly pixels (6,636).
- **Season variation**: Acquisition dates span February through November, providing seasonal diversity (winter, summer, autumn).
- **Sensor mix**: Dataset 2 is Landsat 8; the others are Landsat 9.

## File Structure

```
benchmarking/thermal/
├── 1/
│   ├── LC09_L2SP_141045_20250604_20250605_02_T1_ST_B10.TIF
│   └── gt.tif
├── 2/
│   ├── LC08_L2SP_138045_20250215_20250226_02_T1_ST_B10.TIF
│   └── gt.tif
├── 3/
│   ├── LC09_L2SP_150044_20251009_20251010_02_T1_ST_B10.TIF
│   └── gt.tif
├── 4/
│   ├── LC09_L2SP_147049_20251121_20251122_02_T1_ST_B10.TIF
│   └── gt.tif
├── thermal_benchmark_exploration.ipynb
└── BENCHMARK_SUMMARY.md
```

## Data Format

- **Thermal TIF**: Landsat Collection 2 Level-2 Surface Temperature Band 10 (uint16 DN values). Convert to Celsius via: `T(°C) = DN × 0.00341802 + 149.0 − 273.15`
- **Ground Truth (gt.tif)**: Binary float32 raster. `1.0` = anomaly, `0.0` = normal.
- **NoData**: DN value `0` in the thermal TIF indicates fill/nodata pixels.
