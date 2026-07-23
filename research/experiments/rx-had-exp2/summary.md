# rx-had-exp2: Global RX on Destriped PRISMA Scenes

## Pipeline

HE5 → PrismaDatasetBuilder → CombinedDestriper (FFT + moment-matching with σ guard) → GlobalRXDetector

## Results

| Scene | Bands | Valid Pixels | Score Range | Median | Destripe (s) | RX (s) |
|---|---|---|---|---|---|---|
| PRS_L2D_STD_20260104051849_2026010405185 | 177 | 992,515 | 41.3195–52794.0022 | 144.391 | 32.86 | 4.76 |
| PRS_L2D_STD_20210516050459_2021051605050 | 174 | 925,543 | 37.5332–51378.6774 | 129.752 | 47.11 | 6.57 |
| PRS_L2D_STD_20241205050514_2024120505051 | 176 | 1,016,572 | 38.4508–47028.901 | 144.3889 | 36.27 | 11.78 |
| PRS_L2D_STD_20231229050902_2023122905090 | 176 | 958,606 | 40.4552–168219.1934 | 147.0281 | 52.52 | 7.23 |
| PRS_L2D_STD_20201214060713_2020121406071 | 124 | 641,181 | 17.6459–98102.69 | 93.5207 | 48.79 | 2.15 |