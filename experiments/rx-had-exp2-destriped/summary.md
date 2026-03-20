# rx-had-exp2: Global RX on Destriped PRISMA Scenes

## Pipeline

HE5 → PrismaDatasetBuilder → CombinedDestriper (FFT + moment-matching with σ guard) → GlobalRXDetector

## Results

| Scene | Bands | Valid Pixels | Score Range | Median | Destripe (s) | RX (s) |
|---|---|---|---|---|---|---|
| PRS_L2D_STD_20260104051849_2026010405185 | 218 | 914,778 | 50.5433–163771.1636 | 178.6593 | 33.97 | 6.66 |
| PRS_L2D_STD_20210516050459_2021051605050 | 221 | 688,056 | 50.1691–58040.6299 | 170.1805 | 33.59 | 5.32 |
| PRS_L2D_STD_20241205050514_2024120505051 | 215 | 902,206 | 46.7547–51442.7162 | 179.0379 | 26.82 | 6.69 |
| PRS_L2D_STD_20231229050902_2023122905090 | 214 | 853,186 | 51.5053–176424.0634 | 180.2217 | 41.44 | 5.19 |
| PRS_L2D_STD_20201214060713_2020121406071 | 145 | 575,862 | 19.2289–30619.6025 | 111.1482 | 34.15 | 2.38 |