# Family library: sources

411 reflectance spectra in 17 families (10 man-made, 7 natural), for
`app/spectral_match/family_match.py`. Built by `scripts/curate_family_library.py`; do not
edit by hand. Same layout as `data/splib07_slim`, so the existing per-sensor cache builder
reads it unchanged. Each entry's `chapter` is `<group>.<family>`.

| Family | Entries | Sources |
|---|---|---|
| manmade.building | 11 | USGS 5, ECOSTRESS 6 |
| manmade.glass | 1 | ECOSTRESS 1 |
| manmade.metal | 13 | USGS 6, ECOSTRESS 7 |
| manmade.oil | 40 | USGS 40 |
| manmade.paint | 8 | ECOSTRESS 8 |
| manmade.pavement | 15 | USGS 6, ECOSTRESS 9 |
| manmade.plastic | 39 | USGS 39 |
| manmade.roofing | 27 | USGS 16, ECOSTRESS 11 |
| manmade.textile | 14 | USGS 14 |
| manmade.wood_paper | 8 | USGS 6, ECOSTRESS 2 |
| natural.mineral | 40 | USGS 30, ECOSTRESS 10 |
| natural.npv | 40 | USGS 17, ECOSTRESS 23 |
| natural.rock | 40 | ECOSTRESS 40 |
| natural.snow_ice | 20 | USGS 16, ECOSTRESS 4 |
| natural.soil | 40 | USGS 12, ECOSTRESS 28 |
| natural.vegetation | 40 | USGS 30, ECOSTRESS 10 |
| natural.water | 15 | USGS 13, ECOSTRESS 2 |

`npv` is non-photosynthetic vegetation: dry grass, litter and bark. Families with more than
40 spectra were cut to 40 spectrally spread entries.

## Provenance

- **USGS Spectral Library Version 7** (splib07b, the release convolved to the ASD
  spectrometer; every instrument, including Beckman). Public domain.
  Kokaly, R.F., Clark, R.N., Swayze, G.A., et al., 2017, *USGS Spectral Library Version 7*:
  U.S. Geological Survey Data Series 1035, 61 p., https://doi.org/10.3133/ds1035.
- **ECOSTRESS Spectral Library version 1.0** (JPL / JHU spectra). Freely available; cite as:
  Meerdink, S.K., Hook, S.J., Roberts, D.A., Abbott, E.A., 2019, *The ECOSTRESS spectral
  library version 1.0*: Remote Sensing of Environment 230, 111196,
  https://doi.org/10.1016/j.rse.2019.05.015.
  and Baldridge, A.M., Hook, S.J., Grove, C.I., Rivera, G., 2009, *The ASTER spectral library
  version 2.0*: Remote Sensing of Environment 113, 711-715.

## Processing

- **USGS:** values as distributed on the 2,151-channel ASD axis (350-2500 nm); the USGS
  missing-value sentinel is kept.
- **ECOSTRESS:**
  - micrometres converted to nm, percent to 0-1;
  - linearly interpolated onto the same ASD axis inside each spectrum's measured range;
    missing (sentinel) outside it.
  - Its man-made spectra step about 20 nm in the SWIR, so fine absorption features there
    (e.g. hydrocarbon bands at 1.73 / 2.31 µm) are blunted; USGS resolves them.
- **Curation** follows the rules of the man-made library experiments:
  - pigments, ore, lab chemicals, lab organics, calibration targets and marble are excluded;
  - an entry must cover at least 95 % of the 165-band common grid;
  - each family is capped at 40.
- **`asd_subtype`** is `ASDFR` for every entry. The cache builder needs an ASD FWHM to exist,
  but resamples with the sensor's FWHM.
