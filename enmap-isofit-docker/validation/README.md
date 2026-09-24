# Validation against DLR PACO L2A

Each run compares this pipeline's L2A with DLR's PACO L2A for the same EnMAP
datatakes, using `src/validate_vs_paco.py`. Files per run:

- `validation_<date>.csv` - one row per scene and class (land / water, split by
  PACO's `QL_QUALITY_CLASSES`)
- `validation_<date>_bands.csv` - median PACO, median ours, median bias and
  Pearson r for 13 sample bands
- `validation_<date>.txt` - the console output of the run

Columns: `mae_*` is mean |ours − PACO| in reflectance units over all bands,
460–2450 nm (the Allotrope common grid), below 700 nm and above 800 nm;
`bias_460` is the median signed difference at 460 nm; `sam_deg` is the median
per-pixel spectral angle.

## 2026-09-24

Image `enmap-isofit:1.1.3`, analytical line, full resolution (`--step 1`).
DT0000187759 was run with the summer preset's `h2o_min` at 0.2 (commit
`b0693fc`); the other four with 0.5, which does not bind for them - their
presolve H2O sits well above both floors.

| Scene | Site, date | Land MAE | Land MAE 460–2450 | Land SAM |
|---|---|---|---|---|
| DT0000189895 | Eastern India, 22.8°N, Apr 2026 | 0.0097 | 0.0092 | 3.9° |
| DT0000194619 | Rajasthan, 26.6°N, May 2026 | 0.0113 | 0.0115 | 4.5° |
| DT0000122280 | Ganges plain, 25.1°N, Apr 2025 | 0.0134 | 0.0125 | 5.1° |
| DT0000131004 | Hyderabad, 18.0°N, May 2025 | 0.0163 | 0.0163 | 4.8° |
| DT0000187759 | Himalaya, 31.1°N, 1.9–5.9 km, snow, Apr 2026 | 0.0345 | 0.0321 | 4.2° |

What the numbers say:

- **Plains land agrees to 0.009–0.016 mean absolute reflectance.** The blue
  excess at 460 nm varies by scene (+0.006 to +0.035), so it tracks each
  scene's haze rather than a fixed offset.
- **Water is +0.03–0.06 too bright in the NIR on every scene** - adjacency
  from bright surrounding land, which PACO corrects and ISOFIT does not.
- **Band 224 (2445 nm) runs high**, up to +0.05, at the low-signal sensor edge.
- **The Himalayan scene's MAE is pixel scatter, not bias**: median band biases
  stay within about ±0.02, but MAE below 700 nm is 0.065. Bright snow on steep
  terrain points at terrain-illumination handling, which differs between the
  two processors.
