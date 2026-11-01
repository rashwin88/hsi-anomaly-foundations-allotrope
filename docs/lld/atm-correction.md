# Atmospheric-correction post-processing (`atm_post`) and harmonisation

**Files:** `isofit-atm-docker/src/{atm_post,rfl_qc,adjacency,terrain,sixs_parts}.py`,
`isofit-atm-docker/harmonise/`. The same code also sits in `enmap-isofit-docker/`, the predecessor, which is no
longer developed.
**Called by:** `isofit-atm-docker/entrypoint.sh` stage 2b (`atm_post`, after `isofit apply_oe`) and stage 2c
(`harmonise`, with the rule's model), before the writers (stage 3).
**Whole stack:** [`isofit-atm-docker.md`](isofit-atm-docker.md) covers readers, speed levers, writers, report,
image and tests.
**Evidence:** `research/experiments/atm-correction/EXPERIMENT_LOG.md` (IDs AC26-xx below)

## Purpose

The anomaly models were trained on vendor L2 reflectance:
- EnMAP: DLR PACO L2A;
- PRISMA: ASI L2D;
- AVIRIS-NG: JPL v2 `corr`.

At deployment we run ISOFIT on L1. This stage moves ISOFIT's output toward each vendor's product.
**The target is the vendor, not ground truth.** A step is here only because a vendor does the
equivalent, and it is switched off for the sensors whose vendor does not.

## Interfaces

```python
# atm_post.py
SENSOR_RULES: dict[str, SensorRules]          # "enmap" | "prisma" | "avirisng" | "tanager" | "ghost" | "generic"
rule_name(sensor_id) -> str                   # readers.sensors id -> rule ("aviris_ng" -> "avirisng", unknown -> "generic")
run(sensor, isofit_dir: Path, obs: str, px_m=30.0, aot_prior: Path | None = None) -> dict  # report, also atm_post.json
find_outputs(isofit_dir) -> (rfl, atm_interp, lut_full)
# python -m atm_post --sensor S --isofit-dir D --obs O [--px-m 30] [--report P] [--aot-prior <work>/aot_prior.json]

# rfl_qc.py         in place, return counts
mask_windows(rfl) -> int                      # bands set to nodata
clip_outliers(rfl) -> (low, high, mask_path) # below -0.2: nodata; above 2.5: kept, flagged in <rfl>_outlier
snow_mask(rfl) -> (mask_path, fraction)       # NDSI snow, read-only, <rfl>_snow
# adjacency.py / terrain.py   in place, return small dicts for the report
adjacency.correct(rfl, lut_dir, atm_interp, range_km=1.0, px_m=30.0) -> dict
terrain.correct(rfl, obs, lut_dir, atm_interp, px_m=30.0) -> dict
# sixs_parts.py
scene_state(atm_interp) -> (aot550, h2o, elev_km)
components(lut_dir, aot, h2o, elev_km, wl_nm) -> {"down_dir", "down_dif", "up_dir", "up_dif"}
```

`writers.py` copies the report whole into `<scene>_L2_SR_provenance.json` and turns the two masks into
`_outlier.tif` / `_snow.tif` sidecars. Only with `--paco-layout` (EnMAP) does
`postprocess_to_paco.py --atm-post atm_post.json --emulator <file>` also turn it into METADATA.XML tags:
- `terrain_correction` and `adjacency_correction`: Yes / No / Unknown;
- `outlier_pixels_nodata`;
- `caltab_atm_version`.

## Data flow

1. **Input:** the finished `apply_oe` directory.
   - `output/<id>_rfl` is float32 ENVI, BIL, nodata −9999. Sparse `*_subs_*` files are ignored.
   - `output/<id>_atm_interp` has bands named `AOT550`, `H2OSTR` and, with
     `--pressure_elevation`, `surface_elevation_km`.
   - `lut_full/AOT550-*` holds the 6S text outputs, one per LUT grid point.
2. **Obs cube:** the obs cube the preprocessor wrote. Band 4 is solar zenith, 6 slope, 8 cos i.
3. **Steps, each editing `_rfl` in place through a `spectral` memmap:**
   1. window mask;
   2. outlier clip (negative → nodata; above 2.5 kept and flagged: flares and fires are what the anomaly
      stage looks for);
   3. adjacency, if the rules allow;
   4. terrain, if the rules allow;
   5. NDSI snow mask (read-only, every sensor).
4. **Output:** `isofit/atm_post.json`, read by harmonisation (covariates) and the writers (provenance).

### Where the transmittances come from

With the sRTMnet `.h5` emulator, ISOFIT 4.1.5 runs single-part transmittance:
- `apply_oe` sets `multipart_transmittance` False for `.h5`;
- `6S.lut.zarr` holds totals only.

Adjacency needs the up direct/diffuse split, and terrain needs the down split. The 6S runs that
built the LUT print a "Multipart transmittances" table, so `sixs_parts` reads those. It takes the
nearest point in H2O and elevation and interpolates linearly in AOT between the bracketing points,
at the scene medians.

## Invariants

- **Order is mask → clip → adjacency → terrain.**
  - The clip must precede adjacency. ISOFIT occasionally returns −8971 or +514 (Himalayan snow and
    shadow); one such pixel inside a 1 km kernel shifts its neighbours' background.
  - The prototype ran adjacency before the clip, which is why module and prototype differ by up to
    0.45 on those pixels.
- **Steps only write nodata or corrected values; they never fill nodata.** The band filter
  downstream interpolates nodata bands, which is exactly what the window mask relies on: the
  1960/1970 nm grid bands go from ~0 to 0.198, against PACO's 0.173.
- **The outlier test covers only the 165-band grid range outside absorption windows, with limits
  [−0.2, 2.5].**
  - Tightening it to [−0.05, 2] or testing the edge and absorption bands removed 35–80% of
    AVIRIS-NG pixels and up to 17% of PRISMA's (AC26-F notes).
  - It is a safety net, not a correction. Expected rates: EnMAP ~0.09%, PRISMA ≤ 0.05%,
    AVIRIS-NG ~0.
- **Adjacency validity is "valid in every band that is valid somewhere".** Requiring all bands would
  void the image, because the window-masked bands are nodata everywhere.
- **The pixel size must be right.** `--px-m` sets the kernel and the 0.5 km terrain box. It is 30
  for EnMAP and PRISMA; AVIRIS-NG would need its 4–8 m.
- **The rules key on vendor behaviour.** Do not turn adjacency or terrain on for `prisma` or
  `avirisng` to "improve" physics: both over-correct against ASI and JPL, which do neither
  (AC26-A).

## Failure modes

- **No `*_rfl` or `*_atm_interp`:** `FileNotFoundError`, and the entrypoint stops under
  `set -e`. That means apply_oe did not finish.
- **No `lut_full/AOT550-*`:** `FileNotFoundError` from `components`. This only happens if the 6S
  intermediates were cleaned away before stage 2b.
- **A 6S output without a multipart table:** `ValueError` naming the file.
- **No elevation band** (run without `--pressure_elevation`): the elevation is taken as 0 km, so
  EnMAP would get adjacency. The entrypoint always passes the flag.
- **Run without the stage** (an old work dir): `postprocess_to_paco` writes "Unknown" in the three
  tags rather than guessing.
- **Module names are generic** (`adjacency`, `terrain`). `python -m atm_post` puts the current
  directory first on `sys.path`, so a stray `adjacency.py` there shadows the module:
  `AttributeError: module 'adjacency' has no attribute 'correct'`.
  - Seen 2026-10-24 when plan C ran from a Colab directory holding the prototype scripts.
  - In the image, WORKDIR is `/work` and holds only data. Do not run the stage from a directory that
    contains research scripts.

## Decisions

| Choice | Evidence | Rejected alternative |
|---|---|---|
| Post-hoc adjacency, ρ + q(ρ − ρ_bg), R = 1 km, EnMAP below 2 km | Ganges: water 0.0145 → 0.0073, land 0.0113 → 0.0099 (AC26-02, A) | ISOFIT `--use_background_rfl` needs a `.6c`/MODTRAN engine; with `.6c` it was worse and retrieved a 0.62 km surface on a 40 m plain (AC26-B, C) |
| No adjacency at 2 km and above | Over-corrected the 3.2 km Himalaya scene at AOT 0.38 (AC26-02) | |
| Post-hoc PACO terrain (ATBD §5.8.2) with V = cos²(slope/2) | Same MAE as without, but the SWIR improves (1600 nm r 0.947 → 0.980) (AC26-T) | Horizon sky view: 0.0285 → 0.0297 (AC26-M2). PACO cast-shadow rule: no change, 0.05% of pixels (M6a). ISOFIT native terrain via `.6c`: 0.0328, surface elevation 1.3 km high (M5) |
| Interpolate AOT between 6S points | Adjacency and terrain see the same atmosphere. Effect against the prototype's nearest point: ≤ 0.02 per pixel, MAE +0.0001 | |
| Noise: ISOFIT default for EnMAP/PRISMA, `avirisng_noise.txt` only for AVIRIS-NG | `avirisc_noise.txt` is applied by channel index: wrong weights on EnMAP, crash on PRISMA | |
| AVIRIS-NG AOT pinned 0.10–0.14 (`configs/avirisng_lut.json`) | JPL's fixed 50 km visibility; Jamnagar 0.0123 → 0.0087 (AC26-P) | |
| PRISMA AOT pinned to the climatology month value ± 0.15, floor 0.05 (`aot_prior.py`, before apply_oe) | 15 scenes after H6: 0.0128 free → 0.0118 (AC26-A2, A2b) | A constant ASI-like prior: +0.0006 only (P2). Per-scene ASI AOT helps more (P1), but L2C does not exist at deployment |

Gaps against the vendors that are known and deliberately **not** handled here, but in sensor-level
harmonisation:
- EnMAP snow (AC26-M7);
- the EnMAP blue offset;
- PRISMA beyond 2350 nm (AC26-S1);
- PRISMA visible/NIR vs ASI's aerosol (AC26-P1).

Physics steps here, vendor-matching statistics there.

## Harmonisation (`isofit-atm-docker/harmonise/`)

A standalone step after `atm_post`. `entrypoint.sh` stage 2c calls it with the sensor rule's
`harmonise_model`:
- EnMAP: `enmap_blue600.npz`;
- PRISMA: `prisma_native.npz`;
- AVIRIS-NG, Tanager, GHOSt and generic: none (identity).

`--no-harmonise` skips it. It applies a model fitted on training pairs (ours vs vendor, native bands) to one run. The models are fitted and
frozen in the research folder, and the decisions are in its `LEARNINGS.md` §4.

```python
# python -m harmonise <model.npz> <atm_post.json> <obs>  ->  <rfl>_harm (ENVI BIL) + <rfl>_harm.json
model.load(path) -> (scene, parts)                     # .npz of plain arrays, allow_pickle=False
apply.harmonise(scene, parts, x, wl, z, labels) -> x'  # x (pixels, our bands), NaN = nodata
classes.labels(x, wl) -> "snow" | "water" | "land"    # per pixel, from our reflectance
bands.band_groups(wl) -> core | edge_lo | edge_hi | weak | strong   # per band
```

**Model file.**
- Scene arrays: `sensor`, `zmu`/`zsd` (standardisation), `zlo`/`zhi` (training range of the
  standardised z) and `gate`.
- One part per band group. Each part's kind is one of identity, nodata, const, own
  (`y = (a + A·z)·x + c + C·z`) or ridge (`y = [x_in, z, 1] @ coef`).
- A `core_<class>` part applies to that class; every other part applies to all pixels.

**Covariates** z = [AOT550, H2O, SZA/90, elevation km], taken from `atm_post.json` and obs band 4.
PRISMA adds `aot_prior.aod550_clim`. That is the grid month value the AOT window was pinned
around, not the AOD of that year's month: the latter is not known at deployment (AC26-CLIM).

### Invariants

- **Labels come from our reflectance** (AC26-NWL option a): NDSI snow with `rfl_qc.snow_mask`'s
  thresholds, then NDWI(560, 860) > −0.1 with ρ860 < 0.15 for water.
  - Vendor masks do not exist at deployment, and using ours cost nothing in LOSO.
  - Only land has a core model, so snow and water keep our values. The land model on water
    roughly doubles PRISMA's water error.
- **A model never extrapolates in z.**
  - z is clamped to the training range.
  - The H6 gate skips own and ridge parts in a scene more than `gate` sd outside that range.
  - const and nodata parts use neither x nor z, so they still apply in a gated scene.
- **Ridge inputs are read from the original x**, so the order of the parts does not matter.
- **A pixel with no valid band stays nodata.** const may fill a band masked on a live pixel. That
  is intended: PRISMA's strong-window bands get the vendor's typical value.
- **Bands missing from the cube:** a model band with no cube band within 3 nm is skipped as a
  target. A ridge part that lacks an input band (a scene whose SWIR is cut short) is skipped whole
  (`apply.lacks_inputs`): its bands stay ours, and `<rfl>_harm.json` lists it under `skipped_parts`.
  Nothing is skipped silently: calling `fitted()` directly still raises `ValueError`.

### Failure modes

- **Model sensor ≠ run sensor:** the tool exits 1 and names both.
- **A PRISMA scene whose SWIR ends early** (20260202051921 ends at 2343 nm): the window ridge part is
  skipped and named in `skipped_parts`. The core and the other parts still apply (176 of 186 bands).
- **A 5-covariate (PRISMA) model with no `aot_prior` record in `atm_post.json`:** `KeyError` or
  `TypeError`. This covers runs made before the record existed, including the 2026-10 holdout runs;
  add the record from that run's `aotpin.txt` (hi − 0.15).
- **A research script `harmonise.py` in the current directory** (Colab `/content` holds one)
  shadows the package. `python -m harmonise` then runs the research script. Run from the module
  directory.

### Decisions

| Choice | Evidence | Rejected alternative |
|---|---|---|
| AVIRIS-NG identity | Holdout land MAE 0.0051 with no model | |
| PRISMA H6: per band with z, 1 sd gate, land only | Holdout 165 grid: vs L2C 0.0249 → 0.0095, vs L2D 0.0279 → 0.0121 (AC26-HO165) | H5 spectral ridge: better on 22 training scenes (0.0116 vs 0.0123, AC26-S4), worse on the 15 the lock was made on |
| EnMAP HB: blue-only own+z, core 464–599 nm + edge_lo (`enmap_blue600.npz`), cut 600 nm chosen on training LOSO; **holdout-informed, declared** (model family chosen after AC26-HON) | Training LOSO land 0.0117 → 0.0104; holdout land 0.0098 → 0.0086 with all 5 scenes better, bias at 460 nm +0.023 → +0.001 (AC26-HB). The blue offset matters for SAM-based spectral matching | H5 spectral ridge over the whole core: fixed the blue on every holdout scene but worsened the core on 2 of 5 (0.0098 → 0.0100, AC26-HON) |
| Tanager, GHOSt, generic: identity | No vendor-paired model fitted. Tanager vs Planet basic SR without one: land MAE 0.0061 / 0.0144 / 0.0079 on three scenes (AC26-ACC) | |
| No EnMAP water model | Borderline (AC26-NW): NDWI reaches P/R ≥ 0.9 against DLR's flag only pooled, with the rule tuned on the same data; the per-scene median precision is 0.88, and one scene fails | |
| Outside the core, per sensor (AC26-NBW): PRISMA strong const, weak ridge, edge_lo own+z, edge_hi ridge ≤ 2470 / const ≥ 2477 nm; EnMAP strong nodata (as PACO) except 1449 nm own+z, weak ridge, edge_lo own+z | Per-group LOSO comparison of interp / ridge / const / own+z | One rule for all sensors |
