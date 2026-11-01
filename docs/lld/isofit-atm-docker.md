# isofit-atm-docker: the L1 → L2 atmospheric-correction stack

**Folder:** `isofit-atm-docker/`, a stack of its own. It shares nothing with `docker/` (the product) and replaces
`enmap-isofit-docker/`, which is left as it was.
**Post-processing and harmonisation:** [`atm-correction.md`](atm-correction.md).
**Evidence:** `research/experiments/atm-correction/EXPERIMENT_LOG.md` (AC26-*). The 12-item plan is
`SPEED_MODULE_PLAN.md` in the same folder.

## Purpose

The stack turns any delivered L1 cube from EnMAP, PRISMA, AVIRIS-NG, Tanager, GHOSt or an unknown sensor into
the organisers' L2 product: surface reflectance as GeoTIFF and BSQ, int16 ×10000, water bands dropped. It runs
ISOFIT 4.1.5 with the sRTMnet emulator on an air-gapped GPU VM, about 4× faster than stock ISOFIT, with the same
outputs. It never refuses an input for missing metadata: it fills what it can, says what it assumed, and grades
the result.

## Interfaces

```bash
entrypoint.sh <input file|folder> <output dir> [--sensor S] [--input-level radiance|toa|sr] [--radiance-scale X]
              [--wavelength-file F] [--season S] [--no-harmonise] [--formats tif,bsq] [--paco-layout]
              [--ref PATH] [--ref-wavelength-file F] [--no-report] [--n-cores N] [--line analytical|empirical]
              [--segmentation-size N] [--no-presolve]   # defaults: the sensor rule (AVIRIS-NG 400 + empirical line)
              [--keep-work] [--stop-after-lut]
# env: DEM_VRT, N_CORES, CONFIG_ROOT, SIXS_FAST=0, SIXS_CACHE=0, SIXS_ARCH=v4, ATM_FAST_AL=0
scripts/run_scene.sh <input> <out> [entrypoint options]   # env: DEM_DIR, DEM_VRT_NAME, REF, GPUS, SHM_SIZE, IMG
```

```python
readers.prep.main([...]) -> 0                 # any delivery -> <fid>_rdn/_loc/_obs (or _rfl), scene.json
readers.level.classify(sample, wl, integer) -> {"level": radiance|toa_reflectance|surface_reflectance|dn|unknown, ...}
readers.scene.Scene.grade() -> "A".."D"; .checklist() -> [{item, value, source, note}]
writers.write(rfl, scene_json, out_dir, report, formats, compress) -> provenance dict
report.build(out_dir, work_dir, ref=None, ref_wl=None) -> Path   # <name>_report.html
fast_al.analytical_line(*a, **k)              # ISOFIT's analytical line with a batched solve (apply_oe uses it)
```

## Data flow

| Stage | Module | Reads | Writes (work dir unless noted) |
|---|---|---|---|
| 1 read | `readers.prep` | the delivery | `<fid>_rdn` (or `_rfl` when passed through), `_loc`, `_obs`, `wavelengths.txt`, `scene.json` |
| 1b prior | `isofit surface_model` | `wavelengths.txt` | `scene_surface.mat`, on the scene's exact band centres |
| LUT | `aot_prior` (PRISMA only) | `configs/aod_clim_1deg.npz` | `lut.json` pinned to the climatology ±0.15, `aot_prior.json` |
| 2 ISOFIT | `apply_oe` → 6S wrapper → `fast_al` | rdn/loc/obs, LUT config, noise file | `isofit/output/<fid>_rfl`, `_atm_interp`; `isofit/lut_full` |
| 2b post | `atm_post` | ISOFIT output, obs | reflectance in place, `_outlier`, `_snow`, `atm_post.json` |
| 2c harmonise | `harmonise` | model `.npz`, `atm_post.json`, obs | `<rfl>_harm` + `.json` |
| 3 write | `writers` | final reflectance, `scene.json`, report | **output dir:** `<fid>_L2_SR.tif/.bsq/.hdr`, `_outlier.tif`, `_snow.tif`, `_latlon.tif` (swath), `_provenance.json` |
| 3c report | `report` | product, work dir, optional `--ref` | **output dir:** `<fid>_L2_SR_report.html` |

`stage_times.txt` in the work dir holds an epoch stamp per stage boundary for the report. The work dir is deleted
at the end unless `--keep-work` is given; the report runs before that.

## The pieces

### Readers (`src/readers/`)
- **Native products keep their validated preprocessors:**
  - an EnMAP L1C folder goes through `enmap_l1c_preprocess`;
  - a PRISMA L1 `.he5` goes through `prisma_l1_preprocess`;
  - an AVIRIS-NG L1 folder is symlinked as delivered.
- **Everything else** (ENVI, GeoTIFF, Tanager HDF5, GHOSt, unknown sensors) goes through `cube` / `tanager`. The
  fallback ladder fills each fact from metadata, else derives it, else assumes it, or marks it missing.
- **Grade:**
  - A: every fact from metadata.
  - B: some facts derived.
  - C: view, altitude, noise or DEM assumed.
  - D: time, location or radiance scale assumed.
  - Grades never stop a run.
- **`level.classify` is asymmetric on purpose.** A cube passes through as surface reflectance only if at least 3
  atmosphere tests ran and every one votes SR. Any TOA vote, too few tests, or a grey-zone value means "correct
  it". Calling SR "TOA" costs a needless correction; the reverse would ship an uncorrected product.
- **Interleave is checked against the data.** `check_interleave` scores coherence × smoothness. A wrong ENVI
  header reads back as plausible garbage, not as an error, so the header alone is not trusted.

### Speed (AC26-6SX, 6SC, W3, FAL, EOE)
- **6S:**
  - the aerosol-only call is removed;
  - the scattering cache is shared across water-vapour points;
  - builds at `-O3` for x86-64-v3 and v4.
  - The wrapper takes ISOFIT's 6S executable name and picks the binary per CPU (v3 by default; it was faster on
    an AVX-512 Xeon).
  - Full-LUT 6S stage: 577 s → 115 s.
- **`isofit_patches.py`:**
  - shard patch: each H2O column stays in one LUT shard, so the cache is reused;
  - `fast_al` hook in `apply_oe`.
- **`fast_al`:** a batched fp64 Cholesky solve in torch, on the GPU with a CPU fallback. EnMAP 1948 → 602 s,
  outputs identical.
- **Surface prior on the scene's exact band centres** (EnMAP: 53 → 30 min). A prior on shifted centres made
  `forward.upsample` interpolate on every evaluation.

### Output (writers, report)
- **Bands:** those outside ISOFIT's inversion windows (1360–1410, 1800–1970, > 2500 nm) and empty bands are
  dropped and listed.
- **Values:** int16 = round(ρ × 10000), nodata −9999.
- **Georeference:**
  - kept from the source when it has a map (AVIRIS-NG's rotated grid included);
  - lat/lon GCPs every 50 px plus `_latlon.tif` for swaths (PRISMA L1, Tanager basic): no resampling;
  - "none" when the delivery has no location.
- **The report is optional:** a failure only logs a warning. `--ref` pairs pixels by location (nearest reference
  pixel within one reference pixel spacing) and reads only the reference rows that hold pairs.

### Image (`Dockerfile`)
- **builder:** Ubuntu 24.04 python3.12 venv from `requirements.txt`, compilers, 6S builds with
  `sixs_check.py` (`-O` equals stock exactly, v3/v4 within 2e-4), sRTMnet and ISOFIT data, patches.
- **runtime:** fresh Ubuntu 24.04 + python3.12 + libgfortran5/libgomp1. The finished trees are copied in, and the
  runtime checks run again: `ldd` of every 6S build, the selftest, `fast_al` wired in, imports.
- **test:** pytest on the runtime image with `--cov-fail-under=30`. `build_image.sh` builds it before tagging.
- **Runs as the caller's uid.** `/root` and the data are world-readable, never writable; caches go to `/tmp`;
  `/work` is mode 1777.

## Invariants

- **Every fast path has an off switch and falls back to stock on error.**
  - `SIXS_FAST=0` / `SIXS_CACHE=0`.
  - `ATM_FAST_AL=0`. A `fast_al` actor also falls back by itself when a chunk's Seps is not diagonal.
  - `apply_oe` takes ISOFIT's own analytical line if `fast_al` fails to import. The image build asserts that it
    is wired in, so that cannot happen silently.
- **The 6S builds must reproduce stock**, or the build fails (`sixs_check.py`). `--selftest` re-checks them on the
  target CPU.
- **TOA must never pass through as SR** (`level.classify`; the tests cover it).
- **Nothing is silently dropped or assumed.**
  - Dropped bands, assumed facts and overrides go into `scene.json` and the provenance JSON.
  - Each stop names its override: `--radiance-scale`, `--input-level`, `--wavelength-file`.
- **NA-mode file ids never end in `_rdn`.** ISOFIT builds names as `fid + "_rdn"` with `_rdn` replaced, so the
  entrypoint links the radiance under the bare fid.
- **No location means a pseudo 30 m lat/lon grid, not a constant.** With a constant, ISOFIT's spatial
  interpolation writes AOT = H2O = 0 everywhere.
- **No DEM:** ISOFIT sets its pressure-elevation grid from the loc cube's min−2 .. max+2 km. Pixel (0, 0)'s
  placeholder becomes 3 km, so the grid spans 0–5 km.

## Failure modes

- **Raw counts with no gain:** `prep` stops: "pass --radiance-scale".
  - Same for an undecidable level ("pass --input-level") and no spectral axis ("pass --wavelength-file").
- **SR input:** passed through with no correction, and the log says so. `--input-level toa` corrects it anyway.
- **`apply_oe` failure:** the entrypoint stops (`set -e`); `isofit.log` is in the work dir.
- **No GPU or no CUDA:** `fast_al` solves on the CPU. The log's "Per-pixel solve" line names the device.
- **Docker's default 64 MB `/dev/shm`:** Ray spills to disk and slows down. `run_scene.sh` and compose set 16 GB.
- **DEM VRT with absolute tile paths:** the tiles are not found inside the container, and elevation falls back
  to 0 m with a note. The VRT must say `relativeToVRT="1"`.
- **Cosmetic ISOFIT warning** "Center wavelengths provided in surface model file do not match". `apply_oe`
  converts the cube's wavelengths to microns before comparing them with the prior's nanometres. The centres
  match exactly (checked 2026-10-31).

## Known limitations

- **Level-test thresholds and harmonisation scores are in-sample.** The confusion table covers 28 cubes; GHOSt is
  unseen. The EnMAP blue model is holdout-informed and declared as such.
- **PRISMA and Tanager basic stay on the swath grid**, with GCPs and per-pixel lat/lon but no map projection.
  Compare against L2C (swath), not L2D: on relief, nearest-location pairing against an orthorectified product
  mismatches pixels. On the report's all-band metric, the Himalaya trial scored MAE 0.056 against L2D; the flat
  Maharashtra scene scored 0.0145 against L2C.
- **Free-AOT error depends on the surface.**
  - On the two Tanager AERONET overpasses it read high: +0.07 (Thessaloniki) and +0.09 (ROCX2025, AERONET 0.033).
    Planet read +0.16 / +0.21.
  - On a bright Permian desert it sank to the 0.05 LUT floor. The May climatology there is 0.19 and Planet read
    0.99, so the blue there runs +0.047 at 451 nm against Planet.
  - Water vapour reads +17% / +8%.
  - PRISMA's climatology prior would not have changed any of the three: each retrieval already sat inside its
    ±0.15 window, and ROCX's AERONET value lies below its window. Core-band land MAE against Planet is 0.0089–0.0114;
    the effect is confined to the blue below 600 nm.
- **Dark water can retrieve below −0.2 and is set to nodata** (EnMAP Sundarbans: 1.5% of pixels). DLR's L2A is
  nodata on the same pixels.
- **`fast_al` handles diagonal Seps only.** A chunk with off-diagonal noise terms falls back to ISOFIT's solve for
  that actor (2 of 12 actors on the EnMAP coast trial).
- **AVIRIS-NG runs at superpixels of 400 with the empirical line** (sensor rule; `--line analytical` /
  `--segmentation-size` override).
  - Core MAE against JPL is 0.0057, against 0.0056 at seg 50 analytical.
  - The darkest ~1% of pixels lose spectral fidelity: the angle against JPL at p99.9 is 6.9° against 4.9°. These
    are mostly shadow and dark soil, which anomaly detection may see.
  - The 460–550 nm error grows by +0.002, and the AOT map is smoother.
  - Not tested over water: the test window was dryland.
  - Seg 400 *analytical* keeps the seg-50 tail, at 2× the time: the per-pixel solve is ~90 min for a 4 Mpix line on
    the L4.
- **The report's MAE is a sanity check, not the accuracy claim.** It compares native bands, by nearest location,
  over all land, with no shift search and no cloud or shadow exclusion. The submission accuracy figures are the
  research scorer's (AC26-*).
- **No destriping and no map projection:** both out of scope.

## Decisions

| Choice | Evidence | Rejected |
|---|---|---|
| pip venv on Ubuntu 24.04 python3.12, no conda | Nothing at run time needs GDAL tools; rasterio wheels carry GDAL 3.12.1. Clean venv: 76/76 tests (AC26-D10b) | conda base (mambaforge retired; environment.yml's gdal=3.9 with rasterio 1.5 never solved) |
| Separate stack, new folder | User decision 2026-10-31: do not touch the existing stacks | Editing `enmap-isofit-docker/` in place |
| GeoTIFF + BSQ both | < 1% of a run (2–12 s); GeoTIFF half the size (AC26-W8) | One format |
| 6S v3 default, v4 opt-in | v3 1.29 s vs v4 1.38 s per point on an AVX-512 Xeon (clock drop) | `-march=native` (bakes in the build host's CPU) |
| int16 ×10000, water bands dropped | Organisers' mail 2026-10-31 | float32 |
| Tests gate the build (≥ 30% line coverage) | User rule: no tests, no ship. 79 tests, 53% (AC26-D10b, T11) | Build-time checks and end-to-end runs only |
| AVIRIS-NG: superpixels of 400 + empirical line (`SensorRules.segmentation_size`, `line_mode`); every other sensor 50 + analytical | 600-line window against JPL: ISOFIT 40.8 → 7.5 min, core MAE 0.0056 → 0.0057, median angle 2.15° → 2.19°. It is the only configuration that fits a 14–16 GB line in the 2 h budget, which includes anomaly detection (AC26-SEG, EL, FAST; user 2026-11-01) | seg 50: 6.2 h for a full line on Colab. seg 200 empirical: worse dark-pixel tail (p99 5.3°). seg 400 analytical: the seg-50 tail but 14.8 min per window, ~2–3 h for 14–16 GB on the VM. No presolve: saves only 1.3 min |
| Report pairs reference pixels at exactly the located rows/cols, and judges the reference scale on quiet bands | T7: a decimated read paired pixels N/2 apart (report MAE 0.038 against 0.005 by index), and JPL's junk water bands made `corr` look ×10000 (AC26-T11) | GDAL `out_shape` decimation; all-band p99 |
