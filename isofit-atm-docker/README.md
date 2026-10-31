# isofit-atm-docker

L1 → L2 surface reflectance for **EnMAP, PRISMA, AVIRIS-NG, Tanager, GHOSt** and unknown sensors, with
ISOFIT 4.1.5 + sRTMnet. A self-contained Docker stack, separate from the product's `docker/` stack and from the
older `enmap-isofit-docker/` (which this replaces and leaves untouched).

Output for every sensor, as the organisers asked: `<scene>_L2_SR.tif` and `<scene>_L2_SR.bsq/.hdr`.
- **Values:** int16 reflectance ×10000, nodata −9999.
- **Bands:** water-absorption bands dropped and listed.
- **Metadata:** wavelengths and FWHM in both files.
- **Sidecars:** `_outlier.tif`, `_snow.tif`, `_latlon.tif` (swath products) and `_provenance.json`.
- **Report:** a self-contained `_report.html` quick-look.

## Layout

| Path | What |
|---|---|
| `Dockerfile` | `builder` (compilers, venv, 6S builds + checks, ISOFIT data) → `runtime` (no compilers) → `test` (pytest, coverage gate) |
| `requirements.txt` | pip pins, verified end to end 2026-10-31; `requirements-dev.txt` adds pytest |
| `docker-compose.yml` | one-scene run with GPU and shm |
| `entrypoint.sh` | stage 1 read → 2 ISOFIT → 2b post-process → 2c harmonise → 3 write → 3c report |
| `src/readers/` | any delivery (native, ENVI, GeoTIFF, Tanager HDF5) → ISOFIT inputs + `scene.json` (grade, checklist) |
| `src/fast_al.py` | batched per-pixel solve (CUDA torch, CPU fallback; `ATM_FAST_AL=0` = ISOFIT's own) |
| `build/` | fast 6S (aerosol call removed, H2O scattering cache, -O3 v3/v4), wrapper, equality checks, ISOFIT patches |
| `src/atm_post.py` | per-sensor rules: noise, LUT, AOT prior, adjacency, terrain, harmonisation model |
| `src/writers.py`, `src/report.py` | submission product; HTML quick-look (optional `--ref` comparison) |
| `harmonise/`, `configs/harmonise/` | harmonisation towards the vendor L2 (EnMAP blue, PRISMA) |
| `tests/` | unit + integration tests on synthetic scenes (no ISOFIT run, no data, no network) |
| `scripts/` | `build_image.sh`, `deploy_airgap.sh`, `run_scene.sh`, `stage_dem.sh` |

## Build, test, ship (connected build PC)

```bash
./scripts/build_image.sh   # test target (pytest, >=30 % coverage) -> runtime image -> non-root smoke test
                           # -> trivy (if installed) -> dist/isofit-atm-docker-1.3.0.tar.gz + .sha256
```

Tests without Docker, given a Python 3.12 env:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-dev.txt
python -m pytest           # 76 tests, ~30 s; fails under 30 % coverage (53 % on 2026-10-31)
```

## Run (air-gapped VM)

```bash
./scripts/deploy_airgap.sh isofit-atm-docker-1.3.0.tar.gz isofit-atm-docker-1.3.0.sha256
DEM_DIR=/data/dem ./scripts/run_scene.sh /data/in/PRS_L1_STD_OFFL_....he5 /data/out/prisma1
REF=/data/ref/PRS_L2D_....he5 ./scripts/run_scene.sh ...      # adds the reference comparison to the report
GPUS=none ./scripts/run_scene.sh ...                             # CPU-only host
```

Entrypoint options (`--help`):
- **Overrides:** `--sensor`, `--input-level radiance|toa|sr`, `--radiance-scale`, `--wavelength-file`, `--season`.
- **Steps:** `--no-harmonise`, `--formats tif,bsq`, `--no-report`.
- **Run control:** `--n-cores`, `--keep-work`, `--stop-after-lut`.

Grades and flags never stop a run. Every stop names its override.

Switches: `SIXS_FAST=0` (stock 6S), `SIXS_CACHE=0`, `SIXS_ARCH=v4`, `ATM_FAST_AL=0`.

Evidence and history: `research/experiments/atm-correction/EXPERIMENT_LOG.md` (AC26-*).
