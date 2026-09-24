# EnMAP L1C → L2A via ISOFIT + sRTMnet (Air-Gap Deployment)

A self-contained Docker pipeline that converts DLR EnMAP L1C products to L2A
surface reflectance using **ISOFIT 3.x** with the **sRTMnet** radiative
transfer emulator, then repacks the output in the DLR PACO L2A layout so a
downstream anomaly-detection model trained on PACO L2A can consume it
unchanged.

Designed for the two-node pattern: build once on a connected node, transfer a
single tarball to the air-gapped inference node.

---

## Pipeline stages

```
EnMAP L1C directory
        │
        ▼
┌─────────────────────────┐
│ enmap_l1c_preprocess.py │   parse METADATA.XML, apply gain/offset,
│                         │   convert mW→µW, sample Cop-DEM,
│                         │   emit ENVI radiance / location / observation
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ isofit surface_model    │   rebuild the multicomponent prior on THIS
│                         │   scene's wavelength grid
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ isofit apply_oe         │   presolve → per-segment OE inversion
│  (sRTMnet emulator,     │   → per-pixel reflectance via analytical line
│   analytical line,      │   LUT bounds from the season preset
│   6S-backed sRTMnet)    │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ postprocess_to_paco.py  │   float→int16 (×10000), GeoTIFF w/ deflate,
│                         │   PACO-style METADATA.XML with ISOFIT
│                         │   provenance, copy L1C sidecars through
└──────────┬──────────────┘
           ▼
   L2A product directory (PACO-compatible layout)
```

---

## Alignment with PACO training conditions

Choices below trace back to the metadata harvest of your L2A training set:

| PACO field (training) | This pipeline |
|---|---|
| `correction_type = Land_Mode` | Terrestrial multicomponent surface prior, no aquatic model |
| `terrain_correction = Yes/No` (mixed) | Always **Yes**: DEM-derived slope/aspect/cos-i drive the observation cube. Uniform terrain handling is one less nuisance variable at inference. |
| `cirrus_haze_removal = No` | No pre-flight cirrus mask; L1C `QL_CIRRUS` copied through for downstream use |
| `smile_correction_applied = no` | Nominal per-band λ / FWHM from L1C metadata, no per-column shift |
| `band_interpolation = No` | All 224 bands preserved; water-vapor bands flagged in `bbl` but not filled |
| `season = summer / winter` | Auto-detected from acquisition month and passed to `apply_oe --lut_config_file` as `configs/enmap_{summer,winter}.json` (H2O + AOD LUT bounds) |
| `dem_database = TDM_COP1ARC_OCEAN_v010300` | **Copernicus GLO-30 with ocean fill** (same underlying dataset) |
| `caltab_atm_version = 04.01.00` | Cannot spoof PACO's MODTRAN 5.4 LUT; provenance tag reports `ISOFIT-3.x-sRTMnet_v100` honestly |
| `reflectance_unit = binary abs value` | int16, scale ×10000, nodata −32768 (as PACO) |

**Read the caveat**: sRTMnet + OE and PACO/ATCOR do not produce numerically
identical L2A. Run at least one shared scene through both before trusting your
reconstruction-error thresholds. See §7 in the design doc.

---

## Repo layout

```
enmap-isofit-docker/
├── Dockerfile
├── environment.yml
├── docker-compose.yml
├── entrypoint.sh
├── src/
│   ├── enmap_l1c_preprocess.py   # L1C → 3 ENVI cubes for ISOFIT
│   ├── postprocess_to_paco.py    # ISOFIT output → PACO-style L2A
│   └── validate_vs_paco.py       # our L2A vs DLR PACO L2A, per scene
├── validation/                   # dated validation results + notes
├── configs/
│   ├── enmap_summer.json           # LUTConfig overrides (H2O / AOD bounds)
│   ├── enmap_winter.json
│   ├── enmap_wavelengths.txt       # nominal grid, build-time surface prior
│   └── surface_multicomponent.json # surface prior recipe (build + per scene)
├── scripts/
│   ├── build_image.sh            # connected node: build + export
│   ├── deploy_airgap.sh          # air-gap node: verify + load
│   ├── stage_dem.sh              # connected node: fetch Cop-DEM for AOI
│   └── run_scene.sh              # convenience wrapper
└── README.md
```

---

## One-time build (connected node)

Requires `docker` (or `podman` with docker alias) and internet.

```bash
# 1. Build image and export tarball
./scripts/build_image.sh
# → dist/enmap-isofit-1.1.4.tar.gz  (~9.6 GB compressed, 24.7 GB loaded)
# → dist/enmap-isofit-1.1.4.sha256
# → dist/aux-manifest.txt

# 2. Stage Cop-DEM tiles for your AOI (or per-scene)
./scripts/stage_dem.sh bbox 68 8 98 37 dist/dem   # India bbox example
# → dist/dem/Copernicus_DSM_COG_10_*.tif
# → dist/dem/cop30_mosaic.vrt
```

Everything under `dist/` is the transfer bundle. Ship it (physical media,
one-way diode, whatever) to the air-gap host.

---

## Deploy (air-gap node)

Requires `docker` (or `podman`). No internet.

```bash
# 1. Verify + load the image
./scripts/deploy_airgap.sh enmap-isofit-1.1.4.tar.gz enmap-isofit-1.1.4.sha256

# The script runs a smoke test importing isofit + tensorflow and confirming
# sRTMnet weights are present.

# 2. Place the DEM directory somewhere persistent, e.g. /srv/aux/dem/
```

---

## Run one scene

```bash
./scripts/run_scene.sh \
    /data/enmap/l1c/ENMAP_L1C_20240615T091234_000_V010500 \
    /data/enmap/l2a \
    /srv/aux/dem \
    --season auto \
    --n-cores 16
```

Or directly with docker:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /data/enmap/l1c/SCENE:/data/l1c:ro \
    -v /data/enmap/l2a:/data/l2a \
    -v /srv/aux/dem:/aux/dem:ro \
    enmap-isofit:1.1.4 \
    /data/l1c /data/l2a --season auto --n-cores 16
```

The output tree per scene is named with the **L2A** scene id, as PACO names it
(`____L1C-` in the input id becomes `____L2A-`):

```
/data/enmap/l2a/ENMAP01-____L2A-DT..._V010506_.../
    <id>-SPECTRAL_IMAGE.TIF        # int16 reflectance, ×10000, nodata −32768
    <id>-METADATA.XML              # L1C metadata + <atmosphericCorrection> block
    <id>-HISTORY.XML               # ┐
    <id>-QL_PIXELMASK.TIF          # │ every L1C sidecar, renamed - the same
    <id>-QL_QUALITY_{CLASSES,CLOUD,CLOUDSHADOW,HAZE,CIRRUS,SNOW,TESTFLAGS}.TIF
    <id>-QL_VNIR.TIF, QL_SWIR.TIF  # ┘ 13-file set PACO ships (see Known gaps)
    <id>-RFL_UNC.TIF               # extra: int16 reflectance uncertainty
    <id>-ISOFIT_STATE.TIF          # extra: float32 AOT550, H2OSTR (g/cm²),
                                   #        surface_elevation_km, per pixel
    <id>-PROVENANCE.json           # extra: pipeline + ISOFIT version, config hash
```

Options beyond the ones above: `--line analytical|empirical` (see Tuning),
`--keep-work` to retain `_work/`, and `--stop-after-lut` for LUT diagnostics.

---

## Batch processing

The container processes one scene per invocation, which keeps memory bounded
and makes retries simple. For a queue, wrap `run_scene.sh` with GNU parallel:

```bash
find /data/enmap/l1c -maxdepth 1 -mindepth 1 -type d \
  | parallel -j 2 --line-buffer \
      ./scripts/run_scene.sh {} /data/enmap/l2a /srv/aux/dem --n-cores 8
```

`-j 2` × `--n-cores 8` for a 16-core box. Tune to your host.

---

## Tuning / troubleshooting

- **Memory**: TensorFlow + Ray + `apply_oe` peak around 8–12 GB per scene at
  `segmentation_size=50`. Reduce concurrency or increase segment size if OOM.
- **Slow first scene**: the first `apply_oe` per container startup builds the
  emulator LUT interpolation cache. Subsequent scenes reuse it if you keep the
  container running (use `docker run -it --entrypoint bash` for interactive
  batch).
- **Season override**: if scene metadata is ambiguous or you have local
  climatology, pass `--season summer|winter` explicitly.
- **Per-pixel solve**: `--line analytical` (default) runs OE on each SLIC
  superpixel, interpolates the atmospheric state, then solves each pixel
  analytically. `--line empirical` uses the older empirical line, which only
  works because the Dockerfile patches it. Neither matches what PACO/ATCOR
  does, so if you are tuning for agreement with a PACO L2A product, run both
  and compare rather than assuming.
- **Custom AOD / H2O priors**: edit `configs/enmap_{summer,winter}.json`. These
  are passed to `apply_oe --lut_config_file`, which is a **flat override of
  `LUTConfig`'s own attributes** — not a `forward_model` config. Set AOD via
  `aerosol_2_range` / `aerosol_2_spacing`: with a non-`.jld2` emulator (sRTMnet
  is one) ISOFIT copies `aerosol_2_*` onto `aot_550_*` and zeroes
  `aerosol_2_spacing`, so writing `aot_550_*` directly is silently discarded.
  Grids are `linspace(min, max, ceil((max-min)/spacing)+1)`, so irregular
  spacings are not expressible. Under `--presolve` the H2O range is replaced by
  the 2nd/98th percentile of the presolve retrieval — your bounds govern the
  presolve LUT and clamp the result rather than fixing the final grid.
- **Surface prior wavelengths**: the prior is rebuilt per scene from
  `_work/wavelengths.txt` so it matches the real band centres (~5 s; the
  build-time prior on the nominal grid stays as a fallback). Its log is
  `_work/surface_model.log`.
- **`check_surface_model` always warns** that "center wavelengths ... do not
  match", and **rebuilding will not silence it** — the check compares
  incompatible units. `surface_model` forces nm before `savemat`
  (surface_model.py:155), `get_wavelengths` forces microns
  (template_construction.py:1399), and `check_surface_model` compares the two
  with `atol=0.01` (template_construction.py:592). Ignore it. The check that
  matters is the channel-count test on the line above, which *raises* rather
  than warns.
- **`Failed to register worker ... to Raylet. IOError: [RayletClient] Unable
  to register worker with raylet`**, usually right after
  `ModuleNotFoundError: No module named 'pkg_resources'`: `setuptools` is
  missing. Ray 2.9's dashboard agent imports `pkg_resources`; when the agent
  dies the raylet dies with it, so no worker can register and the process
  aborts natively — **exit code 1, no Python traceback**, which reads like an
  OOM but is not (`docker inspect --format '{{.State.OOMKilled}}'` says
  `false`). Fixed in 1.1.1 by installing `setuptools<81`. Do not dismiss the
  `pkg_resources` line as a cosmetic dashboard warning; it is the cause.
- **`ValueError: array must not contain infs or NaNs`** in `svd_inv_sqrt`,
  preceded by `6S path not valid` and `Failed to parse any data for file:
  .../lut_h2o/H2OSTR-*`: the 6S binary is missing. sRTMnet emulates 6S but its
  `preSim` still runs the real thing to build the transmittance base, so
  without it every LUT entry is NaN and the first inversion dies. Check the
  image with `docker run --rm --entrypoint bash <tag> -c 'ls /root/.isofit/sixs/sixsV*'`.
  Do not match the name exactly: the `isofit/6S` release currently builds
  `sixsV2.1` plus a `sixsV2.1-lutaero` variant, while the repo's `main`
  Makefile emits `sixsV2.1.2.CO2`. Glob `sixsV*` and drop `*lutaero*`, which is
  what `get_exe()` does.
- **LUT diagnostics in ~8 minutes**: `--stop-after-lut` runs stages 1-2 up
  to the finished full LUT (the 4th `Loading LUT into memory` log line), then
  exits keeping `_work/`. Inspect `_work/isofit/lut_full/6S.lut.nc` and the
  per-point `LUT_*.inp` 6S inputs. Do not judge completion by file size:
  ISOFIT pre-allocates `lut.nc` full-size and fills it in place.
- **`rhoatm is partially NaN (A/B, X%), replacing with 0s`** is never
  cosmetic. `A` is the NaN count and `B` the *valid* count, so
  `454608/170478` means 73 % NaN, not 267 %. Zeroed path reflectance leaves
  the atmosphere in the product, worst in the blue. First check the 6S input
  says `-99.00 (sensor level)`; a small number there means the sensor
  altitude is wrong (see 1.1.3).
- **QA against PACO** (measured, 1.1.3, scene DT0000122280, land pixels,
  25.1°N Ganges plain in April): bias vs PACO +0.039 at 418 nm, +0.028 at
  497 nm, +0.011 at 577 nm, then within ±0.017 from 670 nm to 2445 nm;
  spatial correlation 0.86 at 418 nm, 0.95-0.997 from 497 nm on; mean
  absolute difference 0.013, median spectral angle 5.1°. Compare on land
  only (`QL_QUALITY_CLASSES == 1`) - water runs high, see Known gaps. A new
  scene well outside these numbers needs investigation. Five-scene results,
  including a snow-covered Himalayan scene, are in
  [validation/README.md](validation/README.md). To check new scenes, run
  `python -m validate_vs_paco --paco <PACO root> --ours <our root> --csv
  <out.csv>` inside the image; its docstring has the full `docker run` line.

---

## Known gaps you may want to close

1. **Blue runs +0.03-0.04 above PACO below 500 nm** (see QA above). First
   suspect: `--atmosphere_type` is left at `ATM_MIDLAT_SUMMER` — it sets the
   H2O upper bound and the aerosol lower bound, and it cannot be set from the
   season JSON because `LUTConfig` reads it *before* applying the config
   override. It has to be an `apply_oe` CLI flag. For low-latitude scenes
   (the Ganges-plain test scene is 25.1°N) `ATM_TROPIC` is the better fit.
   The aerosol model is the second suspect. Change one at a time.
2. **Per-pixel view geometry** — L1C only gives scene-mean angles. For very
   wide FOV effects (EnMAP is ~30 km, so small) you could derive per-pixel
   view geometry from orbit ephemeris in the L1C XML. Not implemented; the
   scene-mean approximation matches SISTER's DESIS treatment.
3. **DEM caching** — DEM sampling is currently per-pixel via `dem.sample()`.
   For very large scenes this could be sped up by windowed reprojection to
   the scene grid. Not a bottleneck for standard 1000×1024 tiles.
4. **Cloud mask refinement** — L1C cloud masks are copied through unchanged.
   If you find them too permissive, add a Fmask-style post-filter using the
   ISOFIT state cube (extreme H2O or AOT indicates likely cloud).
5. **Water runs +0.03 high in the NIR**, where it should be near zero —
   consistent with adjacency from bright surrounding land, which PACO
   corrects and ISOFIT does not. Treat water-pixel reflectance with caution.
6. **`QL_VNIR` / `QL_SWIR` are the L1C quicklooks**, i.e. top-of-atmosphere
   radiance previews; PACO's are rendered from reflectance. Visual only; no
   model input reads them.

---

## Changelog

### 1.1.4

1. **Summer `h2o_min` lowered 0.5 → 0.2 cm** (ISOFIT's default and the
   presolve floor). `apply_oe` floors the final H2O range at `h2o_min` and
   drops presolve pixels at or below it from the percentiles that set that
   range, so on the Himalayan scene DT0000187759 water vapour pinned at
   0.50 cm and pressure elevation compensated ~1 km high. Now: H2O
   0.21–0.52 cm, retrieved elevation within +0.11 km of the DEM (r 0.90),
   977/1128 nm bias vs PACO halved. Plains scenes are unaffected - their
   H2O ranges start well above either floor.
2. **`src/validate_vs_paco.py`** compares this pipeline's L2A with DLR PACO
   L2A per scene (land / water, per band, summary, optional CSV). The
   five-scene results are in [validation/](validation/README.md).
3. First image built from the real `Dockerfile` via `scripts/build_image.sh`
   since 1.1.1; 1.1.2 and 1.1.3 were `Dockerfile.hotfix` derivatives.

### 1.1.3

**Output file names and nodata changed - update any loader built on 1.1.2.**

1. **The observation cube's path length is written in metres.** ISOFIT reads
   obs band 0 as metres (`template_construction.py` and `core/geometry.py`
   both apply `m_to_km`); the preprocessor wrote km, so ~655 km of path
   became 0.655 km and every 6S run put the sensor ~0.7 km above ground.
   LUT points at or above that height came back `rhoatm = NaN` (73 % of the
   full LUT, zero-filled by ISOFIT), and even the valid points left most of
   the Rayleigh column above the "sensor". Result: +0.17 reflectance at
   418 nm fading by ~700 nm, which no AOD setting could move. **Every product
   from 1.1.2 and earlier carries this error.** After the fix the 418 nm bias
   vs PACO fell to +0.039, the 2060 nm water-vapour bias from −0.025 to 0.000,
   and the median spectral angle from 11.9° to 5.1°.
2. **Products are named with the L2A scene id**, as PACO names them.
3. **Every L1C sidecar is copied**, which yields PACO's exact 13-file set
   (adds `QL_QUALITY_SNOW`, `QL_VNIR`, `QL_SWIR`, `HISTORY.XML`). The previous
   copy split names on the first `-`, which is *inside* the EnMAP id, so every
   QL file carried the scene id twice.
4. **nodata is −32768**, PACO's value; valid reflectance is clipped at −32767
   so it can never collide.
5. **`ISOFIT_STATE.TIF` is written again.** Analytical line writes no
   full-image `*_state`; its per-pixel atmosphere is `*_atm_interp`, and band
   names now come from that file's header.
6. **`PROVENANCE.json` reports the real version**, from `ENV PIPELINE_VERSION`
   (it said `1.0.0` regardless).
7. **Summer AOD ceiling raised 0.40 → 0.90.** The retrieval sat at 0.40;
   after the change the land median is 0.40 with p99 0.66. Doubles the AOD
   LUT dimension (5 → 11 points).
8. **`--stop-after-lut`** diagnostic mode (see Tuning).

### 1.1.2

Two ISOFIT 3.7 defects, both reached only once the LUT finally built.

1. **Surface-prior wavelengths are de-collided before `surface_model`.**
   ISOFIT names surface states `"RFL_%04i" % int(wl)` and `construct_full_state`
   de-duplicates them with `set()`. EnMAP's VNIR/SWIR overlap puts two band
   pairs inside the same integer nm (911.572/911.968 and 993.145/993.338), so
   the output statevector came out 222 long against a 224-element forward
   model and every write raised
   `IOError: len(fm.statevec) > len(self.full_statevec)`. Colliding bands are
   nudged across the integer boundary - at most 0.15 nm against a ~6.5 nm
   FWHM - for the prior only; the radiance cube keeps its true band centres.
   Fixed upstream in 4.1.5, which names states `"RFL_%04i_%.3f"`.
2. **`--analytical_line` is now the default, and `empirical_line` is patched.**
   `empirical_line` sizes its output reflectance file with the *location*
   cube's band count (always 3) while the metadata carries the radiance bands,
   so it dies with `could not broadcast (L,3,S) into (L,224,S)`. This is not
   EnMAP-specific - it breaks every sensor, arrived with the
   `initialize_output` helper in 3.6.0, and is still unfixed in 4.1.5, which
   says plainly that analytical line is the path upstream maintains. Select
   with `--line analytical|empirical`; the Dockerfile patches the one-token
   bug so both work, and fails the build loudly if a future ISOFIT changes the
   line it anchors on.

### 1.1.1

**`setuptools` is a hard runtime dependency.** Without it Ray 2.9's dashboard
agent fails to import `pkg_resources`, the raylet follows it down, and
`apply_oe` aborts at `Segmenting...` with exit 1 and no traceback. Installed as
`setuptools<81` in a layer *below* the sRTMnet download, so the fix costs a
~1 minute rebuild rather than re-pulling 5.8 GB. The `HEALTHCHECK` now imports
`pkg_resources` so a future image cannot regress silently.

### 1.1.0

Six fixes, the first of which explains why the previous five were invisible.

1. **`scripts/build_image.sh` pinned ISOFIT 3.4.0** via `--build-arg`,
   overriding the Dockerfile's own default. Every build claiming to be 3.7.x
   was in fact 3.4.0. Now defaults to `3.7.7`, and the image tag moves to
   `1.1.0`.
2. **6S is built by ISOFIT itself.** 3.7's `isofit download sixs` pulls the
   `isofit/6S` release, patches the Makefile with `-std=legacy` and runs
   `make`, so the hand-rolled `gfortran` block is gone. (The
   `NameError: name 'validate_' is not defined` that killed that step was a
   3.4.0 bug in its `download_cli`.) `download()` swallows `make` failures, so
   the layer now asserts the binary exists and that `get_exe()` resolves it.
3. **The 6S binary is located by glob**, in the entrypoint preflight and the
   `HEALTHCHECK`, matching `sixsV*` minus `*lutaero*` as `get_exe()` does. Both
   previously hardcoded `sixsV2.1`; the output name varies between the pinned
   release and `main`, so an exact match can kill a perfectly good image.
4. **`find_isofit_outputs` no longer picks `*_subs_*`.** It took the first
   `rglob` hit, so the choice between the full cube and the sparse superpixel
   retrieval was filesystem order — per product, meaning reflectance and
   uncertainty layers could describe different geometries.
5. **The surface prior is rebuilt per scene** on the real wavelength grid,
   clearing ISOFIT's `check_surface_model` mismatch warning.
6. **Season presets are actually applied.** They were shaped as
   `forward_model.radiative_transfer.*`, which `--lut_config_file` does not
   read; both files are now flat `LUTConfig` overrides and wired into
   `apply_oe`. See the Tuning section for the `aerosol_2_*` trap.

If you have a 1.0.0 run in flight: drop any bind-mount of
`patches/template_construction.py`. That was a 3.4-era patch and it will
shadow 3.7.7's module.

---

## Provenance

- ISOFIT: <https://github.com/isofit/isofit> (v3.7.7, `ARG ISOFIT_VERSION`)
- 6SV2.1: <https://github.com/isofit/6S> (built at image build; sRTMnet's
  `preSim` needs it — see Troubleshooting)
- sRTMnet: DOI [10.5281/zenodo.4096627](https://doi.org/10.5281/zenodo.4096627)
- Copernicus DEM: `s3://copernicus-dem-30m/` (Copernicus Programme, ESA)
- Sensor: EnMAP (DLR / GFZ)
