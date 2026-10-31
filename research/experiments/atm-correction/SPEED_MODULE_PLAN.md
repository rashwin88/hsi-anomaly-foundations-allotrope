# Atm-correction module: speed + five sensors, 12-item plan (2026-10-30, draft for agreement)

Target: worktree `C:/wt/atm-isofit4`, folder `enmap-isofit-docker/` (branch `vinoth/atm-isofit4`, uncommitted).
Evidence: EXPERIMENT_LOG AC26-BSC, AC26-6SX, AC26-6SC, AC26-PRF. User scope: memory `atm-module-scope-2026-10-30`.

## Scope
- Sensors: EnMAP, PRISMA, AVIRIS-NG, Tanager, GHOSt.
- Inputs: native (EnMAP L1C, PRISMA L1 HE5, AVIRIS-NG L1B ENVI, Tanager HDF5), ENVI .dat/.hdr, single-file GeoTIFF.
- Output format mirrors the input. Self-contained offline HTML quick-look per scene.
- Speed levers (exact): 6S aerosol-only call removed; 6S scattering cached across H2O (two-wave run order);
  6S -O3 built for x86-64-v3 and v4, chosen at run time; batched per-pixel solve (CUDA torch, CPU fallback).
  Every lever has an off switch and falls back to stock ISOFIT on error.
- Harmonisation: PRISMA H6, AVIRIS-NG identity, EnMAP blue-only own+z (holdout-informed, declared);
  Tanager and GHOSt identity (no vendor pairs fitted yet).
- Image: trivy critical < 100; DEM mounted (sneakernet), not baked.

## Findings that shape it
- `entrypoint.sh` is EnMAP-only; PRISMA / AVIRIS-NG prep is Colab prototypes; `ang_prep.py` crops 600 lines.
- 1.2.0 never built as a container. Base `condaforge/mambaforge:24.9.2-0` is deprecated (likely CVE source).
- Torch is the CPU wheel; GPU needs cu128 (VM driver 580 / CUDA 13).
- `src/sixs_parts.py` reads the 6S "Multipart transmittances" table: must be checked under noaer + cache.
- EnMAP blue-only needs no package code (`own` parts already apply per band).
- Tanager radiance HDF5 has sun/sensor angles, lat/lon, time per pixel. ENVI exports (`*_ortho_sr`) are SR.
- GHOSt seen so far: dataset1 SR (QUAC), dataset2 TOA-like with no CRS, dataset3 DN with no scale. ISOFIT needs
  radiance, time, location and geometry.
- Docker's default /dev/shm is 64 MB; Ray needs GBs (`shm_size`).

## The 12 items
1. P0 checks (Colab, no repo edits): Multipart table under noaer / cache; ISOFIT 4.1.5 hook points; EnMAP blue
   LOSO and model freeze.
2. 6S build: patches, v3/v4 binaries, CPU wrapper, build-time equality check vs stock.
3. 6S two-wave run order, per-group cache key, `SIXS_CACHE=0`.
4. Batched per-pixel solve `src/fast_al.py`, apply_oe hook, CUDA torch, `ATM_FAST_AL=0`, fallback.
5. Reader layer: one interface (rdn, loc, obs, wl, fwhm, time, sensor) for every sensor and format.
6. Sensor rules for Tanager and GHOSt (noise, LUT preset, AOT rule, post-processing, harmonisation).
7. Entrypoint: `--sensor` / auto-detect, DEM with fallback, PRISMA AOT prior, harmonise step.
8. Writers: output mirrors input (native, ENVI, COG GeoTIFF).
9. HTML quick-look report (offline, single file).
10. Image hardening: multi-stage build, current base, shm, GPU compose, LF `.gitattributes`, trivy, size.
11. Trials: L4 PRISMA, V6e multi-sensor (no code); stress set (snow, winter haze, coast, flares); one Tanager scene
    with an AERONET station in frame: our AOT550 and H2O vs AERONET (and vs Planet SR per-pixel AOD / water vapour).
12. Docs and air-gap kit: LLD, README, handover, deploy scripts, dependency manifest, VM probe commands.
