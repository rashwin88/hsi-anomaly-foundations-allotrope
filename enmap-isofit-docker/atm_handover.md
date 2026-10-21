# Atmospheric correction - handover

State of the L1 -> L2 atmospheric-correction work as of 2026-10-21, for
whoever (human or agent) picks it up on another machine. Read this first,
then [README.md](README.md) (the ISOFIT pipeline and its four ISOFIT
workarounds) and [validation/README.md](validation/README.md) (every number
measured so far).

## Why this exists

Allotrope's anomaly models were trained on vendor L2 products - DLR PACO L2A
for EnMAP, ASI L2D for PRISMA. At deployment only L1 arrives, and neither
vendor processor is openly available. This repo builds an air-gappable L1 ->
L2 Docker pipeline whose output matches those L2 products closely enough that
the models see the same thing. "Good" here means *close to PACO / ASI*, not
close to ground truth.

## Where the code lives

| Copy | What it is |
|---|---|
| Standalone repo, `C:\Users\FITT\Downloads\enmap-isofit-docker\enmap-isofit-docker` on the original box | **Source of truth.** Branch `main` = ISOFIT pipeline, tags `v1.1.3`, `v1.1.4`. Branch `vinoth/acolite-engine` = `main` + the ACOLITE work + this file. No git remote. |
| Allotrope repo, branch `vinoth/enmap-isofit-l2a`, folder `enmap-isofit-docker/` | `main` of the standalone repo up to 1.1.4, mirrored with `git subtree`. Left as is. |
| Allotrope repo, branch `vinoth/acolite-engine`, folder `enmap-isofit-docker/` | `vinoth/acolite-engine` of the standalone repo, mirrored the same way. **Continue from here.** |

Mirror command, run from an Allotrope worktree on the target branch:
`git subtree pull --prefix=enmap-isofit-docker <standalone repo path> <branch>`.
If you continue on a new machine without the standalone repo, just work
directly in `enmap-isofit-docker/` on the Allotrope branch; it is a plain folder.

## What is NOT in git (bring it separately or rebuild)

- **Scene data** - `C:\Users\FITT\Documents\gaic_stage3_datasets\stage3_20260906\`
  with `without ac\` (EnMAP L1C `ENMAP01-____L1C-DT*`, PRISMA `PRS_L1_STD_OFFL_*`)
  and `with ac\` (PACO L2A `ENMAP01-____L2A-DT*`, ASI `PRS_L2D_STD_*`).
  5 EnMAP datatakes (DT0000122280, 131004, 187759, 189895, 194619) and 5 PRISMA
  pairs (2021-03-05, 2023-06-20, 2024-05-11, 2024-09-19, 2026-03-05).
  `acolite/run_*.sh` default `SRC` to that path under `/mnt/c`; override `SRC`.
- **ISOFIT image** - `enmap-isofit:1.1.4`, rebuild with `scripts/build_image.sh`
  (downloads ~5.8 GB of sRTMnet) or copy `dist/enmap-isofit-1.1.4.tar.gz`
  (9.6 GB, gitignored).
- **ACOLITE image** - `acolite:8e04b97`, `docker build -t acolite:8e04b97 acolite/`.
- **ACOLITE outputs** - WSL Ubuntu `~/acolite_work/` (~30 GB): `out/` (EnMAP),
  `prisma/`, `isofit_rescore/`, `cache/` (LUTs, MERRA-2), `score_*.txt`. The
  scores are already copied into `validation/*.csv|txt`; the rest regenerates.

## Where it stands

**ISOFIT engine (v1.1.4, EnMAP).** Land MAE vs PACO 0.010-0.016 on four
plains scenes, 0.035 on the Himalayan snow scene. Details and tables in
`validation/README.md`.

**ACOLITE engine (checkpoints `c314ca2`, `99e48ae`, 2026-09-30).** Added to go
after ISOFIT's water adjacency error.
- EnMAP, `dsf_aot_estimate=fixed`: water error roughly halves on the plains
  scenes, but land is worse than ISOFIT on all 5 (Rajasthan desert worst, AOT
  over-estimated at 0.67). 50 of 224 bands come out blank.
- PRISMA L1 vs ASI L2D: works only with `src/prisma_l1_geometry.py` standing
  in for ASI's L2C. `settings/prisma_fixed_swir.txt` (dark-pixel search
  400-2400 nm) beats ACOLITE's PRISMA default on 4 of 5 scenes; land MAE
  0.013-0.045.

## Open items, roughly by value

1. **Hybrid: ISOFIT on land + ACOLITE on water**, split by the L1C
   `QL_QUALITY_CLASSES` (EnMAP). Not tried yet. The numbers suggest it takes
   the better of both engines.
2. **`dsf_aot_estimate=ancillary_fixed`** - ACOLITE author Quinten
   Vanhellemont's recommendation (email 2026-09-08; tested in
   doi:10.3390/rs17233790), expected to fix the bright-desert AOT failure.
   Blocked on a NASA EarthData login: put credentials in WSL `~/.netrc` and
   `run_acolite.sh` / `run_prisma.sh` mount it. Settings files
   `settings/{enmap,prisma}_ancillary_fixed.txt` are ready. Open question: it
   downloads ancillary data per scene date, which conflicts with the air gap.
3. **PRISMA 2023-06-20 is -0.11 in NIR whatever the AOT** (828 nm: ASI 0.343
   vs ours 0.236). Next check: is ASI's surface NIR above the L1 TOA NIR? If
   so the gap is ASI's, not ours.
4. **PRISMA L2D-layout product.** Allotrope reads PRISMA L2D straight from the
   `.he5` (`app/templates/sensor_data_templates/prisma_hyperspectral.py`:
   `PRS_L2D_HCO` VNIR/SWIR_Cube uint16 + `L2Scale*`, `List_Cw_*`; georef in
   `app/georef/from_scene_file.py` fits an affine to Latitude/Longitude). So a
   PRISMA output must be a gridded HE5 in that layout. Not built.
5. **PRISMA geometry**: relative azimuth from `prisma_l1_geometry.py` is 9.5
   deg vs ASI's 11.9 deg on 2024-05-11, cause unknown. L1 geolocation is off
   from L2D by up to ~200 m cross-track (~600 m along-track on 2021-03-05),
   varying across the swath; `validate_prisma_vs_asi.py` fits a 6x6 tile
   shift field to compare.
6. **ISOFIT residuals**: blue excess +0.006..+0.036 at 460 nm (tracks haze;
   next suspect aerosol model / `ATM_MIDLAT_SUMMER` at 25 N); terrain
   illumination on snow (Himalaya MAE is scatter, not bias); band 224
   (2445 nm) up to +0.05 high.
7. **Housekeeping**: `src/postprocess_to_paco.py:146` hard-codes
   `sRTMnet_v100` into METADATA.XML, and README provenance says v100 too.
   Earlier notes say the image actually ships `sRTMnet_v120.h5` -
   confirm with `ls /root/.isofit/srtmnet` in the image, then fix the label.
   `Dockerfile.hotfix` still FROMs 1.1.1 and its header says to delete it
   after a clean build (1.1.4 was a clean build) - undecided.

## Facts worth knowing before you start

- **PRISMA processor versions** (read from HDF5 attrs 2026-10-10): every L1 is
  `PRS_L1_STD`, `L1_A_EO` 4.6-0 (`_OFFL` is only in the filename); every L2D
  is processor `L2D` 02.07, `Atm_Lut_version` 1.1.2, built on L1 4.6-0 -
  identical 2021-2026, so ASI reprocessed the archive. For harmonization
  training, order **L1 + L2D + L2C per datatake**: L2C is the same reflectance
  in L1 swath geometry, so no co-registration is needed.
- **ASI L2D is bright in the blue** (water 0.06-0.09 at 407-468 nm). Matching
  ASI means matching that, too.
- **RAdCor** (Vanhellemont's adjacency correction) does not support EnMAP or
  PRISMA.
- ACOLITE names bands by rounded nm, so EnMAP's VNIR/SWIR overlap pairs
  (911.6/912.0, 993.1/993.3) collide - the same root cause as ISOFIT bug 3 in
  the README.

## Environment gotchas (Windows + WSL box)

- Docker runs inside **WSL Ubuntu**, not Docker Desktop. Run the `acolite/*.sh`
  scripts from WSL.
- WSL `/mnt/c` bind mounts occasionally come up empty inside a container
  (empty log, or `No module named ...`). Rerun; it is not your bug.
- Git Bash on Windows rewrites container paths that start with `/`
  (`/bin/bash` -> `C:/Program Files/Git/bin/bash`). Put `docker run` lines
  with Windows paths in a script file.
- The ACOLITE image uses its own conda env, not miniforge base (changing
  python in base breaks mamba's cache), and needs `PROJ_DATA` / `GDAL_DATA`
  set by hand - already done in `acolite/Dockerfile`.
- ISOFIT runs ~25-29 min per EnMAP scene at `--n-cores 16`; ACOLITE ~6 min.
- Commits in the standalone repo use the repo-local identity
  `Vinoth <vinothvital@gmail.com>`.
