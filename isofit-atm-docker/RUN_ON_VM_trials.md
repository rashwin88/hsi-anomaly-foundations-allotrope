# Trial scenes for the VM: one per sensor, with the vendor reference

## 1. Get the files (Google Drive; the Colab tunnel was too slow at ~100 KB/s)

Everything is already on Google Drive under `My Drive/PS_11/`. Download these and arrange them as below on the hard
disk; Drive desktop or the web UI works.

| Put here (on the disk) | From Drive (`My Drive/PS_11/…`) | Size |
|---|---|---|
| `prisma/input/` | `PRISMA Training Set` shortcut → `20260514055637/PRS_L1_STD_OFFL_20260514055637_20260514055641_0001.he5` | 0.8 GB |
| `prisma/reference/` | same folder → `PRS_L2C_STD_20260514055637_20260514055641_0001.he5` | 1.4 GB |
| `enmap/input/` | `EnMAP_datasets/enmap_L1C/ENMAP01-____L1C-DT0000056279_20240102T052511Z_001_V010502_20260127T164258Z/` (whole folder) | 0.6 GB |
| `enmap/reference/` | `EnMAP_datasets/stage_3/ENMAP01-____L2A-DT0000056279_20240102T052511Z_001_V010502_20260127T164258Z/` (whole folder) | 0.4 GB |
| `tanager/input/` | `tanager_datasets/scenes/20250919_170238_88_4001/` → `…_basic_radiance_hdf5.h5` and `…_stac_item.json` | 0.6 GB |
| `tanager/reference/` | same folder → `20250919_170238_88_4001_basic_sr_hdf5.h5` | 1.2 GB |
| `aviris_ng/input/ang20151219t081738/` | `aviris_datasets_L1/ang20151219t081738/` (whole folder) | 6.9 GB |
| `aviris_ng/reference/` | `aviris_datasets/ang20151219t081738/` → `ang20151219t081738_corr_v2m2_img.bin` + `.hdr` | 6.9 GB |
| `dem/` | `atm_trial_scenes/dem/` (Copernicus 30 m tiles + `cop30_mosaic.vrt`, relative paths) | 2.0 GB |

The PRISMA pair (`prisma/`) is already complete in this local folder; the rest of the local copy is partial and
can be deleted.

## 2. Run (from the `isofit-atm-docker/` folder, after `scripts/deploy_airgap.sh`)

Each run writes its products and `<scene>_L2_SR_report.html` to the output folder. Section 5 of the report
compares the product with the reference.

```bash
T=/data/atm_trial_scenes          # this folder on the VM
O=/data/atm_trial_out             # outputs
export DEM_DIR=$T/dem

# PRISMA L1 (Maharashtra coast near Mumbai, 19.3 N 72.8 E, 2026-05-14) vs ASI L2C
REF=$T/prisma/reference/PRS_L2C_STD_20260514055637_20260514055641_0001.he5 \
  ./scripts/run_scene.sh $T/prisma/input/PRS_L1_STD_OFFL_20260514055637_20260514055641_0001.he5 $O/prisma

# EnMAP L1C (Sundarbans coast, 2024-01-02, winter haze) vs DLR L2A
REF=$T/enmap/reference/ENMAP01-____L2A-DT0000056279_20240102T052511Z_001_V010502_20260127T164258Z \
  ./scripts/run_scene.sh $T/enmap/input/ENMAP01-____L1C-DT0000056279_20240102T052511Z_001_V010502_20260127T164258Z $O/enmap

# Tanager basic radiance (ROCX2025, Rochester, 2025-09-19, AERONET site in frame) vs Planet basic SR
REF=$T/tanager/reference/20250919_170238_88_4001_basic_sr_hdf5.h5 \
  ./scripts/run_scene.sh $T/tanager/input/20250919_170238_88_4001_basic_radiance_hdf5.h5 $O/tanager

# AVIRIS-NG L1 full flight line (5592 x 722 px, India, 2015-12-19) vs JPL L2 corr (no DEM needed: it carries loc_ort)
# Runs at superpixels of 200 by default (AVIRIS-NG rule, AC26-SEG). Colab estimate for this 4 Mpix line ~2.5 h,
# of which the per-pixel solve is ~70-90 min. On the VM's H200 + 16 cores it should be faster. Please time it.
REF=$T/aviris_ng/reference/ang20151219t081738_corr_v2m2_img.hdr \
  ./scripts/run_scene.sh $T/aviris_ng/input/ang20151219t081738 $O/aviris_ng
```

What to send back for each run:
- **Timing:** the last lines of the console log (`DONE`), and stage times from the report's section 1.
- **Report section 5:** MAE, bias and pixels compared.
- **Any `WARNING` / `ERROR` lines.**

## 3. Expected (Colab trials 2026-10-31 / 11-01, L4 GPU + 12 vCPU; the VM should be faster)

The report's MAE averages all written bands. It includes the noisy water-band shoulders and the > 2450 nm edge,
where vendor products differ most. The core-band figure (460–2450 nm outside the water bands, land) is the like-for-like
number. The final table is filled in from the corrected reports (AC26-T11).

| Sensor | Colab time | Report: all-band MAE | Core bands, land MAE (report pairing) |
|---|---|---|---|
| PRISMA | 33 min | 0.0125 | 0.0124 (research scorer, 165-band grid with shift search: 0.0097 raw / 0.0057 harmonised) |
| EnMAP | 33 min | 0.0162 | 0.0157 |
| Tanager | 44 min | 0.0126 | 0.0089 |
| AVIRIS-NG | 6.2 h at seg 50 (now seg 200: ~2.5 h est.) | 0.0136 | 0.0060 (pixel-index paired: 0.0051; seg 200 same, AC26-SEG) |

The report pairs by nearest location on native bands, over all land, with no sub-pixel shift search or cloud and
shadow exclusion. It is a sanity check on the VM: a run within about ±30% of these numbers is behaving as on Colab.
The accuracy figures for the submission are the research scorer's (EXPERIMENT_LOG AC26-*).

Notes:
- **CPU-only host:** add `GPUS=none`.
- **Faster turnaround:** `--no-report` skips the HTML.
- **Inspecting intermediates:** `--keep-work` keeps the work folder.
