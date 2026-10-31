# Atmospheric correction (L1 → L2): handover

Status at 2026-10-25 morning, for whoever picks this up next, human or agent.

**Start with [LEARNINGS.md](LEARNINGS.md)** (2026-10-27). It is the consolidated reference: results,
harmonisation choices, vendor fingerprints, band gaps, FLAASH and speed, and next steps.

**Latest: plan C is done (AC26-C45).** 45 training scenes ran through the frozen module (hash
75385785732aa4c9); results in EXPERIMENT_LOG and `results/planC_45_aggregate.md`.

| Sensor | Land MAE median |
|---|---|
| EnMAP vs PACO | 0.0146 |
| PRISMA vs ASI **L2C** | 0.0241 |
| AVIRIS-NG vs JPL | 0.0081 |

- Harmonisation chosen (AC26-H/H6, 2026-10-27, LOSO on the 45 training pairs; `scripts/harmonise.py`):
  - PRISMA: H6, land 0.0242 → 0.0128 with free AOT; 0.0118 with the A2 climatology prior (AC26-A2b).
    The A2 prior uses the offline grid `atm_corr/aod_clim_1deg.npz` (`scripts/aod_grid.py`).
  - EnMAP: H5, land 0.0119 → 0.0094.
  - AVIRIS-NG: identity.
  - PRISMA AOT prior: POWER climatology (A2) is recommended, but is not yet in the module.
- Next: user review of these choices, then the holdout evaluation.
- The holdout is reserved, untouched, until module and harmonisation are frozen:
  - EnMAP and PRISMA: set3;
  - AVIRIS-NG: 5 flights.

**Repo fixes in progress:** branch `vinoth/atm-isofit4`, worktree `C:/wt/atm-isofit4`, cut from
`origin/vinoth/acolite-engine`. It carries ISOFIT 4.1.5, the bug fixes, and the new modules
`sixs_parts`, `rfl_qc`, `adjacency`, `terrain` and `atm_post`. Not committed; the user transfers
files. Every run, with its version
and score, is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The code used is in [scripts/](scripts/).

## The goal

Our anomaly models were trained on vendor L2 reflectance:
- EnMAP: DLR PACO L2A, `Land_Mode`;
- PRISMA: ASI L2D;
- AVIRIS-NG: JPL v2 `corr`.

At deployment only L1 radiance arrives. The atmospheric-correction module has to reproduce each
vendor's L2 closely enough that the models see the same thing. **"Good" means close to the vendor
product, not close to ground truth.**

The trigger for this work was the stage-3 test: the same EnMAP scene through PACO and through our
v1.1.4 ISOFIT gave completely different anomaly maps.
- Root cause: our water stayed too bright in the NIR (uncorrected adjacency), so `NDWI > 0.4` caught
  **0.003%** of PACO's water pixels, against 69% with PACO's own product.
- The rivers therefore stayed in `keep_mask` and were scored as anomalies.

## Where things live

| What | Where |
|---|---|
| Module source (v1.1.4) | Allotrope branch `vinoth/acolite-engine`, folder `enmap-isofit-docker/`. Its own handover: `enmap-isofit-docker/atm_handover.md` |
| Experiment scripts, 2026-10-21..24 | this folder, `scripts/`; Drive copies in `MyDrive/PS_11/stage3_experiments/atm_corr/v6e_2026-10-23/scripts/` |
| Results (char/final logs) | Drive `PS_11/stage3_experiments/atm_corr/` → `colab_2026-10-21/`, `colab_char/` (rounds 1–2), `v6e_2026-10-23/{A,B,C,P,T6c}/`, `summary_18scenes.txt` |
| Round 5 (2026-10-24) | Drive `atm_corr/v6e_2026-10-24/`: `<arm>/` char logs for A, M1–M6b, P1, P2; `m6_split.txt`, `m7_split.txt`, `prs_l2c_atm.tsv`, `inventory_*.tsv`; `logs/` (every queue and setup log); `scripts_colab/` (the exact scripts as run, incl. the prototypes) |
| Plan C, 45 training scenes (AC26-C45) | Drive `atm_corr/planC/<sensor>_<id>/` {`pairs.npz` (20k paired px), `pairs.txt`, `run.log`, `atm_post.json`, `char_l2d.txt`}; `aggregate_c.md`; `_module/modC_75385785732aa4c9.tgz` (the frozen module as run). Local copy of the aggregate: `results/planC_45_aggregate.md` |
| Harmonisation, 2026-10-27..28 (AC26-H … HON) | Scripts: this folder, `scripts/` (incl. `export_native.py`, `score_native_holdout.py` and the run `.sh` files); Drive copy `atm_corr/scripts_2026-10-27/`. Module tests (run on Colab against a copy of the worktree at `/content/modW`): `scripts/harmonise_tests/`. Frozen models: Drive `atm_corr/final_models/` (+ `FROZEN_AT.txt`; native: `final_models/native/` with a package snapshot). Holdout scores: `atm_corr/holdout_scores_165grid.md`, `holdout_native_2026-10-28/`. Logs of the last Colab session: `atm_corr/colab_archive_2026-10-28/run_logs.tgz` |
| Holdout pairs (5 EnMAP, 5 PRISMA) | Drive `PS_11/stage3_datasets/set3_20260906/` (L1C/L1 + L2A/L2D) |
| EnMAP training pairs (60) | `PS_11/EnMAP_datasets/enmap_L1C/` + `.../stage_3/` (155 L2A). Some newer products use `*_COG.tiff` names |
| PRISMA training pairs (13 with L1 + L2C + L2D) | shared Drive folder `1XhTlJ-NSfjnZjwUIDWkzoEwI9PDo9PN2/PRISMA_Training_Set/<start-time>/`, mounted at `/content/drive/.shortcut-targets-by-id/<id>/`. 93 folders in total, most L2D-only. Older L2D-only set: `PS_11/Prisma Data sets/August 2026 /` (trailing space) |
| AVIRIS-NG pairs (20) | L1 `PS_11/aviris_datasets_L1/<fid>/` (`rdn_<ver>_{img,glt,obs_ort[,loc_ort]}`, ver ∈ v2m2/v2m3/v2s1/v2s2), L2 `PS_11/aviris_datasets/<fid>/*corr*`. 14 flights lack `loc_ort`; see Open items |

## The shipped recipe (what the repo fixes implement)

ISOFIT **4.1.5** with the public **sRTMnet v120 `.h5`** emulator, 6S v2.1:
- presolve, analytical line, segmentation 50, `--pressure_elevation`;
- season LUT presets as in v1.1.4.

| Step | EnMAP | PRISMA | AVIRIS-NG |
|---|---|---|---|
| Instrument noise | ISOFIT default (no AVIRIS files) | default | `avirisng_noise.txt` only |
| Aspect / cos i | **fixed**: downslope `atan2(-dz_dx, dz_dy)` | our preprocessor (already correct) | from `obs_ort` |
| AOT | free (LUT preset) | free | **pinned 0.10–0.14** (`ang_lut.json`, JPL's 50 km VIS) |
| Bands outside inversion windows | nodata (`mask_windows.py`) | nodata | nodata |
| Outlier pixels (grid band outside [−0.2, 2.5]) | nodata (`clip_outliers.py`) | nodata | nodata |
| Adjacency, post-hoc 1 km (`adjacency.py`) | **on** if median elevation < 2 km | off (ASI does none) | off (JPL does none) |
| Terrain, post-hoc PACO-style (`terrain.py`) | **on** if > 1% of pixels have slope > 6° | off (ASI does none) | off |

Result on 6 scenes per sensor, land MAE against the vendor product:

| Sensor | Range | Notes |
|---|---|---|
| EnMAP | 0.008–0.015 plains; 0.0285 Himalaya | v1.1.4 was 0.0126 on Ganges |
| PRISMA | 0.018–0.038 | limited by L1↔L2D misregistration and ASI quirks |
| AVIRIS-NG | 0.004–0.006 | 0.009 at low sun; 0.020 on 5 km snow |

### Is outlier clipping conclusive?

Yes, as a safety net, with this exact rule: a pixel becomes nodata if any band inside the 165-band
grid leaves [−0.2, 2.5].
- What it catches is pathological ISOFIT output (−8971 and +514 seen on Himalayan snow and shadow),
  which would otherwise score as anomalies.
- Rates on the 18 scenes: EnMAP about 0.09%, PRISMA ≤ 0.05%, AVIRIS-NG about 0.
- **Do not tighten it to [−0.05, 2] or test edge or absorption bands.** That version removed 35–80% of
  AVIRIS-NG pixels and up to 17% on PRISMA.
- It is a mask, not a correction. Report the count in the product metadata.

## Bugs found in v1.1.4

1. **Aspect points upslope** in `enmap_l1c_preprocess._slope_aspect_cosi`, so cos i is
   uncorrelated with the image. Only matters once anything uses cos i (terrain step, `.6c`), but it
   is wrong.
2. **The 1960/1970 nm grid bands read about 0.** ISOFIT only fits bands inside its inversion
   windows. Fix: write out-of-window bands as nodata so the band filter interpolates them.
3. **`--channelized_uncertainty_path avirisc_noise.txt`** is a fixed 224-channel file applied by
   channel index. It silently mis-weights EnMAP and crashes PRISMA. **`avirisng_noise.txt`** is the
   wrong sensor for EnMAP.
4. **Provenance says sRTMnet v100**, but the image ships v120.
5. ISOFIT occasionally returns wild values (see outlier clipping).
6. Outside the atmospheric-correction scope, for the pipeline owner: newer EnMAP products use
   `*-SPECTRAL_IMAGE_COG.tiff`, which the preprocessor does not find.

## What the vendors actually do (sources in EXPERIMENT_LOG and the research notes below)

- **PACO** (EN-PCV-TN-6007, Issue 2.5):
  - `Land_Mode` corrects land **and water** pixels and applies adjacency to all of them
    (R = 1 km).
  - Terrain correction with a modified-Minnaert BRDF.
  - AOT from dark dense vegetation, falling back to a fixed 23 km visibility, so its scene AOT is a
    weak target on desert and snow.
  - Rural aerosol model.
  - EOMAP's water processor is used only in `Combined` or `Water` mode.
- **ASI L2D:** MODTRAN LUT processor, AOT from dark vegetation, per-pixel water vapour, **no terrain
  correction, probably no adjacency** (our adjacency over-corrects against it). Reflectance collapses
  beyond 2350 nm (0.019–0.09 at 2450 nm on bright soils).
- **JPL AVIRIS-NG v2 `corr`:** ATREM, **fixed 50 km visibility**, no adjacency, no terrain.

## Open items

1. **High-mountain snow, all three sensors.** EnMAP Himalaya 0.0285, PRISMA Himalaya 0.038,
   AVIRIS-NG Himachal 0.020 (+0.13 at 460 nm).
   - Not aerosol: pinning AOT left it unchanged.
   - Not fixed by Lambertian + Minnaert + terrain-reflected correction; only the SWIR improves.
   - **Resolved 2026-10-24 (AC26-M1…M7): a vendor difference on snow, not our physics.**
     - On EnMAP the whole −0.2 blue tail is snow; PACO's snow is 4–12% brighter than ours in the
       VNIR.
     - On AVIRIS-NG it is JPL's known snow "hook".
     - AOT, sky view, elevation cap, cast shadow and ISOFIT `.6c` native terrain were each tested,
       and none helps.
     - Handled in harmonisation with a snow / non-snow class: the L1C `QL_QUALITY_SNOW` flag for
       EnMAP; flag the scene for AVIRIS-NG.
2. **PRISMA's two regimes.** On hazy or humid scenes ours is +0.03–0.055 brighter in the visible and
   −0.03–0.06 darker in the NIR than ASI.
   - **Measured 2026-10-24 (AC26-P1/P2):** ASI's L2C AOT runs 0.4–0.8 while ours can collapse to
     0.1. Pinning to ASI's per-scene AOT halves the visible bias. A constant prior gains only 0.0006
     MAE, so it is not adopted; AOT becomes a harmonisation covariate.
   - The NIR deficit does not respond to AOT. Leads are in `research_notes.md` §3b: Ångström
     exponent, SISTER flat-field and smile (experiment P3).
   - The training set has **33** L1 + L2C pairs, not 13. Only 13 of them have L2D.
3. **PRISMA beyond 2250 nm.** Done (AC26-S1). The gains are stable up to 2350 nm; ASI collapses
   beyond it. A leave-one-scene-out test validates per-band brightness-binned gains, which go into
   PRISMA harmonisation.
4. **DONE 2026-10-24: all 20 AVIRIS-NG flights now have `loc_ort`.** The 14 rebuilt ones carry a
   `LOC_ORT_DERIVED.txt` marker and agree with the UTM grid to within about 1.5 m on 4 m pixels.
   How it was done, for reference:
   - ORNL DAAC 2095 has the raw flight-line `<fid>_rdn_<ver>_loc.{hdr,bin}` (and `_igm`) under
     `https://data.ornldaac.earthdata.nasa.gov/protected/aviris/AVIRIS-NG_L1B_radiance/data/`.
   - `scripts/ang_loc_fetch.py` downloads it with an Earthdata bearer token read from
     `/content/edl_token.txt` and orthorectifies it through the `_glt`. The GLT holds 1-based
     sample/line indices, negative values mean nearest-neighbour fill, 0 means outside the swath.
   - It writes `_loc_ort` plus a `LOC_ORT_DERIVED.txt` marker.
   - Not run yet: the runtime died first. **Never commit the token.** The user supplies it each
     session.
5. **PRISMA L1↔L2D misregistration** of 15–22 px along track on some scenes caps correlation at
   about 0.85. Use the L2C (swath geometry) for training pairs instead of L2D.

## Next plan (agreed 2026-10-24)

1. **Repo fixes, atmospheric correction only.**
   - Scope: aspect/cos i, inversion-window mask, outlier clipping, noise files, ISOFIT 4.1.5
     upgrade (Dockerfile + `environment.yml`: Python 3.12, no TensorFlow), sRTMnet label, and the
     per-sensor adjacency/terrain/AOT rules, as new post-processing modules (`adjacency.py`,
     `terrain.py`) beside `postprocess_to_paco.py`.
   - Excluded: COG names and other pipeline work.
   - Where: `enmap-isofit-docker/` on a new branch cut from `vinoth/acolite-engine`, e.g.
     `vinoth/atm-isofit4`, worked in a separate git worktree so `feat/indradhanu-se-pretrain` is
     untouched.
   - Method: the iterative nano-chunking workflow (design → agreement → chunks of ≤ 20 lines → docs
     and LLD).
   - PRISMA needs a production preprocessor; `scripts/prisma_l1_preprocess.py` is the prototype.
2. **Experiments on the mountain snow scenes and PRISMA**, following the literature review.
3. **45-scene run, 15 per sensor, spread across terrains**, with the fixed recipe:
   - EnMAP: 5 holdout + 10 from the 60 pairs;
   - PRISMA: 5 holdout + 10 of the 13 training pairs;
   - AVIRIS-NG: 15 of the 20 flights, once the `loc_ort` files are built.

   On a 44-vCPU runtime at 3 × 14 cores: about 30 min per EnMAP or PRISMA scene, about 20 min per
   AVIRIS-NG subset. So 45 scenes is roughly 6–7 h.
4. **Sensor-level harmonisation.**
   - Per-band affine for EnMAP.
   - For PRISMA, a model conditioned on AOT/H2O, or spectrum-to-spectrum.
   - A light correction for AVIRIS-NG.
   - Train on the 45-scene outputs plus the remaining pairs; validate on the holdout scenes; track
     each version in EXPERIMENT_LOG.

## If the Colab session drops during the holdout (2026-10-27)

Everything needed is on Drive `PS_11/stage3_experiments/atm_corr/`:
- `scripts_2026-10-27/`: every script, `qh.txt`, `qx.txt`, `aod_clim.tsv`, `aod_clim_1deg.npz`;
- `final_models/`: the frozen models plus `FROZEN_AT.txt` with their sha256;
- `holdout/<sensor>_<id>/`: finished holdout scenes;
- `planC2/`, `planA2/`: the training runs.

To resume:
1. Run the fresh-runtime setup below (`setup_v6e.sh`, then `setup_c.sh`, which builds `/content/modC`
   and the symlink).
2. Copy `scripts_2026-10-27/*` to `/content/`, then `ln -s /content/modC /content/enmap-isofit-docker`.
   `prisma_l1_preprocess.py`, `char_prisma.py` and `pairs_l2d.py` import `prisma_l1_geometry` from
   that path. Without it every PRISMA run fails in about 2 min, and `queue.sh` still logs `END 0`,
   because each queue line ends with `rm` (2026-10-27: all 5 PRISMA holdout scenes). Check
   `grep "apply_oe exit" run.log`, not the queue status.
3. Restart with `nohup bash queue.sh qh.txt 3 > qh.log 2>&1 &`. `runh.sh` skips scenes whose results
   are already on Drive. Then run `queue.sh qx.txt 3`, which `runc.sh` also skips when done.
4. Do **not** refit the models: score with `harm_score.py final_models/<model>.npz …`.
5. PRISMA vs L2C, once the team's link arrives: `bash score_l2c_hold.sh <L2C dir>`, then
   `harm_score.py final_models/prisma_H6_A2.npz prisma atm_corr/holdout`.

## How to resume on a fresh Colab runtime

1. Add the runtime as an SSH alias (see memory `colab-ssh-workflow`). Upload `scripts/*` and
   `atm.tgz` (`git archive origin/vinoth/acolite-engine enmap-isofit-docker`) to `/content/`.
2. Run `bash /content/setup_v6e.sh`, about 30 min. It builds the ISOFIT 4.1.5 env, runs
   `isofit download all`, copies the holdout set with `pcopy.sh`, and stages the DEM.
3. For `.6c` experiments only: build CO2 6S into a separate `HOME`
   (`isofit download sixs --tag "v2.1.2+u-co2"`), patch `sRTMnet.py` (Drive copy) and download
   `20251206_6c_3layer_-1.6c`.
4. Process one scene with `bash run6.sh <en|pr|ang>:<scene> <ARM> [apply_oe args]`, then
   `ARM=<ARM> bash final6.sh <sensor:scene>`. Optional environment variables: `NC`, `EMU`, `LUTCFG`,
   `HOME`.
5. Summarise per sensor with `ARM=<ARM> python aggregate.py`.

**Plan C style runs (module code, training scenes) on a fresh runtime:**
1. Run `setup_v6e.sh` as above.
2. Copy the module into `/content/modC`: `enmap-isofit-docker/{entrypoint.sh,src,configs,scripts}`
   from branch `vinoth/atm-isofit4`, or untar `planC/_module/modC_*.tgz`.
3. **Do not leave the prototype `adjacency.py` or `terrain.py` in the directory that jobs run from**
   (`/content` for `queue.sh`). `python -m atm_post` would import them instead of the module. Put
   them in `/content/proto/`.
4. Stage scenes with `stage_c.sh`, which also adds the COG-name symlinks. Run
   `queue.sh <jobs file> 3` with `runc.sh` lines, then `aggregate_c.py <planC dir>`.
5. To stop a queue, find PIDs with `ps aux | grep "[q]ueue"` and kill them by number. Never run
   `pkill -f` over ssh: the pattern matches the remote shell and any chained waiter.
6. The holdout copy that `setup_v6e.sh` makes (`pcopy.sh`) is for the final evaluation only.
