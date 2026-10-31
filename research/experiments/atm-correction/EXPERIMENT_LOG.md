# Atmospheric correction: experiment log

Versioned record of every L1 → L2 run made on the EnMAP, PRISMA and AVIRIS-NG holdout and training pairs,
so a configuration can be rolled back to, reproduced, or mixed with another. The narrative and open
work are in [README.md](README.md).

Each experiment has an ID, `ACyy-NN`. That ID is what any later report or model card should cite.

## Conventions

- **Metric.** Every number compares our reflectance against the vendor's L2 product for the same
  acquisition, after both are resampled to the Allotrope 165-band common grid (460–2450 nm, 10 nm,
  absorption windows removed). The script is `scripts/char_*.py`.
  - `MAE` is the mean |ours − vendor| over land pixels and all grid bands.
  - Land and water come from the vendor's own class map. PRISMA has none, so there water is
    ρ860 < 0.05 and ρ560 > ρ860 on the ASI L2D side.
- **"Vendor"** means:
  - EnMAP: DLR L2A, PACO, `Land_Mode`;
  - PRISMA: ASI L2D;
  - AVIRIS-NG: JPL v2 `corr`.
- **Results on Drive** are under `MyDrive/PS_11/stage3_experiments/atm_corr/`. Each run directory
  holds `run.log` and `char*.txt`; `final.log` and `char_final.txt` hold the shipped-recipe
  product.
- **Software.**
  - ISOFIT 3.7.7 runs in the `enmap-isofit:1.1.4` environment: Python 3.10, TensorFlow 2.15.
  - ISOFIT 4.1.5 runs in a micromamba env: Python 3.12, PyTorch sRTMnet, GDAL 3.9.
  - The emulator is sRTMnet v120 (`.h5`) unless the run says otherwise.
  - 6S is v2.1. The `.6c` runs need the CO2 build, `v2.1.2+u-co2`.
- **Hardware.**
  - Rounds 1–2: Colab L4, 12 vCPU. One EnMAP scene took about 80 minutes.
  - Rounds 3–4: Colab TPU v6e-1 runtime, 44 vCPU. One EnMAP scene took about 30 minutes at 14 cores.

## Scenes

| Key | Sensor | Acquisition | Terrain |
|---|---|---|---|
| en_122280 | EnMAP | DT0000122280 2025-04-01, 25.1N 87.8E | Ganges floodplain + river |
| en_131004 | EnMAP | DT0000131004 2025-05-07, 18.0N | Hyderabad, peninsular |
| en_179439 | EnMAP | DT0000179439 2026-02-12 tile 032, 12.0N 79.8E | Puducherry coast (COG-named product) |
| en_187759 | EnMAP | DT0000187759 2026-04-12, 31.1N | Himalaya, 1.9–5.9 km, snow |
| en_189895 | EnMAP | DT0000189895 2026-04-14, 22.8N | Eastern India |
| en_194619 | EnMAP | DT0000194619 2026-05-25, 26.6N | Rajasthan desert |
| pr_20210305060117 | PRISMA | 2021-03-05, 27.2N 71.6E | Rajasthan desert |
| pr_20230620045859 | PRISMA | 2023-06-20, 21.8N 86.1E | Odisha forest / mining, humid |
| pr_20240511052012 | PRISMA | 2024-05-11, 18.9N 78.2E | Hyderabad plateau + reservoirs |
| pr_20240919061516 | PRISMA | 2024-09-19, 23.8N 69.3E | Kutch salt flats |
| pr_20260305054623 | PRISMA | 2026-03-05, 32.2N 76.5E | Himalaya |
| pr_20260927050402 | PRISMA | 2026-09-27, 16.6N 81.9E | Godavari delta, monsoon (training-set pair) |
| ang_20160128t054914 | AVIRIS-NG | 15.0N 77.4E, 4 m | Bellary / Hampi rocky, mining |
| ang_20160202t060649 | AVIRIS-NG | 24.3N 72.8E, 7.9 m | Aravalli hills |
| ang_20160203t060655 | AVIRIS-NG | 23.7N 74.3E, 4 m | Mahi reservoir / hills |
| ang_20160214t114143 | AVIRIS-NG | 22.6N 70.0E, 8 m, SZA 70° | Jamnagar coast, low sun |
| ang_20160217t082454 | AVIRIS-NG | 32.7N 77.2E, 3.8 m, ~5 km | Himachal / Spiti snow |
| ang_20160223t050929 | AVIRIS-NG | 25.6N 85.2E, 4 m | Ganges at Patna |

AVIRIS-NG runs use a 600-line, mid-strip window cut by `ang_prep.py`.

## Experiments

### AC26-00: baseline `enmap-isofit:1.1.4` (ISOFIT 3.7.7)

Run 2026-10-21 on the L4. Recipe: `entrypoint.sh` as shipped, with:
- presolve, analytical line;
- season LUT preset (`enmap_{summer,winter}.json`);
- segmentation 50, `--pressure_elevation`;
- `avirisng_noise.txt` as instrument noise **and** `avirisc_noise.txt` as channelized uncertainty;
- bad bands 1355–1450 and 1795–1970 nm.

PRISMA ran through the same apply_oe recipe via `oe_run4.sh` with no AVIRIS noise files, because the
224-channel `avirisc` file crashes on PRISMA's 210 bands.

| Scene | Land MAE | Notes |
|---|---|---|
| en_122280 | 0.0126 | VNIR gain 0.68–0.82 (adjacency); water NDWI 0.10 vs 0.48; 1960 nm 0.001 vs 0.173 |
| en_194619 | 0.0128 | AOT 0.52 vs PACO 0.31 |
| en_187759 | 0.0285 | p10 error −0.20 at 460 nm (shaded slopes) |
| pr_20210305 | 0.0292 | ASI collapses >2350 nm |
| pr_20240511 | 0.0351 | |
| pr_20260305 | 0.0386 | |

### AC26-01: ISOFIT 4.1.5, otherwise as AC26-00

Run 2026-10-21/22 on the L4.
- Same results as AC26-00 on Ganges and Himalaya.
- On the desert, AOT moves to **0.26** (PACO 0.31), helped by 4.1.x's lower AOT prior. MAE 0.0131.
- `--use_background_rfl` refuses to run with an `.h5` emulator. It needs multipart transmittance:
  MODTRAN or a `.6c` emulator.

### AC26-02: post-hoc adjacency (`adjacency.py`) on AC26-00/01

The correction is ρ = ρ_ret + q(ρ_ret − ρ_bg).
- q = t_up_dif / t_up_dir, from the 6S multipart table at the scene-median AOT, H2O and elevation.
- ρ_bg is an exponential-kernel mean over a 1 km range (0.5 km for airborne data).

| Scene | Result |
|---|---|
| en_122280 | land 0.0126 → 0.0115, water 0.0145 → **0.0078**, VNIR gain → 0.95 |
| en_194619 (AOT 0.52) | over-corrects, gain 1.28 |
| en_194619 on AC26-01 (AOT 0.26) | gain 0.93–1.05, r ≥ 0.99, leaves a near-constant per-band offset |
| en_187759 (AOT 0.38) | harmful |

### AC26-A: "fixed" recipe on ISOFIT 4.1.5 (shipped-recipe base)

Run 2026-10-23 on the v6e, arm `A` in `run6.sh`:
- no AVIRIS noise files; AVIRIS-NG gets `avirisng_noise.txt` only;
- the same LUT presets, presolve and analytical line;
- `mask_windows.py`: bands outside ISOFIT's inversion windows [350–1360], [1410–1800],
  [1970–2500] nm are written as nodata. This fixes the 1960 nm grid band: 0.198 vs PACO 0.173.

Raw results:

| Scene | Land MAE | Note |
|---|---|---|
| en_122280 | 0.0113 | SAM 5.3° → 3.1° |
| en_187759 | 0.0279 | |
| pr_20240511 | 0.0340 | |
| pr_20210305 | 0.0279 | |
| ang_20160128 | 0.0053 | |
| ang_20160203 | 0.0036 | |

Post-hoc adjacency on these runs:
- **en_122280: 0.0099**, water 0.0073.
- PRISMA and AVIRIS-NG over-correct (gain above 1), which confirms that ASI and JPL do no adjacency
  correction.

### AC26-B: `.6c` emulator with native adjacency + terrain

Run 2026-10-23 on the v6e.
- Emulator: `20251206_6c_3layer_-1.6c` (JPL experimental server).
- Flags: `--use_background_rfl --skyview_factor slope`.
- Needs the CO2 6S build and a one-line patch of ISOFIT 4.1.5 `sRTMnet.py`, where `self.points` is
  never set (patched copy: `scripts/../v6e_2026-10-23/scripts/sRTMnet_6c_points_patch.py` on Drive).
- `20260706_combined_model_random.6c` uses a newer layout that no released ISOFIT reads.

| Scene | Land MAE | Notes |
|---|---|---|
| en_122280 | 0.0116 | blue +0.026, best SAM 2.56°, but gain 0.86–0.90 and water unchanged |
| en_187759 | **0.0458** | worse. Ran with the upslope-aspect cos i bug, found later |

**Verdict:** not adopted.

### AC26-C: `.6c` + sky view, no native adjacency, then post-hoc adjacency

en_122280: 0.0116, against 0.0099 for A + post-hoc. Retrieved a surface height of 0.62 km on a
plain that sits at about 40 m. **Not adopted.**

### AC26-T: post-hoc terrain (`terrain.py`) on AC26-A, en_187759

PACO ATBD §5.8.2 method:
- Lambertian illumination: Tdir·cos β/cos θs + Tdif·V_sky, plus terrain-reflected light
  ρ_t·(1 − V_sky) with ρ_t a 0.5 km box mean;
- then the modified-Minnaert cap.

It needs the **aspect fix** (`fix_obs.py`): `enmap_l1c_preprocess` writes upslope aspect, so cos i
had r = −0.004 with radiance before the fix and +0.20 after.

Result: MAE 0.0285, about the same as raw, but the SWIR improves (1600 nm r 0.947 → 0.980). The
visible negative tail (−0.20) does not respond.

### AC26-F: final per-sensor recipe on 6 scenes per sensor

Run 2026-10-23/24 on the v6e. `run6.sh` arm A, then `final6.sh`:
- all sensors: `clip_outliers.py`, which sets a pixel to nodata if any grid band leaves [−0.2, 2.5];
- EnMAP:
  - adjacency, only if the median retrieved elevation is below 2 km;
  - terrain, only if more than 1% of pixels have slope > 6°;
- PRISMA and AVIRIS-NG: nothing more.

| EnMAP vs PACO | MAE | PRISMA vs ASI | MAE | AVIRIS-NG vs JPL | MAE |
|---|---|---|---|---|---|
| en_189895 | 0.0076 | pr_20240919 | 0.0183 | ang_20160203 | 0.0036 |
| en_122280 | 0.0091 | pr_20210305 | 0.0279 | ang_20160223 | 0.0040 |
| en_194619 | 0.0105 | pr_20230620 | 0.0279 | ang_20160128 | 0.0053 |
| en_131004 | 0.0147 | pr_20260927 | 0.0337 | ang_20160202 | 0.0061 |
| en_179439 | 0.0153 | pr_20240511 | 0.0339 | ang_20160214 | 0.0123 |
| en_187759 | 0.0285 | pr_20260305 | 0.0380 | ang_20160217 | 0.0204 |

Pixels removed as outliers: EnMAP about 0.09%, PRISMA ≤ 0.05%, AVIRIS-NG about 0.

Mean land bias (ours − vendor) and its standard deviation across the six scenes. The full table is
`summary_18scenes.txt` on Drive.

| nm | 460 | 560 | 860 | 1250 | 1960 | 2200 | 2400 | 2450 |
|---|---|---|---|---|---|---|---|---|
| EnMAP mean | +.023 | +.010 | −.005 | −.009 | +.029 | +.008 | +.006 | +.022 |
| EnMAP std | .008 | .004 | .005 | .006 | .024 | .009 | .013 | .026 |
| PRISMA mean | +.035 | +.022 | −.012 | −.028 | +.041 | +.011 | +.058 | +.099 |
| PRISMA std | .014 | .014 | .017 | .025 | .056 | .007 | .044 | .077 |
| AVIRIS mean | +.018 | −.002 | +.004 | +.016 | +.004 | +.001 | +.007 | +.016 |
| AVIRIS std | .052 | .013 | .010 | .013 | .010 | .001 | .005 | .011 |

### AC26-P: AVIRIS-NG with AOT pinned to JPL's assumption

Run 2026-10-24 on the v6e. `LUTCFG=ang_lut.json`: `aerosol_2_range` 0.10–0.14, matching ATREM's
fixed 50 km visibility.

| Scene | MAE |
|---|---|
| ang_20160214 | **0.0123 → 0.0087** |
| ang_20160217 | 0.0204 → 0.0208 (not aerosol) |
| other four | within ±0.0002 |

**Adopted for AVIRIS-NG.**

### Round 5 (2026-10-24, fresh v6e runtime): mountain-snow and PRISMA experiments

Results on Drive under `atm_corr/v6e_2026-10-24/<arm>/`. The harness gained overrides, all in
`scripts/`:
- `AOTPIN="lo hi"`: pins the LUT AOT range (`pin_lut.py`, derived from the season preset);
- `ELEVCAP=<m>`: caps the loc elevation in a copy of the input dir (`cap_elev.py`);
- `NOPE=' '`: drops `--pressure_elevation`;
- `ARM=` / `SVF=` in `final6.sh`.

**Reproducibility check.** Arm A on en_187759, rebuilt on the new runtime: final 0.0285, identical
to AC26-F.

#### AC26-M1 / AC26-M2: EnMAP Himalaya (en_187759), PACO-like aerosol and ray-traced sky view

| Arm | AOT | Land MAE | SAM | 460 nm bias [p10, p90] | 860 nm bias [p10, p90] |
|---|---|---|---|---|---|
| A (shipped recipe, terrain on) | 0.383 | **0.0285** | 3.37° | +0.016 [−0.203, +0.054] | +0.002 [−0.096, +0.098] |
| M1: `AOTPIN="0.05 0.20"`, `ELEVCAP=5500` (1.2% capped) | 0.158 | 0.0313 | 3.90° | +0.025 [−0.219, +0.063] | +0.002 [−0.107, +0.111] |
| M2: A + `isofit.utils.skyview` horizon, 64 angles (median V 0.86) | 0.383 | 0.0297 | 3.58° | +0.016 [−0.231, +0.055] | −0.001 [−0.119, +0.074] |

- **M1 refutes the research-notes hypothesis.** A PACO-like low AOT makes the blue negative tail
  deeper and the mean bias higher. The −0.2 tail is not caused by our higher AOT. **Not adopted.**
- **M2:** the ray-traced sky view lowers V, raises the terrain-reflected term and deepens the tail.
  cos²(slope/2), which is PACO's simpler option, stays. **Not adopted.**
- **Where the tail must come from instead:** shaded and steep pixels, where PACO's own terrain
  model brightens more than ours. A physics knob has not closed it, so it is handed to
  harmonisation and flagged as mountain terrain.

#### AC26-M5 / AC26-M6: EnMAP Himalaya, `.6c` native terrain and PACO's cast shadow

Suggested by the second literature pass (research_notes §3b).
- **M5:** AC26-B rerun with the aspect fixed. `20251206_6c_3layer_-1.6c`, CO2 6S, the patched
  `sRTMnet.py`, horizon SVF from M2 via `--skyview_factor <file>`, `--use_background_rfl`, and no
  post-hoc terrain (`NOTERRAIN=1`).
- **M6a:** the arm-A cube, with `terrain.py` setting the direct term to 0 in cast shadow
  (`SHADOW=`, from `isofit.utils.skyview.create_shadow_mask`, via `m6_shadow.py`).
- **M6b:** `ELEVCAP=5500` on its own, AOT free, plus M6a's terrain.
- `m6_split.py` compares ours − PACO at 460 nm pixel by pixel on the shared L1C grid, by illumination
  class.

| Arm | AOT / elev | MAE | Cast shadow (0.05%) | cos i < 0.3 (0.85%) | 0.3–0.6 (14%) | ≥ 0.6 (85%) | elev > 5.5 km (1.2%) |
|---|---|---|---|---|---|---|---|
| A | 0.38 / 3.21 | **0.0285** | −.168 (p10 −.33) | −.091 (−.41) | +.008 (−.14) | +.018 (**−.20**) | −.075 (−.25) |
| M1 (low AOT) | 0.16 / 3.07 | 0.0313 | −.095 | −.003 | +.033 | +.024 (−.22) | −.064 |
| M2 (horizon SVF) | 0.38 / 3.21 | 0.0297 | −.170 | −.097 | +.006 | +.018 (−.23) | −.072 |
| M6a (cast shadow) | 0.38 / 3.21 | 0.0286 | −.145 | −.091 | +.008 | +.018 (−.20) | −.075 |
| M6b (cap 5.5 km + M6a) | 0.38 / 3.21 | 0.0286 | −.145 | −.092 | +.008 | +.018 (−.21) | −.074 |
| M5 (`.6c` native) | 0.20 / **4.55** | 0.0328 | −.089 | −.105 | −.047 | −.029 (−.27) | −.079 |

Each cell is the median 460 nm bias, with p10 in brackets.

- **The negative blue tail is not a terrain-illumination effect.** It sits in the well-lit 85% of
  pixels (cos i ≥ 0.6, p10 −0.20), not in shadow or on steep slopes.
- With the sun at 24° zenith, cast shadow covers only 0.05% of pixels. PACO's shadow rule moves
  those from −0.168 to −0.145 and the scene MAE not at all.
- That explains why no terrain, sky-view or AOT change (M1, M2, M5, M6) closes it.
- The elevation cap on its own (M6b) changes nothing; the pixels above 5.5 km (1.2%) are −0.075.
- **M5:** native `.6c` terrain darkens everything by about 0.03 in the visible and again retrieves a
  surface elevation 1.3 km too high (4.55 km against 3.2; compare AC26-C: 0.62 km on a 40 m plain).
  **Not adopted; `.6c` is closed for us.**
- **None adopted.**
#### AC26-M7: the EnMAP Himalaya blue tail is PACO's snow, not illumination

`scripts/m7_split.py`, arm A, ours − PACO at 460 nm by PACO's own quality layers and brightness:

| Class | Fraction | Median (p10) |
|---|---|---|
| snow (QL_QUALITY_SNOW) | 42.2% | −0.066 (−0.250) |
| snow interior / edge (2 px) | 40.1% / 2.0% | −0.071 (−0.253) / +0.026 (−0.092) |
| clear non-snow | 56.6% | **+0.026 (+0.016)** |
| PACO ≥ 0.8 | 25.8% | **−0.132 (−0.283)** |
| QL_PIXELMASK ≠ 0 | 0.15% | +0.024 |

- **Every tail pixel is snow.** 16.7% of pixels have ours − PACO < −0.1, and **100% of them are
  snow**. PACO's median at 460 nm on them is **1.113**; ours is 0.936.
- **It is not saturation:** no L1C pixel sits at the 459 nm band maximum (65535; p99.9 is 62791).
- **It is not edges or misregistration:** the snow interior carries the tail; the edges don't.
- **PACO/ours on snow, using the L1C snow flag** (identical to the L2A flag on 100% of pixels, so it
  is available at deployment):

  | nm | 430 | 460 | 560 | 660 | 860 | 1000 | 1250 | 1600 | 2200 |
  |---|---|---|---|---|---|---|---|---|---|
  | snow | 1.12 | 1.08 | 1.07 | 1.03 | 1.04 | 0.94 | 0.95 | 1.02 | 0.91 |
  | non-snow (ρ > 0.05) | 0.51 | 0.61 | 0.84 | 0.94 | 0.97 | 1.01 | 1.00 | 1.00 | 0.98 |

  The interquartile range on snow is about ±0.1.
- **Interpretation:**
  - PACO's snow is 4–12% brighter than ours in the VNIR and darker in the 1000–1250 and 2200 nm ice
    absorption shoulders. That is a different snow spectral shape, plausibly from PACO's BRDF/terrain
    handling of bright targets.
  - Separately, on dark non-snow pixels PACO is far darker in the blue (0.5–0.6×): the familiar EnMAP
    blue offset (+0.023 mean, AC26-F).
- **Action: harmonisation, not physics.**
  - The EnMAP harmonisation (AC26-H) gets a two-class per-band model, snow (L1C `QL_QUALITY_SNOW`) vs
    non-snow, fitted over the mountain scenes of the 45-scene run.
  - The M-series has now ruled out AOT (M1), sky view (M2), elevation cap (M6b), cast shadow (M6a)
    and ISOFIT native terrain (M5).
- Hypotheses that M7 replaced (kept for the record): the tail is bright snow. 47% of pixels are "bright" by the scorer
  and PACO's median is 0.418 at 460 nm. Candidates:
  - VNIR saturation or PACO's saturated-pixel handling (ATBD: flag > 0.9·2¹⁴ DN at ~470 nm);
  - PACO's dehaze, active since v01.04.00 with snow masked and interpolated;
  - small misregistration at snow/rock edges.

  Split the residual by PACO brightness, by the L1C quality / saturation mask, and by distance to a
  snow edge.

#### AC26-M3: AVIRIS-NG Spiti, emulate JPL's 4 km LUT ceiling

`LUTCFG=ang_lut.json` (AOT 0.10–0.14) plus loc elevation capped at 4000 m, which affects 66.6% of
pixels.

| Arm | Retrieved AOT / H2O / elev | Land MAE | 460 nm bias | 660–860 nm bias |
|---|---|---|---|---|
| AC26-P (no cap) | | 0.0208 | | |
| M3 (cap + `--pressure_elevation`) | 0.135 / 0.21 / **4.63 km** | 0.0208 | +0.110 | −0.04 |

The cap did nothing: with `--pressure_elevation` ISOFIT still retrieved 4.63 km. A true test needs
the pressure retrieval off, which is arm M3b.

#### AC26-M4: PRISMA Himalaya (pr_20260305), AOT pinned low

`AOTPIN="0.05 0.20"`. The scene was already on the winter preset (0.02–0.30) and retrieved 0.17.

| Arm | AOT / H2O / elev | Land MAE | 460 | 560 | 860 | 1250 | 2450 |
|---|---|---|---|---|---|---|---|
| AC26-F | | 0.0380 | | | | | |
| M4 | 0.172 / 0.66 / 2.62 | 0.0382 | +.049 | +.031 | −.019 | −.030 | +.038 |

- **Pinning AOT low changes nothing.**
- ASI's 460 nm reflectance is 0.031 against our 0.082. ASI removes more path radiance than any AOT
  of 0.2 or below would. The adjacent training tile 20260305054602 has an all-fill L2C AOT map, so
  ASI's value is unknown here.
- Correlation is r ≈ 0.80–0.93 with a −2 to −7.6 px column shift: misregistration still caps this
  scene.

**Arm M4b: AOT pinned high (`AOTPIN="0.35 0.45"`, retrieved 0.43).** MAE 0.0382 → **0.0372**; the
blue bias goes from +0.049 to +0.034 at 460 nm and from +0.031 to +0.022 at 560 nm, and NIR/SWIR are
unchanged. The sign is confirmed: ASI assumes more aerosol than we retrieve, here as elsewhere (see
AC26-P1). The gain is small because misregistration dominates this scene.

**Arm M3b: AVIRIS-NG Spiti with `--pressure_elevation` off**, loc capped at 4 km, AOT 0.10–0.14:
MAE 0.0209, 460 nm +0.113, so nothing changes. Neither aerosol (AC26-P) nor pressure altitude
explains the blue excess. It is consistent with the "erroneous hook" in JPL's operational snow
reflectance (Bair et al. 2025). **Accepted as a vendor artifact**: flag snow and mountain AVIRIS-NG
scenes in harmonisation and do not tune physics toward it.

#### AC26-P1: PRISMA, AOT pinned to ASI's own L2C value

Three L1 + L2C + L2D training pairs, chosen for hazy scenes. AOT is pinned to ASI's L2C scene median
± 0.03, read with `prs_l2c_atm.py`. Everything else is the arm-A recipe.

| Scene | ASI AOT / WV | Ours, free: AOT / H2O | MAE A → P1 | 460 bias A → P1 | 660 | 860 | 1250 |
|---|---|---|---|---|---|---|---|
| 20260927050402 Godavari | 0.80 / 2.96 | 0.52 / 3.60 | 0.0337 → **0.0312** | +.055 → +.029 | +.040 → +.027 | −.036 → −.035 | −.059 → −.060 |
| 20251017055224 | 0.76 / 2.27 | **0.09** / 2.54 | 0.0354 → **0.0318** | +.055 → +.016 | +.028 → +.015 | −.022 → −.018 | −.046 → −.044 |
| 20250923053816 | 0.43 / 2.78 | **0.10** / 3.38 | 0.0222 → **0.0203** | +.035 → +.019 | +.021 → +.016 | −.010 → −.005 | −.025 → −.024 |

- **It works on all three.** Pinning to ASI's AOT removes half to two-thirds of the visible bias
  and lowers MAE by 0.002–0.004.
- Our free retrieval can land far below ASI (0.09 against 0.76), so the PRISMA regime split is
  largely an aerosol disagreement.
- **What P1 does not touch:**
  - the NIR deficit (−0.02 to −0.06 at 860–1250 nm);
  - the SWIR excess beyond 2350 nm, where ASI collapses to 0.006–0.035 at 2400–2450 nm (gain 2–10).
    S1 handles that.
- **Water vapour (P2 check):** our H2O runs 0.3–0.6 g/cm² above ASI's L2C median on all three, and
  is about the same with AOT pinned (3.49 / 2.46 / 3.33). That is near the 0.5 threshold. Do not
  bound presolve yet; the NIR deficit sits in window bands (860, 1250 nm), not absorption bands, so
  it is not a water-vapour effect.
- **Deployment cannot use L2C.** AC26-P2 tests a constant prior instead: the 33-pair median of
  ASI's AOT is 0.51, so `AOTPIN="0.45 0.57"` on these three plus three holdout scenes.

#### AC26-P2: PRISMA, one constant AOT prior (`AOTPIN="0.45 0.57"`, the 33-pair ASI median 0.51)

Arm A is also the first shipped-recipe run of the 5 PRISMA holdout scenes on this runtime. Its MAE
matches AC26-F to ±0.0001 on all five.

| Scene | Our free AOT | A MAE | P2 MAE | P1 MAE | 460 bias A → P2 | 860 | 1250 |
|---|---|---|---|---|---|---|---|
| 20210305 desert | 0.05 | **0.0279** | 0.0286 | | +.012 → .000 | +.006 → +.013 | −.003 → .000 |
| 20230620 Odisha humid | 0.67 | **0.0279** | 0.0284 | | +.040 → +.043 | −.031 → −.031 | −.063 → −.062 |
| 20240511 Hyderabad | 0.12 | 0.0339 | **0.0334** | | +.023 → +.012 | | |
| 20250923 (train) | 0.10 | 0.0222 | **0.0198** | 0.0203 | +.035 → +.015 | | |
| 20251017 (train) | 0.09 | 0.0354 | 0.0336 | **0.0318** | +.055 → +.036 | | |
| 20260927 Godavari (train) | 0.52 | 0.0337 | 0.0335 | **0.0312** | +.055 → +.052 | | |
| mean of 6 | | 0.0302 | 0.0296 | | | | |

- **The constant prior helps where our free AOT collapses below about 0.15:** 4 scenes, blue bias
  roughly halved. It hurts slightly where we already retrieve 0.5 or more, or over bright desert,
  where NIR goes from +0.006 to +0.013.
- **Net effect is −0.0006 MAE**, against −0.002 to −0.004 for the per-scene ASI value (P1). Most of
  P1's gain needs the scene's own aerosol, which L2C supplies only in training.
- **Not adopted as a physics rule.**
  - **Harmonisation:** include our retrieved AOT and ASI's AOT, where available, as covariates.
  - **Optional production variant, if visible fidelity matters more than MAE:** an AOT floor of
    0.35–0.4 in the PRISMA LUT config. It only affects scenes whose retrieval collapses.
- The NIR deficit at 860–1250 nm is unchanged by any AOT choice: see the P3 hypotheses in
  research_notes §3b (Ångström exponent, SISTER flat-field).

#### AC26-S1: PRISMA per-band gains beyond 2250 nm (ASI / ours)

`s1_swir_gain.py` on the arm-A runs of 8 PRISMA scenes (5 holdout + 3 training), using 5000 paired
land pixels per scene and binning by our reflectance.

> **Holdout overlap.** 5 of these 8 scenes are the PRISMA holdout:
> `PS_11/stage3_datasets/set3_20260906/` 20210305060117, 20230620045859, 20240511052012,
> 20240919061516, 20260305054623. The other 3 are training pairs from the shared
> `PRISMA_Training_Set`: 20250923053816, 20251017055224, 20260927050402.
>
> Agreed 2026-10-25: the holdout is exposed only once the module is frozen. **These gains are
> diagnostic only.** The harmonisation gains must be refitted on training pairs alone (plan C).
> The same applies to every round-1–5 number on holdout scenes:
> - EnMAP en_122280, 131004, 187759, 189895, 194619;
> - PRISMA as above.
>
> Those numbers shaped design decisions, but no fitted parameter from them may enter harmonisation.

| nm | ρ < 0.1 | 0.1–0.2 | 0.2–0.3 | ≥ 0.3 |
|---|---|---|---|---|
| 2260–2300 | 0.94–0.97 ± .055 | 0.98–0.99 ± .015 | 0.95–0.98 ± .017 | 0.96–0.97 ± .01 |
| 2320 | 0.81 ± .05 | 0.86 ± .02 | 0.85 ± .02 | 0.86 |
| 2350 | 0.77 ± .06 | 0.82 ± .04 | 0.82 ± .06 | 0.86 |
| 2360–2400 | 0.37–0.56 ± .08–.13 | 0.34–0.53 | 0.46–0.62 | 0.54–0.67 |
| 2410–2450 | 0.09–0.28 ± .13–.15 | 0.09–0.25 | 0.13–0.44 | 0.08–0.49 |

The ≥ 0.3 bin has few scenes, hence its ±0.000 entries.

- **Up to 2350 nm the gains are stable** across scenes (std ≤ 0.02 in the main bins) and close to
  1, apart from a consistent dip of about 0.86 at 2320 nm.
- **From 2360 nm ASI collapses** to 0.1–0.6 of ours, with a scene-to-scene std of 0.07–0.17.
- **Leave-one-scene-out** (gains pooled from the other 7), MAE over > 2250 nm land:
  - 20230620 0.032 → 0.004; 20240919 0.023 → 0.004; 20250923 0.043 → 0.007;
    20260927 0.032 → 0.008; 20251017 0.051 → 0.016; 20240511 0.078 → 0.024;
  - desert 20210305 0.087 → 0.064, because bright desert is not covered;
  - Himalaya 0.029 → 0.028.
- **Adopted for harmonisation, not physics.** The gains reproduce an ASI artifact (PRISCAV: PRISMA
  "marked drop in performance" > 2300 nm), so they belong in the PRISMA harmonisation layer
  (AC26-H), never in `atm_post`.
  - Fit them again on the 15 PRISMA scenes of the 45-scene run, with a separate bin for very bright
    soils.
  - Headline scores should also report MAE with ASI's > 2350 nm bands excluded.

### Plan C (2026-10-24/25): 45 training scenes through the frozen module, ID AC26-C45

- **Training folders only.** The holdout (set3 EnMAP/PRISMA and 5 untouched AVIRIS-NG flights) is
  exposed only after the module and harmonisation are frozen.
- **Module:** branch `vinoth/atm-isofit4` `enmap-isofit-docker/` (ISOFIT 4.1.5 + `atm_post`), copied
  to `/content/modC`. Content hash **75385785732aa4c9** (sha256 of entrypoint + src + configs), which
  every run log records.
- **How each sensor is run** (`scripts/runc.sh`):
  - EnMAP: the module's own `entrypoint.sh`, run natively.
  - PRISMA and AVIRIS-NG: the module's `apply_oe` recipe (`SENSOR_RULES` noise and LUT), then
    `python -m atm_post --sensor <s>`.
- **Training products are COG-named** (`*_COG.tiff`). `stage_c.sh` adds classic `*.TIF` symlinks;
  the preprocessor fix stays out of atmospheric-correction scope (README bug 6).
- **Scoring** (`scripts/pairs_sample.py`) compares on the same pixel grid as the vendor product:
  - EnMAP: PACO L2A;
  - PRISMA: **ASI L2C**, in swath geometry, with a residual integer shift search of ±4 px; L2D is
    additionally scored with `char_prisma.py` where it exists;
  - AVIRIS-NG: the JPL corr window.

  It reports land / land-without-snow / snow / water MAE, with and without > 2350 nm, and saves a
  20,000-pixel paired sample with class flags and scene covariates (AOT, H2O, elevation, SZA) for
  AC26-H.
- **Results:** Drive `PS_11/stage3_experiments/atm_corr/planC/<sensor>_<id>/`. Inventory:
  `scripts/inventory_c.py` → `atm_corr/v6e_2026-10-24/inventory_c.tsv`.

**EnMAP (15 of 60 pairs), datatake_tile:**
- 071060_016 Nilgiri / Western Ghats;
- 113132_003 Deccan plateau;
- 103656_016 Andhra coastal plain;
- 120451_017 Goa coast;
- 219322_006 Eastern Ghats;
- 056279_001 Sundarbans delta;
- 100387_018 Mizoram hills;
- 127112_004 Rann of Kutch salt;
- 128486_001 Thar desert;
- 072188_017 Punjab farmland;
- 075520_007 Brahmaputra floodplain;
- 209516_003 Arunachal Himalaya;
- 201686_001 Tibetan plateau, 5.2 km;
- 191215_002 Kashmir snow;
- 178884_010 Karakoram snow, SZA 48°.

**PRISMA (15 of 33 L1 + L2C pairs; 9 with L2D):**
- 20260204055049 Goa coast;
- 20250220053534 Western Ghats;
- 20250806051456 Tamil Nadu hills;
- 20250209054417 Karnataka, hazy;
- 20260518052710 Telangana;
- 20260927050402 Godavari, hazy;
- 20260514055637 Konkan coast;
- 20260604051851 Chhattisgarh forest;
- 20250510045541 Bengal delta, humid;
- 20251017055224 Gujarat, hazy;
- 20251113051858 MP plateau, dry;
- 20260509060824 Rajasthan desert;
- 20250807052709 Uttarakhand foothills;
- 20251115055012 Punjab, low sun;
- 20260426054314 Ladakh snow.

20260305054602 was excluded: it is the neighbouring tile of the PRISMA Himalaya holdout scene.

**AVIRIS-NG (15 of 20 flights; 600-line mid-strip windows):**
- the 6 development flights 20160128, 20160202, 20160203, 20160214, 20160217, 20160223;
- plus 20151227 Odisha coast, 20151228 Bengal coast, 20180227t084640 Godavari delta, 20180313
  Karnataka hills, 20180318 TN coast, 20180319 Kerala backwaters, 20180321 Kerala lowland, 20180331
  Aravalli, 20180429 Jharkhand.

**AVIRIS-NG holdout** (never run in any round; exposed only at the end):
- ang20151219t081738 Hyderabad;
- ang20180227t043325 Godavari coast;
- ang20180320t062955 Nilgiri, 1.7 km;
- ang20180326t062335 Gujarat;
- ang20180514t055115 Sundarbans.

**Run notes:**
- The first two EnMAP jobs (103656, 113132) failed at the end and were rerun:
  - `python -m atm_post` ran from `/content`, where the prototype `adjacency.py` and `terrain.py`
    shadowed the module (LLD failure mode). The prototypes were moved to `/content/proto/`.
  - `pairs_sample.py` assumed `to_grid` returns a 3-D cube.
- First clean result: en 071060 Nilgiri, land MAE **0.0082**.
- The PRISMA queue's chained waiter was killed by a `pkill -f` aimed at the queues (it matched the
  waiter's command line), so PRISMA started 3 h late (00:58 UTC 25 Oct) instead of at 21:58.

#### AC26-C45 results (finished 2026-10-25 08:08 UTC; all 45 scenes scored)

Full per-scene tables: [results/planC_45_aggregate.md](results/planC_45_aggregate.md) (also on
Drive, `planC/aggregate_c.md`). Paired samples (20,000 px per scene) are in each
`planC/<sensor>_<id>/pairs.npz`.

| Sensor (vendor) | Scenes | Land MAE median | Range | ≤ 2350 nm median |
|---|---|---|---|---|
| EnMAP (PACO L2A) | 15 | **0.0146** | 0.0058–0.0274 | 0.0145 |
| PRISMA (ASI **L2C**) | 15 | **0.0241** | 0.0094–0.0294 | 0.0171 |
| AVIRIS-NG (JPL corr) | 13 with land | **0.0081** | 0.0038–0.0207 | 0.0078 |

Two AVIRIS-NG windows are water only (20180227t084640, 58% water; 20180319, 87%). They have water
scores only: 0.030 and 0.008.

**Cross-scene bias, ours − vendor, land without snow (mean ± std over scenes):**

| nm | 460 | 560 | 660 | 860 | 1000 | 1250 | 1600 | 2200 | 2350 | 2400 | 2450 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EnMAP | +.023 ± .018 | +.016 ± .012 | +.009 ± .011 | −.003 ± .009 | −.003 | −.009 ± .009 | −.002 | +.001 | −.002 | −.006 | −.005 |
| PRISMA | +.035 ± .022 | +.023 ± .016 | +.021 ± .014 | −.009 ± .015 | −.021 ± .023 | −.025 ± .022 | −.005 | +.011 | +.020 | +.053 ± .033 | +.090 ± .057 |
| AVIRIS-NG | +.009 ± .028 | +.008 ± .017 | +.007 | +.004 | +.002 | +.004 | −.002 | +.002 | +.004 | +.009 | +.023 |

**EnMAP snow class** (5 scenes with flagged snow, incl. Rann salt flagged as snow by DLR):

| nm | 460 | 560 | 660 | 760 | 860 | 1000 | 1250 | 2200 |
|---|---|---|---|---|---|---|---|---|
| bias | +.005 ± .070 | −.021 | −.023 | −.038 | −.027 | +.021 | .000 | +.008 |

This confirms on training data the PACO snow-shape difference found on the holdout (AC26-M7).

**Findings for harmonisation (AC26-H):**
- **EnMAP:**
  - A consistent blue/green excess that falls off with wavelength. The scatter between scenes
    (±0.018 at 460 nm) is comparable to the mean, so a pure per-band offset will leave much of it.
    Expect a covariate model (AOT, SZA).
  - Snow needs its own class; its scatter is large (±0.07 at 460 nm).
  - Water is mildly high in the blue (+0.018).
- **PRISMA:**
  - **L2C is the right reference:** every pair aligned at shift (0, 0). The 9 L2D-equipped pairs
    score 0.011–0.035 against L2D versus 0.009–0.029 against L2C.
  - **The aerosol split shows up on training data.** Our free AOT is ≤ 0.10 on 8 of 15 pairs; these
    carry most of the visible excess and score 0.013–0.029. The AOT ≥ 0.26 pairs score 0.009–0.017,
    except the two humid ones (H2O > 3.2), at 0.029.
  - So PRISMA harmonisation needs AOT and H2O as covariates, the S1-style brightness-binned SWIR gains
    (> 2350 nm: +0.05 to +0.09), and the 1000–1250 nm deficit (−0.02 to −0.025).
- **AVIRIS-NG:**
  - Small mean bias, but the **2018 campaign scores worse**: 0.017–0.020 on 20180313/0318/0321,
    against 0.004–0.009 for the 2016 flights and 20180331/0429.
  - It does not follow the L1/L2 version cleanly (v2s1 vs v2s2). The two humid ones (H2O ≈ 4)
    suggest a water-vapour or campaign effect. Investigate before fitting.
  - Spiti snow: 0.0207, the JPL snow hook.

### Round 6 (2026-10-26): diagnostics on the C45 samples, no new ISOFIT runs (`scripts/harm_diag.py`)

#### AC26-D1: AVIRIS-NG 2016 vs 2018

- **JPL processing difference.** The `corr` headers of the 2016 products carry per-channel
  `correction factors` (0.96–1.02; Thompson 2015's final empirical per-channel step). **Every 2018
  product has all 1.0.**
- **But this does not explain the poor flights:** the good 2018 flights 0331 and 0429 also lack the
  factors and score 0.008.
- **The poor flights have per-flight broadband differences:**

  | Flight | Pattern (ours − JPL) | JPL/ours at 860 nm |
  |---|---|---|
  | 0313 | ours brighter everywhere: 460 +0.044, 860 +0.024, 1600 +0.013 | 0.90 |
  | 0318 | ours brighter everywhere: 460 +0.027, 860 +0.021, 1600 +0.013 | 0.84 |
  | 0321 | visible +0.057, NIR −0.02…−0.05 | 1.08 |
  | 2016 0214 (SZA 70°) | darker | 1.30 |
  | 2016 Spiti | snow hook | — |
  | other 2016 | — | 0.97–1.02 |

- **Not a campaign-wide effect.** Something scene-specific in JPL's processing (illumination, haze,
  or which radiance its L2 came from) differs per flight.
- **Harmonisation implication:** a per-flight gain is not deployable. Either accept it as noise
  or find an observable driver. Next: overlay the windows on the quicklooks for haze/cloud, and
  compare our radiance with JPL reflectance × our modelled TOA to see which side is off.

#### AC26-N1/N2/N3: PRISMA vs ASI L2C on the 15 C45 pairs

- **N1, per-band fit ASI = a·ours + b** (land, no water or snow):
  - a = 1.07 / 1.09 / 1.10 / 1.10 at 660 / 860 / 1000 / 1250 nm, scene sd 0.09–0.14;
  - b = −0.028 / −0.013 / −0.007 / −0.004.
  - **a and the visible b track ASI's own L2C AOT**: corr(a, AOT_ASI) = +0.67…+0.85 and
    corr(b460/660, AOT_ASI) = −0.90. They do **not** track our AOT (|r| ≤ 0.3) or SZA.
  - DOY is moderate (+0.25…+0.55).
  - So ASI's aerosol assumption drives both the visible offset and an NIR gain of about 1.1.
  - It is not deployable as is, because L2C does not exist at deployment. **Deployable covariate:
    an ASI-like DDV AOT computed by us** (SWIR-to-blue/red dark-vegetation relation on our
    reflectance or on L1).
- **N2, the 1250 nm peak:** ASI is 0.005–0.035 above its 1000–1200 / 1280–1320 nm neighbours on
  **all 15** pairs. This confirms ASI's 1250 nm artifact (Musacchio 2024); learn it per band.
  - The broad 1000–1200 nm deficit is scene-dependent: about 0 on 7 pairs, −0.02…−0.05 on the
    hazy and humid ones (consistent with N1).
- **N3, the Ångström-shape hypothesis: rejected.** After P1's AOT pinning the visible residual is
  flat across 460–660 nm (+0.029/+0.027, +0.016/+0.015, +0.019/+0.016), not steep. The leftover
  visible gap is an additive offset.

#### AC26-G0: band provenance on the 165-band grid

The common grid runs 460–2450 nm at 10 nm with four interior gaps (`BandFilterConfig` exclusions:
912–978, 1131–1152, 1350–1450, 1800–1950 nm). That leaves 10 gap-edge grid bands. The pipeline
(`app/utils/data_transformations/`) works like this:
- the band filter drops excluded bands and bands with < 20% valid pixels, so our nodata bands go;
- `interpolate_spectral_gaps` fills partially valid pixels;
- `resample_to_common_grid` interpolates linearly between the nearest kept native bands, with
  constant extrapolation only at the ends of the spectrum.

**Consequence:** a grid band beside a gap is interpolated *across* the gap whenever the native bands
next to it were dropped. The 1960 grid band is the clearest case. `atm_post` sets everything below
ISOFIT's 1970 nm window to nodata, so 1960 = about 0.92 × ρ(~1975) + 0.08 × ρ(~1795), and ρ(1795)
is much brighter.

Mean land bias (ours − vendor) on the gap-edge grid bands, C45:

| Sensor | 460 | 910 | 980 | 1130 | 1160 | 1340 | 1460 | 1790 | 1960 | 2450 | median \|bias\|, interior |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AVIRIS-NG | +.009 | +.003 | +.003 | −.005 | +.001 | **+.022** | −.004 | +.008 | −.010 | **+.023** | .003 |
| EnMAP | **+.023** | −.002 | −.010 | **−.029** | −.004 | −.002 | **−.022** | −.005 | +.005 | −.005 | .004 |
| PRISMA | **+.035** | −.010 | −.025 | **+.050** | +.004 | +.021 | +.003 | **+.035** | **+.038** | **+.090** | .012 |

- **Gap-edge bands carry 2–8× the interior bias.** The 1130 and 1340 bands sit on water-vapour
  shoulders. 1960 and 2450 are at the end of a window.
- AVIRIS-NG has no gaps beyond the shared absorption windows.
- **G1 (next):** in `atm_post`, fill the out-of-window bands within 20 nm of a window edge by constant
  extrapolation from the first in-window band, instead of nodata. That stops cross-gap
  interpolation.
  - It can be tested offline on the full-band cubes the class-flag rerun now saves.
  - If it helps, it is a module change (nano-chunk).
  - Harmonisation is per band in any case, so it will absorb the means; G1 targets the scatter.

#### AC26-A2: offline aerosol climatology as the AOT prior (setup and first comparison)

**Source.** NASA POWER, open API with no login: `power.larc.nasa.gov/api/temporal/climatology/point`,
parameters `AOD_55,AOD_84`.
- CERES SYN1deg, 1°, monthly climatology 2001–2025; actual monthly values up to 2025.
- `scripts/aod_clim.py` → `v6e_2026-10-24/aod_clim.tsv`, for all 45 C45 scenes.
- Air-gap path: a monthly 1° global grid of 12 × 360 × 180 values is a few MB. Fetch it once with the
  regional API or POWER's AWS data store; not yet scripted.

**Comparison** on the 8 PRISMA pairs with a valid ASI L2C AOT:

| Scene | Climatology AOD550 (month) | ASI L2C | Ours (free) |
|---|---|---|---|
| 20250209 | 0.27 | 0.64 | 0.29 |
| 20250220 | 0.40 | 0.44 | 0.26 |
| 20250806 | 0.50 | 0.36 | 0.07 |
| 20250807 | 0.59 | 0.72 | 0.07 |
| 20251017 | 0.43 | 0.76 | 0.09 |
| 20251113 | 0.40 | 0.75 | 0.15 |
| 20260204 | 0.32 | 0.15 | 0.29 |
| 20260927 | 0.58 | 0.80 | 0.52 |

- **Ours vs climatology:** r ≈ 0. Our free AOT collapses to 0.05–0.15 on 8 of 15 PRISMA scenes where
  the climatology is 0.40–0.74, so the collapse is **unphysical**.
- **Climatology vs ASI:** r ≈ 0.4. ASI is about 0.14 higher on average; its fallback constants are
  part of that.
- **Ångström exponent:** the climatology gives 0.25–0.96 everywhere, so ASI's `AEX_Map` ≈ 3.1 is not
  a physical aerosol property. Treat it as an internal parameter.
- EnMAP: our 0.20–0.47 is in line with the climatology (0.12–0.73). AVIRIS-NG is pinned at
  0.10–0.14 by design (JPL's 50 km visibility).

**A2 run, queued after the planC2 class rerun:** the 8 collapsed PRISMA scenes (20250510, 20250806,
20250807, 20251017, 20251113, 20251115, 20260509, 20260518), with AOT pinned to climatology ± 0.15
through `runc_a2.sh` (AOTPIN), into Drive `planA2/`. To compare against C45 (free) and P1 (ASI
pinned).

#### AC26-K: class-flag rerun (planC2) and per-class bias

- **What the rerun is.** All 45 C45 scenes re-run through the same frozen module, results in Drive
  `planC2/`. The sampler now also stores:
  - sample row/col and our own shadow flag;
  - EnMAP DLR QL classes / cloud / cloud shadow / haze / cirrus / snow;
  - PRISMA L1 `Cloud_Mask` / `LandCover_Mask` / `SunGlint_Mask`;
  - the full-band `rfl_int16.tif` (int16 × 10⁴, about 270 MB per EnMAP scene).
- **PRISMA failed on the first pass.** `prisma_l1_preprocess.py` imports `prisma_l1_geometry` from
  `/content/enmap-isofit-docker/src`, which `setup_c.sh` did not create. Fixed with a symlink to
  `/content/modC` (now in `setup_c.sh`) and requeued.

**EnMAP per-class bias** (`class_bias.py`, 15 scenes, scene-weighted median, ours − PACO):

| Class | Samples | Scenes | MAE | 460 | 560 | 660 | 760 | 860 | 1000 | 1250 | 2200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clear land | 209k | 15 | 0.0111 | +.024 | +.016 | +.010 | +.007 | −.003 | −.003 | −.009 | +.001 |
| water | 21k | 8 | 0.0110 | +.021 | +.001 | +.005 | +.004 | +.002 | .000 | −.005 | +.002 |
| **snow** | 23k | 4 | **0.0320** | +.022 | −.025 | −.033 | −.050 | −.047 | +.012 | −.008 | +.004 |
| **cloud** | 12k | 14 | **0.0378** | +.012 | +.008 | +.006 | −.002 | −.001 | +.008 | −.003 | −.001 |
| cirrus thin / medium | 10k / 13k | 6 / 4 | 0.010 / 0.014 | +.030 | +.019 | +.012 | | −.002 / −.009 | | −.009 / −.016 | |
| cirrus thick | 8k | 4 | 0.0168 | +.029 | +.017 | +.012 | −.002 | −.013 | −.011 | −.019 | −.003 |
| haze | 4k | 7 | 0.0129 | +.022 | +.014 | +.008 | +.004 | −.007 | −.008 | −.014 | .000 |
| cloud shadow | 0.6k | 3 | 0.0066 | +.018 | +.012 | +.005 | +.009 | +.005 | +.001 | −.003 | .000 |

- PACO's haze and cirrus removal leaves **no large systematic class gap**. Haze and thin/medium
  cirrus ≈ land; thick cirrus is slightly more NIR-negative.
- **Snow needs its own model:** distinct shape, confirming M7 on training data.
- **Cloud: small bias but large scatter**, so mask or flag it; do not harmonise.
- Water ≈ land except the blue.

**G1 on EnMAP:** at 1960 nm it is better on 7 scenes and worse on 4, by ≤ 0.006. Not worth a module
change for EnMAP. EnMAP's big edge errors are water-vapour shoulders (1130 nm: 0.03–0.10; 1460 nm:
0.01–0.06), not window edges. PRISMA is pending.

#### AC26-H0…H4 (v0): EnMAP harmonisation, leave-one-scene-out (`harmonise.py`)

Median over 15 held-out scenes (mean in brackets), cloud excluded:

| Model | Land MAE | Land ≤ 2350 | Snow (4 sc.) | Water (7 sc.) | \|bias460\| |
|---|---|---|---|---|---|
| H0 identity | 0.0119 (0.0116) | 0.0115 | 0.0296 | 0.0124 | 0.019 |
| H1 per-band affine | 0.0118 (0.0106) | 0.0120 | 0.0281 | 0.0135 | 0.015 |
| H2 + class groups | 0.0120 (0.0103) | 0.0121 | 0.0274 | 0.0130 | 0.016 |
| H3 + covariates (ridge 1e-3) | 0.0105 (0.0105) | 0.0108 | 0.0349 | 0.0140 | 0.016 |
| H4 spectral ridge (1e-5) | **0.0098 (0.0094)** | **0.0095** | 0.0498 | 0.0174 | 0.014 |

- **Land:** modest gain, −18% at best (H4).
- **Snow and water get worse.** Snow has only 4 scenes, so the held-out tests fit on 3 and overfit.
  Water is already close to PACO.
- Some scenes degrade: Mizoram 0.0126 → 0.018–0.022; Tibet 0.0119 → 0.025 under H3.
- So most of EnMAP's remaining error is **scene-specific**, not systematic.
- Next: choose per class (land H4 or H1, water and snow identity) and re-score; tune ridge by nested
  CV.

**AVIRIS-NG** (ridge 1e-5; 13 flights with land, 2 water-only):

| Model | Land MAE, median (mean) |
|---|---|
| H0 | **0.0081 (0.0093)** |
| H1 | 0.0088 (0.0103) |
| H2 | 0.0071 (0.0100) |
| H3 | 0.0111 (0.0215) |
| H4 / H5 | 0.0129 (0.0230) |

- H3/H4 **blow up on Spiti** (0.0087 → 0.15–0.18). Held out, its elevation (4.76 km) and H2O (0.20)
  lie outside every training flight's range, so the covariate terms extrapolate.
- **Decision: AVIRIS-NG gets no harmonisation (identity).** Its residuals are per-flight (AC26-D1);
  there is no systematic term to learn.
- **Rule for every covariate model:** clamp z to the training range, and fall back to a weaker model
  outside it.
  - Implemented in `harmonise.py` (per-fold `zrange`).
  - With clamping, AVIRIS-NG H3 is 0.0090 (mean 0.0102, was 0.0215) and H4 0.0111 (0.0106). Neither
    beats H0 on the mean, so **identity stands for AVIRIS-NG**.

#### PRISMA, all 15 planC2 reruns (finished 2026-10-26 21:00 UTC)

Outputs were copied to Drive `atm_corr/`: `cb_pr15.txt`, `g1_pr15.txt`, `cb_a2.txt`,
`cb_a2c.txt`, `h_pr.txt`. Run with `/content/mamba/envs/isofit4/bin/python`, because the system
Python has no rasterio.

**AC26-K, PRISMA per class** (ASI L1 land-cover classes, ours − ASI):

| Class | Scenes | MAE | 460 | 660 | 860 | 1000 | 1250 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|
| forest | 13 | 0.0237 | +0.038 | +0.024 | −0.014 | −0.030 | −0.032 | +0.009 | +0.044 |
| bare | 15 | 0.0227 | +0.035 | +0.021 | −0.007 | −0.016 | −0.021 | +0.011 | +0.054 |
| crop | 14 | 0.0253 | +0.037 | +0.018 | −0.010 | −0.020 | −0.027 | +0.010 | +0.082 |
| urban | 11 | 0.0132 | +0.031 | +0.016 | +0.002 | −0.003 | −0.010 | +0.011 | +0.054 |
| water | 3 | 0.0065 | −0.036 | −0.004 | −0.000 | −0.005 | −0.004 | +0.002 | +0.004 |
| snow | 1 | 0.0160 | +0.025 | +0.011 | +0.028 | +0.051 | +0.008 | +0.006 | +0.003 |
| cloud | 5 | 0.0365 | +0.026 | +0.009 | −0.003 | −0.006 | −0.024 | +0.023 | +0.139 |

- The land classes share one bias shape: visible high, 1000–1250 nm low, 2400 nm high. One land group
  is enough, so per-cover models are not needed.
- Water has the opposite sign in the blue: ASI is brighter there.

**AC26-G1, PRISMA, 15 scenes.** G1 changes only the 1960 nm gap-edge band. It improves 11 of 15
scenes and worsens 4. Mean |bias| goes from 0.041 to 0.037.
- **Decision: no module change.** The gain is on one band and is smaller than what H3 removes anyway.

**AC26-A2, climatology-pinned AOT vs free AOT** (the same 8 PRISMA scenes, land):

| Run | Land MAE | 460 | 560 | 860 | 1250 | 2400 |
|---|---|---|---|---|---|---|
| C45 free AOT (planC2) | 0.0244 | +0.037 | +0.022 | −0.011 | −0.026 | +0.073 |
| A2 POWER climatology | 0.0233 | +0.024 | +0.017 | −0.009 | −0.026 | +0.070 |

- The climatology prior removes about a third of the visible offset. NIR and SWIR do not change.
- So the 1000–1250 nm deficit and the 2400 nm excess are not aerosol. They match ASI's own artefacts
  (AC26-N1/N2 and the collapse beyond 2350 nm).
- **A2 is worth adopting as the PRISMA AOT prior.** Its gain is small but physical, and the
  climatology is air-gap friendly. It is not yet a module change.

**AC26-H, PRISMA harmonisation** (LOSO, 15 scenes, clim covariate, clamped, ridge 1e-3; median (mean)):

| Model | Land MAE | Land ≤2350 | Snow (1) | Water (2) | \|bias460\| |
|---|---|---|---|---|---|
| H0 | 0.0242 (0.0215) | 0.0182 | 0.0160 | 0.0084 | 0.045 |
| H1 | 0.0169 (0.0165) | 0.0163 | 0.0370 | 0.0150 | 0.024 |
| H2 | 0.0147 (0.0161) | 0.0155 | 0.0384 | 0.0107 | 0.023 |
| **H3** | **0.0128 (0.0138)** | **0.0134** | 0.0300 | 0.0136 | 0.019 |
| H4 | 0.0144 (0.0170) | 0.0149 | 0.0579 | 0.0116 | 0.015 |
| H5 | 0.0144 (0.0170) | 0.0149 | 0.0160 | 0.0084 | 0.015 |

- **PRISMA gains much more than EnMAP:** land MAE falls 47% (median), against 18% for EnMAP.
  PRISMA's residual is systematic, ASI-specific and covariate-dependent.
- **H3 beats H4.** At 15 scenes, the 165×165 spectral ridge overfits.
- H3 improves 11 of 15 scenes. It worsens the three already-close scenes (0.009–0.013) slightly, and
  worsens the high-Himalaya scene **20260426054314** (32.5° N, 77.5° E) badly: 0.0216 → 0.037. That
  scene is the only one of its kind, so it is out of domain even with clamping.
- Ridge 1e-5 is worse on that scene (0.062) and no better elsewhere, so **1e-3 stays**.
- **Proposed PRISMA model (H6):** H3 for land; identity for water, snow and cloud; identity for any
  scene whose elevation or H2O lies outside the training range.

**AC26-H6, out-of-domain gate** (`GATE` = the margin, in standard deviations, beyond the training
scenes' covariate range; outputs `h6_*.txt` on Drive):

| Sensor, ridge | Model | Land MAE, median (mean) | Snow | Water |
|---|---|---|---|---|
| PRISMA 1e-3 | H3 | 0.0128 (0.0138) | 0.0300 | 0.0136 |
| PRISMA 1e-3 | H6, gate 0.5 | 0.0129 (0.0138) | 0.0160 | 0.0084 |
| PRISMA 1e-3 | **H6, gate 1.0** | **0.0128 (0.0128)** | 0.0160 | 0.0084 |
| EnMAP 1e-3 | H6, gate 1.0 | 0.0105 (0.0104) | 0.0296 | 0.0124 |
| EnMAP 1e-5 | H4 | 0.0094 (0.0093) | 0.0418 | 0.0174 |
| EnMAP 1e-5 | **H5** | **0.0094 (0.0093)** | 0.0296 | 0.0124 |

- **PRISMA, gate 1.0:** it gates only the high-Himalaya scene, which goes back to identity (0.0216
  instead of 0.0368). Every other scene keeps H3.
- **PRISMA, gate 0.5:** it also gates 20251115 (good: 0.0182 → 0.0129) and 20260927 (bad: 0.0091 →
  0.0289), so it is too tight.
- **Harmonisation choice, all with clamping (frozen before the holdout):**
  - **PRISMA: H6** (per-band covariate H3 on land, identity elsewhere, gate 1.0, ridge 1e-3, clim
    covariate). Land 0.0242 → 0.0128 median.
  - **EnMAP: H5** (spectral ridge on land, identity on snow and water, ridge 1e-5). Land 0.0119 →
    0.0094 median, and snow and water are unchanged.
  - **AVIRIS-NG: identity.**

#### AC26-A2b: all 15 PRISMA training scenes with the climatology prior, then harmonisation (2026-10-27)

The 7 missing A2 runs were queued in `qa3.txt` (AOT window = POWER month clim55 ± 0.15). All 15 are in
Drive `planA2/`; outputs `cb_a2_15.txt` and `h_a2.txt`.

**Raw, land, 15 scenes:**

| Run | MAE | 460 | 560 | 1250 | 2400 |
|---|---|---|---|---|---|
| free AOT | 0.0215 | +0.037 | +0.025 | −0.020 | +0.052 |
| A2 | 0.0204 | +0.025 | +0.020 | −0.020 | +0.051 |

- Water: 0.0065 → 0.0069. ASI's water stays brighter in the blue under either prior.

**Harmonised (LOSO, ridge 1e-3, gate 1.0), land MAE median (mean):**

| Model | Free AOT | A2 |
|---|---|---|
| H0 | 0.0242 (0.0215) | 0.0220 (0.0205) |
| H3 | 0.0128 (0.0138) | **0.0118** (0.0130) |
| H6 | 0.0128 (0.0128) | **0.0118 (0.0127)** |

- **A2 + H6 is the best PRISMA chain:** 0.0242 → 0.0118, −51% against the current free-AOT module.
- **Gate:** the high-Himalaya scene is again gated to identity (0.0210).
- **Where H6 still hurts:** 20251115, 20260204 and 20260514 are already close (≤0.013) and get
  slightly worse.
- **Recommendation:**
  - Adopt A2 in the module for PRISMA, with the AOT window = climatology ± 0.15.
  - Train the PRISMA H6 on A2 outputs.

**Offline grid (`scripts/aod_grid.py`, `atm_corr/aod_clim_1deg.npz`, 255 KB compressed):**
- POWER regional climatology, global, 1°, monthly, AOD 550 and 840 nm; no missing cells.
- **Check:** the grid cell equals the point-API value on all 15 PRISMA scenes.
- **Month value vs interpolated prior:**
  - `prior()` interpolates linearly between monthly means. It can differ from the plain month value
    by up to 0.13 in fast-changing months (20260604: month 0.77, interpolated 0.64 — the early-June
    monsoon switch).
  - The A2 runs and the harmonisation above used the **month value**, so deployment must use the
    same lookup.
  - Use the month value unless the harmonisation is retrained on interpolated-prior runs.

### Round 7 (2026-10-27): frozen models, holdout launch, gap experiments

#### AC26-FZ: final harmonisation models, frozen before any holdout result exists

The models were fitted on all training scenes (`harmonise.py --save`, `FINAL=<kind>`). They are in
Drive `atm_corr/final_models/` with `FROZEN_AT.txt` (09:2x UTC) and these sha256 hashes:

| File | Model | sha256 |
|---|---|---|
| `prisma_H6_A2.npz` | H6 on planA2 (15 scenes), ridge 1e-3, gate 1.0, clim covariate | 6e2d42cb…e99e66 |
| `enmap_H5.npz` | H5 on planC2 (15 scenes), ridge 1e-5 | 7723c91c…f2250cd |

- AVIRIS-NG is identity, so it has no model file.
- **Sanity check:** `harm_score.py` applied in-sample to planA2 PRISMA gives land 0.0220 → 0.0079,
  against 0.0118 LOSO. The in-sample figure is optimistic, as expected; this only checks that the
  scorer reproduces `harmonise.predict`.

#### AC26-HO: holdout runs (launched 09:27 UTC, queue `qh.txt`, 3 parallel)

- **Scenes:**
  - EnMAP set3: 122280, 131004, 187759, 189895, 194619;
  - PRISMA set3: 20210305060117, 20230620045859, 20240511052012, 20240919061516, 20260305054623;
  - AVIRIS-NG: the 5 untouched flights.
- **Scripts:** `scripts/stage_h.sh` + `scripts/runh.sh`. These are the frozen module (hash 75385785732aa4c9)
  plus the PRISMA A2 window from `aod_clim_1deg.npz` (`aod_grid.py month`, exactly the rule that was
  trained). Results go to Drive `atm_corr/holdout/`.
- **PRISMA reference.** The set3 holdout has only L2D, so `pairs_l2d.py` scores against L2D
  (co-registered with a per-tile shift field; slightly pessimistic). The team has the L2C for the 5
  scenes. When it arrives, `scripts/score_l2c_hold.sh <dir>` scores against L2C without rerunning ISOFIT,
  because `runh.sh` keeps rfl / loc / obs in `holdout/pr_<id>/cube/`.
- **Scoring:** `harm_score.py <model> <sensor> <holdout dir> [--glob pairs_l2d.npz]`. It applies the frozen
  model only; there is no fitting.

#### AC26-SN: NDSI snow mask on our own reflectance (deployable; `scripts/ndsi_test.py`)

- **Rule:** NDSI(560, 1600) > 0.4 and ρ860 > 0.11 (excludes water) and ρ560 > 0.10. Scored on the planC2
  samples, with cloud and water excluded.
- Quicklooks are in Drive `atm_corr/ndsi/*.png`.

| Sensor / scene | NDSI snow | Vendor snow | Precision | Recall | Land MAE all → outside NDSI snow | Inside |
|---|---|---|---|---|---|---|
| EnMAP 178884 Karakoram | 42.2% | 41.0% (DLR) | 0.93 | 0.96 | 0.0281 → 0.0166 | 0.0438 |
| EnMAP 191215 Kashmir | 45.7% | 48.1% | 0.99 | 0.94 | 0.0176 → 0.0124 | 0.0238 |
| EnMAP 201686 Tibet | 12.5% | 12.7% | 1.00 | 0.99 | 0.0141 → 0.0120 | 0.0286 |
| EnMAP 127112 Rann (salt) | 20.6% | 22.3% | 1.00 | 0.92 | 0.0146 → 0.0101 | 0.0321 |
| PRISMA 20260426 Ladakh | 99.0% | 90.2% (ASI L1) | 0.91 | 1.00 | 0.0165 → 0.0103 | 0.0166 |
| AVIRIS-NG Spiti 20160217 | 98.8% | 98.6% (NDSI on JPL) | 1.00 | 1.00 | 0.0207 → 0.0091 | 0.0209 |

- **It matches the vendors' own snow flags** with precision 0.91–1.00 and recall 0.92–1.00. The
  quicklooks confirm it covers snow slopes, including shaded snow.
- **Masking snow lowers land MAE** on every snow scene, by 15–55%. The snow pixels themselves carry
  0.017–0.044.
- **Salt is flagged as snow** (Rann of Kutch), exactly as DLR's own flag does it.
- **False alarms** on low-lying scenes (< 1 km) at ρ560 > 0.10: up to 1.1%. They are surf and foam on
  the Odisha beach, a turbid river edge on the Godavari, and bright roofs.
- **Brightness threshold sweep** (lowland false alarms vs snow recall):

  | ρ560 > | Max lowland false alarm | Recall |
  |---|---|---|
  | 0.10 | 1.1% | 0.94–1.00 |
  | 0.20 | 0.23% | 0.92–1.00 |
  | 0.25 | 0.07% | 0.87–1.00 |
  | 0.30 | 0.01% | 0.83–1.00 |

- **Recommended rule:** ρ560 > 0.20, plus pixel DEM > 1000 m if salt must be separated from snow.
  This becomes the AVIRIS-NG snow flag (JPL ships none) and a cross-check for the others.

#### Gap scans (2026-10-27)

**EnMAP, more water.** The inventory (`v6e_2026-10-24/inventory_c.tsv`, DLR QL water fraction over the
60 pairs, holdout excluded) has these unused water-rich tiles:
- 098959_017 Chilika lagoon, 71%;
- 071630_017 Kerala coast, 63%;
- 179513_002 Gulf of Kutch, 30%;
- 179439_032 Puducherry coast, 29%;
- 067061_017 Sundarbans, 28%;
- 198820_002 Mansarovar / Rakshastal lakes at 4.6 km, 13%: high-altitude water.

All six are queued (`qx.txt`, planC2, frozen module) after the holdout. They are for a v2 EnMAP model
with a water group.

**PRISMA, high altitude.** `scripts/prs_alt_scan.py` → Drive `prs_alt_scan.tsv`. It covers all 33 L1
in PRISMA_Training_Set; every one has an L2C. Two besides the training Ladakh scene are above 1 km:
- **20260111053240** Ladakh / Pangong, 4957 m median, cloud 21%, snow 8%, L2C + L2D. Unused, so it is
  queued in `qx.txt` (planA2, climatology window 0.05–0.20).
- **20260305054602**, 4559 m. This is the adjacent tile of holdout 20260305054623 (same datatake, 21 s
  apart), so it **stays out** under the holdout rule.

The August-2026 L2D folder (119 files) adds no L1. Only 2 of its L2Ds match a training L1:
20250807052644 and 20260208051738. So there is no further PRISMA training data there.

- **Consequence for the PRISMA gate:** even with 20260111, high-altitude PRISMA training has only 2
  scenes. Gating the Himalaya to identity remains the safe default until more L1 + L2C pairs exist.

#### AC26-NBW: vendor values outside the core bands, and how well we reproduce them

**Setup:**
- Scripts: `scripts/pairs_native.py` (native-band samples at the pairs.npz pixels) and
  `scripts/window_char.py`.
- Training scenes only: EnMAP planC2, 21 scenes including the 6 water scenes; PRISMA planA2, 16 scenes
  including 20260111. Land pixels, cloud excluded.
- Four LOSO predictors per vendor band:
  - interp: our neighbouring core bands;
  - ridge: all our core bands plus z;
  - const: the vendor median of the other scenes;
  - own+z: gain/offset on our own band plus z (the H3 form).
- Reports on Drive: `atm_corr/window_char_{enmap,prisma}.md`.
- Core reference after harmonisation: PRISMA 0.0118, EnMAP 0.0094.

Group medians (MAE; best in bold):

| sensor | group | bands | interp | ridge | const | own+z |
|---|---|---|---|---|---|---|
| PRISMA | edge_lo <460 | 7 | 0.029 | 0.034 | 0.094 | **0.024** |
| PRISMA | weak | 13 | 0.037 | **0.028** | 0.095 | 0.031 |
| PRISMA | strong | 26 | 0.173 | 0.017 | **0.012** | 0.018 (4 bands) |
| PRISMA | edge_hi >2450 | 7 | 0.092 | **0.027** | 0.026 | 0.041 |
| EnMAP | edge_lo | 9 | 0.034 | **0.026** | 0.073 | 0.026 |
| EnMAP | weak | 16 | 0.021 | **0.016** | 0.117 | 0.019 |
| EnMAP | strong (1449 only) | 1 | 0.040 | 0.037 | 0.104 | **0.029** |

**Findings:**
- **ASI fills every L2C band and flags none off.** It does *not* interpolate across the strong windows
  (self-interp error 0.09–0.20): it reports near-total absorption, median 0.002. Our ISOFIT keeps only
  1417–1449 nm there, and those bands track ASI poorly (r 0.0–0.5, gain 0.0–0.2).
- **PACO leaves 1367–1390 nm as nodata in every scene.** EnMAP has no bands in 1800–1950 nm, and 1355 nm
  is ours only. At 1449 nm PACO has a real surface value (median 0.16), best fitted on our own band.
- **PRISMA strong windows, per class:** a near-zero constant wins everywhere except snow, which has less
  water vapour above it. Snow rests on only 2 scenes.

  | class | const | ridge |
  |---|---|---|
  | forest | 0.0025 | 0.006 |
  | crop | 0.0023 | 0.009 |
  | bare | 0.0026 | 0.012 |
  | urban | 0.0029 | 0.009 |
  | water | 0.0024 | 0.015 |
  | **snow** | **0.069** | **0.030** |

- **Weak windows:**
  - PRISMA: the residual follows H2O (r up to +0.77) and elevation (down to −0.62), i.e. ISOFIT and ASI
    disagree on water vapour. At 1131–1142 nm the error stays at 0.043–0.057 whatever is tried.
  - EnMAP: the residual follows elevation only (−0.3 to −0.84); gain is about 1.0.
- **Blue edge (<460 nm):** we read about 0.02 high against both vendors (ref ≈ ours − 0.02), the known
  blue offset. own+z or ridge removes about a fifth of it.
- **PRISMA above 2470 nm:** our bands thin out (valid 36–74%, r ≤ 0.35), while ASI is smooth there. From
  2477 nm a constant beats everything.
- **Implementation:**
  - Our PRISMA cubes have 197–230 bands (the SWIR tail is trimmed per scene), so `window_char.py` maps
    each scene onto a common grid.
  - Two first-run crashes are fixed in the script: all-NaN PACO bands made the joint ridge singular, and
    band counts differ between scenes.

**Proposed handling per group (chunk 5):**

| sensor | group | handling |
|---|---|---|
| PRISMA | edge_lo | own+z |
| PRISMA | weak | ridge |
| PRISMA | strong | constant per band; ridge on snow |
| PRISMA | edge_hi | ridge up to 2477 nm, constant beyond |
| EnMAP | edge_lo | ridge |
| EnMAP | weak | ridge |
| EnMAP | strong 1355–1390 nm | nodata, as PACO |
| EnMAP | strong 1449 nm | own+z |
| AVIRIS-NG | all bands | identity |

Each group is scored and reported on its own, never mixed into the core number.

**Decided by the user (2026-10-28):** the "Proposed handling" table above, with sensor-level models.
The full decision table is in LEARNINGS.md §4.

#### AC26-NBC: core harmonisation on native bands vs the 165 grid (and a covariate fix)

> **Correction (AC26-CLIM):** the "covariate fix" below was wrong. It is reverted, and the PRISMA rows
> are superseded. The EnMAP rows stand, because EnMAP has no climatology covariate.

`harmonise.py` gained two modes:
- `NATIVE=1`: the vendor's core bands, each paired with our band within 3 nm. Only bands that have data
  on both sides in every scene are kept. EnMAP keeps 192 bands; PACO leaves 1343 nm empty. PRISMA keeps
  177 bands, ending at 2436 nm because our SWIR tail is trimmed per scene.
- `SCENES=<frozen model npz>`: restricts the run to the frozen scene list.

**Covariate fix (applies to both modes):**
- PRISMA's climatology covariate was `clim55` (interpolated) in training, while scoring and `aot_prior`
  use `month55`, the value the run was pinned with.
- It is now `month55` everywhere (`harmonise.py`, `harm_score.py`).
- The frozen 2026-10-27 PRISMA model is superseded. It was never scored on the holdout.

LOSO land MAE, median over held-out scenes (mean in brackets). Frozen settings: EnMAP ridge 1e-5;
PRISMA ridge 1e-3, gate 1.0.

| run | scenes | bands | H5 | H6 |
|---|---|---|---|---|
| EnMAP 165 grid | 15 | 165 | 0.0094 (0.0093) | 0.0106 |
| EnMAP native | 15 | 192 | **0.0094** (0.0095) | 0.0112 |
| PRISMA 165 grid, month55 | 15 | 165 | 0.0125 | 0.0100 (0.0115) |
| PRISMA native, month55 | 15 | 177 | 0.0123 | **0.0099** (0.0113) |
| EnMAP native, all training scenes | 21 | 192 | **0.0077** (0.0092) | 0.0110 |
| PRISMA native, all training scenes | 16 | 177 | 0.0114 | **0.0111** (0.0122) |

- Native bands are as good as the 165 grid for both sensors, so the native refit is accepted.
- **The `month55` fix alone moved PRISMA H6 from 0.0118 to 0.0100.** The mismatched covariate was
  costing accuracy, not only consistency.
- With the 6 water tiles (13 scenes with water), EnMAP water: identity 0.0131, H2 water group 0.0113.
  A water group is an option for the frozen EnMAP model.
- 20260111 (the high-altitude scene) is hard for PRISMA, and the gate sends it to identity.
- Outputs: Colab `/content/nbh/*.md`.

#### AC26-LOCK (2026-10-28): native design locked before any holdout score is viewed

The user asked for the holdout score of the 165-grid models. To keep the native holdout blind in practice,
the native design was fixed in writing first:
- the band-group picks (AC26-NBW decision);
- core EnMAP H5 (ridge 1e-5) and PRISMA H6 (ridge 1e-3, gate 1.0, `month55`);
- a freeze on all training scenes (EnMAP 21, PRISMA 16);
- the EnMAP water group (per-band affine, H2 form) only if NDWI matches DLR's water flag with precision
  and recall ≥ 0.9 on training scenes; otherwise water stays identity.

None of these may change in response to the 165-grid holdout numbers.

The 165-grid PRISMA model frozen on 2026-10-27 carries the `clim55` mismatch. It is refitted on the same
15 scenes with `month55` (`prisma_H6_A2_m55.npz`) and hashed before scoring. EnMAP H5 has no climatology
covariate and is scored as frozen.

#### AC26-HO165: holdout score of the 165-grid models (first holdout result, 2026-10-28)

> **Correction (AC26-CLIM):** the PRISMA model scored here (`prisma_H6_A2_m55.npz`) was trained on the
> wrong covariate (`month55`) and scored with the pinned values (`clim55`), so its rows are invalid. The
> valid 165-grid PRISMA number is in AC26-CLIM, from the original `prisma_H6_A2.npz`. The EnMAP and
> AVIRIS-NG rows stand.

- Models: `enmap_H5.npz` (as frozen) and `prisma_H6_A2_m55.npz` (sha256 0496769d…, refitted before
  scoring). AVIRIS-NG is identity.
- Scored with `harm_score.py`; the full tables are on Drive in `atm_corr/holdout_scores_165grid.md`.
- Land MAE, median over the 5 holdout scenes per sensor:

| sensor | reference | identity | model | training LOSO (identity → model) |
|---|---|---|---|---|
| PRISMA | ASI L2C | 0.0249 | **0.0115** (all 5 improve, −34% to −52%) | 0.0220 → 0.0100 |
| PRISMA | ASI L2D | 0.0279 | **0.0149** (all 5 improve) | (not trained on L2D) |
| EnMAP | DLR PACO | **0.0103** | 0.0121 (3 improve slightly, 2 worsen +25% / +57%) | 0.0119 → 0.0094 |
| AVIRIS-NG | JPL | **0.0051** | (identity) | 0.0081 |

**PRISMA generalises.**
- The holdout gain matches LOSO, and the model also helps against L2D, which it never saw.
- Blue bias is overcorrected on 2 scenes: at 460 nm, 20230620 goes +0.019 → −0.041 and 20260305 goes
  +0.044 → −0.030.

**EnMAP H5 does not generalise to this holdout.**
- It fixes the blue bias everywhere (+0.038 → +0.006 on 122280), but the median land MAE rises from
  0.0103 to 0.0121.
- 189895 goes 0.0077 → 0.0121 and 194619 goes 0.0105 → 0.0125.
- Not a PACO version shift: all 5 holdout scenes are V010506, as are 5 of the 15 training scenes, and in
  LOSO H5 helped 4 of those 5.
- The training LOSO already showed fragility: H5 worsened 100387 (+42%) and 127112 (+11%). The median
  gain hid scene-level losses.
- Raw ISOFIT is already close to PACO on these scenes (0.0103 against 0.0119 in training).

The EnMAP decision was raised with the user, since the AC26-LOCK design says H5 and any change now would
be informed by the holdout.

**User decision (2026-10-28):** keep the strict protocol.
- The native EnMAP core stays H5 as locked and will be scored on the holdout once, as is.
- The user intends to **drop EnMAP harmonisation (identity) before the final commit**. That choice is
  recorded in LEARNINGS.md §4 as holdout-informed, so it must be declared when reported.
- The holdout identity score (0.0103) stays unbiased, because identity has no fitted parameters.

#### AC26-NW: NDWI water mask on our reflectance vs the vendor water flags (training scenes)

- Script: `scripts/ndwi_test.py`; full tables on Drive in `atm_corr/ndwi_test.md`.
- Our NDSI snow pixels (AC26-SN rule) are removed first. Water is NDWI(560, 860) > t, optionally with
  ρ860 < 0.15. Vendor cloud is excluded.

| sensor (flag) | rule | pooled P / R | median per scene P / R | worst scene |
|---|---|---|---|---|
| EnMAP (DLR QL, 13 scenes with water) | NDWI > −0.1, ρ860 < 0.15 | **0.927 / 0.922** | 0.880 / 0.976 | 201686 (0.02, 0.00) |
| EnMAP | NDWI > 0.0, ρ860 < 0.15 | 0.952 / 0.880 | 0.926 / 0.954 | 201686 (0.00, 0.00) |
| PRISMA (ASI LandCover, 3 scenes) | best: NDWI > 0.1 | 0.29 / 0.93 | 0.63 / 0.99 | 20260204 (0.46, 0.99) |
| AVIRIS-NG (6 scenes) | NDWI > 0.0 | 0.64 / 1.00 | 0.77 / 1.00 | ang20151228 (0.05, 1.00) |

**Findings:**
- **EnMAP is borderline.**
  - Only one rule passes the AC26-LOCK bar (P and R ≥ 0.9) on pooled pixels, and it was chosen on this
    same data.
  - The per-scene median precision is 0.88, and 201686 fails completely.
  - Our water has NDWI near 0, not clearly positive: ISOFIT water is bright in the NIR (adjacency,
    turbidity).
- **PRISMA and AVIRIS-NG: NDWI disagrees with the vendor water flags** (precision 0.1–0.6), and ASI marks
  very little water (3442 samples in 3 scenes).
- **Consequence beyond the water model.** At deployment the labeller also decides which pixels get the
  core *land* model. In training and holdout scoring that was decided by vendor masks, which a deployed
  scene does not have. This is raised with the user before chunk 9.

#### AC26-NWL: deployable labels (ours) vs vendor labels at prediction time (training LOSO, native bands)

> **Correction (AC26-CLIM):** the PRISMA rows used the wrong covariate (`month55`), and AC26-CLIM reruns
> them. The EnMAP rows stand.

`harmonise.py LABELS=vendor|ndwi|snowonly`:
- Fitting always uses the vendor's land pixels.
- The labels only decide which pixels get the land model at prediction:
  - `ndwi` (option a): our NDSI snow (AC26-SN) and NDWI water (NDWI(560, 860) > −0.1 and ρ860 < 0.15)
    stay identity;
  - `snowonly` (option b): only our snow stays identity.
- Scoring always uses the vendor classes. "all" means every non-cloud pixel.
- Runs: EnMAP 21 scenes (ridge 1e-5), PRISMA 16 scenes (ridge 1e-3, gate 1.0, `month55`). Outputs are on
  Drive in `atm_corr/nwl_2026-10-28/`.

Median over held-out scenes, with the mean in brackets:

| sensor, model | labels | all non-cloud | land | water |
|---|---|---|---|---|
| EnMAP H0 (identity) | - | 0.0118 (0.0131) | 0.0117 | 0.0131 (0.0124) |
| EnMAP H5 | vendor | 0.0107 (0.0109) | 0.0077 | 0.0131 (0.0124) |
| EnMAP H5 | **ndwi (a)** | **0.0094 (0.0107)** | 0.0077 | **0.0131 (0.0119)** |
| EnMAP H5 | snowonly (b) | 0.0093 (0.0113) | 0.0077 | 0.0143 (0.0147) |
| PRISMA H0 (identity) | - | 0.0205 (0.0200) | 0.0216 | 0.0072 |
| PRISMA H6 | vendor | 0.0111 (0.0115) | 0.0111 | 0.0072 |
| PRISMA H6 | **ndwi (a)** | **0.0111 (0.0114)** | 0.0111 | **0.0072** |
| PRISMA H6 | snowonly (b) | 0.0111 (0.0115) | 0.0111 | 0.0144 |

**Option (a) wins for both sensors.**
- The two options tie on all pixels by median: EnMAP 0.0094 vs 0.0093, PRISMA 0.0111 each.
- Option (a) is better on the mean (EnMAP 0.0107 vs 0.0113).
- On water it is clearly better: the land model on water roughly doubles the PRISMA water error
  (0.0072 → 0.0144) and worsens EnMAP's (0.0131 → 0.0143).

**Deployable labels cost nothing against the vendor masks.** EnMAP is even better on all pixels (0.0094 vs
0.0107), because NDWI also holds back dark, wet pixels that the land model handles badly. So the earlier
vendor-label numbers were not optimistic.

The NDWI threshold is shared by both sensors (it was set on EnMAP); PRISMA's ASI water flag is too sparse
to tune on. Chunk 9 implements option (a).

#### AC26-CLIM: the PRISMA climatology covariate was right all along (correction, overnight 2026-10-28)

**What happened:**
- `aod_clim.tsv` has two monthly AOD columns:
  - `clim55` (column 6) is the 1° grid's climatological month value. It equals the pinned value scene for
    scene (`aotpin` hi − 0.15) and is what `aot_prior` records.
  - `month55` (column 9) is that particular year's POWER month. It is NaN for 7 scenes from 2026 and is
    not deployable.
- AC26-NBC mistook `clim55` for an interpolated value and switched training to `month55`. `harmonise.py`
  silently turned the NaNs into the mean.
- That produced the invalid `prisma_H6_A2_m55.npz` (marked INVALID in `final_models/FROZEN_AT.txt`) and
  wrong PRISMA rows in AC26-NBC, AC26-HO165 and AC26-NWL.
- The 7 rows appended for new scenes (20260111 and the 6 extension scenes) also had the grid value only
  in `month55`.

**Fixes:**
- `harmonise.py` and `harm_score.py` are back on `clim55`.
- `clim55` is filled for the appended rows. The backup is `aod_clim.tsv.bak_20261028`.

PRISMA reruns (Drive `atm_corr/fixc_2026-10-28/`):

| run | bands | H5 | H6 |
|---|---|---|---|
| 165 grid, frozen 15 scenes, vendor labels | 165 | 0.0142 | **0.0118** (reproduces AC26-H6) |
| native, frozen 15 scenes, vendor labels | 177 | 0.0146 | **0.0120** |
| native, all 22 scenes, vendor labels | 163 | **0.0116** | 0.0123 |
| native, all 22 scenes, ndwi labels (all non-cloud) | 163 | **0.0116** (0.0120) | 0.0123 (0.0125) |
| native, all 22 scenes, snowonly labels (all non-cloud) | 163 | 0.0117 (0.0123) | 0.0124 (0.0128) |

- On native bands H6 matches the 165 grid (0.0120 vs 0.0118).
- Option (a) labels still win for PRISMA: water 0.0067 vs 0.0135 for snowonly.
- **On all 22 scenes H5 beats H6** (0.0116 vs 0.0123), whereas on the frozen 15 it was the other way
  round. This is for the user: the lock says H6.
- **The common native band set shrinks to 163 bands (≤ 2343 nm)** because 20260202051921 (Kerala coast)
  ends at 2343 nm. The fix needs a decision: drop that scene, or fit core bands on the scenes that have
  them.

**Valid 165-grid PRISMA holdout score.** This is the original `prisma_H6_A2.npz` (6e2d42cb…), frozen
2026-10-27 and scored for the first time here. Land MAE, median (mean) over the 5 scenes:

| reference | identity | H6 | per scene |
|---|---|---|---|
| ASI L2C | 0.0249 (0.0216) | **0.0095** (0.0088) | all 5 improve, −34% to −72% |
| ASI L2D | 0.0279 (0.0274) | **0.0121** (0.0162) | all 5 improve |

This is better than both the invalid `m55` score (0.0115) and training LOSO (0.0118). The blue
overcorrection is milder: the worst 460 nm bias is −0.026 on 20230620, against −0.041 for `m55`.

#### AC26-S4: native rerun and window analysis on all training scenes (EnMAP 27, PRISMA 22)

**EnMAP, ndwi labels:**
- H5: all non-cloud 0.0089 (0.0100), land 0.0077, water 0.0131 (identity).
- H6: 0.0100.
- H0: 0.0118.
- 192 bands.

**Window analysis (Drive `atm_corr/window_char_{enmap,prisma}.md`):** the picks hold under the
own+z-unless-ridge-is-≥10%-better rule.

| sensor | group | own+z | ridge | constant | pick |
|---|---|---|---|---|---|
| EnMAP | edge_lo | 0.0230 | 0.0221 | – | own+z (ridge only 4% better) |
| EnMAP | weak | 0.0180 | 0.0152 | – | ridge |
| EnMAP | 1449 nm | 0.0268 | – | – | own+z |
| PRISMA | edge_lo | 0.0221 | 0.0254 | – | own+z |
| PRISMA | weak | 0.0302 | 0.0270 | – | ridge (11% better) |
| PRISMA | strong | – | 0.0128 | 0.0089 | constant (land classes 0.0016–0.0021; snow ridge 0.032 vs const 0.070) |
| PRISMA | edge_hi | – | 0.0215 | 0.0224 | per band, as before |

#### AC26-FZN: native-band models frozen in the module format (2026-10-28 07:46 UTC, before any native holdout score)

- **Script:** `scripts/export_native.py`. **Models:** Drive `atm_corr/final_models/native/`, with sha256
  in `FROZEN_AT.txt`:
  - `enmap_native.npz`: 30050e33…
  - `prisma_native.npz`: d59710e5…
  - A snapshot of the `harmonise/` package used is saved alongside them.
- **Training sets:**
  - EnMAP: 27 scenes.
  - PRISMA: 21 scenes. 20260202051921 (Kerala coast) is excluded by user decision; its SWIR ends at
    2343 nm.
  - Both use `clim55`.
- **Fits** are the ones whose LOSO chose them:
  - core land: PRISMA H6 (1e-3, gate 1 sd), EnMAP H5 (1e-5);
  - non-core bands: `window_char.py` forms with ridge 1e-3 and the 4 covariates. The PRISMA clim
    weight is 0 there.
  - Picks as AC26-NBW / S4. EnMAP 1355–1390 nm is nodata (as PACO).
- **Checks, all passed:**
  - core coefficients equal `harmonise.py NATIVE=1 --save`, max |diff| 0;
  - `harmonise.apply` reproduces the direct core prediction (EnMAP 0, PRISMA 4e-16);
  - in-sample land MAE: EnMAP 0.0130 → 0.0072, PRISMA 0.0287 → 0.0091.
- **Two band-matching bugs found and fixed in the first export attempt** (nothing was saved from it):
  - **EnMAP:** a common our-band grid merged within 1 nm cross-pairs the VNIR/SWIR overlap bands, which
    sit 0.2–0.4 nm apart (911.6/912.0, 993.1/993.3). Ours is now matched per scene, as
    `harmonise.native()` and `apply.cols()` do. `window_char.py` still uses the merged grid, so its EnMAP
    weak-band numbers in that overlap are slightly off. That does not change any pick.
  - **PRISMA:** ASI keeps 3 overlap pairs (943.4/944.6, 967.0/969.8, 977.4/979.2), but our cube has one
    band there. Only the vendor band nearest ours is modelled. A vendor band with no band of ours within
    3 nm (e.g. 956 nm) has no output band and is not modelled.
- **Differences from the LOSO-evaluated non-core design:**
  - Ridge inputs are ours at the vendor core bands, not our merged-grid core bands.
  - The gate also switches PRISMA's own/ridge window parts to identity in a gated scene. const stays
    on.
- **Deployment caveat:** a PRISMA scene whose SWIR is trimmed below an input band of the ridge parts
  (up to 2436 nm) raised `ValueError` in `apply.cols`. This is the Kerala case.
  - **Resolved by module chunk 12 (gentle fallback):** a ridge part lacking an input band is skipped, its
    bands stay ours, and `<rfl>_harm.json` lists it under `skipped_parts`.
  - On Kerala (ours to 2343 nm), only `win_ridge` is skipped, and 176 of 186 bands are still
    harmonised.
- **Holdout preparation (no score computed):**
  - `native_pairs.npz` was built for the 10 EnMAP and PRISMA holdout scenes from their `rfl_int16.tif`
    (Colab `nbhold.sh`).
  - The 5 PRISMA holdout `atm_post.json` files got the `aot_prior` record from `aotpin.txt`; the
    originals are kept as `atm_post.json.orig`.
  - A dry check found every ridge input band present in all 10 scenes.
  - Some PRISMA scenes lack const/ridge *target* bands: our cube drops 4–24 strong-window and tail bands
    per scene. The package skips those targets, and the output has no band there.

#### AC26-HON: native holdout, scored once with the frozen AC26-FZN models (2026-10-28)

- **Script:** `scripts/score_native_holdout.py`.
  - The model sha256 was checked against `FROZEN_AT.txt`, and the package used was diffed identical to
    the frozen snapshot.
  - Labels: the model is selected with ours (option a); scoring uses the vendor's labels, as in LOSO.
- **Output:** Drive `atm_corr/holdout_native_2026-10-28/`. No scene was gated.
- **Metric:** MAE, median over 5 scenes. "Delivered" = the share of vendor-valid values our output also
  has.

| sensor | land core H0 → model | all non-cloud core | edge_lo / 1449 (own+z) | weak, edge_hi ≤2470 (ridge) | strong, edge_hi ≥2477 (const) |
|---|---|---|---|---|---|
| EnMAP | 0.0098 → 0.0100 (2 better, 2 worse, 1 level) | 0.0097 → 0.0109 | 0.0263 → 0.0162 | 0.0148 → 0.0205 | nodata as PACO (1355–1390) |
| PRISMA | 0.0244 → 0.0099 (all 5 better) | 0.0244 → 0.0103 | 0.0212 → 0.0097 | 0.0611 → 0.0194 | 0.2817 → 0.0078; delivered 0.15–0.50 → 1.00 |

**Findings:**
- **PRISMA native generalises like the 165 grid:**
  - land MAE 0.0244 → 0.0099, against 0.0249 → 0.0095 on the 165 grid (AC26-HO165);
  - every band group improves;
  - in the strong windows, ours is mostly nodata and wrong where present, while the const delivers the
    vendor level.
- **EnMAP native repeats AC26-HO165:**
  - the core is not improved (land level, all non-cloud worse);
  - edge_lo own+z helps (0.0263 → 0.0162);
  - the weak-band ridge hurts (0.0148 → 0.0205).
  - This supports the "drop EnMAP harmonisation" pointer. A narrower variant (edge_lo own+z only) is
    holdout-informed, and would have to be declared as such.

### AC26-SPD: stage timings of the shipped pipeline (2026-10-28)

- **Question:** where does the time go, per stage, so the speed work against FLAASH (about 3 min/scene)
  targets the right part?
- **Splits (`scripts/time_splits.py`, parsed from run.log timestamps):** prep, segment, presolve LUT,
  presolve OE, main LUT 6S, main LUT sRTMnet, main OE, atm interpolation, analytical line (per pixel),
  atm_post, repack, plus `MARK` lines for harmonisation (`scripts/speed.sh`).
- **Baseline:** 94 existing logs (holdout, planC2, planA2): 14 workers, 3 scenes sharing a 44-vCPU v6e host.
  Medians: EnMAP 2025 s, PRISMA A2 1055 s, AVIRIS-NG 600-line 2874 s. Per-pixel solve 40–55%, the three
  ISOFIT solves together about 75%, 6S LUT 10–20% (to 36% on mountain PRISMA), own code under 10%.
- **Solo L4 runs (12 vCPU = 6 physical cores, Xeon 2.2 GHz), challenging scenes, current module
  (`setup_speed.sh`, `speed.sh`):**

| scene | prep | seg | ps LUT / OE | 6S / emu | main OE | interp / per-pixel | post / write | harm | total |
|---|---|---|---|---|---|---|---|---|---|
| EnMAP en_187759 (Himalaya, snow, 3.2 km) | 228 | 70 | 18 / 912 | 573 / 9 | 1766 | 44 / 1906 | 34 / 51 | 9 | 5620 s (94 min) |
| PRISMA 20260305054623 (Himachal, 18.5% snow) | 79 | 47 | 16 / 251 | 1351 / 14 | 543 | 30 / 1378 | 4 | 7 | 3720 s (62 min) |
| AVIRIS-NG ang20151219t081738 (600 lines, 0.43 Mpix, 425 bands) | 94 | 26 | 19 / 473 | 1167 / 12 | 1253 | 19 / 2464 | 4 | – (identity) | 5531 s (92 min) |

- **The L4 is 2–2.7× slower than the v6e on every stage alike** (6 physical cores; hyperthreads don't help
  dense numerics). Rounds 1–2 had already shown about 80 min/EnMAP scene on an L4. Shares are comparable
  across runtimes, absolute times are not.
- **Bound by:** CPU everywhere. Sampled during the AVIRIS-NG per-pixel solve: 84% user + 16% sys, 0% iowait,
  zero I/O and memory pressure, about 1 GB resident per worker.
- **py-spy of the per-pixel solve (AVIRIS-NG):** about 75% is dense n×n linear algebra in
  `invert_analytical` (Seps product 19%, `L'PL` 14%, two dpotrf/dpotri pairs about 30%), not Python overhead.
  Outputs: Drive `atm_corr/speed_2026-10-28/`.

### AC26-BAT: exact restructuring / batching of the per-pixel solve (2026-10-28/29, done)

- **Finding 1 (exact):** for AVIRIS-NG, `Seps` is fully diagonal (Sy parametric diagonal; Kb = per-channel
  `Cal_Relative`, diagonal; Sb diagonal), and `H` (`analytical_model`) is diagonal. ISOFIT still forms,
  factors and inverts them as dense 425×425 matrices per pixel.
- **`scripts/al_batch.py`** re-runs ISOFIT's own per-pixel code on n random pixels of a kept work dir and the
  rewrite on the same pixels. Rewrite v1 (diagonal Seps, elementwise `L'PL`, one Cholesky + its inverse for
  the uncertainty): **2.0× per pixel, rfl max diff 5e-15, uncertainty 3e-15** (40 pixels, AVIRIS-NG,
  contended CPU).
- **Plan (overnight):**
  1. Rerun EnMAP + PRISMA keeping the work dirs, with py-spy sampling every 5 min (`chain_keep.sh`,
     `prof_loop.sh`) → per-stage profiles of all stages, and the dirs for steps 2–4.
  2. `al_batch.py` on EnMAP and PRISMA (is their Seps diagonal too?).
  3. Profile the rewrite: share left in Cholesky/inverse vs the per-pixel forward-model calls.
  4. Batch the remaining Cholesky across pixels (numpy stacks on CPU; torch fp64 on the L4 GPU), same
     exactness check; report the option of skipping the uncertainty inverse (rfl unchanged).
- Research code only; nothing goes into the module without a design and nano-chunks.
- **Ops note:** `prof_loop.sh` first parsed an empty PID (`ps` pads the PID with a space), so EnMAP's keep
  rerun has profiles from 20:16 UTC only (per-pixel stage on). Fixed copy runs as `prof_loop3.sh`; PRISMA's
  rerun is profiled from its start.
- **Step 3 done (cProfile of the rewrite, AVIRIS-NG, 100 pixels, 36 ms/pixel on a contended core; ref
  about 65 ms in the same run, 1.8×):** of the per-pixel loop,
  - `fm.Sa` 18% (rebuilds the full-state block_diag prior every call; only the surface block is used and it
    depends only on the component index and a scalar norm, so it can be cached exactly);
  - `fm.Kb` 9.5% and `fm.Sb` 3.8% (dense n×n builds of what is diagonal here);
  - Cholesky factor + solve 12%, plus the inverse for the uncertainty inside the remaining self time
    (about 35%, which also holds the 425×425 indexing copies);
  - setup (algebraic initial guess) 12%; atmosphere LUT calls only about 2%.
  - So after the exact rewrite the per-pixel cost is still dominated by dense n×n bookkeeping, not by the
    forward model: caching the prior and building Kb/Sb as vectors should take it to about 3–4× over ISOFIT
    before any batching.
- **Step 2, EnMAP en_187759 (kept dir, 40 pixels, contended core):** ref 18.9 → rewrite 13.5 ms/pixel,
  **1.4×, rfl max diff 2e-15, uncertainty 2e-15**. 220 window bands of 224; Kb is 224 per-channel columns
  (as AVIRIS-NG); no model discrepancy. The gain is smaller than AVIRIS-NG's because n³ matters less at
  n = 224 and the per-pixel bookkeeping (prior, Kb/Sb builds) is a larger share.
- **Clean single-core rerun (both chains done, 60 pixels each), `al_batch.py`:** PRISMA 9.2 → 6.3 ms (1.47×,
  206 window bands of 230), EnMAP 9.9 → 6.7 ms (1.49×), AVIRIS-NG 32.7 → 17.2 ms (1.90×). All max diffs ≤ 3e-15.
  Seps is diagonal on all three sensors (Kb square per-channel `Cal_Relative`, no model discrepancy).
- **Step 4, batched solve (`scripts/al_batch2.py`, 200 pixels per sensor, single core):** per pixel only the
  vectors are computed in Python (atmosphere terms, theta, diagonal Seps, prior component + norm; prior
  inverse cached per component, exact); A = Sa_inv/norm² + diag(θ²/Seps) is then solved for the whole batch.

| ms per pixel | ISOFIT ref | vectors + assemble | solve numpy | solve torch CPU | solve torch CUDA (L4) | total, CPU torch | total, CUDA |
|---|---|---|---|---|---|---|---|
| PRISMA (230 states) | 9.14 | 2.55 | 4.05 | 1.37 | 0.17 | 3.9 (2.3×) | 2.7 (3.4×) |
| EnMAP (224) | 9.72 | 3.16 | 3.66 | 1.28 | 0.15 | 4.4 (2.2×) | 3.3 (2.9×) |
| AVIRIS-NG (425) | 33.38 | 4.17 | 18.08 | 5.23 | 0.76 | 9.4 (3.6×) | 4.9 (6.8×) |

  - **Exact:** every path (numpy, torch CPU, torch CUDA fp64) matches ISOFIT to ≤ 1.2e-14 in reflectance and
    ≤ 9e-15 relative in uncertainty.
  - numpy's batched LAPACK is no faster than the per-pixel rewrite; torch's batched Cholesky is 3× faster on
    CPU, and the L4 GPU makes the solve negligible.
  - **What is left is the Python vector part, 2.5–4 ms/pixel:** the forward-model atmosphere terms and the
    Kb/Sy/Sb builds, one pixel at a time. Vectorising those (batched LUT interpolation) is the next lever.
  - Projected per-pixel stage at 12 workers on the L4 (ratio to ref): EnMAP 1906 s → about 650 s with CUDA
    (870 s torch CPU); AVIRIS-NG 2464 s → about 360 s. Not yet measured end to end.
- **Step 5, stage profiles (py-spy, 60 s every 5 min; Drive `speed_2026-10-28/prof/`):**
  - **Per-pixel solve (EnMAP, PRISMA, AVIRIS-NG alike):** `invert_analytical` dense algebra + `Seps` about
    65–75% (lines 327–334 and forward.py:643), `Sa` 3–4% self; matches the AVIRIS-NG sample above.
  - **Main OE on superpixels (PRISMA):** scipy `least_squares` (TRF + `lsmr`) 69%, ISOFIT Jacobian 15%,
    posterior at output (`calc_posterior`, dense) 13%, prior `Sa` 10% of which `svd_inv_sqrt` (an eigh of the
    prior per call) about 10%, initial guess (`invert_simple`/`heuristic_atmosphere`) about 10%. The same
    diagonal-Seps and cached-prior facts apply here, so part of the OE is exactly reducible too.
  - **Repack (EnMAP):** 87% in `write_geotiff` (I/O-ish, about 50 s; small).
  - **6S LUT:** not sampled (the Python driver idles while the Fortran 6S runs), so no py-spy samples; it
    is known to be Fortran CPU.
  - EnMAP keep rerun missed presolve/OE profiles (profiler bug, fixed at 20:16).

## Rejected or dead ends (kept for the record)

- **ACOLITE.** The earlier device trial plus ACIX-III: AOT error about 5× PACO's, blanks about 50
  bands, `ancillary_fixed` breaks the air gap.
- **Dark-dense-vegetation AOT (`legacy_round1/ddv_aot.py`).** Gave 0.68–0.72 on Ganges, against
  PACO's 0.456.
- **ISOFIT native adjacency and terrain via `.6c`.** See AC26-B and AC26-C.

### AC26-VM: the airgapped H200 VM as the speed target (2026-10-29)

- **Machine:** 16 vCPU Xeon 6760P (AVX-512, AMX), 62 GB RAM, 1.8 TB free, H200 NVL (idle). Full facts are in
  `OneDrive/GAIC/H200_SYSTEM.md`.
- **Benchmarks:** the batched fp64 Cholesky + inverse at 425 states (the AVIRIS-NG per-pixel solve), in
  torch, on the VM and on the Colab L4.

  | ms per pixel | VM | L4 |
  |---|---|---|
  | GPU, 2000 pixels | 0.11 | 0.74 |
  | 1 thread | 3.9 | 4.5 |
  | 16 threads in one process | 4.3 (no scaling) | 9.0 at 16 threads (best is 2.5 at 4) |
  | N processes × 1 thread | 16 procs at 3.9 each | 12 procs at 8.6 each |

- **Readout:**
  - CPU throughput with every core busy is about **3× the L4**.
  - Parallelise across processes (as ISOFIT's Ray workers already do), and pin each worker to one thread.
  - The H200 does the batched solve at 6.7× the L4 GPU.
  - Rough expectation for the current code: about 30 / 20 / 30 min (EnMAP / PRISMA / AVIRIS-NG). This is
    not measured, and the 6S Fortran step was not benchmarked.
- **Not yet available:** there is no ISOFIT image on the VM. An airgapped atmospheric-correction image is
  needed.
- **Drive:** the copy at `PS_11/claude_experiments/atm_speed_2026-10-29/` (kept work dirs, scripts, this log)
  was taken before this entry.

### AC26-6S: what sets the 6S table size (2026-10-29, read-only analysis of the kept work dirs)

- **The tables are smaller than the file count suggests.** Each 6S point writes three files, so the real grids
  are:

  | Scene | Points | Grid |
  |---|---|---|
  | PRISMA | 2088 | 29 elevation × 3 AOT × 6 H2O × 2 view zenith × 2 relative azimuth |
  | EnMAP | 1122 | 34 elevation × 11 AOT × 3 H2O (nadir) |
  | AVIRIS-NG | 1728 | 12 elevation × 3 AOT × 4 H2O × 3 view zenith × 4 relative azimuth |

  - About 6–8 CPU-seconds per point (from the AC26-SPD stage times at 12 workers).
  - Solar zenith and date are fixed per scene, so tables are **not reusable across scenes**. A library over
    sun and view geometry would need about 1M points (thousands of core-hours). The "reuse the 6S table" lever
    in AC26-SPD is withdrawn.
- **Elevation is the largest dimension, because of `--pressure_elevation`.** ISOFIT then widens the grid to
  DEM min − 2 km (floored at 0) … DEM max + 2 km (`template_construction.get_metadata_from_loc`):

  | Scene | Valid-pixel DEM | Grid |
  |---|---|---|
  | PRISMA | 0.69–4.91 km | 0–6.91 km |
  | EnMAP | 1.58–6.17 km | 0–8.22 km |
  | AVIRIS-NG | 0.52–0.55 km | 0–2.55 km |

- **The retrieved pressure elevation wanders far from the DEM** (superpixels, retrieved − DEM):

  | Scene | p1 / p50 / p99 | Notes |
  |---|---|---|
  | PRISMA | −0.61 / +0.31 / +1.81 km | |
  | EnMAP | −0.98 / −0.12 / +3.44 km | some pinned at the 8.22 km grid top |
  | AVIRIS-NG | −0.46 / −0.39 / −0.10 km | some pinned at 2.54 km |

  So it absorbs other errors rather than measuring pressure. A narrower margin is therefore *not* exact.
- **Prior evidence:** AC26-M3b (AVIRIS-NG Spiti, pressure retrieval off) gave MAE 0.0209 vs 0.0208, i.e.
  no change.
- **Candidate (accuracy trade, to test):** drop `--pressure_elevation`. The elevation grid becomes DEM
  min..max: about −38% 6S points on PRISMA, −44% on EnMAP and −85% on AVIRIS-NG. It also removes one state
  from the presolve and main inversion. Test on a subset of the training pairs per sensor, against vendor L2,
  before any module change.

### AC26-BSC: the per-pixel stage on whole scenes, five ways (2026-10-29, Colab L4, 12 workers, done)

- **Set-up:** `scripts/al_scene.py` runs the analytical line (ISOFIT's per-pixel stage) on every pixel of the three
  kept work dirs (`chain_bsc.sh`). Same forward model and inputs in every mode; outputs compared with ISOFIT's own run
  over all pixels.
- **Exactness:** every mode matches ISOFIT to float32 rounding (reflectance ≤ 6e-8 absolute, uncertainty ≤ 1.2e-7
  relative). No pixel was solved by one and skipped by the other.

| Wall s (speed-up) | PRISMA 881k px | EnMAP 1.20M px | AVIRIS-NG 381k px |
|---|---|---|---|
| ISOFIT (ref) | 1391 | 1948 | 2490 |
| exact per-pixel rewrite (`fast`) | 905 (1.5×) | 1301 (1.5×) | 1210 (2.1×) |
| batched numpy CPU (`np`) | 804 (1.7×) | 1125 (1.7×) | 1197 (2.1×) |
| batched torch CPU (`tcpu`) | 557 (2.5×) | 817 (2.4×) | 617 (4.0×) |
| batched torch GPU (`cuda`) | **349 (4.0×)** | **557 (3.5×)** | **358 (7.0×)** |

- **ms per pixel, single-core CPU time by level (`cuda` mode):**

  | Level | PRISMA | EnMAP | AVIRIS-NG |
  |---|---|---|---|
  | set-up (algebraic initial guess) | 1.8 | 1.6 | 2.6 |
  | forward-model vectors | 2.7 | 3.7 | 5.2 |
  | GPU assemble + solve | 0.18 | 0.16 | 0.81 |

  - The CPU solve in `tcpu` is 2.4 / 2.2 / 10.2 ms, and in `np` 5.9 / 5.2 / 28.5 ms.
  - On the GPU the solve is negligible. What remains is about 4.5–8 ms of Python per pixel: the set-up and the
    forward-model vectors. That is the next lever: vectorise them, i.e. batched LUT interpolation.
- **Ops:**
  - AVIRIS-NG `cuda` first ran out of GPU memory: 3 × 4096 × 425² fp64 is about 18 GB on the 22 GB L4. Rerun with
    1024-pixel GPU batches. `al_scene.py` now sizes GPU batches to about 1.5 GB per matrix.
  - `chain_bsc.sh` keeps only the last log line, and Ray printed after the ref JSON on AVIRIS-NG; that line was
    added by hand.
- **Effect on the whole L4 run (other stages unchanged):** PRISMA 62 → about 45 min, EnMAP 94 → about 71 min,
  AVIRIS-NG 92 → about 57 min.
- Results: Drive `claude_experiments/atm_speed_2026-10-29/bsc/` (`results.jsonl` + logs).

### AC26-6SP: inside one 6S point (2026-10-29, Colab L4, PRISMA point AOT 0.06 / H2O 0.2 / z 0 km)

- **A point is one grid combination** (AOT, H2O, elevation, view zenith, relative azimuth). One `sixsV2.1` run per
  point computes 350–2500 nm at 2.5 nm (861 wavelengths), printing per wavelength: gas transmittance, down / up
  scattering transmittance, spherical albedo, atmospheric reflectance. That is sRTMnet's input.
- **gprof** (6S rebuilt with `-pg`; output identical to stock), 7.3 s:
  - `OSPOL` (polarised successive orders of scattering) 66%, 60 calls: scattering at 20 reference wavelengths ×3,
    interpolated to the 861 outputs;
  - `KERNELPOL` (polarised phase-function kernels) 32%;
  - `ABSTRA` (gas absorption) 0.3%;
  - everything else about 1%.
- **Compiler flags** (ISOFIT builds 6S with plain `-O`):

  | Build | Time | Output vs stock |
  |---|---|---|
  | `-O2` | 7.76 s | identical |
  | `-O3` | 7.38 s | identical |
  | **`-O3 -march=native`** | **4.86 s (1.6×)** | 1 of 1722 printed table lines differs in the last digit |
  | `-Ofast -march=native` | 3.46 s (2.2×) | 2 lines differ in the last digit |

- **Water vapour does not touch the scattering.** Points differing only in H2O (0.2 vs 0.56 vs 2.0 cm; AOT 0.21,
  z 2.47 km) have byte-identical scattering down / up / spherical albedo on all 861 wavelengths. Only gas
  transmittance and atmospheric reflectance differ, and both come from the cheap post-scattering gas step.
  So computing the scattering once per (AOT, elevation, geometry) and only the gas step per H2O node is exact.
  - Expected gain on the 6S stage: about H2O-node count × 0.98 of the time, i.e. 6S time ×(0.98/n + 0.02).
  - n = 6 PRISMA, 3 EnMAP, 4 AVIRIS-NG → 4.4× / 2.9× / 3.8×.
- **Projected 6S stage, L4 (from AC26-SPD), with `-O3 -march=native` plus scattering reuse across H2O:**

  | Scene | Now | Projected | Gain |
  |---|---|---|---|
  | PRISMA | 1351 s | about 155 s | 8.7× |
  | EnMAP | 573 s | about 125 s | 4.6× |
  | AVIRIS-NG | 1167 s | about 195 s | 6.0× |

  Pressure elevation off would cut the points further (AC26-6S). Not implemented; the reuse needs a 6S (Fortran
  main) or ISOFIT-engine change.
- Not recommended: polarisation off (scalar OS) or fewer quadrature angles / layers. Both change 6S's output away
  from what sRTMnet was trained on.

### AC26-6SX: drop 6S's aerosol-only scattering call (2026-10-29, Colab L4, done)

- **Why it is spare:** `ATMREF.f` calls `OSPOL` three times per reference wavelength: Rayleigh only (cheap, 2
  Fourier terms), aerosol only, and mixed. The per-wavelength path reflectance ISOFIT reads uses only the mixed and
  Rayleigh results (`ratm2 = (romix − rorayl)·tgp2 + rorayl·tgp1`, main.f). The aerosol-only result feeds only
  6S's printed band-integrated summary.
- **Patch:** `scripts/sixs_noaer_patch.py` removes that call in a scratch copy (`/content/sixs_noaer`) and zeroes
  roaero / rqaero / ruaero. Built with ISOFIT's own flags (`-O`).
- **Test:** 9 points, 3 per sensor (first, last and middle `.inp` of the kept LUT dirs: AOT 0.05–0.90, elevation
  0–8.2 km, all geometries).
  - **All 1722 per-wavelength table rows are identical to stock on every point.**
  - Only 2–6 summary lines change (the aerosol column becomes NaN: "reflectance I/Q/U", "degree of polar.").
  - ISOFIT's `SixSRT.parse_file` reads only the primary 11-column table and the multipart 5-column table, and
    stops at "integrated values of", so it never reads them. Exact for ISOFIT.
- **Speed:** 1.83–1.92× per point (PRISMA 7.75 → 4.08 s, EnMAP 5.50 → 2.96 s, AVIRIS-NG 8.48 → 4.43 s).
- **Stacked with `-O3 -march=native` (AC26-6SP) and scattering reuse across H2O nodes, projected (L4, 12
  workers; not measured end to end):**

  | Scene | 6S stage now | Projected | Gain |
  |---|---|---|---|
  | PRISMA | 1351 s | about 85 s | about 16× |
  | EnMAP | 573 s | about 65 s | about 9× |
  | AVIRIS-NG | 1167 s | about 105 s | about 11× |

  The `-march` gain is assumed to carry over to the patched build; check before relying on it.
- **Module route:** ISOFIT builds 6S from source (`isofit/data/cli/sixs.py`, `patch_makefile`). The Docker image
  can apply this patch plus the flags at build time. It is a build-step change, not an ISOFIT code change.

### AC26-OEB: main inversion and presolve timed standalone (2026-10-29, Colab L4, 12 Ray workers)

- **Set-up:** `scripts/oe_bench.py` runs ISOFIT's main inversion (`*_isofit.json`) or presolve (`*_h2o.json`) on the
  kept superpixel inputs, outputs redirected; `chain_oeb.sh`; states compared per group by `oe_compare.py`.
- **Exact patches** (`scripts/oe_patch/`, applied in every Ray worker via sitecustomize when OE_PATCH=1):
  - `seps`: diagonal Seps, so elementwise residual and Jacobian scaling;
  - `post`: posterior via Cholesky with a diagonal Seps.

| Wall s | PRISMA 17,054 sp | EnMAP 24,001 sp | AVIRIS-NG 7,624 sp |
|---|---|---|---|
| main ref | 559 | 1792 | 1296 |
| main patch (exact) | 530 (1.05×) | 1712 (1.05×) | 1242 (1.04×) |
| main patch + pressure elevation off | 407 (1.37×) | 1478 (1.21×) | 625 (2.07×) |
| presolve ref | 259 | 939 | 493 |
| presolve patch (exact) | 254 (1.02×) | 916 (1.03×) | 487 (1.01×) |

- **Exactness:** every patched run has bit-identical states (max difference 0.0 in every group).
- **Pressure elevation off changes the states.** Max / p50 / p99 of the per-superpixel difference:

  | Group | PRISMA | EnMAP | AVIRIS-NG |
  |---|---|---|---|
  | AOT | 0.30 / 0.023 / 0.23 | 0.84 / 0.085 / 0.47 | 0.040 / 0.003 / 0.038 |
  | H2O | 1.09 / 0.018 / 0.17 | 0.48 / 0.021 / 0.16 | 0.28 / 0.16 / 0.21 |
  | surface group | 0.44 / 0.007 / 0.29 | 0.31 / 0.011 / 0.17 | 2.03 / **0.48** / 1.09 |

  - The AVIRIS-NG surface-group change is implausibly large for reflectance. Check what that group holds (any
    non-reflectance surface states) before reading it as a physics result.
  - Accuracy of pressure elevation off is the planned 5-scene-per-sensor test against vendor L2.
- **The exact patches are worth only about 5%.** The cost is scipy's trust-region loop and the forward-model
  evaluations, not the algebra ISOFIT does around them.
- **The `sa` patch moves the solver's end point.** It caches the prior blocks, a change at rounding level only
  (rel. 1e-16). On 100 PRISMA superpixels it moved the solver end point on a minority: AOT max 0.126, p99 0.0036,
  median 5e-7; 24% of superpixels > 1e-6 in surface states. ISOFIT itself is deterministic (two identical runs:
  0.0 difference). ISOFIT's OE stops at max_nfev 20 / ftol 0.01, so its answer is defined only to about 1e-3 AOT
  on its worst 1% of superpixels. Any faster OE (a batched solver included) must be judged by accuracy, not bit
  equality. `sa` is off by default.
- **Ops:** the AVIRIS-NG presolve pair first failed (h2o config not path-remapped). It is rerun by
  `chain_after.sh` with `al_batch_h2o.json`.
- Drive `claude_experiments/atm_speed_2026-10-29/oeb/`.

### AC26-PRF: one PRISMA run profiled end to end (2026-10-29, Colab L4, alone on the machine)

- **Run:** `chain_prof.sh` ran `speed.sh pr:20260305054623` into `/content/outs_prof`, with `prof_cont.sh`
  sampling it: back-to-back 30 s py-spy windows over the pipeline process and all its descendants, plus vmstat
  every 5 s. `prof_stage.py` mapped samples to stages from the run log.
- **Profiler overhead:** the profiled run took 4399 s against 3720 s clean (AC26-SPD).
  - The 6S stage took 1841 s against 1351 s, with system CPU 24%. py-spy `--subprocesses` attaching to thousands
    of short-lived 6S processes overran its 60 s timeout, so 4 windows (about 31 min) were lost in that stage.
  - Other stages were about 8% slower.
  - So stage times come from AC26-SPD, and only the composition from this run.
- **Per stage** (clean s; vmstat user / system / idle %; hot functions):
  - **Prep** (79 s): 14 / 1 / 84, single thread. rasterio `sample_gen` 63% (DEM sampled point by point),
    radiance calibration 11%.
  - **Segmentation** (47 s): 28 / 5 / 68. kmeans2/vq 29%, superpixel extraction 29%, Ray worker imports 28%.
  - **Presolve inversion** (251 s): 91 / 7 / 2. least_squares 75% (lsmr 40%), matmat 10%, Sa 4%, eigh 3%.
  - **6S** (1351 s): 76 / 24 / 1. Fortran; the inside is AC26-6SP.
  - **Emulator:** 14 s.
  - **Main inversion** (543 s): 93 / 5 / 2. least_squares 71% (lsmr 44%), matmat 11%, eigh 7%, Sa 3%.
  - **Per-pixel solve** (1378 s, plus 30 s interpolation): 91 / 7 / 2. `invert_analytical` 84%, of which its dense
    n×n lines 38%; Seps 23%, Sa 13%.
  - **atm_post** 4 s; **harmonise** 7 s. The HARM_DONE mark was glued to the end of the harmonise JSON (no
    newline), so `prof_stage.py` labelled those 7 s "tail".
- **Bound type:** I/O wait ≤ 2% in every stage. The parallel stages are CPU-saturated; prep and post are single
  thread.
- **Projected with the exact levers measured so far** (L4: 6S 124 s, per-pixel 349 s GPU, inversion shortcuts):
  3720 → about 1454 s (2.6×). Not run end to end.
- **Report page** for the team: https://claude.ai/artifact/DL6WEYGzjAq2Z5NvSK79du
- Drive `claude_experiments/atm_speed_2026-10-29/prof/` (stages.json, splits.tsv, vmstat, run.log, py-spy raw).

### AC26-6SC: the whole 6S stage with each 6S build (2026-10-29, Colab L4, 12 processes, done)

- **Set-up:** `scripts/sixs_stage.py` runs every point of a kept LUT dir (PRISMA 2088, EnMAP 1122, AVIRIS-NG 1728)
  with one build (`build_6s_variants.sh`, run by `chain_6sc.sh`), and compares every per-wavelength table row
  with the original ISOFIT run.
- **Builds:**
  - **stock:** ISOFIT's 6S, `-O`;
  - **noaer:** aerosol-only OSPOL call removed (AC26-6SX);
  - **cache:** noaer + DISCOM cache across H2O nodes (`sixs_cache_patch.py`). One point per group runs first,
    then the rest read the cached scattering;
  - **N:** the same compiled with `-O3 -march=native`.

| Wall s (× vs stock) | PRISMA | EnMAP | AVIRIS-NG |
|---|---|---|---|
| ISOFIT 6S stage, clean run (AC26-SPD) | 1351 | 573 | 1167 |
| stock (harness) | 1298 (1.0×) | 524 (1.0×) | 1120 (1.0×) |
| noaer | 686 (1.9×) | 283 (1.9×) | 589 (1.9×) |
| noaerN | 517 (2.5×) | 211 (2.5×) | 444 (2.5×) |
| **cache** | **124 (10.5×)** | **99 (5.3×)** | **155 (7.2×)** |
| **cacheN** | **93 (14.0×)** | **74 (7.1×)** | **116 (9.7×)** |
| H2O nodes / cache groups | 6 / 348 | 3 / 374 | 4 / 432 |

- **Exactness:**
  - stock, noaer and cache: **0 table rows differ** on every point (3.6M / 1.9M / 3.0M rows).
  - The `-march=native` builds differ only in the last printed digit (max 1e-4), on 1122 / 1001 / 711 rows
    (820 / 632 / 567 points).
- **The cache gain follows the H2O node count:** noaer → cache is 5.5× / 2.9× / 3.8× against 6 / 3 / 4 nodes.
  The leader runs (phase 1) are the whole cost; the cached runs take 0.1 s each.
- **The harness stock run matches ISOFIT's stage time to 4%,** so the stage-level speed-ups transfer.
- **Module route:**
  - noaer and the compile flags are a 6S build-step change in the Docker image.
  - The cache needs ISOFIT's 6S engine to run one point per (AOT, elevation, geometry) group first and pass
    `SIXS_DISCOM_CACHE`, or a 6S main that loops over H2O itself.
  - Design plus nano-chunks.
- Drive `claude_experiments/atm_speed_2026-10-29/6sc/`.

### AC26-MPC: 6S speed builds vs the "Multipart transmittances" table (2026-10-30)

- **Question:** `src/sixs_parts.py` (adjacency, terrain) reads each 6S output's Multipart table, which AC26-6SC did not
  compare. Do the speed builds leave it unchanged?
- **Script:** `scripts/mp_check.py`, builds from `build_6s_variants.sh`; 3 H2O groups per sensor from the kept LUT dirs
  (AVIRIS-NG 12 points, EnMAP 9, PRISMA 18), each output compared with ISOFIT's own stock output.

| build | Multipart rows differing | ISOFIT rows differing |
|---|---|---|
| stock, noaer, cache | **0** of 10332 / 7749 / 15498 | 0 |
| noaerN, cacheN (-O3 -march=native) | 0 / 1 / 6 (max 1e-4, last printed digit) | 9 / 5 / 8 (max 1e-4) |

- **Finding:** noaer and cache are exact for the Multipart table too. The -march builds change the last printed digit
  only, as in AC26-6SC. Item 2 of `SPEED_MODULE_PLAN.md` can proceed.

### AC26-HB: EnMAP blue-only harmonisation (2026-10-30). HOLDOUT-INFORMED: designed after AC26-HON

- **Why:** H5 fixed EnMAP's blue bias on every holdout scene but worsened the core on 2 of 5 (AC26-HON). The user
  chose a blue-only model (the blue offset matters for SAM-based spectral matching).
- **Model HB:** the H6 form (per band, own + 4 scene covariates, ridge 1e-3, gate 1 sd) fitted on land, applied only
  to core bands below the cut; identity on every other band. Scripts: `harmonise_blue.py` (LOSO), `export_blue.py`
  (freeze), `score_blue_holdout.py` (holdout). Drive `atm_corr/hb_2026-10-30/`.
- **Training LOSO** (27 scenes, native bands, ndwi labels; median land MAE, all core bands):

| cut | H0 | HB | land below cut H0 → HB | H5 (ref) |
|---|---|---|---|---|
| 500 nm | 0.0117 | 0.0111 | 0.0216 → 0.0138 | 0.0094 |
| 550 nm | 0.0117 | 0.0108 | 0.0185 → 0.0126 | 0.0094 |
| 600 nm | 0.0117 | **0.0104** | 0.0161 → 0.0114 | 0.0094 |

  - |bias460| median 0.024 → 0.012 at every cut. Six already-close scenes get slightly worse (e.g. 100387 0.0117 →
    0.0138, 113132 0.0060 → 0.0075 at 600).
  - **Cut 600 nm chosen on training LOSO** (best land median), before any holdout score.
- **Frozen:** `final_models/native/enmap_blue600.npz` sha256 4dbb8d3f…d594 (FROZEN_AT.txt): core own+z 464–599 nm
  (28 bands), edge_lo own+z (9 bands, 418–459 nm), PACO nodata windows (1355–1390 nm) nodata, all else ours.
  In-sample land 0.0130 → 0.0109; package reproduces the direct fit to 2e-16.
- **Holdout, scored once** (5 scenes, all core bands, vendor scoring labels; none gated):

| | identity | HB 600 |
|---|---|---|
| land MAE | 0.0098 | **0.0086** (all 5 improve) |
| all non-cloud | 0.0097 | **0.0086** |
| land below 600 nm | 0.0136 | **0.0055** |
| bias at 460 nm | +0.023 | **+0.001** |
| edge_lo bands | 0.0270 | **0.0117** (4 of 5; 187759 0.087 → 0.089) |

- **Finding:** unlike H5 (0.0098 → 0.0100, 2 of 5 worse), the blue-only model improves every holdout scene. The score is
  holdout-informed (the model family was chosen after AC26-HON) and must be declared as such.

## AC26-W3: 6S wrapper + shard patch, real EnMAP LUT build (2026-10-30, item 3)
- Colab L4 runtime, 12 vCPU, n_cores 12, EnMAP 187759 kept work cubes, apply_oe stopped after the full LUT.
- Full-LUT 6S stage (1122 points, 11 shards x 102): stock 577 s (17:24:46-17:34:23) -> wrapper 115 s (18:00:31-18:02:26), **5.0x**.
  Presolve 6S (10 points): 9 s -> 5 s. Cache files 375 = 374 AOT x elevation groups + 1 presolve group.
- Rows vs stock: lut_full 632 of 1122 points differ, all in the last printed digit (max 1e-4); lut_h2o 1 of 10, same.
  Unit: wrapper == binary byte for byte, cache on == off, SIXS_FAST=0 == stock.
- Per point (fixture, AOT 0.39): stock 3.0 s, v4 1.38 s, **v3 1.29 s** (v3 faster on this AVX-512 Xeon), cached read 0.03 s.
  -> wrapper default v3; SIXS_ARCH=v4 opt-in, to be timed on the VM.
- Shard patch (isofit_patches.py): 64 cores now pick 34 shards with H2O whole instead of 66 with H2O split;
  per shard 11 x 1.3 + 22 x 0.03 ~ 15 s vs 17 x 1.3 ~ 22 s. 12 and 192 cores: layout unchanged (already whole).
- Found: entrypoint --stop-after-lut waited for "Loading LUT into memory", which 4.1.5 never logs; fixed to the
  4th "Running any post-sim functions".

## AC26-FAL: src/fast_al.py inside ISOFIT's analytical line (2026-10-30, item 4, Colab L4, 12 actors)
- Hook: fast_al.analytical_line runs ISOFIT's own analytical_line with its Ray Worker swapped for FastWorker (subclass;
  only run_chunks changes: per-pixel vectors in the actor, Cholesky batched per chunk with torch fp64). Ray 2.59 leaves
  CUDA_VISIBLE_DEVICES unset for num_gpus=0 actors, so all 12 actors use the L4 (batches of ~740 px at 224 states).
- EnMAP 187759 kept work dir (atm_interp reused, so per-pixel stage only):
  - 150-line crop, 83,545 px: ISOFIT 154 s, fast GPU 59 s, fast CPU torch 82 s; rfl and unc **identical (max diff 0.0)**
    on every pixel for GPU, CPU and the forced fallback (ATM_FAST_AL_DEVICE=bogus -> every actor logs and uses ISOFIT).
  - Full scene, 1,201,607 px: fast GPU **602 s** vs ISOFIT 1948 s (AC26-BSC ref) = 3.2x (fork-pool prototype 557 s).
    Uncertainty vs the kept ISOFIT run: no pixel missing either side, max rel 6e-8 (float32 rounding).
    119 chunks via fast_al, 0 fallbacks.
- Per-chunk fallback tested by accident first: a transposed test crop made ISOFIT itself fail; every fast chunk logged
  the error and reran ISOFIT's run_chunks, which then failed the same way (no silent different answer).

## AC26-EOE: why EnMAP's main inversion costs ~2x PRISMA's per superpixel (2026-10-31, Colab L4)
- Kept work dirs en_187759 / pr_20260305054623, ISOFIT main inversion on superpixels 4000-5199 (oe_bench.py), every
  scipy least_squares call counted (scripts/nfev_patch, eoe.sh / eoe_sum.py):

  | variant | nfev mean / p90 | s per evaluation | s per superpixel |
  |---|---|---|---|
  | EnMAP as run (free AOT 0.05-0.90) | 7.8 / 12 | 0.093 | 0.72 |
  | EnMAP, AOT bounded to POWER clim +/- 0.15 | 7.0 / 11 | 0.094 | 0.66 |
  | PRISMA as run (A2 window) | 6.6 / 10 | 0.047 | 0.32 |

  Iterations are not the cause (+18%); each evaluation costs 2x. The AOT window saves 9% and moves AOT by up to 0.34
  (p90), surface rfl by 0.0025 mean: dropped.
- py-spy, 60 s of 4 workers each (eoe2.sh / eoe2_sum.py): EnMAP spends **40% of inversion samples in
  forward.upsample** (scipy interp1d set-up and calls, inside the Jacobian K: 45.6% vs 10.6%); PRISMA 1.6%.
- Cause: entrypoint.sh stage 1b builds the per-scene surface prior on a NUDGED grid (bands sharing an integer nm moved
  up to 0.15 nm, an ISOFIT 3.7 state-name collision workaround), so surface wl != instrument wl and upsample
  interpolates on every evaluation (its closeness test is np.allclose, ~0.01 nm). ISOFIT 4.1.5 names surface states
  "RFL_%04i_%.3f", so the collision is gone: EnMAP has 2 same-integer-nm pairs, 0 duplicates at 3 decimals,
  min gap 0.193 nm. ISOFIT's own log warned "Surface and atmosphere wavelengths differ".
- Test queued: EnMAP 113132 + 056279 with the nudge removed (/content/modF2, acc2.sh) -> Drive acc_nonudge_2026-10-31.
- Ops: eoe.sh first carried a `ray stop --force`, which killed the concurrent 113132 campaign rerun; removed, rerun
  requeued, memory rule added.
- **Result (no nudge, /content/modF2):** EnMAP surface prior on the true band centres. "Surface and atmosphere
  wavelengths differ" gone.

  | EnMAP | presolve s | main OE s | per-pixel s | total min | land MAE raw / harmonised |
  |---|---|---|---|---|---|
  | 113132 nudged (campaign module) | 831 | 1281 | 522 | 53 | 0.0059 / 0.0064 |
  | 113132 true centres | **316** | **550** | **424** | **30** | 0.0059 / 0.0064 |
  | 056279 nudged | 865 | 1522 | 671 | 59 | 0.0165 / 0.0143 |
  | 056279 true centres | **358** | **679** | **528** | **34** | 0.0165 / 0.0143 |

  Accuracy identical to the 4th decimal (<=2350 nm on 056279: 0.0171 -> 0.0170). Proposed module change: drop the
  stage 1b nudge in entrypoint.sh (pending user approval).

## AC26-ACC: speed module + harmonisation, 12 training scenes (2026-10-30/31, Colab L4, 12 cores)
- Drive `atm_corr/acc_2026-10-31/` (summary.md, per-scene logs, pairs, splits). Module = worktree items 2-4 (6S wrapper
  + H2O cache + shard patch + fast_al on the L4), harmonisation enmap_blue600 / prisma_native (H6).
- **Raw land MAE equal to the stock-module score on all 12 scenes** (EnMAP + AVIRIS-NG vs Plan C, PRISMA vs A2):
  ANG 0.0062 / 0.0207 / 0.0081; EnMAP 0.0165 / 0.0059 / 0.0156; PRISMA 0.0097 / 0.0220 / 0.0162.
- Harmonised (IN-SAMPLE: the models were fitted on these scenes): EnMAP 0.0143 / 0.0064 (113132 worse: blue flips
  +0.012 -> -0.015, an already-close scene as in the LOSO) / 0.0148; PRISMA 0.0057 / 0.0112 / 0.0162 (94% snow,
  untouched by design).
- Tanager (first ISOFIT runs, readers.prep + apply_oe tanager, vs Planet basic SR): 0.0061 / 0.0144 / 0.0079.
- Totals: ANG 31-39 min (600-line windows), PRISMA 23-33 min (stock speed run 62 min), EnMAP 52-59 min (nudged;
  30-34 with the fix above), Tanager 33-49 min (the 49 is the hazier India scene with 25% more superpixels).
- Ops: EnMAP 113132 first scored nothing (COG naming; the scorer needs stage_c.sh's .TIF symlinks, added to
  acc_setup.sh); its rerun was killed by eoe.sh's ray stop and run a third time.

## AC26-LVL: item 5 reader layer tests (2026-10-31, Colab)
- Interleave check (cube.layout_score = spectral coherence x weakest of band / sample / line smoothness): AVIRIS-NG
  radiance written BIL / BIP / BSQ under all 3 headers: 9 / 9 resolved to the true layout, data read back exact.
  Neighbour-band correlation alone failed (BSQ read as BIL looks smooth; BIL read as BIP 0.99 vs 0.90).
- Level classifier, numbers only (r3_conf.py): 9 radiance, 9 TOA (from each radiance with its own sun zenith and
  ISOFIT's irradiance), 10 vendor SR (JPL corr x3, Planet SR x4, DLR L2A, ASI L2D x2). First pass 24/28 (4 SR -> TOA,
  safe side); thresholds retuned on this table (O2 SR < 0.06 / TOA > 0.12; water SR < 0.30 / TOA > 0.40; CO2 and
  blue TOA-only; 1380 counts only when it votes): **28 / 28, in-sample**. No TOA or radiance cube ever called SR.
- Native: AVIRIS-NG L1 folder, PRISMA L1 HE5, EnMAP L1C (incl. *_COG.tiff naming, fixed in enmap_l1c_preprocess),
  Tanager basic + ortho radiance (grade A) and SR (product label).
- Formats (r2.sh): AVIRIS-NG GeoTIFF x10 units, no unit tags -> scale 0.1 derived, grade C; AVIRIS-NG ENVI BIP data
  under a BIL header, no CRS / time -> layout fixed, grade D; Tanager TOA as GHOSt-style float64 GeoTIFF, no CRS ->
  toa_reflectance (6 TOA votes), grade D; unlabelled Tanager SR GeoTIFF -> SR (4 / 4); EnMAP L2A COG -> SR x1e-4,
  wavelengths from METADATA.XML (red edge checked).
- **Found and fixed: EnMAP L1C raw counts given as a bare GeoTIFF were classified SR** (p99 48000, no absorption at the
  band centres without gains). Now: reflectance-shaped values that fit no reflectance scale -> "unknown" (the run stops
  and asks for --input-level); a SPECTRAL_IMAGE inside an L1C folder routes to the native preprocessor; a bare one
  without its XML is refused (no wavelengths). Sensor name rules now read the file name only (a folder named
  ghost_* had labelled Tanager data as GHOSt).

## AC26-E7: all-sensor entrypoint end to end (2026-10-31, Colab L4, scripts/e7.sh)
- EnMAP 113132 native: 29 min, land MAE 0.0059 raw / 0.0064 harmonised (= AC26-ACC). PRISMA 20260514 native, AOT prior
  via SENSOR_RULES: 31 min, 0.0097 / 0.0057 (= AC26-ACC). Tanager SR GeoTIFF: pass-through, no ISOFIT.
- Generic path (Tanager 20240925 TOA as a GHOSt-style float64 GeoTIFF, 300 lines, no CRS / time / DEM: grade D) found
  two bugs, both fixed:
  1. ISOFIT NA- mode names files fid + "_rdn" with "_rdn" swapped ("x_rdn" stem -> "x_lbl_lbl"); analytical line could
     not find its segmentation. Entrypoint now passes a link named after the bare fid.
  2. Without a location every pixel had lat/lon 0, 0; atm_interpolation (spatial neighbours) then wrote AOT = H2O =
     elevation = 0 everywhere (MAE 0.0210). Readers now write a pseudo 30 m lat/lon grid for ISOFIT only.
- After the fixes, grade D vs Planet basic SR (same 300 lines): **land MAE 0.0062**, retrieved AOT 0.054 / H2O 0.87 /
  elevation 0.97 km with no DEM (0-5 km widened grid) against the full-metadata run's 0.054 / 0.85 / 0.95 km (DEM).
  The assumed 35 deg sun cancels to first order here (used both to convert TOA -> radiance and inside ISOFIT); a
  radiance delivery without time would feel it more.

## AC26-W8: item 8 writers (2026-10-31, fresh Colab, light env, readers outputs; no ISOFIT)
- GeoTIFF (tiled, DEFLATE + predictor, all cores) and ENVI BSQ (GDAL ENVI driver + spectral header keys), int16 x10000,
  nodata -9999, water bands (outside the inversion windows) and empty bands dropped. Read back with GDAL and spectral:

  | case | kept / dropped | georeference | values | GeoTIFF / BSQ write s (MB) |
  |---|---|---|---|---|
  | EnMAP L2A 1157x1249 | 218 / 1 | UTM 45N = source | exact | 2.1 (294) / 1.3 (601) |
  | AVIRIS-NG 5592x722, 90 deg rotated | 380 / 45 | UTM 43N rotated = source | exact | 8.0 (1555) / 11.7 (2926) |
  | PRISMA L1 swath | 206 / 24 | 441 GCPs (GeoTIFF), 4 geo points (ENVI) + _latlon.tif | exact | 1.4 / 0.8 |
  | Tanager ortho | 382 / 44 | UTM 44N = product grid | exact | 1.4 / 1.1 |
  | no-location ENVI | 380 / 45 | none (stated) | exact | 0.3 / 0.2 |

  Both formats cost < 1% of a run: keep both, compression on. Outlier / snow sidecars and the entrypoint stage 3 are
  exercised end to end only in item 11 (needs ISOFIT outputs).
- Readers: the solar irradiance is now read only when radiance units must be inferred (a labelled unit returns first).

## AC26-R9: item 9 HTML quick-look report (2026-10-31, Colab light env, item 8 outputs; no ISOFIT)
- `src/report.py`, called by entrypoint.sh after the writers (before the work dir is removed; a failure only logs a
  warning). One offline HTML per scene, `<fid>_L2_SR_report.html`, images base64-embedded: run / grade / checklist /
  stage times, L1 vs L2 true and false colour, flag overlay, AOT / H2O / elevation maps, spectra / sample pixels /
  histograms / pixel counts, optional `--ref` comparison (pairing by location: nearest reference pixel within one
  reference pixel spacing; by index when neither side has location and sizes match).
- Atmosphere, flags, atm_post and stage times exercised with a fake fixture (t9_fake.py) on the EnMAP case.

  | case | report s (no ref) | with --ref | reference | paired pixels | MAE |
  |---|---|---|---|---|---|
  | EnMAP L2A | - | 25 | the same L2A folder (self-check) | 261,474 / 261,534 | 0.0000 |
  | AVIRIS-NG rotated | - | 25 (351 before reading only the needed rows) | ang L2 corr_v2m2 | 71,840 / 72,686 | n/a (rdn stand-in) |
  | PRISMA L1 swath | 6 | 50 | PRS L2D .he5 (map-projected) | 193,882 / 220,393 | n/a (rdn stand-in) |
  | Tanager ortho | 6 | 168 | ortho SR .h5 | 132,260 / 132,459 | n/a (rdn stand-in) |
  | no location | 4 | - | - | - | - |

  Fixes found by the test: reference water bands (NaN) no longer invalidate every pixel; PRISMA L2 swath chosen by
  its VNIR_Cube; reference read limited to the rows that hold paired pixels. Report size 0.3-1.4 MB.

## AC26-D10: item 10 image hardening (2026-10-31; files only, built and scanned on the build PC)
- Dockerfile v1.3.0, two stages on ubuntu:24.04. builder: Miniforge 26.7.2-1 + compilers, conda env, CUDA torch,
  ISOFIT 4.1.5, 6S builds + sixs_check, sRTMnet / data / examples, surface prior, ISOFIT patches. runtime: fresh
  ubuntu:24.04 + apt upgrade + libgfortran5 / libgomp1 only; /opt/conda, /root/.isofit, /opt/sixs, /opt/aux,
  /opt/build and the pipeline copied in. Runtime re-checks: ldd of every 6S build, 6S selftest, fast_al wired into
  apply_oe, pipeline imports. Old base condaforge/mambaforge:24.9.2-0 (mambaforge is retired).
- Non-root: run_scene.sh runs as the caller's uid; HOME=/root kept for ~/.isofit/isofit.ini, /root and the data made
  world-readable (not writable), MPLCONFIGDIR / XDG_CACHE_HOME in /tmp, /work 1777.
- **Latent build break found:** environment.yml pinned gdal=3.9.* with rasterio=1.5.*; every rasterio 1.5 build needs
  libgdal >= 3.12, so the env never solved (the 1.2.0 Dockerfile had not been built with it). Repinned gdal=3.12.*.
  Checked on Colab: Miniforge 26.7.2-1 (base python 3.14.8) takes the env into base in 17 s -> python 3.12.16,
  gdal 3.12.4, mamba 2.10; isofit 4.1.5 pip-installs on top; nothing in isofit or src/ imports osgeo.
- compose: GPU reservation, shm_size 16g (Ray object store; Docker default 64 MB), generic IN_DIR / INPUT / OUT_DIR /
  DEM_DIR. run_scene.sh: any input file or folder, GPUS / SHM_SIZE / DEM_DIR (optional) / REF (report comparison).
  build_image.sh: ISOFIT default was still 3.7.7 (would have overridden 4.1.5) -> 4.1.5; non-root smoke test; trivy
  CRITICAL table + CRITICAL/HIGH json with a per-type count. deploy_airgap.sh: 6S selftest + CUDA check as non-root.
  .dockerignore is now an allowlist of what the Dockerfile copies.

## AC26-D10b: separate stack isofit-atm-docker, pip requirements, unit tests (2026-10-31)
- User: a separate Docker stack for atmospheric correction; leave the existing stacks alone. New top-level folder
  `isofit-atm-docker/` (own Dockerfile, docker-compose.yml, requirements.txt); `enmap-isofit-docker/` and the
  product's `docker/` untouched in the commit. Image name isofit-atm-docker:1.3.0.
- No conda: Ubuntu 24.04's python3.12 + venv from requirements.txt (pins = the versions of the verified Colab env;
  torch 2.14.1+cu130 via --extra-index-url). Nothing at run time uses GDAL tools or bindings; rasterio's wheel
  carries GDAL 3.12.1. Checked on Colab: a clean 3.12 venv installs requirements-dev.txt in 322 s.
- Unit tests (user rule: no tests, no ship; >= 30 % coverage): `tests/`, 76 tests on synthetic scenes (solar
  spectrum x gas transmittance x smooth soil / vegetation / water mixtures; no ISOFIT run, data or network), ~29 s.
  Covered: level classifier (radiance / dn / TOA never SR / SR / x1e4 / unknown), sensor id / time / units, grades,
  ENVI in all three layouts + wrong-header detection, GeoTIFF tags / descriptions / reversed bands, geometry, prep end
  to end (radiance, TOA via irradiance, SR pass-through + override, raw counts stop + override, no location / time,
  Tanager HDF5 radiance and SR), rfl_qc, atm_post (ghost rule) on a fake apply_oe output, aot_prior, harmonise
  (model format, own / const / nodata / ridge / class parts, gate, CLI, shipped models), writers (map, GCP swath,
  none), report (all sections, self-reference MAE 0, graceful gaps), and a chain prep -> atm_post -> writers -> report.
  Coverage 53.2 % (conda env and clean pip venv alike); 0 % left on what needs real products or a live ISOFIT run
  (EnMAP / PRISMA native preprocessors, fast_al, the research validators), which the item 11 trials exercise.
- Gate: pytest.ini --cov-fail-under=30; Dockerfile `test` stage (FROM runtime) runs it; build_image.sh builds
  --target test before tagging --target runtime.
- Fixes found while writing tests: none in the module; three fixture mistakes (radiance floor 1e-4 by design,
  shape test needs a 2200 nm band, mixed water pixels too bright in NIR).
