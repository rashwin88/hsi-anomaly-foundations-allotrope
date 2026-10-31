# Atmospheric correction: what we learned and what we built (as of 2026-10-27)

For the team and for future sessions. This is the consolidated reference: read it first. Then use
`README.md` for status and how to resume, `EXPERIMENT_LOG.md` for the evidence behind every claim
(IDs `AC26-*`), and `research_notes.md` for the literature.

## 1. The goal and the yardstick

- **What we do.** Turn L1 radiance into L2 surface reflectance for EnMAP, PRISMA and AVIRIS-NG, using
  ISOFIT 4.1.5 with the sRTMnet emulator. The output must be close enough to each vendor's own L2 that
  the anomaly models trained on vendor L2 see the same domain.
- **References.** EnMAP = DLR PACO L2A. PRISMA = ASI **L2C** (not L2D: the L2C pairs align at shift
  (0, 0) and score better). AVIRIS-NG = JPL `corr` v2.
- **Score.** Mean absolute reflectance difference (MAE) over land pixels on the 165-band common grid,
  with clouds excluded. Medians are over scenes, and harmonisation is scored leave-one-scene-out
  (LOSO).
- **Data split.** 15 training scenes per sensor (45 in total), spread across Indian terrain.
- **The holdout is never fitted on.** It is opened once, after the module and harmonisation are
  frozen.

## 2. The chain

```
L1 radiance --ISOFIT 4.1.5 (sRTMnet v120, 6S, presolve, analytical line, segmentation 50)--> rfl
  --atm_post: window mask -> outlier clip -> adjacency (EnMAP < 2 km) -> PACO-style terrain (steep EnMAP)--> rfl
  --harmonisation (per sensor, fitted on the 45 training pairs)--> "vendor-like" L2
  --app pipeline: band filter -> gap interpolation -> 165-band grid--> anomaly models
```

- **Module code.** Branch `vinoth/atm-isofit4`, worktree `C:/wt/atm-isofit4`, folder
  `enmap-isofit-docker/`. Uncommitted; the user transfers files manually.
- **Design.** LLD `docs/lld/atm-correction.md` on that branch.
- **Frozen-module hash:** `75385785732aa4c9`.

## 3. Results on the 45 training scenes (land MAE vs vendor, median over held-out scenes)

| Sensor | Raw module | Chosen harmonisation | After | Status |
|---|---|---|---|---|
| AVIRIS-NG | 0.0081 | identity, i.e. none (residuals differ per flight) | 0.0081 | done |
| EnMAP | 0.0119 | H5: spectral ridge on land; snow and water untouched | 0.0094 | chosen |
| PRISMA (free AOT) | 0.0242 | H6: H3 on land; others untouched; 1-SD gate | 0.0128 | superseded |
| **PRISMA + climatology AOT** | 0.0220 | H6 | **0.0118** | proposed (needs a module change) |

For reference, inter-vendor agreement in the literature is in the 0.01–0.02 range. So EnMAP and
AVIRIS-NG are at that level, and PRISMA gets there after harmonisation.

## 4. Key learnings (each with its log ID)

### ISOFIT behaviour

- **ISOFIT ignores terrain with the `.h5` sRTMnet emulator.** cos_i, sky view and background
  reflectance are only used in multipart mode (`.6c`/MODTRAN). That is why adjacency and terrain are
  post-processing modules we wrote (AC26-B/C/T).
- **ISOFIT writes nodata outside its inversion windows.** The windows are 350–1360, 1410–1800 and
  1970–2500 nm. `atm_post` sets nodata there and clips outliers to [−0.2, 2.5].
- **Do not use `--pressure_elevation` together with an elevation cap.** The cap is ineffective
  (AC26-M3).
- **Plain DDV and free AOT are both unreliable on PRISMA.** Our free AOT collapses to 0.05–0.15 where
  the climatology says 0.4–0.75 (AC26-A2). The NASA POWER monthly climatology, used as an AOT window
  (± 0.15), fixes a third of the visible offset (AC26-A2b).

### Each vendor has a fingerprint, and harmonisation learns it

- **PACO (EnMAP):**
  - **Snow.** PACO gives snow its own spectral shape. Our "mountain blue tail" is that shape, not a
    terrain error (AC26-M7). Snow is therefore left untouched by harmonisation, using DLR's
    `QL_QUALITY_SNOW` mask.
  - **Haze and cirrus.** PACO removes them without leaving a class gap (AC26-K).
- **ASI (PRISMA):**
  - Its AOT is high, 0.4–0.8 (AC26-P1). Its visible offset and NIR gain of about 1.09 track ASI's own
    AOT, not ours (AC26-N1).
  - It has a spurious peak at 1250 nm (AC26-N2), and its reflectance collapses beyond 2350 nm
    (AC26-S1).
  - All land covers share one bias shape (AC26-K), so one land model is enough.
- **JPL (AVIRIS-NG):**
  - JPL uses ATREM with 50 km visibility. The 2016 products carry per-channel correction factors; the
    2018 products have all factors at 1.0.
  - The poorer 2018 flights differ by per-flight broadband gains (AC26-D1), which cannot be learned
    from deployable inputs.
  - JPL's L2 ships only reflectance and water vapour: **no cloud or snow mask**.

### Harmonisation (`scripts/harmonise.py`)

- **What it is.** A small statistical model fitted on our-vs-vendor pixel pairs from the training
  scenes. It maps our reflectance to the vendor's: `y = f(x, z)`, where x is our spectrum and z the
  scene covariates.
- **H-ladder:**
  - **H0** identity.
  - **H1** per-band `y = a·x + c`.
  - **H2** H1 per class group (land, water, snow).
  - **H3** per band, with a and c varying linearly with the scene covariates.
  - **H4** "spectral ridge": every output band is a linear mix of **all** 165 input bands plus the
    covariates, so it can learn cross-band artefacts.
  - **H5** H4 on land, identity on snow and water.
  - **H6** H3 on land, identity elsewhere, plus an out-of-domain gate.
- **Ridge penalty.** This is the regularisation strength (`RIDGE`) that shrinks the fitted weights
  toward "no correction".
  - Larger values are safer with few scenes.
  - EnMAP H4/H5 works best at 1e-5, because it has many pixels and a stable residual.
  - PRISMA H3/H6 works best at 1e-3, because it has fewer comparable scenes and a covariate-dependent
    residual.
- **Scene covariates.** Per-scene numbers known at deployment:
  - our retrieved AOT;
  - our retrieved water vapour (H2O);
  - solar zenith / 90;
  - mean elevation (km);
  - the POWER climatology AOD (PRISMA only).

  They are standardised across the training scenes. **Never** use a vendor-only quantity such as
  ASI's own AOT.
- **Clamping and the gate.** At prediction, covariates are clamped to the training range. H6 adds a
  further rule: a scene more than **1 standard deviation** beyond the training range on any covariate
  gets no correction.
  - On PRISMA this catches only the one high-Himalaya scene (32.5° N, 77.5° E). H3 there went
    0.0216 → 0.037; gated, it stays at 0.0216.
  - A 0.5 SD gate is too tight: it also skips a scene that harmonisation fixes (0.0091 vs 0.0289).
- **What did not work:**
  - Harmonising snow and water: too few scenes, so it overfits.
  - The H4 spectral ridge on PRISMA: overfits at 15 scenes.
  - Covariate models on AVIRIS-NG: they blew up on Spiti before clamping was added.

### Bands and gaps

- **The common grid.** 165 bands, 460–2450 nm at 10 nm. Four exclusions: 912–978, 1131–1152,
  1350–1450 and 1800–1950 nm.
- **Only one grid band falls in ISOFIT's nodata: 1960 nm** (ISOFIT's window starts at 1970). This is
  the same for all three sensors, AVIRIS-NG included.
  - The app pipeline fills 1960 by interpolating across the 1800–1950 gap, which pulls it toward the
    much brighter 1795 nm.
  - Bias at 1960: AVIRIS-NG −0.010, EnMAP +0.005, PRISMA +0.038 (AC26-G0).
- **G1** (extrapolate from the window edge instead) helps only modestly: PRISMA 11 of 15 scenes,
  0.041 → 0.037. It is **not adopted**; harmonisation absorbs the mean.
- **The other "bad" gap-edge bands** (1130, 1340, 1460 nm) are water-vapour shoulders, not fills.
  Harmonisation handles them per band.

### Native bands outside the core: options (decision pending, 2026-10-28)

Harmonisation now targets each instrument's native bands. Every band belongs to one group: core,
edge_lo (<460 nm), edge_hi (>2450 nm), weak (912–978, 1131–1152 nm) or strong (1350–1450,
1800–1950 nm). Each group is fitted and scored on its own, never mixed into the core number
(`harmonise/bands.py`).

Evidence: AC26-NBW, LOSO over the training scenes, MAE against the vendor. For comparison, the core
after harmonisation is PRISMA 0.0118 and EnMAP 0.0094.

**What the vendors do:**
- **ASI (PRISMA L2C)** fills every band. In the strong windows it reports near-total absorption
  (median 0.002). It does not interpolate across them.
- **PACO (EnMAP L2A)** leaves 1343 nm and 1367–1390 nm as nodata in every scene. At 1449 nm it has a
  real value. EnMAP has no bands in 1800–1950 nm.
- **ISOFIT (ours)** returns nodata outside 350–1360 / 1410–1800 / 1970–2500 nm.
- **FLAASH** inverts every band. Users then mark the strong-window bands as bad, because they are noise.

**Strong windows: the open choice**

| option | PRISMA error | for | against |
|---|---|---|---|
| A. constant per band (vendor training median, ~0.002) | 0.012; land classes 0.002–0.003 | closest to ASI | synthetic: the same value everywhere, carries no surface information |
| B. nodata | n/a | honest; what ISOFIT, PACO (EnMAP) and FLAASH users do | penalised if the organisers score every band against the vendor |
| C. ridge from our core bands + covariates | 0.017; snow 0.030 (const 0.069) | physically informed; best on snow | worse than A on ordinary land |
| D. our own ISOFIT value where it exists | PRISMA 1417–1449 poor (r ≤ 0.5); EnMAP 1449 good (0.029) | measured | mostly absent; poor on PRISMA |

Any filled band would be flagged as synthetic in the output metadata (ENVI `bbl` and a group tag).
For EnMAP, "match the vendor" means B for 1343 and 1355–1390 nm, and D (own band + z) at 1449 nm.

**What decides between A and B** is how the organisers score. If they compare every band against
vendor L2, A wins. If they drop bad bands, as FLAASH users do, it doesn't matter and B is the more
defensible choice. Ask them.

**Weak windows and the edges: the open choice**

| choice | PRISMA | EnMAP | note |
|---|---|---|---|
| spectral ridge (core bands + z) | weak 0.028, edge_hi 0.027 | weak 0.016, edge_lo 0.026 | best; uses every core band, so more fragile on unusual scenes |
| own band + z (H3 form) | edge_lo 0.024, weak 0.031 | weak 0.019, edge_lo 0.026 | simpler and more robust |
| identity (raw ISOFIT) | edge_lo 0.027, weak ~0.04 | edge_lo 0.031, weak 0.021 | no fitting |
| PRISMA > 2477 nm: constant | 0.024–0.033 | (none) | our bands are thin there (valid 36–74%) |

**Also open:**
- whether the H6 scene gate also applies to these groups (proposed: yes, out of range → identity);
- snow in the strong windows (ridge, A, or B).

**Decided by the user (2026-10-28): sensor-level, one model per band group per sensor.**

Rule: use own+z unless the ridge is more than ~10% better.

| sensor | group | handling |
|---|---|---|
| PRISMA | strong | constant per band (A) |
| PRISMA | edge_lo | own+z |
| PRISMA | weak | ridge |
| PRISMA | edge_hi ≤ 2470 nm | ridge |
| PRISMA | edge_hi ≥ 2477 nm | constant |
| EnMAP | strong 1343, 1355–1390 nm | nodata, as PACO |
| EnMAP | strong 1449 nm | own+z |
| EnMAP | edge_lo | own+z |
| EnMAP | weak | ridge |
| AVIRIS-NG | all bands | identity |

- Gate-out scenes fall back to identity.
- The constant/nodata choice stays a switch, with these defaults.
- PRISMA 1131–1142 nm stays at 0.043–0.057 under every model. It is reported as is, as the worst group.

### Pointers for the final commit (choose before release)

1. **EnMAP: drop harmonisation (core and non-core → identity).**
   - The user intends this, for two reasons:
     - On the 165-grid holdout (AC26-HO165), H5 made the median land MAE worse: identity 0.0103,
       H5 0.0121. 2 of 5 scenes got worse by 25–57%.
     - The training LOSO already showed scene-level losses of up to 42%.
   - The native holdout (AC26-HON, 2026-10-28) repeats this:
     - core land 0.0098 → 0.0100, all non-cloud 0.0097 → 0.0109;
     - the weak-band ridge is worse (0.0148 → 0.0205);
     - only edge_lo own+z helps (0.0263 → 0.0162).
   - Raw ISOFIT is already close to PACO.
   - Cost: the blue +0.02 bias stays.
   - **Variant:** keep only the edge_lo part (< 460 nm). It is holdout-informed, so declare it as such.
   - **Declare it:** this choice was made after seeing the holdout. The holdout identity score is still
     unbiased, because identity has nothing fitted.
   - **The user decides at the final commit.**
2. **PRISMA: ship the native model `final_models/native/prisma_native.npz`** (AC26-FZN, sha d59710e5…).
   - H6 core plus the non-core picks, 21 training scenes.
   - Native holdout (AC26-HON): land 0.0244 → **0.0099**, all 5 scenes better, every band group better.
   - The strong-window constant now delivers the vendor's values (identity delivered 15–50% of them,
     and badly).
   - The 165-grid holdout agrees: 0.0249 → 0.0095 against L2C, 0.0279 → 0.0121 against L2D.
   - Watch the blue end: the worst bias at 460 nm is −0.026 (165 grid).
   - **Settled:**
     - H6, not H5 (user, 2026-10-28);
     - Kerala 20260202051921 is excluded from fitting and analysis (user);
     - scenes whose SWIR ends early are handled by the module fallback, which skips the window ridge
       part and records it in `skipped_parts`.
3. **AVIRIS-NG: identity.** Holdout 0.0051.

## 5. Operations learnings (Colab)

- **Runtime.** v6e-1 TPU runtime, 44 vCPU, with 3 scenes in parallel at 14 cores each.
  - About 23–30 min per EnMAP or PRISMA scene, and about 20 min per AVIRIS-NG 600-line subset.
  - A 45-scene batch takes about 6–7 h.
  - Measured: one PRISMA A2 run took 23 min end to end on 14 cores.
- **Never `pkill -f <pattern>` inside `ssh '...'`.** It kills its own shell and any waiter whose
  command line matches.
- **Don't leave prototype `adjacency.py`/`terrain.py` in `/content`.** They shadow the module's own
  copies under `python -m`.
- **Use `/content/mamba/envs/isofit4/bin/python` for analysis.** The system Python has no rasterio.
- **Where things are.** Drive `PS_11/stage3_experiments/atm_corr/`:
  - `planC2/`: the 45 scenes, with paired samples and int16 cubes;
  - `planA2/`: the 15 PRISMA runs with the climatology AOT;
  - analysis tables: `*.txt`;
  - the global AOD grid: `aod_clim_1deg.npz`, 255 KB.

## 6. What is next (agreed order)

1. **Module change (nano-chunked).** PRISMA AOT window = POWER monthly climatology ± 0.15, read from
   the bundled `aod_clim_1deg.npz`. Use the plain month value.
   - **The covariate is `clim55`** (`aod_clim.tsv` column 6). It is exactly the grid month value the run
     was pinned with (`aotpin.txt` hi − 0.15, and what `aot_prior` records).
   - **`month55` (column 9) is that particular year's month.** It is not deployable and is NaN for recent
     scenes.
   - On 2026-10-28 it was briefly and wrongly switched to `month55`, then reverted (AC26-CLIM).
   - **Done** (module chunk 2).
2. **Fit the final harmonisation models.** **Done:**
   - the 165 grid (AC26-FZ);
   - native bands in the module format, on all training scenes (AC26-FZN: EnMAP 27, PRISMA 21).
3. **Freeze, then run the holdout once.** **Done:** AC26-HO165 and AC26-HON.
   - The module `harmonise/` is complete (chunks 5–12; the CLI is `python -m harmonise`).
   - It is uncommitted on worktree branch `vinoth/atm-isofit4`.
4. **Open items (the next phase):**
   - the EnMAP final-commit decision (section 4, pointer 1);
   - a FLAASH comparison, if the organiser can share outputs;
   - a speed study (section 7);
   - an AVIRIS-NG snow mask from our own reflectance (NDSI);
   - an EnMAP water-specific check (blue +0.02).

## 7. FLAASH and speed (for the organiser discussion)

- **What FLAASH is.** ENVI FLAASH (Spectral Sciences / AFRL) is MODTRAN-based.
  - It computes radiative-transfer tables for the scene geometry once.
  - It retrieves water vapour per pixel from a band ratio, and uses one aerosol value per scene
    (fixed visibility, or a dark-pixel retrieval).
  - It inverts every pixel with a **closed-form equation** and does adjacency with a spatial blur.
  - The per-pixel work is a few arithmetic operations per band, so a scene takes minutes. The cost is
    mostly the MODTRAN calls.
- **What ISOFIT does instead.** An iterative Bayesian (optimal-estimation) fit of surface and
  atmosphere together, per superpixel, with an emulator call at each iteration. That makes it slower,
  but it gives uncertainty and is more robust.
  - JPL's own `corr` product (ATREM) is close to the FLAASH family.
- **Similarity.** Our output should be close to FLAASH in spectral shape; both are physics-based.
  - Absolute levels will differ by aerosol and water-vapour assumptions, as they do between any two
    vendors (0.01–0.03).
  - If FLAASH is the judged reference, the harmonisation machinery can be refitted to FLAASH pairs
    with no code change. It only needs FLAASH outputs for a few training scenes.
- **Measured (AC26-SPD, AC26-BAT, 2026-10-28/29):**
  - Solo L4 (6 physical cores), challenging scenes: EnMAP 94 min, PRISMA 62 min, AVIRIS-NG 600-line 92 min.
    The L4 is 2–2.7× slower than the v6e on every stage alike.
  - Everything is CPU-bound (no I/O or memory wait). The three ISOFIT solves are about 75% of the time; the
    6S LUT 10–36% (mountain PRISMA); our own code under 10%.
  - Per-pixel solve: about 70% dense n×n algebra on matrices that are diagonal here (Seps, H). An exact
    rewrite gives 1.5–1.9×; batching the remaining Cholesky in torch gives 2.2–3.6× on CPU and 2.9–6.8× with
    the L4 GPU, identical to ISOFIT to 1e-14. The per-pixel Python forward-model part (2.5–4 ms) is next.
- **Speed levers still open:**
  - use all 44 cores on one scene;
  - pin AOT (A2), which makes the LUT and presolve smaller;
  - use a larger segmentation size;
  - use ISOFIT's empirical-line mode;
  - or a "FLAASH-like fast path" on our own LUT: per-pixel H2O plus the climatology AOT, with a
    closed-form inversion.
