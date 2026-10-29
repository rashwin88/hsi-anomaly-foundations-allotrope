# Shipping FT04 in the product

> **2026-10-29: Indradhanu V2 now points to Indradhanu Heavy** (FTH3 epoch 5, `v2.1.0`,
> research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/), behind the same codename and architecture key,
> with cloud exclusion narrowed to thick cloud (`thin_cloud_is_cloud=False`). FT04 (`v2.0.0`) stays in the models
> volume as the manifest's first alternative. Everything below describes the FT04 ship and still holds for the
> serving path.

**Status:** proposed 2026-10-19, for agreement. **Model:** FT04 (research/experiments/indradhanu-finetune/
FT04-final-joint/) - native-band reconstruction plus a segmentation head; with cloud exclusion on its own masks the
best real-bench result (pAUC 0.170 vs shipped v0.2.0 0.159; like for like with exclusion, 0.170 vs 0.165).

## Why
One model gives reconstruction at each sensor's own bands, better reconstruction on all three sensors, and the masks
that cloud exclusion needs (v0.2.0 has no segmentation head). Detection is a near-tie with v0.2.0; the case is the
masks, native bands and one model instead of two.

## What exists today (read-only map, 2026-10-19)
- The inferencer factory has no spectral-stem entry; the resolver has no capabilities entry or config builder;
  `_anomaly_scoring_run` never calls `set_wavelengths` or drops untrained bands.
- `band_filter_apply` can keep native bands (`use_common_wavelength_grid=False`); defaults resample to 165, which
  v0.2.0 needs. One band-filter output feeds every model picked in an action; nothing checks the grid.
- Masks reach scoring only as `keep_mask.tif`, fed into the score's fit (= `fit_on_kept=True`). What was evaluated is
  the opposite: score on all valid pixels, zero cloud afterwards (`fit_on_kept=False`).
- Codename routing is last-wins; reusing "Indradhanu" would silently re-route the hyperspectral defaults.
- `scene_segmentation`'s brightness assumes ascending bands - wrong on native PRISMA (descending).

## Design - two phases

**Phase 1: FT04 available as an opt-in model next to v0.2.0 (no defaults change).**
1. `app/` - one implementation shared by product and evaluator:
   - register `spectral_stem_finetune` / `spectral_stem_mae` in the inferencer factory -> `SpectralStemInferencer`;
   - move the tiled segmentation (`EvalModel.segment`: 128 px, stride 64, sigmoid > 0.5, priority cloud > snow >
     shadow > water) into `SpectralStemInferencer.segment_full_scene`, plus a `read_bands(cube, wl)` helper that
     applies `trained_band_mask`; `app/evaluation/models.py` calls them.
2. `backend/allotrope/foundation_models/resolver.py`:
   - capabilities entry for the new architecture: methods as v0.2.0, patch 128 / stride 32, `pixel_stats_relpath=None`
     (normalisation travels in the checkpoint), new fields `band_grid="native"` and `exclude_classes=("cloud",)`;
   - config builder returns the manifest's embedded `model_config` (the api cannot torch.load);
   - **a new codename** (your choice - see Decisions), so v0.2.0 / "Indradhanu" routing is untouched.
3. `_anomaly_scoring_run.py`, for stem models: read bands (trained mask, `set_wavelengths`), reconstruct, segment,
   score on the plain keep mask, **then** zero `exclude_classes` - exactly the evaluated semantics. Write
   `models/<slug>/mask_cloud.tif` (+ class map) and record the exclusion and band count in stats / summary
   (additive keys only - the frontend reads by key).
4. `anomaly_scoring.validate_config` (+ the api's output-ref check): a native-grid model requires an upstream band
   filter run with `use_common_wavelength_grid=False`, and a 165-grid model the opposite; clear error otherwise.
5. Fix `scene_segmentation`'s brightness for descending band lists (select VNIR bands by wavelength, not position).
6. Tests: `test_imports` for the new paths; unit tests for the resolver builder and the grid check; a parity test -
   PRISMA bench set1 through the product path vs the evaluator's cached FT04 score (same numbers within float16).
7. Ops: FT04 `best.pt` + `current.json` (full manifest) into `<models>/spectral_stem_finetune/`; snapshot bundle;
   rebuild api and worker; `docker compose ps -a` shows the worker `running` (the stack runs in WSL Ubuntu).
8. Docs: 04-models, 06-backend, 09-known-issues, LLDs (spectral-stem, finetune, evaluation), ROADMAP row.

**Phase 2 (after Phase 1 is verified in the running stack): FT04 the default** - hyperspectral defaults route to its
codename and band_filter_apply defaults to native; v0.2.0 stays selectable with a 165-grid band filter.

## Decisions for you
1. **Codename** for FT04 (must differ from "Indradhanu" or v0.2.0 is renamed deliberately).
2. **Phase 2 in the same piece of work, or after you have used Phase 1?** I recommend after.
3. **EnMAP:** the default band filter already removes provider cloud / shadow / haze pixels before scoring, unlike
   the PRISMA evaluation. Keep (product convention), or turn it off for FT04 so its own masks decide?

## Risks
- CPU cost in the worker: native bands (~200-370) at stride 32 plus a segmentation pass; measure on the stack.
- No behavioural backend tests exist beyond imports - the parity test is the guard.
- Frontend: a new architecture shows no erosion knob and no flow diagram (cosmetic; another session owns frontend).
