# 61 - Shipping FT04 as Indradhanu V2

> **The one thing this part teaches:** putting FT04 into the product took six small decisions, each
> forced by something the product already did - a new codename because codename routing is last-wins,
> a manifest that carries the whole model config because the api cannot load a checkpoint, a band-grid
> check because one band filter feeds every model in an action, `read_bands` because the model must be
> told which wavelengths it is reading, cloud zeroed *after* scoring because that is the order the
> benchmark measured, and a parity run that proves the product path gives the evaluator's numbers.

## Where we are

Part 60 ended on "next is product wiring". On 2026-10-19 that happened: FT04 is served in
`anomaly_scoring` under the codename **Indradhanu V2** (Devanagari script `इंद्रधनु V2`), opt-in next to
the shipped v0.2.0 "Indradhanu", which stays the default. The design was agreed first, in two phases:
Phase 1 (done) makes V2 selectable without changing any default; Phase 2 (later, after V2 has been used
in the product) would make it the hyperspectral default.

| file | what it holds |
|---|---|
| `docs/proposals/ship-ft04.md` | the design: what existed, the two phases, the decisions |
| `backend/allotrope/foundation_models/resolver.py` | `ModelCapabilities.band_grid` / `exclude_classes`, the `spectral_stem_finetune` entry, `_model_config_from_manifest`, `list_catalog`'s collision warning |
| `scripts/make_model_manifest.py` | writes `current.json` from a checkpoint, with the `model_config` embedded |
| `app/foundation_models/inferencers/spectral_stem_inferencer.py` | `read_bands`, `segment_full_scene`, `CLASS_NAMES` |
| `backend/allotrope/action_types/_anomaly_scoring_run.py` | `_check_band_grid`, the native-band branch, exclusion after scoring, the new outputs and stats keys |
| `tests/test_backend/test_native_band_models.py` | the resolver and grid-check unit tests |
| `research/experiments/indradhanu-finetune/FT04-final-joint/product_parity.py` | product path against the evaluator, on the PRISMA bench |
| `docs/lld/spectral-stem.md`, section "Serving (Indradhanu V2, since 2026-10-19)" | the authoritative summary |

What was shipped, from the manifest at `C:/es/ship/spectral_stem_finetune/current.json` (dev laptop):

| field | value |
|---|---|
| architecture | `spectral_stem_finetune` |
| version, epoch | `v2.0.0`, epoch 10 (of fine-tuning) |
| params | 5,886,451 (about 5.9 M) |
| val_loss | 0.05578 |
| checkpoint file | `spectral_stem_finetune_v2.0.0_epoch10.pt` |

> **Analogy.** A hospital takes on a new specialist who trained abroad. Before her first patient, the
> hospital gives her a name badge that no other doctor on the ward shares (or the switchboard puts
> calls through to whoever was added last), files her credentials in a form the front desk can read
> without phoning the overseas university, checks every referral is in the units she works in, and
> has her see a few patients the old consultant already examined - to confirm she reaches the same
> findings by the same route before anyone relies on her.

## 1. Why the codename had to change

A **codename** is the name an analyst picks in `anomaly_scoring` (part 30): "Indradhanu", "MNF-RX". The
worker turns it into a model with one dictionary:

```python
# backend/allotrope/action_types/_anomaly_scoring_run.py (and anomaly_scoring.py's validate_config)
by_codename = {m.codename.lower(): m for m in catalog}
```

`catalog` comes from `list_catalog`, which walks the model directories in sorted order
(`for p in sorted(models_root.iterdir())`). A Python dict built this way keeps the **last** value for a
repeated key. So if two manifests both said "Indradhanu", the one whose directory sorts last would win.

Work it by hand. The two directories are `hyperspectral_segformer_mae` (v0.2.0) and
`spectral_stem_finetune` (FT04). Sorting compares the first letters: `h` is character 104, `s` is 115.

```
104 < 115   ->   hyperspectral_segformer_mae sorts first, spectral_stem_finetune last
"indradhanu" -> v0.2.0, then overwritten by FT04       ->  FT04 wins
```

Every default hyperspectral action (`_DEFAULT_PRISMA`, `_DEFAULT_ENMAP`, `_DEFAULT_AVIRIS_NG` in
`anomaly_scoring.py` all say `"model_codenames": ["Indradhanu"]`) would have been re-routed to FT04
without anyone choosing it - and, with section 3's check, every one of them would then fail, because the
default band filter resamples to the 165-band grid. `list_catalog` does log a warning on a collision
("codename collision ... Rename one of them"), but a log line is not a guard. Hence a new name.

## 2. A manifest that carries the whole config

The api lists models and validates actions; it never loads torch (the repo's lazy-import rule: action-type modules import torch only inside the worker's `run()`). So
everything the resolver needs must be readable from `current.json`, a JSON file next to each checkpoint.
For v0.2.0, `_model_config_from_manifest` rebuilds the model config from a few summary fields. For the
stem models that is not possible:

```python
# resolver.py, _model_config_from_manifest
if arch in ("spectral_stem_finetune", "spectral_stem_mae"):
    # The stem configs carry far more than a manifest summary can rebuild (sources, band weighting, the stem
    # variant), and the api cannot torch.load; the manifest therefore embeds the checkpoint's full
    # config.model_config (scripts/make_model_manifest.py).
    cfg = current.get("model_config")
    if not isinstance(cfg, dict) or cfg.get("model_type") != arch:
        raise KeyError(f"{arch}: current.json must embed the checkpoint's model_config")
    return cfg
```

`scripts/make_model_manifest.py` is the one place that does load the checkpoint: it copies
`ck["config"]["model_config"]` into `current.model_config`, counts the parameters (skipping the
`stats.*` and `trained_wl*` buffers) and copies the checkpoint next to it as
`<architecture>_<version>_epoch<N>.pt`. If the embedded config is missing or names another
architecture, the resolver raises `KeyError` - and `list_catalog` skips manifests that raise `KeyError`,
so a bad manifest makes the model disappear from the picker rather than load wrongly.

The capabilities entry sets `pixel_stats_relpath=None`: there is no per-band stats file, because the
normalisation curve travels inside the checkpoint (`stats.wl`, `stats.mean`, `stats.std`; part 49).

## 3. One band filter, two kinds of model: the grid check

v0.2.0 reads the 165-band common grid (part 03). V2 reads each sensor's **native** bands - the
wavelengths the instrument actually measured. `band_filter_apply` can produce either
(`use_common_wavelength_grid`, default `true`), but an `anomaly_scoring` action reads **one** upstream
band-filter output for every model it runs. So the resolver now records which grid each model needs:

```python
# resolver.py, ModelCapabilities
band_grid: str = "common165"              # v0.2.0 and every older model
# the spectral_stem_finetune entry:
band_grid="native", exclude_classes=("cloud",)
```

and the worker checks the vendable before running any foundation model on a hyperspectral scene:

```python
# _anomaly_scoring_run.py, _check_band_grid
grid = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
on_grid = len(wavelengths) == len(grid) and np.allclose(wavelengths, grid)
if band_grid == "native" and on_grid:      raise ValueError("... reads native bands: run band_filter_apply with use_common_wavelength_grid=false ...")
if band_grid == "common165" and not on_grid: raise ValueError("... needs the 165-band grid: run band_filter_apply with use_common_wavelength_grid=true ...")
```

The test is "do these wavelengths equal the grid", not "are there 165 of them". A native list that
happens to have 165 bands still fails `np.allclose` and counts as native. Classical detectors (MNF-RX
and friends) skip the check: they take either grid.

The check runs in the worker, not in `validate_config`, because the api cannot see the upstream band
filter's settings at submit time (`docs/09-known-issues.md`). The consequence: **one action cannot run
V2 and v0.2.0 together** - whichever does not match the band filter makes the action fail.

## 4. `read_bands`: tell the model what it is reading

A spectral stem reads a band list of any length, but only if it is told the wavelengths (parts 31 to 42).
The product's older path never called `set_wavelengths`. The new helper does that and drops the bands
the model has never trained on:

```python
# spectral_stem_inferencer.py
def read_bands(self, cube, wavelengths_nm):          # cube: (N, H, W)
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    keep = self.model.inner.trained_band_mask(wl).cpu().numpy()   # (N,) bool, within 15 nm of trained_wl
    self.set_wavelengths(wl[keep])
    return cube[keep], wl[keep], np.flatnonzero(keep)            # (N_kept, H, W), (N_kept,), indices
```

`trained_band_mask` is part 49's rule: keep a band within 15 nm of any wavelength in the `trained_wl`
record, which grew during fine-tuning as FT04 read EnMAP, PRISMA and AVIRIS-NG. Work three bands
against a record ending `... 2395, 2405, 2440`:

```
band 2400 nm   nearest 2395 or 2405: |2400 - 2395| = 5     5 <= 15   keep
band 2452 nm   nearest 2440:         |2452 - 2440| = 12   12 <= 15   keep
band 2470 nm   nearest 2440:         |2470 - 2440| = 30   30 >  15   drop
```

The docstring states the trap in its second half: it "also sets the wavelengths for the next
predict_full_scene / segment_full_scene". Skip it and the model keeps the previous scene's wavelengths
(`docs/lld/spectral-stem.md`). In the worker, the branch is only taken for `band_grid == "native"`, and
everything after it - reconstruction, scoring, segmentation, the reconstruction preview - uses the
kept cube and wavelengths. The stats record how many bands the model read (`bands_read`).

## 5. Score first, then zero the cloud

V2 has a segmentation head (part 47). `segment_full_scene` runs it over the scene in 128 px windows at
stride 64, averages the sigmoid probabilities where windows overlap, and assigns classes by priority
cloud (cloud or thin cloud) > snow > shadow > water > land:

```python
CLASS_NAMES = ("land", "water", "cloud", "shadow", "snow")   # class map ids 0..4, 0 outside valid
```

The worker then zeros the model's `exclude_classes` - for V2, cloud:

```python
# _anomaly_scoring_run.py, after compute_score(...)
class_map, _ = inferencer.segment_full_scene(model_cube, spatial_valid.astype(bool))
excluded = np.isin(class_map, [inferencer.CLASS_NAMES.index(c) for c in m.exclude_classes])
score = np.where(excluded, 0.0, score).astype(np.float32)
```

Why after, and not "remove the cloud from the mask, then score"? Residual Mahalanobis (parts 45-46)
first fits its whitening - the spread of the residual - over every pixel in the mask, and measures each
pixel in units of that spread. Which pixels are in the fit changes every score. A one-number
illustration (this course's, not a measurement): four land residuals 0.02, -0.02, 0.02, -0.02, two cloud
residuals 0.06, -0.06, and a target residual 0.08.

```
fit on all six (mean 0):
  squares   4 x 0.0004 + 2 x 0.0036 = 0.0016 + 0.0072 = 0.0088
  variance  0.0088 / 6 = 0.001467        spread = sqrt(0.001467) = 0.0383
  target    0.08 / 0.0383 = 2.09 spreads

fit on land only:
  variance  0.0016 / 4 = 0.0004          spread = sqrt(0.0004)   = 0.0200
  target    0.08 / 0.0200 = 4.00 spreads
```

Same pixel, twice the score. Neither order is wrong in principle, but **the benchmark measured "fit on
everything valid, then zero the cloud"**, and the product must run what was measured. The evaluator's
other order (`fit_on_kept: true` in `app/evaluation/scorers.py`) was tried on FT03 and was no better
(15.7% against 16.5% caught at the top 0.1%, `docs/lld/evaluation.md`).

A note on the keep mask: if the analyst also attaches an upstream `scene_segmentation` keep mask, that
mask does feed the fit, as it always has. The parity run below used no upstream keep mask - only the
valid pixels - which is the configuration the numbers describe.

Three things are written per model, under `models/<codename>/`:

| output | contents |
|---|---|
| `mask_cloud.tif` | uint8, 1 where the score was zeroed |
| `class_map.tif` | uint8, the head's classes 0 land .. 4 snow |
| `stats.json` keys | `bands_read`, `excluded_classes` (`["cloud"]`), `excluded_percent` |

`excluded_percent` is `round(100 x excluded pixels / valid pixels, 3)`. On a 4 x 4 tile with 14 valid
pixels of which 3 are cloud:

```
100 x 3 / 14 = 300 / 14 = 21.42857...   ->   21.429
```

The keys are additive: the frontend reads stats by key, so new keys break nothing - and nothing in the
viewer shows them yet (`docs/09-known-issues.md`).

## 6. The parity run: the guard

There are no behavioural backend tests in this repo; `tests/test_backend/test_native_band_models.py`
checks the resolver (embedded config read, missing or mismatched config refused) and the grid check
(each grid accepted by its own kind, refused by the other) - but not the numbers. The guard for the
numbers is `product_parity.py`: it builds the inferencer **from the shipped manifest** through the
resolver's `_from_manifest` and `get_inferencer`, then runs `read_bands`, `predict_full_scene`,
`compute_score(method="residual_mahalanobis")`, `segment_full_scene` and the zeroing, exactly as the
worker does, and compares against the evaluator's FT04 numbers on the PRISMA bench.

Result (2026-10-19, `docs/lld/spectral-stem.md`):

| scene | bands read | pAUC plain | pAUC cloud-excluded | excluded |
|---|---|---|---|---|
| set1 | 185 | 0.2460 | | |
| set4 | 186 | 0.2106 | 0.2199 | 11.83% of valid pixels |

Identical to the evaluator's run; ROC-AUC within 0.0005. Two things made "identical" possible: the
evaluator (`app/evaluation/models.py`) now calls the same `read_bands` and `segment_full_scene`, so
there is one implementation of band reading and segmentation, not two that could drift; and part 60's
fear of a quiet mismatch - the sorting bug that a cross-check found - is the reason a parity run exists
at all.

These per-scene pAUCs are not comparable with part 60's four-scene 0.170: they are single scenes. What
matters here is only that product and evaluator agree.

## 7. What did not change

- **Defaults.** Every hyperspectral default still picks "Indradhanu" (v0.2.0); `band_filter_apply` still
  defaults to `use_common_wavelength_grid: true`. Phase 2 would change both.
- **EnMAP's band filter** still removes the provider's cloud, cloud-shadow and haze pixels
  (`quality_masks_to_apply`) before scoring, unlike the PRISMA evaluation behind V2's numbers. Kept as
  the product's convention (decision, 2026-10-19).
- **The detection case.** Like for like, both with cloud exclusion on FT04's masks, the real bench reads
  v0.2.0 0.165 against FT04 0.170 pAUC (`FT04-final-joint/README.md`, 2026-10-19) - a tie. Part 60 wrote
  that this comparison had not been run; it has now. What V2 adds is the masks, native bands and better
  reconstruction, in one model.
- **One product fix rode along.** `scene_segmentation`'s brightness took the mean of "band 0 up to the
  VNIR edge". On native PRISMA, whose bands run downwards (anomaly course, part 22), band 0 is the
  far SWIR. `app/utils/spectral_indices/scene_indices.py` now picks the VNIR bands by wavelength.

Still open (`docs/09-known-issues.md`): the CPU cost on the stack (native bands at stride 32 plus a
segmentation pass) is unmeasured, and the viewer has no panel for the masks.

## Common confusions

**"Indradhanu V2 replaced Indradhanu."**
No. It is opt-in. "Indradhanu" is still v0.2.0 and still the default.

**"The codename collision would only matter if someone picked the name."**
It matters for everyone: the defaults pick "Indradhanu" for you. A collision would have re-routed every
default action to FT04 without a single config change.

**"The grid check counts bands."**
It compares wavelengths with the grid. 165 native bands are still native.

**"Zeroing cloud before scoring is the same thing, earlier."**
No. It changes which pixels the whitening is fitted on, so it changes every score, not only the cloud's.

**"`mask_cloud.tif` and the upstream `keep_mask.tif` are the same kind of mask."**
`keep_mask.tif` comes from `scene_segmentation` (thresholds) and decides the fit. `mask_cloud.tif` comes
from V2's own head and is applied after scoring. Both can be present.

**"The unit tests prove the product gives the benchmark's numbers."**
They prove the plumbing (config, grid). The numbers are guarded by the parity run, which needs the
checkpoint and the bench scenes and is not part of `pytest`.

## Check yourself

1. Suppose FT04's directory had been named `a_spectral_stem` and both manifests said "Indradhanu". Which
   model would the defaults route to, and why?
2. Why can the api not build V2's model config the way it builds v0.2.0's, and what happens if
   `current.json` lacks `model_config`?
3. A trained record contains 1385 and 1460 nm and nothing between. Does `read_bands` keep a band at 1400
   nm? At 1420 nm? Show the distances.
4. A 6 x 5 tile has 26 valid pixels; V2 calls 4 of them cloud. Compute `excluded_percent` by hand.
5. Set4 had 11.83% of its valid pixels excluded. If set4 has 1,000,000 valid pixels, how many were scored
   0 by the exclusion?
6. An analyst picks both "Indradhanu" and "Indradhanu V2" in one `anomaly_scoring` action on a default
   band filter. What happens, and what should they do instead?

<details>
<summary>Answers</summary>

1. v0.2.0. `a` (97) sorts before `h` (104), so `a_spectral_stem` loads first and
   `hyperspectral_segformer_mae` last; the dict keeps the last value, v0.2.0. Directory names deciding
   routing is exactly why the names must differ.
2. The api cannot `torch.load`, and a stem's config (sources, band weighting, stem variant) cannot be
   rebuilt from summary fields. Without an embedded `model_config` (or with one naming another
   architecture) the resolver raises `KeyError` and `list_catalog` skips the manifest: V2 vanishes from
   the catalog.
3. 1400: `|1400 - 1385| = 15`, and `15 <= 15`, so kept. 1420: `|1420 - 1385| = 35`, `|1460 - 1420| = 40`;
   the nearest is 35, more than 15, so dropped.
4. `100 x 4 / 26 = 400 / 26 = 15.3846...`, rounded to three places `15.385`.
5. `0.1183 x 1,000,000 = 118,300` pixels.
6. The action fails: the default band filter resamples to the 165-band grid, so V2's grid check raises
   "reads native bands: run band_filter_apply with use_common_wavelength_grid=false". Run two scoring
   actions: v0.2.0 on a default band filter, V2 on a second band filter with
   `use_common_wavelength_grid=false`.

</details>

---

That is the course so far. The anomaly-detection course's part 23 covers the same change from the
analyst's side: which actions to create, the error you get otherwise, and where the new outputs land.
