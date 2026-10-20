# Part 23 - Using Indradhanu V2 in an action chain

> **The one thing this part teaches:** to score a scene with Indradhanu V2 you change two settings -
> the band filter keeps the sensor's native bands (`use_common_wavelength_grid: false`) and the scoring
> action picks "Indradhanu V2" - and in return you get a score map with cloud already set to 0, plus
> the cloud mask and class map it used; get the band filter wrong and the action fails with an error
> that names the setting, and that is also why one scoring action cannot run V2 and the old Indradhanu
> together.

## Where we are

Part 21 settled the scope: cloud is excluded, water is kept for vessels. Part 22 fixed how the
benchmark reads PRISMA. Both pointed at the same next step, "wire the mask fine-tune into the product".
On 2026-10-19 it was wired: the fine-tuned model FT04, the release candidate, is now selectable in
`anomaly_scoring` under the codename **Indradhanu V2**.

Two words first. A **codename** is the name you pick a model by in the scoring action (part 10:
"Indradhanu", "MNF-RX"). **Native bands** are the wavelengths the sensor actually measured - PRISMA's
own list, EnMAP's own list - as opposed to the 165-band common grid that `band_filter_apply` resamples
to by default (part 7).

| file | what it holds |
|---|---|
| `backend/allotrope/action_types/band_filter_apply.py` | `use_common_wavelength_grid` and the per-sensor defaults |
| `backend/allotrope/action_types/anomaly_scoring.py` | `model_codenames`, the per-sensor defaults, the output layout |
| `backend/allotrope/action_types/_anomaly_scoring_run.py` | `_check_band_grid`, the exclusion, `mask_cloud.tif` / `class_map.tif`, the new stats keys |
| `backend/allotrope/foundation_models/resolver.py` | which grid each model needs (`band_grid`) and what it excludes (`exclude_classes`) |
| `docs/09-known-issues.md`, "Spectral-stem models - open gaps" | what is not done yet |

The model side - why V2 needs native bands, how it is loaded, and the parity run that checks the product
gives the benchmark's numbers - is the Indradhanu course's part 61. This part is the analyst's side.

> **Analogy.** A film lab with two developers. The old one only takes 35 mm film; the new one only
> takes medium format. You choose the format when you load the camera, long before you reach the lab.
> Hand the new developer a 35 mm roll and it is refused at the door with a note saying "medium format
> only - reload the camera" - and one trip to the lab cannot develop both, because one camera loaded
> one kind of film.

## Step 1 - what is the same and what is new

V2 is opt-in. Nothing you already run changes:

- Every hyperspectral default in `anomaly_scoring.py` (`_DEFAULT_PRISMA`, `_DEFAULT_ENMAP`,
  `_DEFAULT_AVIRIS_NG`) still says `"model_codenames": ["Indradhanu"]` - the shipped v0.2.0.
- Every band-filter default still says `"use_common_wavelength_grid": True`.

What V2 brings, from `docs/04-models.md` and `final design/ROADMAP.md` (OB-14):

| | Indradhanu (v0.2.0) | Indradhanu V2 (FT04, `v2.0.0`) |
|---|---|---|
| reads | the 165-band common grid | each sensor's native bands |
| default score | residual Mahalanobis | residual Mahalanobis |
| segmentation head | none | water, cloud, thin cloud, shadow, snow |
| cloud | scored like anything else | scored 0, from its own mask |
| extra outputs | none | `mask_cloud.tif`, `class_map.tif`, three stats keys |

On detection the two tie: on the real PRISMA bench, both with cloud exclusion, pAUC 0.165 for v0.2.0
against 0.170 for FT04 (`FT04-final-joint/README.md`, 2026-10-19). V2's case is the masks, the native
bands and better reconstruction in one model, not a detection win.

## Step 2 - the chain, setting by setting

```
Scene ──► Project
            │
            ▼
   band_filter_apply           use_common_wavelength_grid: false     <- change 1
            │                  (everything else: the sensor's defaults)
            ▼
   anomaly_scoring             model_codenames: ["Indradhanu V2"]    <- change 2
            │                  (scene_segmentation keep mask: optional, see Step 5)
            ▼
   anomaly_detection_prep      unchanged - reads models/indradhanu_v2/anomaly_score.tif
            │
            ▼
   spectral_library_match      unchanged - reads the native onboarding cube, not the band filter
```

The scoring default for V2 is residual Mahalanobis (`default_scoring_method` in the resolver); `combined`,
`L1` and `SAM` stay selectable through the per-model override, as for v0.2.0. The per-model folder name
is the codename lower-cased with spaces turned into underscores (`_safe_dirname`):

```
"Indradhanu V2"  ->  "indradhanu v2"  ->  "indradhanu_v2"
```

`spectral_library_match` is untouched because it never read the band filter's output: its docstring says
it loads "the **native onboarding vendable** (NOT band_filter_apply's ...)".

## Step 3 - the error you get otherwise

Forget change 1 and keep the default band filter. The action is accepted (the api cannot see the upstream
band filter's settings when you submit), the worker starts, and before it loads the model it runs
`_check_band_grid`. The action fails with:

```
model 'Indradhanu V2' reads native bands: run band_filter_apply with use_common_wavelength_grid=false
(this one resampled to the 165-band grid)
```

The reverse mistake - picking the old "Indradhanu" on a native band filter - fails the same way with
"needs the 165-band grid: run band_filter_apply with use_common_wavelength_grid=true (this one kept
native bands)". Before 2026-10-19 nothing checked this at all.

How does the worker know which grid it was given? It compares the vendable's wavelengths with the grid:

```python
on_grid = len(wavelengths) == len(grid) and np.allclose(wavelengths, grid)
```

So it is not a band count. A native list that happened to have exactly 165 bands would still not equal
460, 470, ... 2450 nm and would count as native. Classical detectors (MNF-RX, part 8) skip the check and
run on either grid; so do thermal scenes.

## Step 4 - why one action cannot run both

`anomaly_scoring` takes one upstream band filter (`input_band_filter_output_id`), and every model in
`model_codenames` reads that one output. V2 needs native, v0.2.0 needs the grid, so whichever does not
match fails the action. To compare them on one scene, build two branches in the same Project:

```
                       ┌─► band_filter_apply (grid: true)  ─► anomaly_scoring ["Indradhanu"]
Scene ──► Project ─────┤
                       └─► band_filter_apply (grid: false) ─► anomaly_scoring ["Indradhanu V2"]
```

A classical detector can ride along in either branch. One consequence to know: `anomaly_detection_prep`
reads a single scoring output (`input_anomaly_scoring_output_id`), so today you cannot weight V2 and
v0.2.0 into one composite score.

## Step 5 - what comes out, and how to read it

Under the scoring action's output, in `models/indradhanu_v2/`, next to the usual `anomaly_score.tif`,
`reconstruction.tif` and `stats.json`:

| output | contents |
|---|---|
| `mask_cloud.tif` | uint8, 1 where the score was set to 0 (V2 called the pixel cloud or thin cloud) |
| `class_map.tif` | uint8, V2's classes: 0 land, 1 water, 2 cloud, 3 shadow, 4 snow |
| `stats.json` `bands_read` | how many bands V2 read (bands it never trained on are dropped first) |
| `stats.json` `excluded_classes` | `["cloud"]` |
| `stats.json` `excluded_percent` | 100 x cloud pixels / valid pixels, to 3 places |

Note what is excluded: cloud only. Water is in the class map but **not** zeroed - part 21's scope, kept
for vessels.

The order matters, and it is the one part 19's `exclude` scorer measured: V2 scores every pixel in the
keep mask first and sets cloud to 0 **afterwards**. A cloud pixel still takes part in fitting the
residual Mahalanobis statistics; it is only its own score that becomes 0. If you also attach a
`scene_segmentation` keep mask upstream, that mask still decides which pixels the statistics are fitted
on, as it always has. The parity check behind V2's numbers used no upstream keep mask.

A pixel scored 0 is not "unknown"; it sinks to the bottom of the ranking (part 21). Read
`excluded_percent` as the share of the scene you cannot find anything in. Work it on a small scene:
1,000 x 1,200 pixels, 6% invalid, 90,240 pixels called cloud.

```
total pixels    = 1,000 x 1,200          = 1,200,000
invalid 6%      = 0.06 x 1,200,000       =    72,000
valid           = 1,200,000 - 72,000     = 1,128,000
excluded_percent = 100 x 90,240 / 1,128,000
                 = 9,024,000 / 1,128,000  = 8.000
```

On the PRISMA bench, set4 had 11.83% of its valid pixels excluded - close to the 11% that part 21
found FT03 calling cloud on the same scene, most of which was a cloud / smoke streak and haze. So on a
hazy scene, expect V2 to blank out the haze too. Open `mask_cloud.tif` before trusting an empty patch.

The viewer does not show any of this yet: the masks and the three keys are on disk only
(`docs/09-known-issues.md`). The stats keys are additive, so nothing that reads `stats.json` breaks.

## Step 6 - two caveats

- **EnMAP is cleaned twice.** The default EnMAP band filter applies the provider's quality masks
  (`quality_masks_to_apply: ["cloud", "cloud_shadow", "haze"]`) before scoring; V2 then zeros its own
  cloud on top. Kept deliberately as the product's convention (decision 2026-10-19), but it is not how
  V2's PRISMA numbers were measured.
- **Native PRISMA runs downwards** (part 22). `scene_segmentation` used to take its brightness as the
  mean of "band 0 up to the VNIR edge", which on a native, descending PRISMA list is the far SWIR. It now
  picks the VNIR bands by wavelength (`app/utils/spectral_indices/scene_indices.py`), so a native-band
  chain gets the right brightness too.

Also unmeasured: how long V2 takes on the stack's CPU worker - more bands than 165, plus a segmentation
pass.

## Common confusions

**"Indradhanu V2 is the new default."**
No. The defaults still pick "Indradhanu" (v0.2.0) on a 165-band filter. Making V2 the default is a later
phase.

**"The error means the model is broken."**
It means the band filter is on the wrong grid. The message names the setting to change.

**"I can pick both models in one action and compare them."**
Not on one band filter. Build two branches (Step 4).

**"`mask_cloud.tif` is the same as the keep mask."**
No. The keep mask comes from `scene_segmentation`'s thresholds and decides which pixels the statistics
are fitted on. `mask_cloud.tif` comes from V2's own head and is applied after scoring.

**"Water is excluded too, since it is in the class map."**
The class map shows water; only cloud is zeroed (`exclude_classes = ("cloud",)`).

**"`spectral_library_match` needs the native band filter now."**
It never used the band filter's output; it reads the native onboarding cube.

## Check yourself

1. You have a PRISMA Project with a default band filter and an `anomaly_scoring` action with
   `model_codenames: ["Indradhanu V2", "MNF-RX"]`. What happens, and which single setting fixes it?
2. Where on disk is V2's cloud mask for that action, exactly?
3. A scene is 800 x 900 pixels; 20,000 are invalid; V2 calls 35,800 valid pixels cloud. Compute
   `excluded_percent` by hand.
4. A band filter kept PRISMA's native bands and, after exclusions, exactly 165 remain. Does a v0.2.0
   scoring action accept it? Why?
5. Why is a vessel on a lake still findable with V2, and what would a cloud-shadowed vessel's score be?
6. You want V2's and v0.2.0's maps side by side on one scene. Sketch the actions.

<details>
<summary>Answers</summary>

1. The action fails: V2 reads native bands and the default band filter resampled to the 165-band grid,
   so `_check_band_grid` raises "reads native bands: run band_filter_apply with
   use_common_wavelength_grid=false". Set `use_common_wavelength_grid: false` on the band filter (MNF-RX
   takes either grid).
2. `models/indradhanu_v2/mask_cloud.tif` under that scoring action's output folder.
3. `800 x 900 = 720,000` total; `720,000 - 20,000 = 700,000` valid; `100 x 35,800 / 700,000 =
   3,580,000 / 700,000 = 5.114` (5.11428... to three places).
4. No. The check compares the wavelengths with the grid (`np.allclose`), not only the count; native
   PRISMA wavelengths are not 460, 470, ... 2450, so it counts as native and v0.2.0 refuses it.
5. Only cloud is zeroed; water is kept (part 21), so a vessel on a lake is scored normally. Shadow is not
   excluded either, so a shadowed vessel is scored normally too - its score is whatever residual
   Mahalanobis gives it, not 0.
6. Two band filters on the same Scene, one with `use_common_wavelength_grid: true` and one with `false`;
   an `anomaly_scoring` on each, picking "Indradhanu" and "Indradhanu V2" respectively.

</details>

---

Next: part 24, three ways to improve the score without changing the model - and why none was kept.
