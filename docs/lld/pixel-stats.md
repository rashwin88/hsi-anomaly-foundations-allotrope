# Pixel statistics resolution

**File:** `app/foundation_models/pixel_stats.py`

## Purpose

Every model with a `PixelNormalize` layer needs a per-band mean and standard deviation when
it is constructed. This decides where those two lists come from, and which source wins when
there is more than one.

## The problem it solves

Normalisation turns raw sensor values into something a network can learn from. A thermal
model trained on Landsat sees Celsius, roughly 290 plus or minus 10. Feed the same model
HotSat-1, which ships raw digital numbers around 5000 plus or minus 400, and the input is
nowhere near the distribution it learned.

Work the arithmetic. The model normalises with `(x - mean) / std` using its baked-in
Celsius figures:

```
HotSat DN value       = 5000
baked-in mean         =  290
baked-in std          =   10

(5000 - 290) / 10     = 471.0
```

A network that saw inputs in roughly the -3 to +3 range during training now receives 471.
It does not degrade gracefully; it reconstructs noise, and since the anomaly score is the
reconstruction error, every pixel looks anomalous and none of them usefully so.

The fix is to recompute mean and std from the scene itself. That is what
`PixelStatsOverride` carries.

## Interfaces

```python
def resolve_pixel_stats(
    stats_path: str | None,
    override: Any | None = None,
) -> tuple[list[float] | None, list[float] | None]
```

Returns `(mean, std)`, or `(None, None)` when neither source is set - which the model
constructors read as "skip normalisation entirely".

`override` is typed as `Any` deliberately. It only needs `.mean`, `.std` and `.source`.
Typing it as `PixelStatsOverride` would make `app/foundation_models/` import from
`app/models/training/`, a dependency this layer does not otherwise have.

## Data flow

```
InferenceConfig.pixel_stats_override ──┐
                                       ├──> resolve_pixel_stats ──> (mean, std) ──> PixelNormalize
InferenceConfig.pixel_stats_path ──────┘                                            (registered buffers)
   or TrainingConfig.data.pixel_stats_path
```

The JSON files live under `app/constants/`: `thermal_pixel_consts.json` holds one mean and
one std; `hyperspectral.json` holds 165 of each, one per band on the common grid.

## Invariants

**The override wins outright. It never merges with the file, and never falls back to it.**

That is not a stylistic choice. When `_anomaly_scoring_run.py` sets an override it also sets
`pixel_stats_path=None`, precisely so the baked figures cannot be read. If this function
ever fell back to disk on a partial override, a HotSat scene would silently normalise with
Celsius statistics and produce the 471 above.

`len(mean)` must equal `len(std)`, and both must equal the model's `in_channels`.
`PixelStatsOverride` enforces the first with a validator; `_anomaly_scoring_run.py` checks
the second before constructing the config.

## Failure modes

| What happens | What the caller sees |
|---|---|
| `stats_path` points at a missing file | `FileNotFoundError` from `open()`, uncaught |
| JSON lacks `mean` or `std` | `KeyError`, uncaught |
| `mean` and `std` differ in length | `ValidationError` at `PixelStatsOverride` construction, naming both counts |
| Neither source set | `(None, None)`; the model runs unnormalised |

The last one is silent by design - several models genuinely have no normalisation layer -
but it does mean a typo in a config path degrades to "no normalisation" rather than an error,
if the path is `None` rather than wrong.

## Decisions

**Why one function rather than ten copies.** This block was duplicated across four
inferencers and six trainers. They had already drifted: four logged the values, two logged
only the counts, and the two thermal ones did `import json` inside the function while the
others imported at module scope. Ten copies of a priority rule is ten chances to get the
priority wrong.

**Why the log line switches on band count.** `_MAX_BANDS_TO_LOG = 4`. Printing 165 floats
per model load buries every other line in the worker log. Under that threshold the values
are genuinely useful - a thermal mean of 24.58 tells you immediately which stats file loaded.

**What was rejected.** Passing the resolved tuple down from the factory instead, so the
model builders take `(mean, std)` rather than a config. Cleaner in isolation, but it would
have changed the signature of every `build_model` in the repo, and the point of the change
was to remove duplication without touching behaviour.

## If you change this

The override path is currently exercised only by the HotSat branch in
`_anomaly_scoring_run.py`. There is no test that runs a real model with an override end to
end - the checks that exist construct a `SpatialAutoencoder` and read back
`model.normalize.mean`. If you extend the priority rules, extend that check too.
