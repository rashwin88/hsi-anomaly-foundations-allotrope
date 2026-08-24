# 05 · Where Indradhanu lives in the product

> **The one thing this part teaches:** Indradhanu is not a script and not a
> service. It is one selectable option inside one step of a web application.

---

## First, a correction to your mental model

If you have only seen the model code, you probably picture a machine-learning
repository: some data loaders, some model files, a training script.

That is a small corner of what this actually is. **Allotrope is a web
application** — Postgres database, login system, React front end, background job
worker. The model is a component inside it.

Knowing this stops you looking for things in the wrong place.

---

## The domain model, in plain words

```
Scene  --->  Project  --->  Action  --->  Action  --->  ...  --->  Export
```

- A **Scene** is one uploaded satellite acquisition. One file, one moment, one
  patch of ground.
- A **Project** is a workspace attached to exactly one Scene. It is where a
  user's analysis lives.
- An **Action** is one step of analysis. Actions chain together, each one
  consuming the outputs of earlier ones, forming a directed graph.

Think of Actions like cells in a spreadsheet: each one computes something from
earlier ones, and re-running one re-runs what depends on it.

---

## The shipped chain, end to end

```
band_filter_apply                    -> filtered_vendable.pkl
  |
  +-> scene_segmentation             -> keep_mask.tif      (hyperspectral)
  |   or cloud_mask                  -> keep_mask.tif      (thermal)
  |
  +-> anomaly_scoring                -> anomaly_score.tif + reconstruction.tif
        |                               (THIS is where Indradhanu runs)
        v
      anomaly_detection_prep         -> composite_score.tif + anomaly_mask.tif
        |                               (a HUMAN picks the threshold here)
        v
      spectral_library_match         -> matches.parquet + match_map.tif
        |
        v
      export                         -> a georeferenced zip bundle
```

**Indradhanu is one option inside the `anomaly_scoring` Action.** It is not an
Action of its own.

In the user interface, a person opens the "new anomaly scoring" dialog, ticks
one or more models by name, and submits. The Action then loops over whichever
models were ticked, running each and writing its outputs into its own
subfolder.

That is why the code has a loop like this:

```python
for codename in cfg["model_codenames"]:
    ...
```

You can run Indradhanu and MNF-RX on the same scene, in the same Action, and
compare their score maps side by side.

---

## The three-layer rule — learn this before writing any code here

| Layer | Path | The rule |
|---|---|---|
| **Portable science** | `app/` | numpy, torch, rasterio only. **No database. No web framework.** |
| **Shared backend** | `backend/allotrope/` | database models, routes, authentication, the Action registry. Used by both the web server and the worker. |
| **Worker only** | `backend/allotrope_worker/` | the loop that picks up jobs and runs them |

Indradhanu's architecture, its trainer and its inferencer all live in `app/`.
They know nothing about users, jobs, HTTP or Postgres. You could copy the `app/`
folder into a completely different project and it would still work.

That separation is why the same model code can be driven by a training script on
a rented GPU box *and* by a background worker in a Docker container.

### The lazy-import rule (a real trap)

Action modules must **not** import `app.*`, `torch` or `rasterio` at the top of
the file. Those imports must be inside the functions.

Why? The web server imports **every** Action module when it starts up, so it can
show the list of available Actions. If one of those modules imported torch at
the top, the web server would load the entire deep-learning stack — hundreds of
megabytes and several seconds — just to render a dropdown menu.

That is why Actions come in pairs:

```
anomaly_scoring.py          <- light. Name, metadata, config validation.
_anomaly_scoring_run.py     <- heavy. Imports torch INSIDE the function.
```

The leading underscore signals "this is the heavy half".

---

## Codenames

Every foundation model in this project has an Indic codename, and the backend
selects models **by codename**, not by class name.

| Codename | Script | Meaning | Architecture | Sensor |
|---|---|---|---|---|
| Pratibimba | | reflection | spatial autoencoder | thermal |
| Antardhana | | disappearance | masked autoencoder | thermal |
| Tirohita | | vanished | masked autoencoder, L1 loss | thermal |
| Asanskrita | | unrefined | unnormalised variant | thermal |
| Drashta | | seer | normalised masked autoencoder | thermal |
| **Chakshu** | | the eye | `segformer_mae` | thermal |
| **Indradhanu** | इंद्रधनु | rainbow, "Indra's bow" | `hyperspectral_segformer_mae` | hyperspectral |

A rainbow is the full visible spectrum, and this is the model that sees all 165
bands. The manifest explains the choice itself:

```json
"why": "165 spectral bands — the full spectrum, with a learnable spectral
        compressor at the front."
```

---

## How a codename becomes a running model

Three files cooperate. This is worth walking through slowly, because "my model
trained fine but does not appear in the UI" is a common and confusing failure.

### 1. The manifest on disk

[`allotrope_models/hyperspectral_segformer_mae/current.json`](../allotrope_models/hyperspectral_segformer_mae/current.json)

It records which weights file is current, and some facts about it:

```json
"current": {
  "file": "hyperspectral_segformer_mae_v0.2.0_epoch200.pt",
  "version": "v0.2.0",
  "epoch": 200,
  "params": 5507354,
  "encoder_dims": "[32, 64, 160, 256]",
  "spectral_dim_D": 32,
  "decoder_dim": 256,
  "val_loss": 0.04349
}
```

Two older weights files are listed as alternatives, including one from an
earlier design where the compression dimension was 24 instead of 32. You will
meet that number in part 10.

### 2. The resolver

[`backend/allotrope/foundation_models/resolver.py`](../backend/allotrope/foundation_models/resolver.py)

A static table of what each architecture is *capable* of, layered on top of the
manifest. Indradhanu's entry:

```python
"hyperspectral_segformer_mae": ModelCapabilities(
    architecture="hyperspectral_segformer_mae",
    foundation_model_name="hyperspectral_segformer_mae",
    scoring_methods=("L1", "SAM", "combined"),
    default_scoring_method="combined",
    default_patch_size=128,
    default_stride=32,
    pixel_stats_relpath=_HSI_STATS,
)
```

Read that as a set of answers to UI questions: *which scoring methods should the
dropdown offer? which is preselected? what patch size by default?*

### 3. The inferencer factory

[`app/foundation_models/inferencers/inferencer_factory.py`](../app/foundation_models/inferencers/inferencer_factory.py)

A dictionary mapping the architecture's enum value to the Python class that
knows how to run it:

```python
FoundationModelName.HYPERSPECTRAL_SEGFORMER_MAE: HyperspectralSegFormerMAEInferencer,
```

---

## The registration checklist (the step people forget)

Adding a new model means **four** separate registrations:

1. the architecture class in `app/foundation_models/components/`,
2. an entry in the **trainer** factory,
3. an entry in the **inferencer** factory,
4. a capabilities entry in the **resolver** — *and* a `current.json` manifest.

Step 4 is the one that gets missed, and the symptom is baffling: training works
perfectly, the checkpoint exists, and the model simply never appears in the
picker. There is no error, because nothing errored — the catalogue just skips
architectures it has no capabilities entry for:

```python
except (KeyError, ValueError):
    # Unknown architecture (no _CAPABILITIES entry) — skip.
    continue
```

---

## One more safety net worth knowing about

Because codename is the routing key, two models with the same codename would
silently collide. The catalogue check guards against it:

```python
_log.warning(
    "model catalog: codename collision — %r used by both "
    "architecture %r and %r; codename lookups will route to "
    "the last-loaded manifest. Rename one of them.", ...)
```

The warning fires when the process boots, not when somebody runs a scoring job.
Catching it early is deliberate.

---

## Common confusions

**"Is Indradhanu a service I can call?"**
No. It is a Python class loaded inside a background worker process, on demand,
for the duration of one job.

**"Where does the model file live at runtime?"**
Under `allotrope_models/<architecture>/`, mounted into both the API and worker
containers.

**"Can I add a model without touching the front end?"**
Yes — that is the design. The resolver and the Action's metadata drive the UI.
No front-end change is needed for a new model or a new Action type.

---

## Check yourself

1. Is Indradhanu an Action? If not, what is it?
2. Why must Action modules avoid importing torch at the top of the file?
3. Name the three layers and the one rule about `app/`.
4. What are the four places a new model must be registered, and which is
   forgotten most often?
5. What does the resolver's capabilities table actually control?

<details>
<summary>Answers</summary>

1. No. It is one selectable model inside the `anomaly_scoring` Action.
2. The web server imports every Action module at startup; a top-level torch
   import would drag the whole deep-learning stack into the web process.
3. `app/` = portable science with no database or web framework;
   `backend/allotrope/` = shared backend; `backend/allotrope_worker/` = worker
   only.
4. Architecture class, trainer factory, inferencer factory, resolver
   capabilities plus a `current.json`. The resolver entry is the forgotten one,
   and it fails silently.
5. What the UI offers and preselects: scoring methods, default patch size,
   stride, batch size, and where the normalisation statistics live.

</details>

---

**Next:** the central idea of the model itself, in
[06-reconstruct-then-subtract.md](06-reconstruct-then-subtract.md)
