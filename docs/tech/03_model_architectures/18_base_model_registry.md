# 3.18 `BaseModel` registry

File: [app/models/base_models/base_model.py](../../app/models/base_models/base_model.py)

## What the code does

This is a tiny `str, Enum` listing string identifiers for base models used elsewhere in the
pipeline (currently just the cloud-masker,
[base_model.py:15](../../app/models/base_models/base_model.py#L15)).

It is not a model definition — it exists so the rest of the codebase can refer to base
models by a canonical key (e.g. when looking up checkpoints or routing files through the
right inferencer).

### Why this is "just an enum"

A `str, Enum` in Python serves two purposes simultaneously:

1. **Type-safe string constants**: instead of passing the literal string `"cloud_mask"`
   around — at risk of typos that only show up at runtime — code passes
   `BaseModel.CLOUD_MASK`. Static analysers and IDEs catch typos.
2. **Backwards-compatible serialization**: because the enum subclasses `str`, instances
   compare equal to their underlying string value. JSON / pickle / config files store the
   string and the loader can convert back to the enum trivially.

### Where it is used

The cloud-masker registers itself under `BaseModel.CLOUD_MASK` in the model registry. The
inferencer for cloud detection looks up checkpoints under that key. New base models
(e.g. an atmospheric correction net, a coregistration prior) would add another enum value
and slot into the same registry pattern.

## Theory in plain language

### Foundation models vs. base models

Allotrope draws a deliberate line between two model categories:

- **Foundation models** (Sections 3.1-3.17): reconstruction-based anomaly detectors. Seven
  of them, each with a Sanskrit codename. Tracked by slug (`spatial_autoencoder`,
  `segformer_mae`, etc.) and managed via `checkpoints/<slug>/current.json`.
- **Base models**: utility models that other parts of the pipeline depend on, but are not
  themselves anomaly detectors. The cloud-masker is the current sole example. Tracked by
  this enum.

The split is procedural: foundation models are the *output* of the research pipeline, base
models are *inputs* (preprocessing helpers). Keeping them in separate registries makes the
data-flow direction obvious.

### Why mention it in this chapter at all

This section is here for completeness — readers tracking the components folder might wonder
where the cloud-masker fits and why it is not in the architecture map. The answer: it lives
in `app/models/base_models/`, not `app/foundation_models/components/`, and it is named via
this enum.

The seven foundation models above are keyed by their slugs (e.g. `segformer_mae`,
`hyperspectral_segformer_mae`) and tracked via `checkpoints/<arch>/current.json`. The
manifest pattern is the same for both registries — only the lookup key differs.
