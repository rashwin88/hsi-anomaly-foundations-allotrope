# Spectral attention bottleneck

**Status:** proposal — not decided, not scheduled, nothing built.
**Raised:** 2026-08-24.
**Scope:** the spectral stem of the hyperspectral foundation models (Indradhanu).
Deliberately independent of the EnMAP cloud-segmenter work; the two should not be coupled.

---

## The question this answers

*Does Indradhanu support band-level attention today?* No — attention in Indradhanu is
purely **spatial**. This document records what could be done about that, and why it might
be worth doing.

## What exists today

`HyperspectralSegFormerMAE` handles the spectral dimension like this:

```
(B, 165, H, W)
  → PixelNormalize                     per-band z-score, fixed buffers
  → SpectralCompressor                 Conv2d(165, D, kernel=1) + BatchNorm2d
  → SegFormerEncoder                   attention over SPATIAL tokens, D channels
  → SegFormerDecoder
  → SpectralDecompressor               Conv2d(D, 165, kernel=1)
  → PixelDenormalize
```

Source: `app/foundation_models/components/hyperspectral_seg_former_mae.py`,
`app/foundation_models/components/spectral_compressor.py`.

Bands are mixed exactly **twice**, both times by a plain 1x1 convolution — a single
learned linear projection, identical for every pixel of every scene. The compressor's own
docstring describes it as "analogous to MNF but trained end-to-end", which is accurate.

In `app/foundation_models/components/efficient_self_attention.py` a *token* is a spatial
location; queries and keys index positions on the image grid. Past the compressor the
model no longer has 165 bands at all — it has `D` feature channels (24 by default, 32 in
`configs/hyperspectral_segformer_exp_2.json`). Nothing ever attends across wavelength.

### Why this is a ceiling

The projection is a hard bottleneck sitting *before* any of the model's capacity. 165
numbers become `D` by a fixed map. Spectral detail that does not survive that projection
is gone permanently — the model cannot later decide that, for this particular pixel, the
2200 nm region matters. It never sees it.

The existing config notes already circle this. `hyperspectral_segformer_exp_2.json` lists
as a red flag: *"SAM stuck above 0.3 after epoch 30 → compressor bottleneck too narrow
(try compressed_channels=32)"*, and v0.2.0 raised `D` from 24 to 32. That treats the
symptom by widening the pipe rather than making it adaptive.

---

## Three levels of "band attention"

They differ a lot in cost. The cheapest may capture most of the benefit.

### Level 1 — data-dependent mixing (not attention, strictly)

**Squeeze-and-excitation gate.** Pool over `H, W` to a 165-vector, small MLP, sigmoid,
scale each input band, then compress as today. The model can express "in this patch the
SWIR matters, damp the visible."

- Cheap. Existing compressor weights remain loadable.
- One gating vector per patch — not per pixel.

**Per-pixel gating.** Same idea, but gates derived from each pixel's own spectrum via a
pointwise MLP. A shadowed pixel and a bright pixel then weight bands differently.

- Still cheap (1x1 convs).
- Fixes the "one static recipe everywhere" problem, which is arguably the single largest
  limitation.
- Neither variant lets one band inform another.

### Level 2 — bands as tokens attending to each other

Treat a pixel's 165 values as a sequence and let them attend. Bands hundreds of nm apart
can then inform each other, so the model can represent *the depth of an absorption
feature relative to its shoulder* — the actual mechanism of spectral identification.

Cost is the problem: attention is quadratic and paid **per pixel**. A 128x128 patch is
16,384 pixels, each running a 165x165 attention. Naively this does not fit.

Two mitigations:

- **Band-group tokens.** 165 bands → ~33 tokens of 5. Quadratic cost falls roughly 25x.
  Groups can follow physical regions (visible, red edge, NIR, SWIR-1, SWIR-2).
- **Apply after a spatial downsample**, so far fewer pixels pay the cost.

### Level 3 — cross-attention bottleneck (recommended shape)

Learn `D` latent query vectors. Each attends over the 165 band tokens of a pixel and
returns one output channel — "how strong is the red edge here?"

Why this shape fits the codebase:

- It is **the current compressor, generalised.** The 1x1 conv is the degenerate case where
  the attention weights are fixed and identical everywhere. The change is simply: make the
  mixing weights depend on the data.
- Cost is `D x 165` per pixel — **linear in band count, not quadratic.** That is the
  difference between expensive and infeasible.
- Output is still `(B, D, H, W)`, so `SegFormerEncoder`, `SegFormerDecoder` and everything
  downstream are unchanged.

---

## Two unlocks specific to this codebase

These may matter more than the accuracy argument.

### 1. Per-band validity can finally be honoured

Validity in this system is **per-band per-pixel** — `validity_cube` is `(C, H, W)`, and a
pixel can have a contiguous run of dead bands while the rest are fine (see
`app/utils/dataset_builder/prisma_dataset_builder.py`, which builds it from band flags x
`PIXEL_L2_ERR_MATRIX` x invalid-value checks).

Today those dead bands are filled with substitute values
(`app/utils/pixel_fill/nearest_valid_fill.py`) and then linearly mixed into the compressed
vector alongside real measurements. The contamination spreads into all `D` channels and
there is no way to prevent it: **a fixed linear map cannot skip an input.**

Attention can. Masking dead band tokens out of the attention leaves the pixel with a clean
representation built only from its surviving bands. This is a structural capability the
current stem does not have.

### 2. The model stops being hard-wired to 165 bands

If each band token carries **its wavelength** rather than its index, bands become
self-describing: the model learns "1650 nm behaves like this", not "input 87 behaves like
this".

Consequences:

- It could ingest AVIRIS-NG's native 425 bands or PRISMA's native grid **without
  resampling to the common grid**. The resample exists largely because the current
  architecture demands a fixed 165-wide input.
- It fixes a distortion that exists today. The 165-band grid has **gaps** — the
  atmospheric windows were removed (`exclusion_ranges` in
  `app/models/dataset/vendables.py`: 0-450, 912-978, 1131-1152, 1350-1450, 1800-1950 nm).
  Adjacent array positions can therefore be 10 nm or 100 nm apart, and the model has no
  way to tell. A wavelength-valued positional encoding costs nothing and removes this.

---

## Costs, risks, and what would have to be proven

- **No weight reuse for the stem.** A new compressor trains from scratch. Encoder weights
  can still transfer if `D` is held at 32.
- **The stem runs at full spatial resolution**, which is where the compute lands. The
  Level 3 form keeps this tractable; Level 2 needs the grouping trick.
- **A harder-to-optimise bottleneck can lose to a simple one.** "More sophisticated" is
  not evidence. Any version of this must be compared against the linear compressor on the
  same shards, same schedule, same metrics (L1 + SAM, per
  `configs/hyperspectral_segformer_exp_2.json`).
- **BatchNorm placement.** The current compressor carries `BatchNorm2d` to stabilise the
  encoder's input distribution, while the decompressor deliberately has none. Any
  replacement has to make an equivalent decision consciously.
- **Keep `PixelNormalize` as-is** — per-band z-score against the shared PRISMA+EnMAP stats
  file. Changing normalisation and architecture in the same experiment makes the result
  uninterpretable.

## Related

The classical detector **MNF-RX** applies the same trick as the compressor — a fixed
linear projection into a low-dimensional subspace. Making that projection adaptive is a
coherent theme across both the foundation and classical halves of the system, not a
one-off change. See `docs/05-detectors.md`.

## Open questions

- Which level to attempt first. Level 1 per-pixel gating is a cheap probe of whether
  adaptivity helps at all, before committing to Level 3.
- Whether to keep `D = 32` (preserves encoder weight transfer) or treat the retrain as
  total and re-tune it.
- Whether the variable-band-count unlock is wanted as a goal in itself or is a side
  benefit. It has consequences well beyond the model — `band_filter_apply`, the common
  grid, and the patch/shard format all assume a fixed band count.
