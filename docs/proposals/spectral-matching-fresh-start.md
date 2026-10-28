# Spectral matching as a new problem - a learned material namer

**Status:** parked 2026-10-28 on the user's word ("note this down, we will return to it when we have the second
block of capacity available"). Resume at H100 capacity block 2 (`cr-0dccba00b34dc29e1`, 2026-10-31 11:30 ->
2026-11-02 11:30 UTC). Framed fresh at the user's request ("forget all the past, consider this problem as new") - it
deliberately does not inherit the earlier gate in docs/proposals/spectral-matching-foundation-model.md.

## The problem
Given a pixel or small object flagged as anomalous, name its material at family level - metal roof, plastic /
greenhouse film, painted surface, concrete, asphalt, tarp / fabric, PV panel, oil, ... - with a confidence, and
**abstain** when it is none of them.

What makes it hard:
1. **Mixing** - targets are mostly sub-pixel: pixel = fill x target + (1 - fill) x background.
2. **Domain gap** - lab library spectra vs atmospherically corrected satellite reflectance with residual errors.
3. **Band sets** - PRISMA, EnMAP and AVIRIS-NG sample the spectrum differently.
4. **Open world** - most things in a scene are in no library; the model must be able to say "unknown".

## Approach
- **Synthetic mixture generator** (truth known by construction): real background pixels from our scenes
  (SpectralEarth, native EnMAP / PRISMA / AVIRIS-NG shards - billions available) x library spectra (USGS, ECOSTRESS
  and others, resampled to each sensor's band response), mixed at random fill into real local background, plus
  realistic noise and atmospheric-residual error estimated from the scenes. Every sample carries an exact label:
  family + fill.
- **Model:** input = the pixel spectrum + its local background (ring mean), read at native bands through the
  wavelength-aware stem (absolute reflectance kept; never per-scene min-max). Outputs: family probabilities,
  fill fraction, and "none of these".
- **Training:** self-supervised masked spectral modelling on real pixels first (learn the real-pixel manifold,
  HyperSL-style but on absolute reflectance), then supervised on the synthetic mixtures - so the namer does not learn
  generator artefacts. Unlimited cheap samples: a good fit for the H100 block.

## The real bottleneck: measurement - an OpenStreetMap material benchmark
Without real targets of known material we cannot tell whether anything works. Proposal: build a **material-labelled
real benchmark from OSM tags** over the PRISMA / EnMAP / AVIRIS-NG scenes we hold - `landuse=greenhouse_horticulture`
(plastic), `power=plant` + `plant:source=solar` (PV glass), `aeroway=runway` / `surface=asphalt|concrete`,
`building` with `roof:material=metal|concrete|...`, water and harbour features. Potentially hundreds to thousands of
real labelled objects per family with no hand labelling. First check (a few hours): is OSM's material tagging dense
enough over our scene footprints?

## Plan
1. Families and library: fix ~10-12 families; assemble the multi-source library resampled to each sensor (laptop, ~1 day).
2. OSM benchmark: OSM features x scene footprints -> labelled object pixels, with quality checks (laptop, 1-2 days).
   **First, or in parallel - it is the judge.**
3. Generator: mixing, noise and atmospheric-residual models; validate that synthetic mixtures match real pixel statistics.
4. Train: small debug run on the laptop, then the full run on the H100 block 2.
5. Evaluate against the current matchers (SAM, family matching) on the OSM benchmark, with paired-bootstrap intervals
   (app/evaluation/significance.py).

## Open decisions (for when we resume)
- Family list and the "unknown" policy.
- Whether the OSM density check is good enough to make it the judge.
- How much of block 2 goes to this versus Indradhanu Heavy follow-ups (FTH1 / FTB1 may spill into block 2).
