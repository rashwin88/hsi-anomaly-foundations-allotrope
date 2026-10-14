# 38 - One pixel start to finish, and the three traps

> **The one thing this part teaches:** the new compressor and decoder slot around an
> unchanged SegFormer, and the design is only safe if it keeps three rules - never let a pixel
> rebuild its own spectrum, normalise by wavelength rather than by scene, and prove it beats
> the old compressor at finding anomalies before shipping it.

## One SpectralEarth pixel, start to finish

Shapes are for one patch (`B = 1`); the pixel we follow is one of its 16,384.

```
1. raw patch file          (202, 128, 128) int16      SpectralEarthPatch.raw
2. reflectance             (202, 128, 128) float32    x 0.0001, nodata -> masked (part 35)
3. normalise by wavelength (202, 128, 128)            trap 2 below
4. pixels as rows          (16384, 202)               one spectrum per row
5. band tokens             (16384, 202, 32)           value stretch + wavelength tag (part 33)
6. 32 queries read them    (16384, 32)                cross-attention compressor (part 34)
7. back to an image        (32, 128, 128)             same shape the encoder gets today
--------------------------------------------------------------------------------- unchanged
8. spatial masking          hide ~65% of tokens       parts 17 to 19
9. SegFormer encoder        four stages               parts 11 to 16
10. SegFormer decoder       (32, 128, 128)            part 21
--------------------------------------------------------------------------------- unchanged
11. pixels as rows         (16384, 32)
12. 202 wavelength queries (16384, 202)               spectral decoder (part 37)
13. back to an image       (202, 128, 128)            reconstruction, native bands
14. loss                   L1 + lambda * SAM          hidden pixels, valid bands only
```

Rows 8 to 10 are the existing model, untouched. **The new parts run strictly inside each
pixel; everything that moves information between pixels is the old, tested machinery.** That
separation is what makes the next trap answerable.

## Trap 1 - a pixel must never rebuild its own spectrum

Part 06 and part 11's masking rest on one rule: a hidden pixel is reconstructed from its
neighbours, never from itself. Does spectral attention break it?

Not as designed. The compressor (step 6) mixes bands *within* a pixel and nothing between
pixels. A pixel hidden at step 8 has its tokens removed before any spatial attention runs, so
its compressed channels cannot reach any other pixel's reconstruction, and its own
reconstruction (step 12) is built from what the encoder inferred from neighbours. The rule
holds.

It would break if you were tempted by a second kind of masking: hiding some *bands* of a
visible pixel and asking the model to fill them from that pixel's other bands. That trains
spectral completion. A model good at it can reconstruct an anomalous spectrum from the
anomalous spectrum itself - the crashed aeroplane of part 00 painting itself back in.
**Hide pixels, not bands.** If band-masking is ever explored for pretraining, it must be off
when the model scores anomalies, and measured for exactly this leak.

## Trap 2 - normalise by wavelength, never by scene

Part 09 z-scores each of the 165 bands with a mean and standard deviation computed over the
training data (`app/foundation_models/pixel_stats.py`). Those statistics are per *band
position*. With native bands there are no fixed positions.

The fix is statistics as a function of wavelength: compute mean and standard deviation per
SpectralEarth band, then for any sensor read them off at its own wavelengths (interpolating
between neighbours). One curve serves every sensor.

The tempting shortcut is to normalise each scene by its own minimum and maximum. Don't. It
erases absolute reflectance, and absolute reflectance is what spectral library matching
(the `spectral_library_match` step, against USGS splib07) depends on. An external model
reviewed for this project (HyperSL) does exactly that per-scene min-max scaling, and it is the
reason its outputs cannot be matched against the library.

## Trap 3 - "more sophisticated" is not evidence

Parts 31 to 37 make a strong case. None of it shows the new model finds anomalies better. A
harder-to-optimise bottleneck can lose to a simple one. So the design is judged by a
controlled comparison - same SpectralEarth subset, same training budget, same spatial model:

| variant | compressor | input bands |
|---|---|---|
| A | today's 1x1 convolution | 165-band grid |
| B | per-pixel gating: a small network reweights each band, then the 1x1 convolution | 165-band grid |
| C | cross-attention compressor and spectral decoder (this sequel) | native 202 |

Two scores per variant:

- **reconstruction** on the held-out SpectralEarth sites (split `se-v1`), L1 and SAM;
- **anomaly detection** on the labelled PRISMA sets (the stage-1 ground-truth masks), the job
  the model exists for.

B is there to answer a narrower question first: does *any* per-pixel adaptivity help? If B
already matches C, the expensive version is not earning its cost. If C does not beat A on
detection, it does not ship, however elegant.

## Common confusions

**"Does the spatial model need retraining?"** Yes - its inputs change meaning, so the whole
model is trained from scratch (or from a SpectralEarth pretraining run). Its *code* does not
change.

**"Where does masking of dead bands happen in the trace?"** Step 2 marks them; step 6 skips
them (part 35); step 14 leaves them out of the loss (part 37).

**"Why keep the 32-channel interface at step 7?"** So variants A, B and C differ only in the
spectral parts, and the comparison measures the compressor rather than three things at once.

**"Is per-wavelength normalisation a new idea?"** It is the current per-band normalisation
with the band index replaced by its wavelength - the same move the whole sequel makes.

## Check yourself

1. Which steps of the trace are new, and which are untouched?
2. Explain in two sentences why spectral attention inside a pixel does not let a hidden pixel
   reconstruct itself.
3. A training stat curve gives mean 0.20 at 860 nm and 0.24 at 870 nm. A sensor has a band at
   864 nm. Estimate its mean by linear interpolation. Show the steps.
4. Why must per-scene min-max normalisation be avoided here?
5. Variant B beats A, and C ties with B on detection. What would you conclude?

<details>
<summary>Answers</summary>

1. New: 3 (by wavelength), 5 and 6 (tokens, compressor), 12 and 13 (decoder), and the
   valid-band rule in 14. Untouched: 8 to 10 - masking, encoder, decoder.
2. Spectral attention mixes bands within one pixel and moves nothing between pixels, and a
   hidden pixel's tokens are removed before spatial attention. Its reconstruction can only
   come from what the encoder inferred from its neighbours.
3. 864 is `(864 - 860) / (870 - 860) = 4/10 = 0.4` of the way from 860 to 870. Mean
   `= 0.20 + 0.4 * (0.24 - 0.20) = 0.20 + 0.4 * 0.04 = 0.20 + 0.016 = 0.216`.
4. It removes absolute reflectance, which spectral library matching needs, and makes the same
   material look different from scene to scene.
5. Per-pixel adaptivity helps, but the full cross-attention design adds nothing over the cheap
   version on the task that matters - so prefer B, unless C's other benefit (native bands,
   no resampling for new sensors) is worth its cost for its own sake.

</details>

---

That is the sequel. The next step is a written design (shapes, normalisation, evaluation
protocol) in `docs/proposals/spectral-attention-bottleneck.md`, reviewed before any model code
is written.
