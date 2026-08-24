# Part 7 - The band pipeline

> **The one thing this part teaches:** eight filtering stages turn 239 sensor-specific bands
> into 165 bands that mean the same thing on every hyperspectral sensor.

## The problem

PRISMA has 239 bands at its own wavelengths. EnMAP has 224 at different ones. AVIRIS-NG has
more still. To train one model on all three, band 47 must mean the same thing in every cube.

It cannot, natively. So the cubes are resampled onto a shared grid.

## The eight stages

Driven entirely by `BandFilterConfig` in `app/models/dataset/vendables.py`.

```
239 bands (PRISMA)
  1. drop bands the sensor flagged bad
  2. drop atmospheric absorption windows
  3. trim 3 detector-edge bands per end
  4. drop bands with under 20% valid pixels
       |
       v  about 188 bands survive
  5. apply quality masks              (EnMAP only)
  6. invalidate pixels over 40% bad   -> validity becomes binary
  7. PCHIP-interpolate remaining gaps
  8. resample onto the common grid
       |
       v  165 bands, 10 nm apart, 460-2450 nm
```

## Why drop bands at all

**Stage 1 - the sensor already knows.** Detectors fail. The sensor ships a flag per band.
Note the trap from part 4: a flagged band still occupies a slice in the cube, carrying
`FWHM = 0.0`. It is present but meaningless.

**Stage 2 - the atmosphere is opaque at some wavelengths.** Water vapour absorbs so strongly
around 1,400 nm that essentially no signal reaches orbit. The band exists, records noise, and
that noise varies with humidity - so it looks like structure and is not.

The default exclusions, and what each is:

| range (nm) | why |
|---|---|
| 0 - 450 | low signal-to-noise at the blue end |
| 912 - 978 | water vapour |
| 1131 - 1152 | water vapour |
| 1350 - 1450 | strong water vapour |
| 1800 - 1950 | water vapour and CO2 |

**Stage 3 - detector edges are unreliable.** Response falls off at the ends of a detector
array, so three bands at each end of each detector are trimmed.

**Stage 4 - a band mostly invalid is not worth keeping.** Under 20% valid pixels, it goes.

## Stage 6, which changes how you think

After stage 6 **validity is binary**: a pixel has a complete spectrum or nothing at all.

The threshold is `max_invalid_voxel_fraction = 0.4`. Over 40% of a pixel's bands invalid, the
whole pixel is invalidated. Under it, the gaps get filled by stage 7.

This removes an entire category of special-casing downstream. No detector ever has to handle
"valid in these 140 bands but not those 25". You will feel the benefit constantly.

## Stage 7 - filling gaps without inventing features

Bands were removed. A pixel's spectrum now has holes. Interpolation fills them - and the
choice of method matters more than it looks.

The code uses **PCHIP** (piecewise cubic Hermite), not a cubic spline. A spline is smoother,
so why not?

Because a spline **overshoots**. Given points that dip and recover, it can swing below the
lowest input value before coming back. That invented dip looks exactly like a narrow
absorption feature - which is precisely the evidence part 14 uses to identify a material.

**PCHIP is shape-preserving: it will not create a minimum that was not in the data.** It gives
up some smoothness to guarantee it never invents a feature. In a system whose whole purpose
is spotting unusual spectral features, that trade is not close.

## Stage 8 - the common grid

The payoff. Every hyperspectral cube is resampled onto:

**165 bands, 10 nm spacing, 460 to 2450 nm, atmospheric windows excluded.**

Verify it yourself:

```python
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as G
len(G)   # 165
G[0]     # 460.0
G[-1]    # 2450.0
```

The grid is not evenly spaced end to end - the excluded windows leave four gaps:

| gap | from | to |
|---|---|---|
| 1 | 910 | 980 |
| 2 | 1130 | 1160 |
| 3 | 1340 | 1460 |
| 4 | 1790 | 1960 |

Check the count by hand. A full 10 nm grid from 460 to 2450 inclusive:

```
(2450 - 460) / 10 + 1 = 199 + 1 = 200 slots
```

Now subtract the excluded ones. Each gap removes the slots strictly inside it:

```
gap 1: 920..970  = 6 slots
gap 2: 1140..1150 = 2 slots
gap 3: 1350..1450 = 11 slots
gap 4: 1800..1950 = 16 slots
                    -------
                    35 slots removed

200 - 35 = 165
```

**165. Do that subtraction yourself** - it is the clearest way to see that the grid is a full
ladder with four rungs missing, not an arbitrary number.

Why 10 nm: it respects the coarsest sensor's resolution. PRISMA bands are about 12 nm wide,
so a finer grid would claim detail no sensor supplies.

## What this buys

After stage 8, a PRISMA cube and an EnMAP cube have **identical shape and identical
wavelengths**. One model trains on shards mixing both. A lab spectrum resampled onto the grid
compares against either.

That single fact is why the hyperspectral foundation model exists at all.

## One more fix: nearest-valid fill

Not part of the eight stages, but applied straight after by `band_filter_apply`.

The SegFormer models embed the image with a 7x7 kernel that straddles valid/invalid
boundaries. An invalid pixel sitting at 0 normalises to about -2.3, against valid pixels
between -0.7 and +1.7. That is a cliff, and the resulting boundary artefacts **outrank real
detections** in the final ranking.

The fix replaces each invalid pixel with its nearest valid neighbour's **actual measurement** -
never an interpolation. The masks are not modified, so the loss still excludes those pixels;
only the values the convolution sees change. `app/utils/pixel_fill/nearest_valid_fill.py`.

## Common confusions

**"Does resampling lose information?"**
Yes, and it is accepted deliberately. Cross-sensor comparability is worth more than the
last few nanometres of native resolution. Note that part 14 deliberately works on the
*unfiltered* vendable for exactly this reason.

**"Why not interpolate across the atmospheric gaps too?"**
Because there is no signal there to interpolate from. The gap is a hundred nanometres wide;
filling it would be inventing a hundred nanometres of spectrum. The grid keeps the hole.

**"188 bands survive stage 4, but the grid is 165. Where did the rest go?"**
Nowhere - they are different axes. The 188 are surviving *sensor* bands at sensor
wavelengths; the 165 are *grid* slots. Stage 8 resamples one onto the other.

**"Is 0.4 in stage 6 a fraction of bands or of the cube?"**
Of that pixel's bands. A pixel with more than 40% of its own bands invalid is dropped
entirely.

## Check yourself

<details>
<summary>1. Why is PCHIP used rather than a cubic spline?</summary>

PCHIP is shape-preserving and will not overshoot. A spline can invent a dip that was not in
the data, and an invented dip is indistinguishable from a real absorption feature - the
exact evidence used for material identification.
</details>

<details>
<summary>2. Derive the 165-band count from the grid definition.</summary>

```
full ladder 460..2450 step 10 = (2450-460)/10 + 1 = 200 slots
gap 920..970   ->  6 removed
gap 1140..1150 ->  2 removed
gap 1350..1450 -> 11 removed
gap 1800..1950 -> 16 removed
                  ----------
                  35 removed
200 - 35 = 165
```
</details>

<details>
<summary>3. A pixel is invalid in 61 of 165 bands. Kept or dropped?</summary>

```
61 / 165 = 0.3697
0.3697 < 0.4  ->  kept
```

Its 61 gaps get filled by PCHIP. At 67 bands, `67/165 = 0.406 > 0.4`, and it would be
dropped entirely.
</details>

<details>
<summary>4. Why does an invalid pixel at zero cause a problem for the SegFormer models specifically?</summary>

Their patch embedding uses a 7x7 kernel that spans valid and invalid pixels together. A zero
normalises to about -2.3 against a valid range of -0.7 to +1.7, so the kernel sees a cliff.
The resulting boundary artefacts outrank real detections.
</details>

<details>
<summary>5. Why does the nearest-valid fill copy a real measurement rather than interpolate?</summary>

An interpolated value is invented data. Copying a genuine nearby measurement keeps every
value physically plausible, which matters because the whole system hunts for values that do
not fit - and an invented one might not fit either.
</details>

---

Next: [part 8](08-rx-detection.md) - statistical detection, and one banned technique.
