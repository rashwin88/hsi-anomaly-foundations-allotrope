# Frequency-domain destriper

**File:** `app/utils/data_transformations/frequency_domain_destriper.py`
**Tests:** `tests/test_utils/test_data_transformations/test_frequency_destriper_characterisation.py`

## Purpose

Pushbroom sensors build an image one line at a time using a row of detectors. When those
detectors have slightly different gains, the result is stripes running down the scene. This
removes them by finding the stripe direction in the frequency domain and notching it out.

## Why the frequency domain

Stripes are periodic and directional. In the spatial domain they are spread across every
pixel; in the frequency domain that same pattern collapses to a narrow line of energy through
the origin, perpendicular to the stripes.

That concentration is the whole trick. Removing a thin wedge of frequencies removes the
stripes while leaving almost all of the scene untouched, because real terrain has energy
spread over all directions and loses only the sliver that overlaps the wedge.

## Data flow

```
cube (B, H, W)
  |
  |-- stage 1: _find_stripe_angles      per-band power spectra -> candidate angles -> consensus
  |             returns [(angle, strength_in_sigma), ...]   or []  -> return copy unchanged
  |
  |-- stage 2: _build_notch_filter      one wedge per angle, combined by element-wise minimum
  |             adaptive radial_preserve from strength
  |
  |-- stage 3: _apply_filter            batched FFT -> multiply -> inverse FFT
  |
  v
destriped cube, same shape and dtype
```

## The significance test, and its trap

A candidate angle is accepted only if its peak stands `PEAK_SIGMA = 3.0` standard deviations
above the background of the power-versus-angle curve, with the peak's own neighbourhood
excluded (`BACKGROUND_EXCLUSION_DEG = 10.0`):

```python
bg_mean, bg_std = _background_stats(sums, angles_grid, peak_angle)
if bg_std > 0 and sums[peak_idx] > bg_mean + PEAK_SIGMA * bg_std:
    candidate_angles.append(peak_angle)
```

**Read the `bg_std > 0` guard carefully, because it has a counter-intuitive consequence: a
perfectly clean stripe is never detected.**

This is not hypothetical. Writing the characterisation tests, a pure sinusoid on a smooth
background was fed in at amplitudes from 0.25 to 0.6, at periods of 4, 8 and 16 pixels, at
two image sizes. Twelve combinations, zero detections. The instrumented curve explains why:

| scene | peak angle | sigma | outcome |
|---|---|---|---|
| pure sinusoid, amplitude 0.6 | 0.0 | `nan` | rejected, `bg_std == 0` |
| row-offset stripe, amplitude 0.6 | 0.0 | `nan` | rejected, `bg_std == 0` |
| textured scene + column gain 0.02 | 0.5 | 3.45 | accepted |
| textured scene + column gain 0.12 | 0.5 | 5.29 | accepted |

With no texture, the log power spectrum is flat everywhere except the two stripe peaks. The
background away from the peak has *zero* variance, so the divisor collapses, and the guard
rejects it. **The cleaner the scene, the less likely a stripe is found.**

Real scenes always have texture, so this is not a production bug. It is a trap for anyone
writing a synthetic test - which is why the fixture in the test file models correlated
texture plus a per-detector column gain rather than a bare sinusoid.

The inverse also holds and is worth knowing: **correlated texture alone can clear the 3
sigma gate with no stripe present**, and the destriper will then filter real scene structure
out of the data. The consensus step across bands is what usually catches that.

## Why invalid pixels are filled with the band mean

`_band_means` computes the mean of each band's valid pixels, and that value is used both to
fill masked pixels and to pad the border.

Filling with zero would be much simpler and much worse. A reflectance scene sits around 0.4;
a zeroed region is a step of 0.4 at every boundary. A step edge is **broadband** - it puts
energy at every frequency and every angle - so it would raise the background of the very
curve the significance test measures against, and could swamp the stripe peak entirely.

## Why the frame is padded

`PAD_WIDTH = 128` on every side. The FFT treats the image as periodic, so without padding
the right edge is adjacent to the left edge and the bottom to the top. Real scenes do not
match at those seams, and the discontinuity injects a spurious edge - again broadband,
again polluting the spectrum.

## Batching arithmetic

`_choose_batch_size` decides how many bands go through the FFT at once. Each in-flight band
needs a padded float32 input, a complex64 spectrum, a complex64 filtered copy and a float32
result:

```
4 + 8 + 8 + 4 = 24 bytes per padded pixel
```

For PRISMA, 1210 x 1219 padded by 128 each side:

```
padded height = 1210 + 256 = 1466
padded width  = 1219 + 256 = 1475
padded pixels = 1466 * 1475 = 2,162,350
per band      = 2,162,350 * 24 = 51,896,400 bytes = 51.9 MB
all 239 bands = 51.9 MB * 239 = 12.4 GB
```

12.4 GB is why this is batched rather than done in one shot. The budget is 2 GB on CUDA and
500 MB elsewhere, giving 38 bands per batch on GPU and 9 on CPU for a scene that size.

## Invariants

- Output has the same shape and dtype as the input.
- **Only valid pixels are written back.** Invalid ones keep their original values from the
  copy made at the top of `_apply_filter`, so the destriper never invents data where the
  sensor had none.
- When no angle is detected, the input is returned as a **copy**, not filtered and not the
  caller's own array.
- The transform is deterministic. No RNG anywhere; the same input gives bit-identical output.

## Failure modes

| Condition | Result |
|---|---|
| `validity_mask` not passed | `ValueError`, immediately |
| No significant stripe | unchanged copy, `WARNING` in the log |
| Peaks found but no cross-band consensus | unchanged copy, `WARNING` naming the disagreeing angles |
| A band is entirely invalid | skipped during probing; still filtered in stage 3 |

## Decisions

**Consensus across bands, not a single band.** Stripes come from the detector array, so they
appear at the same angle in every band. Scene content does not. Requiring agreement is what
separates the two, and it is the only defence against the false-positive case above.

**The notch combines by element-wise minimum, not by sum.** Several angles each get their own
wedge; taking the minimum means a frequency is removed if *any* wedge removes it. Summing
would let overlapping wedges cancel each other back towards 1.

**Radial preserve is adaptive.** Low frequencies carry the scene's large-scale structure, so
the wedge stops short of the origin. A stronger stripe justifies cutting closer in;
`_strength_to_radial_preserve` maps sigma to that distance.

## Verification

The decomposition of `_apply_filter` was checked by loading the pre-refactor module from git
alongside the refactored one and running both over three scenes - striped, weakly striped and
clean:

```
seed=3  amp=0.12: bit-identical=True max_abs_diff=0
seed=11 amp=0.05: bit-identical=True max_abs_diff=0
seed=7  amp=0.0 : bit-identical=True max_abs_diff=0
```

If you change anything in this file, do the same. The comparison in `_background_stats` was
deliberately left as `x > mean + sigma * std` rather than rearranged to
`(x - mean) / std > sigma`, because those are not identical in floating point.
