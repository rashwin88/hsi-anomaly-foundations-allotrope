# Part 14 - Naming the material, and getting it out

> **The one thing this part teaches:** comparing a satellite spectrum to a laboratory one
> requires simulating the sensor, and the answer that reaches the analyst is a georeferenced
> shortlist, not a verdict.

## The last question

Part 12 gave a shortlist of unusual pixels. An analyst wants more than "unusual". They want
"probably calcite" or "probably a hydrocarbon".

Hyperspectral data can answer that, because a spectrum is close to a material fingerprint
(part 2). Thermal cannot - one band is not a fingerprint.

## The reference library

**USGS splib07** - laboratory spectra measured under controlled conditions. The curated
subset in `data/splib07_slim/` holds **519 spectra**: 307 minerals, 69 organics, 61 soils,
59 artificial materials, 22 vegetation, 1 liquid.

Slimmed from a 6.7 GB distribution to about 40 MB, deliberately excluding the vendor's
pre-resampled copies. Those were already resampled onto other sensors' grids, and resampling
a resampled spectrum compounds the error.

## Why you cannot compare directly

The lab spectra are measured at 2,151 channels across 350 to 2,500 nm. A PRISMA cube on the
common grid has 165. The wavelengths do not line up.

The naive fix is to interpolate the lab spectrum at each satellite wavelength. That is wrong,
and the reason is worth understanding.

**A satellite band is not a point measurement.** A band centred at 1,650 nm with FWHM 10 nm
responds to a *range* around 1,650 - most strongly at the centre, tapering off. What the
sensor reports is a weighted average across that range.

So the lab spectrum must be **convolved with each band's response function**, not sampled at
its centre. That is what `gaussian_resample_to_target` does.

Work the width conversion, which is the one piece of arithmetic here:

```
sigma = FWHM / (2 * sqrt(2 * ln 2))
      = FWHM / 2.3548

for FWHM = 10 nm:
sigma = 10 / 2.3548
      = 4.247 nm
```

The code integrates 3 sigma either side:

```
window = 3 * 4.247 = 12.74 nm
```

so a band at 1,650 nm draws on lab measurements from about 1,637 to 1,663 nm. **This is why
part 4 insisted FWHM matters: without it there is no response function, and no way to
simulate the sensor.**

Bands with fewer than three lab samples in-window stay `NaN`, and that NaN pattern becomes the
validity mask the matcher uses.

## Spectral Angle Mapper, again

Same metric as part 12, used differently. There it compared a spectrum to its reconstruction;
here it compares a spectrum to a library entry.

The property that makes it right for this job: **invariance to brightness**. Two pixels of the
same mineral, one in sun and one in shadow, have spectra differing by a scale factor. SAM
returns the same angle for both, because scaling a vector does not change its direction.

Illumination, slope and shadow change magnitude. They do not change what the material is.

## Making it fast

A few thousand anomalous pixels against 519 library entries, each 165 bands. Done naively
that is a large number of small dot products.

Two tricks:

**Bucket by validity pattern.** Pixels sharing the same set of valid bands can be compared to
the same restricted library in one dense matrix multiply. The pattern is packed into a
bitmask and used as a dictionary key.

**Sub-bucket the library** by its own gap pattern, so each bucket is a single dense `matmul`
with correct per-pair norms.

Claimed result: under 2 seconds on CPU for a few thousand anomalies against about 1,200
entries.

## The cache, and why it is content-addressed

Resampling 519 spectra onto a sensor's grid is not free, and the result depends only on the
sensor. So it is cached - keyed by a SHA-256 hash of everything that could change the answer:

```
sensor_id, target wavelengths, target FWHM, bad-band mask,
chapters, min_coverage, splib07_version
```

Change any input and the key changes, so a stale cache is impossible. Change nothing and a
re-run is a no-op. **This is the pattern to copy whenever you cache a derived artifact.**

The cache is built at scene onboarding, best-effort - a failure there does not fail the
upload, it just means the first match pays the cost.

## Smoothing, and preserving the holes

Before matching, spectra are smoothed with a Savitzky-Golay filter - a local polynomial fit
that reduces noise while preserving peak shapes, which a moving average would flatten.

The detail worth noting: NaN bands are linearly interpolated, smoothed, then **set back to
NaN**. The validity pattern survives the smoothing. Lose it and the bucketing above breaks.

## What the analyst receives

`POST /actions/{id}/export` builds a zip:

```
hyper_<scene8>_<action8>/
    hyper_anomaly_materials.tif      int32 raster, top-1 library index per pixel
    hyper_anomaly_materials.shp      one point per matched pixel  (+ shx, dbf, prj, cpg)
    hyper_material_legend.json       index -> material name
    hyper_match_summary.csv          human-readable table
    MANIFEST.json                    schema allotrope-export-hyper/1
```

Material identity is carried **three ways** - a GeoTIFF metadata tag, DBF columns
`mat1/scr1` through `mat3/scr3`, and the standalone legend. Redundant on purpose: different
GIS tools read different parts, and a shortlist that cannot be opened is worthless.

Each point carries its top matches with angles, plus a `confident` flag set when the angle is
below `DEFAULT_CONFIDENCE_THRESHOLD_DEG = 15.0`.

Georeferencing arrives here, from re-reading the raw scene (part 5). If it cannot be
resolved, the export returns **422 `crs_missing`** rather than shipping a bundle that would
be disqualified downstream for having no projection.

## What the answer is, and is not

The output says: *this pixel, at these coordinates, has a spectrum 8.2 degrees from
laboratory calcite, and 11.7 degrees from dolomite.*

It does not say the pixel **is** calcite. A satellite pixel covers 30 metres and mixes
several materials. The atmosphere is imperfectly corrected. The library is not exhaustive.

**The output is a ranked hypothesis for an analyst to check.** That is the same honesty part
1 started with, carried through to the end: the system narrows, a human decides.

## Common confusions

**"Why not use Euclidean distance instead of SAM?"**
It would rank the same material in sun and shadow as different, because their magnitudes
differ. Brightness invariance is the requirement.

**"Sub-15-degree means it is that material?"**
It means the shape is close. Mixed pixels, atmospheric residue and library gaps all produce
plausible-looking matches. Treat it as a lead.

**"Why not match every pixel instead of only flagged ones?"**
You can - `mode="all_kept"` does. It is far more expensive and rarely what is wanted; the
question is what the *anomalies* are.

**"Why does the library exclude the pre-resampled copies?"**
They were resampled onto other sensors' grids. Resampling them again compounds error. Better
to start from the native 2,151-channel measurements.

## Check yourself

<details>
<summary>1. Why must a lab spectrum be convolved with a band's response rather than sampled at its centre?</summary>

A satellite band integrates over a wavelength range, weighted by its response function, so
what it reports is an average - not the value at one wavelength. Sampling at the centre
ignores the width and misrepresents what the sensor would actually measure.
</details>

<details>
<summary>2. A band has FWHM 12 nm. Compute sigma and the 3-sigma integration window.</summary>

```
sigma  = 12 / 2.3548 = 5.096 nm
window = 3 * 5.096   = 15.29 nm either side
```

A band at 2,200 nm would draw on lab measurements from about 2,184.7 to 2,215.3 nm.
</details>

<details>
<summary>3. Two pixels of the same mineral, one shadowed. Why does SAM give the same answer for both?</summary>

Shadowing scales the whole spectrum by roughly a constant. SAM measures the angle between
vectors, and scaling does not change direction, so the angle is unchanged. A magnitude-based
metric would rank them as different materials.
</details>

<details>
<summary>4. What goes into the cache key, and what problem does that solve?</summary>

Sensor id, target wavelengths, target FWHM, bad-band mask, chapters, minimum coverage and
library version. Any change to an input changes the key, so a stale cache cannot be read -
correctness without manual invalidation.
</details>

<details>
<summary>5. A pixel matches calcite at 6.4 degrees. What may you claim?</summary>

That its spectral shape is close to laboratory calcite and it is worth investigating. Not
that it is calcite: a 30-metre pixel mixes materials, atmospheric correction is imperfect,
and the library is not exhaustive. It is a ranked hypothesis, not an identification.
</details>

---

## You have finished

You can now read the source. Where to go next:

- **`docs/01-orientation.md`** through **`docs/10-code-style.md`** - the reference docs
- **`docs/lld/`** - low-level designs for individual subsystems
- **`docs/09-known-issues.md`** - what is currently broken, and what to check before trusting
  a green test run

One habit worth keeping from this course: **when a comment and the code disagree, the code
wins.** You met three such disagreements here - the regularisation default, the
`MIN_VALID_FRACTION` comment, and the discarded `units` field. There will be more.
