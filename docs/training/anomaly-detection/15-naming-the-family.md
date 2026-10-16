# Part 15 - Naming the material family

> **The one thing this part teaches:** to name what a small object is made of, compare the
> pixel with its own surroundings rather than with the whole library, score whole families
> rather than single spectra, and only give a name when it beats the scene's ordinary pixels
> by a bar you chose in advance.

## Why part 14's answer usually names the grass

Part 14 matched each anomaly pixel's whole spectrum against single library spectra with SAM.
At 30 metres that has a basic problem: **a roof, a truck or a tarpaulin rarely fills a
pixel.** The sensor records a mix.

```
   one 30 m pixel                      what the sensor records
  +-------------------+
  |  grass   grass    |       pixel = 0.8 x grass + 0.2 x roof
  |     +-----+       |
  | grass|roof | grass|       SAM asks: "which library spectrum looks most
  |     +-----+       |       like this whole pixel?"  Answer: grass.
  |  grass   grass    |
  +-------------------+
```

The design doc measured it: on planted targets SAM named the right man-made family for **6 %**
of pixels at 30 % fill (`docs/lld/spectral-family-match.md`, section 1). There was a second
problem too: `data/splib07_slim` has only **3** usable man-made spectra out of 519
(`docs/09-known-issues.md`, item 10).

Since 2026-10-16 the `spectral_library_match` action runs a second matcher after SAM, and
the viewer and the export lead with its answer. This part builds it up one idea at a time.
The code is `app/spectral_match/family_match.py`, about 200 lines.

## Do it on paper first: three bands

Take three bands, so you can do every step by hand. The numbers are reflectance (part 2).

| | band 1 | band 2 | band 3 |
|---|---|---|---|
| grass `g` | 0.05 | 0.40 | 0.25 |
| roof `r` | 0.30 | 0.32 | 0.35 |
| soil `s` | 0.20 | 0.25 | 0.30 |

The pixel is 80 % grass and 20 % roof:

```
x = 0.8 * g + 0.2 * r

band 1:  0.8 * 0.05 + 0.2 * 0.30 = 0.040 + 0.060 = 0.100
band 2:  0.8 * 0.40 + 0.2 * 0.32 = 0.320 + 0.064 = 0.384
band 3:  0.8 * 0.25 + 0.2 * 0.35 = 0.200 + 0.070 = 0.270

x = (0.100, 0.384, 0.270)
```

**What SAM says.** The angle between `x` and each library spectrum, as in part 14:

```
x . g = 0.100*0.05 + 0.384*0.40 + 0.270*0.25
      = 0.0050 + 0.1536 + 0.0675 = 0.2261
|x|   = sqrt(0.0100 + 0.147456 + 0.0729) = sqrt(0.230356) = 0.47995
|g|   = sqrt(0.0025 + 0.1600 + 0.0625)   = sqrt(0.2250)   = 0.47434
cos   = 0.2261 / (0.47995 * 0.47434) = 0.2261 / 0.22766 = 0.99315   ->  6.7 degrees

x . r = 0.0300 + 0.12288 + 0.0945 = 0.24738
|r|   = sqrt(0.09 + 0.1024 + 0.1225) = sqrt(0.3149) = 0.56116
cos   = 0.24738 / (0.47995 * 0.56116) = 0.24738 / 0.26933 = 0.91851  ->  23.3 degrees
```

Grass at 6.7 degrees, roof at 23.3. **SAM confidently names the surroundings.** That is the
6 % in a nutshell.

## Idea 1 - subtract the surroundings

If you knew the background, you could remove it. The pixel minus the background is "what the
object adds". The code estimates the background as the mean of a **ring** around the pixel:

```
RING_OUTER, RING_INNER = 13, 5       # ring = 13x13 window minus the central 5x5
```

(`family_match.py`, top of file.) The inner 5x5 is left out so the object itself, if it
spills into neighbours, does not pollute its own background. Count the ring:

```
13 * 13 = 169 pixels in the window
 5 *  5 =  25 pixels in the hole
169 - 25 = 144 pixels in the ring
```

Anomaly pixels are left out of every ring, so two neighbouring targets do not become each
other's background. If fewer than `MIN_RING_PX = 20` usable pixels remain - next to the scene
edge, or inside a cluster of anomalies - the ring falls back to all valid pixels.

In our example the ring is pure grass, so `m = g`:

```
z = x - m
  = (0.100 - 0.05, 0.384 - 0.40, 0.270 - 0.25)
  = (0.050, -0.016, 0.020)
```

Now compare that with "roof minus background":

```
t = r - m = (0.30 - 0.05, 0.32 - 0.40, 0.35 - 0.25) = (0.25, -0.08, 0.10)
0.2 * t   = (0.050, -0.016, 0.020)        <- exactly z
```

**The difference points exactly along roof-minus-grass, and its length says how much roof
there is.** The 80 % grass has vanished.

## Idea 2 - measure the match with ACE

The score for "does `z` point along `t`" is the squared cosine, kept signed so a pixel that
is *darker* than its background in the roof's direction does not count as roof:

```
ACE = sign(t.z) * (t.z)^2 / (|t|^2 * |z|^2)          # family_match.py, ace()
```

It runs from -1 to 1, and 1 means "exactly along". Work it for roof and for soil:

```
roof:  t = (0.25, -0.08, 0.10)
       t.z   = 0.25*0.050 + (-0.08)*(-0.016) + 0.10*0.020
             = 0.01250 + 0.00128 + 0.00200 = 0.01578
       |t|^2 = 0.0625 + 0.0064 + 0.0100 = 0.0789
       |z|^2 = 0.002500 + 0.000256 + 0.000400 = 0.003156
       ACE   = 0.01578^2 / (0.0789 * 0.003156) = 0.000249 / 0.000249 = 1.00

soil:  t = s - m = (0.15, -0.15, 0.05)
       t.z   = 0.0075 + 0.0024 + 0.0010 = 0.0109
       |t|^2 = 0.0225 + 0.0225 + 0.0025 = 0.0475
       ACE   = 0.0109^2 / (0.0475 * 0.003156) = 0.00011881 / 0.00014991 = 0.79
```

Roof 1.00, soil 0.79. The order has flipped from SAM's.

**One step the paper example skips: whitening.** In a real scene some directions vary all the
time - shade, how green the vegetation is - so a difference along them is unremarkable. Part 8
handled the same problem for RX by dividing by the scene covariance. Here the code multiplies
both `z` and `t` by `W`, the inverse square root of the scene covariance, before the cosine
(`whitener()`, estimated from up to 60,000 valid pixels). That is what makes this the
**Adaptive Coherence Estimator**, the standard detector for a known target that fills part of
a pixel.

## Idea 3 - score a family, not a spectrum

No single "roofing" spectrum matches a real roof: there are tar shingles, clay tiles, metal
sheets, all weathered differently. So a man-made family is scored as a **cone**: any
non-negative mix of its spectra is allowed to explain `z`.

```
Simplex ACE = |projection of z onto {sum a_k t_k, all a_k >= 0}|^2 / |z|^2
```

The best mix is found by non-negative least squares (`scipy.optimize.nnls`, in
`family_scores()`). Because a one-spectrum mix is one of the allowed mixes, the cone score is
never below the best single member - a unit test checks exactly that.

Two rules keep this honest:

- **Natural families never get a cone.** There are dozens of minerals and soils; a
  non-negative mix of them can imitate almost anything. A natural family scores its best
  single spectrum instead.
- **Every family is capped at 40 spectra** (`FAMILY_CAP` in `family_library.py`), picked to be
  spread out. Uncapped, about 1,900 minerals would out-vote about 180 man-made spectra by
  sheer number of chances to fit.

The shipped library (`data/family_library`, 411 spectra) has **10 man-made families**
(building, glass, metal, oil, paint, pavement, plastic, roofing, textile, wood/paper) and
**7 natural ones** (mineral, non-photosynthetic vegetation, rock, snow/ice, soil, vegetation,
water).

## Idea 4 - calibrate against the scene's ordinary pixels

Raw scores do not compare across families. A big, varied family like plastic fits *everything*
a little better than a family of one glass spectrum. They do not compare across scenes either.

The fix is the same idea as judging a runner's time against other runners on the same course
on the same day: a hilly, windy course makes everyone slow. The code scores up to
`NULL_N = 10,000` **ordinary** pixels of the same scene - the "null" - and turns each anomaly
pixel's score into a **percentile per family**: the share of ordinary pixels that scored
strictly lower (`calibrate()`).

```
null scores for roofing (sorted):  0.02  0.05  0.11  0.20  ...  (10,000 of them)
pixel's roofing score:             beats 9,983 of them
percentile:                        9,983 / 10,000 = 0.9983
```

Ordinary means: valid, and not within 3 pixels of any anomaly (`EXCLUDE_BUFFER_PX = 3`). No
ground truth is needed.

The pixel's family is the one with the highest percentile. Its percentile is the
**confidence** you will see in the viewer.

## Idea 5 - only name it above a bar you chose in advance

`alpha` is the share of ordinary pixels you are willing to see given a man-made name. The
action's config calls it `family_alpha` and defaults it to 0.05. The code turns it into a bar:

```
p* = 1 - alpha / F          F = number of man-made families
```

Work it for the shipped library:

```
alpha = 0.05, F = 10
alpha / F = 0.05 / 10 = 0.005
p*        = 1 - 0.005 = 0.995
```

Why divide by F? An ordinary pixel clears one family's bar with probability 0.005, because
that is what a 99.5th percentile means. It has 10 families to get lucky with:

```
P(some man-made name) <= 10 * 0.005 = 0.05 = alpha
```

That is the **union bound**: the chance of at least one of several events is at most the sum
of their chances. So the bar tightens on its own when a family is added - field spectra of a
known target, say - and alpha keeps its meaning.

## What a pixel ends up as

`backend/allotrope/action_types/_spectral_library_match_run.py`, `_family_match()`, gives every
anomaly pixel a code. It is written to `family_map.tif` and `family_matches.parquet`:

| code | meaning | viewer says |
|---|---|---|
| -2 | not an anomaly pixel | - |
| -1 | not scored: nodata on a band the matcher uses | "Not scored" |
| 0 | best family is man-made, below p* | "Man-made, type unconfirmed" |
| 1..17 | that family (sorted name order, plus one) | "Likely material: roofing" or "Natural: soil" |

Two details that look like mistakes and are not:

- **A natural family is shown even below p\*.** "Natural: soil" is useful context at any
  confidence; the bar exists to stop *man-made* claims.
- **"Not scored" exists because nodata is not zero.** Invalid cells in the onboarding cube
  hold -9999 on AVIRIS-NG and the scale minimum on PRISMA (part 5). The matcher keeps a band
  if 99 % of the anomaly pixels are valid on it, so up to 1 % could carry fill values into a
  whitened score. Those pixels are left out rather than named.

Each pixel also carries `best_entry`: the closest single spectrum inside its family. **It is
a hint, not the answer.** The family score came from a mix, and from pixel-minus-ring, not
from this one spectrum.

## Where it runs, and what it costs

Family matching runs after SAM, inside the same action. It uses the **raw** cube: the
experiments did not smooth, and Savitzky-Golay is SAM's step. SAM's files are left exactly as
they were, so actions created before 2026-10-16 still open and export as before.

The cost is the per-pixel least squares: about 45 s for about 18,000 pixels on one CPU, or

```
45 s / 18,000 pixels = 0.0025 s = 2.5 ms per pixel
```

That is why it is skipped in `mode="all_kept"`:

```
1,000,000 pixels * 2.5 ms = 2,500 s = about 42 minutes
```

A failure here is written into `summary.json` as `family.status = "failed"`, never raised.
The SAM result still makes a complete action.

## What the analyst sees

The viewer shows two linked maps side by side. On the left is the RGB with family colours:
saturated for man-made, muted earth tones for natural. On the right is the binary anomaly
mask. Pan or zoom one and the other follows; hover or click either to pin a pixel.

The rail leads with the verdict ("Likely material: metal, confidence 0.9971 >= p* 0.9950"),
then the spectrum chart with the family hint curve, then per-family counts, then SAM's top-K
as before.

The export bundle becomes schema `allotrope-export-hyper/2`:

```
hyper_<scene8>_<action8>/
    hyper_anomaly_materials.tif      int16 family codes, nodata -2
    hyper_anomaly_materials.shp      one point per anomaly pixel (+ shx, dbf, prj, cpg)
                                     family, group, fconf, named, best_ent, mat1..3, scr1..3
    hyper_material_legend.json       code -> family, group, colour; alpha and p*
    hyper_sam_legend.json            SAM's library index -> material, as before
    hyper_match_summary.csv
    MANIFEST.json
```

## What the answer is, and is not

On the four labelled PRISMA scenes, real targets got a man-made name about four times as
often as false alarms on sets 1 and 2 - and no more often on sets 3 and 4. False alarms
outnumber targets about 50 to 1, so **most man-made names in a scene still land on false
alarms** (`docs/lld/spectral-family-match.md`, section 5).

The alpha guarantee is about *ordinary* pixels. False alarms are unusual by definition, which
is why they were flagged, so they get man-made names more often than alpha.

**This is a naming step for pixels the analyst already kept, not a filter that decides which
pixels are real.** Same honesty as part 14, one step further.

## Common confusions

**"Confidence 0.997 means a 99.7 % chance it is roofing?"**
No. It means the pixel's roofing score beat 99.7 % of the scene's ordinary pixels. It says how
unusual the roofing-shaped difference is, not the probability of roofing.

**"Why not give natural families a cone too? It works for man-made."**
Dozens of mineral and soil spectra can mix into almost any shape, so every pixel would fit
the mineral cone well and the scores would carry no information.

**"Why calibrate per scene instead of fixing a score threshold?"**
For the same reason part 12 used percentiles: raw scores shift with the scene's covariance,
its land cover and the family's size. A fixed cut would flag everything in one scene and
nothing in another.

**"Two things here are called 'confidence'."**
SAM's old export flagged `conf` when the angle was under 15 degrees. The family result's
`fconf` is the calibrated percentile. Schema `/2` bundles carry `fconf` and `named`; the
angle-based flag only appears in schema `/1`.

**"If family matching is better, why does SAM still run?"**
The evidence is four scenes with mixed results. SAM costs seconds and keeps old actions
consistent, so it stays alongside until the family result has been checked on more product
scenes.

## Check yourself

<details>
<summary>1. With alpha = 0.02 and a library of 15 man-made families (10 shipped plus 5 field spectra of known targets), what is p*?</summary>

```
alpha / F = 0.02 / 15 = 0.001333
p*        = 1 - 0.001333 = 0.998667
```

Adding families raised the bar, so the chance of an ordinary pixel getting any man-made name
stays at most 15 * 0.001333 = 0.02.
</details>

<details>
<summary>2. The ring were 15 x 15 minus the central 7 x 7. How many pixels?</summary>

```
15 * 15 = 225
 7 *  7 =  49
225 - 49 = 176
```
</details>

<details>
<summary>3. In the 3-band example, compute ACE for a "paint" spectrum p = (0.30, 0.40, 0.30) against z = (0.050, -0.016, 0.020).</summary>

```
t     = p - m = (0.30 - 0.05, 0.40 - 0.40, 0.30 - 0.25) = (0.25, 0.00, 0.05)
t.z   = 0.25*0.050 + 0 + 0.05*0.020 = 0.0125 + 0.0010 = 0.0135
|t|^2 = 0.0625 + 0 + 0.0025 = 0.0650
|z|^2 = 0.003156
ACE   = 0.0135^2 / (0.0650 * 0.003156) = 0.00018225 / 0.00020514 = 0.888
```

Below roofing's 1.00, above soil's 0.79. Paint is a closer impostor than soil because it
shares roofing's rise in band 1.
</details>

<details>
<summary>4. A pixel's best family is plastic at confidence 0.9912, and p* is 0.995. What does the viewer say, and which code goes into the GeoTIFF?</summary>

Plastic is man-made and 0.9912 < 0.995, so "Man-made, type unconfirmed", code 0. Its
`best_entry` is still recorded as a hint.
</details>

<details>
<summary>5. Why must the onboarding cube be reflectance on a 0-1 scale for this matcher, when SAM did not care?</summary>

SAM measures an angle, which ignores brightness. This matcher subtracts the ring mean and
compares the difference with library spectrum minus ring mean. Both terms are in reflectance
units, so a cube on another scale (DN, x10000) would make every difference meaningless. The
onboarding cube is reflectance 0-1 for PRISMA, EnMAP and AVIRIS-NG, and the code logs a
warning if the median pixel value falls outside 0.005-1.5.
</details>

---

## You have finished

You can now read the source. Where to go next:

- **`docs/01-orientation.md`** through **`docs/10-code-style.md`** - the reference docs
- **`docs/lld/`** - low-level designs for individual subsystems; for this part,
  `docs/lld/spectral-family-match.md`
- **`docs/09-known-issues.md`** - what is currently broken, and what to check before trusting
  a green test run

One habit worth keeping from this course: **when a comment and the code disagree, the code
wins.** You met three such disagreements here - the regularisation default, the
`MIN_VALID_FRACTION` comment, and the discarded `units` field. There will be more.
