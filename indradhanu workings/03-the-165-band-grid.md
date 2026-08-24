# 03 · Where the number 165 comes from

> **The one thing this part teaches:** 165 is not a magic number. It is 200
> evenly-spaced wavelengths minus 35 that are useless from space. You can derive
> it on paper in two minutes.

---

## The problem: three satellites that do not agree

PRISMA, EnMAP and AVIRIS-NG all measure "the spectrum". But they do not measure
it at the same places.

Made-up but representative example:

```
PRISMA    measures at ... 1000 nm, 1012 nm, 1024 nm, ...
EnMAP     measures at ... 1003 nm, 1011 nm, 1019 nm, ...
AVIRIS-NG measures at ... 1001 nm, 1006 nm, 1011 nm, ...
```

Different starting points, different spacings, different totals.

So band number 47 from PRISMA and band number 47 from EnMAP mean **different
wavelengths**. If you handed both to the same model, band 47's weights would be
learning two contradictory things.

### Why we care about using one model for all three

Training data is the bottleneck. If each satellite needs its own model:

- each model gets one third of the data,
- each model is three times as much work to train, tune and maintain,
- a new satellite means a whole new model.

If instead they can share, one model gets all the data and any new sensor joins
for free.

---

## The fix: agree on a fixed list of wavelengths

Decide, once, on one official list of wavelengths. Then **resample** every
satellite's measurements onto that list before anything else touches the data.

Resampling means: "you measured at 1001 and 1006; I need a value at 1000, so
estimate one from your neighbours." The estimating is done by interpolation.

That official list lives in
[`app/models/dataset/vendables.py`](../app/models/dataset/vendables.py) and is
called `DEFAULT_COMMON_WAVELENGTH_GRID`.

---

## Building the list, step by step

### Step 1 — lay down an even ladder

```python
full_grid = np.arange(start_nm, end_nm + spacing_nm / 2, spacing_nm)
# start_nm = 460.0, end_nm = 2450.0, spacing_nm = 10.0
```

In plain words: **start at 460 nanometres, take steps of 10, stop at 2450.**

```
460, 470, 480, 490, ... , 2430, 2440, 2450
```

How many rungs is that? The standard fencepost formula:

```
number of rungs = (last - first) / step  +  1

                = (2450 - 460) / 10 + 1
                = 1990 / 10 + 1
                = 199 + 1
                = 200
```

**200 wavelengths.**

> **Why the odd `+ spacing_nm / 2` in the code?** Python's `arange` stops
> *before* the end value, so `arange(460, 2450, 10)` would stop at 2440 and lose
> the last rung. Adding half a step (5) pushes the limit to 2455, which safely
> includes 2450 without adding 2460. It is a fencepost fix, nothing more.

### Step 2 — throw away the wavelengths that do not work from space

The atmosphere is not transparent. At certain wavelengths, water vapour and
carbon dioxide in the air absorb the light almost completely before it reaches
the satellite.

At those wavelengths the sensor is not measuring the ground. It is measuring
noise.

Here is the exclusion list, hard-coded in the same function:

| Range (nm) | Why it is excluded |
|---|---|
| 0 to 450 | signal is too weak to be trustworthy |
| 912 to 978 | water vapour absorption |
| 1131 to 1152 | water vapour absorption |
| 1350 to 1450 | strong water vapour absorption |
| 1800 to 1950 | water vapour and carbon dioxide |

### Step 3 — count what each exclusion removes

A rung at wavelength `x` is deleted when `lo <= x <= hi`. Our rungs are 460,
470, 480, and so on.

**Range 0 to 450.** Our ladder starts at 460. Nothing to delete.
→ **0 removed**

**Range 912 to 978.** Which multiples-of-ten rungs fall inside?

```
910 -> is 910 >= 912?  No.  keep
920 -> yes, delete
930 -> yes, delete
940 -> yes, delete
950 -> yes, delete
960 -> yes, delete
970 -> yes, delete
980 -> is 980 <= 978?  No.  keep
```

→ **6 removed** (920, 930, 940, 950, 960, 970)

**Range 1131 to 1152.**

```
1130 -> 1130 >= 1131?  No.  keep
1140 -> yes, delete
1150 -> yes, delete
1160 -> 1160 <= 1152?  No.  keep
```

→ **2 removed**

**Range 1350 to 1450.** Both ends land exactly on rungs, so use the fencepost
formula again:

```
(1450 - 1350) / 10 + 1 = 100/10 + 1 = 10 + 1 = 11
```

→ **11 removed** (1350, 1360, ..., 1450)

**Range 1800 to 1950.** Same again:

```
(1950 - 1800) / 10 + 1 = 150/10 + 1 = 15 + 1 = 16
```

→ **16 removed** (1800, 1810, ..., 1950)

### Step 4 — total it up

```
removed = 0 + 6 + 2 + 11 + 16 = 35

remaining = 200 - 35 = 165
```

## **That is where 165 comes from.**

Nothing mysterious: a 10 nm ladder from 460 to 2450, minus five stretches the
atmosphere ruins.

> **Do this on paper once.** It takes two minutes and it permanently changes
> 165 from "a constant somebody chose" into "a consequence of two decisions I
> could have made myself".

---

## What the surviving grid looks like

The five surviving stretches are called **spectral families**:

```
460 ....... 910 | gap | 980 .. 1130 | gap | 1160 .. 1340 | gap | 1460 .. 1790 | gap | 1960 .. 2450
<- family 1 ->         <-family 2->        <-family 3->         <-family 4->         <-family 5->
```

The vendable (part 04) records which family each band belongs to, in a field
called `spectral_family_order`.

---

## Two design questions you should be able to answer

### Why 10 nm steps and not 5, for more detail?

Because the grid has to respect the **coarsest** contributing sensor. PRISMA's
bands are about 12 nm wide.

If you asked for a value every 5 nm, you would be claiming detail that PRISMA
never measured. The interpolator would happily produce numbers, but those
numbers would be invented, and the model would learn to reproduce invented
structure.

10 nm is the honest choice: fine enough to be useful, coarse enough that every
value is supported by a real measurement.

### Why cut the gaps out *before* interpolating, rather than interpolating and discarding after?

This is subtler and more important.

The resampling uses an interpolator called PCHIP
([`app/utils/data_transformations/spectral_resampler.py`](../app/utils/data_transformations/spectral_resampler.py)).
An interpolator's job is to draw a smooth curve through known points.

Now imagine you leave 1350–1450 in the target grid. There is no real sensor data
in that stretch. The interpolator does not know that; it just draws a smooth
curve from 1340 across to 1460:

```
real data      real data
    |              |
1340 *            * 1460
      \__________/       <- 11 completely invented values
```

Those 11 fabricated values would go into training. The model would learn to
reproduce them confidently. And at inference, a genuine anomaly in that region
would be compared against fiction.

By deleting those wavelengths from the grid, **the interpolator is never asked
the question.** The problem is designed out rather than cleaned up afterwards.

---

## What this means for Indradhanu specifically

- The model's configuration says `in_channels = 165`. That number is downstream
  of this grid, not an independent choice. (See
  `HyperspectralSegFormerMAEConfig` in
  [`app/models/training/training_config.py`](../app/models/training/training_config.py).)
- The 165 averages and 165 spreads in
  [`app/constants/hyperspectral.json`](../app/constants/hyperspectral.json) are
  lined up with *this exact list, in this exact order*.
- One training file can contain PRISMA patches and EnMAP patches mixed together.
  At the tensor level the model genuinely cannot tell which is which.
- AVIRIS-NG was added to the project later and required **no model change at
  all**, because it comes out of the preprocessing as the same 165-band cube.
  There is a comment saying exactly that in
  [`backend/allotrope/foundation_models/resolver.py`](../backend/allotrope/foundation_models/resolver.py).

---

## The gotcha: this grid is a wire format

Nowhere in the saved model file does it say "band 43 means 1000 nm". The
connection is purely **positional** — band 43 means whatever the 44th entry of
the grid is (counting from zero).

So if somebody edited the exclusion ranges tomorrow:

- the model would still load without complaint (still 165 channels),
- the normalisation statistics would still load (still 165 numbers),
- and every band would silently be about the wrong wavelength.

No error, no warning, just quietly wrong answers. Treat the grid like a database
schema: changing it invalidates everything already stored.

---

## Common confusions

**"Are the 165 bands evenly spaced?"**
No. They are 10 nm apart *within* each of the five families, with big jumps
between families. The list is ascending but not uniform.

**"Does resampling lose information?"**
A little, always. That is the price of comparability. The grid was chosen to
keep the loss small (10 nm, matching the coarsest sensor).

**"Could we just use each sensor's native bands and pad?"**
You could, but then band 47 would mean different things in different rows of the
same training batch, which is precisely the thing that stops one model from
serving three sensors.

---

## Check yourself

1. How many rungs are there from 460 to 2450 in steps of 10? Show the formula.
2. How many rungs does the range 1800–1950 delete?
3. Why are the atmospheric ranges deleted *before* interpolation instead of
   after?
4. Why is the step 10 nm rather than 5 nm?
5. What would break if someone changed the exclusion ranges but kept the
   existing model file?

<details>
<summary>Answers</summary>

1. `(2450 - 460) / 10 + 1 = 200`.
2. `(1950 - 1800) / 10 + 1 = 16`.
3. Because an interpolator asked for a value in a data-free gap will invent a
   smooth one. Deleting the wavelengths means it is never asked.
4. Because PRISMA's bands are about 12 nm wide; a 5 nm grid would claim detail
   no sensor measured.
5. Band index maps to wavelength purely by position. The file would still load
   (still 165 channels) but every band would refer to the wrong wavelength — a
   silent, error-free wrongness.

</details>

---

**Next:** the exact object the model is handed, in
[04-the-vendable.md](04-the-vendable.md)
