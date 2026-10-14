# 33 - A spectrum as tokens that know their wavelength

> **The one thing this part teaches:** each band of a pixel becomes one token made of two
> added parts - its measured value, stretched into a vector, plus a fixed fingerprint of its
> wavelength - so the model reads "0.21 at 864 nm" rather than "0.21 in slot 87".

## Where the tokens live

In part 11 a token was a small square of the image. Here a token is **one band of one
pixel**. A SpectralEarth pixel has 202 bands, so it becomes 202 tokens; an EnMAP pixel 224;
a PRISMA pixel 239.

```
one pixel's spectrum:  [0.084, 0.091, ..., 0.183]        (202,)    reflectance
as tokens:             202 vectors of d numbers          (202, d)
```

`d` is the token width. The design studied here uses `d = 32`, matching the current `D = 32`
channels so the parameter budgets compare cleanly (part 34). Every pixel is handled
separately: nothing in this part mixes one pixel with another.

## Part 1 of a token: the value, stretched

A single reflectance number cannot take part in dot products with 32-number queries. It is
stretched with a tiny linear layer, `nn.Linear(1, d)`: multiply by a learned vector, add a
learned bias.

Worked with `d = 2` so it fits on paper. Suppose training gave

```
w = [0.5, -1.0]     b = [0.1, 0.2]
```

A band measuring `0.3`:

```
0.3*0.5  + 0.1 =  0.15 + 0.10 =  0.25
0.3*-1.0 + 0.2 = -0.30 + 0.20 = -0.10

value part = [0.25, -0.10]
```

Parameters: `d` weights plus `d` biases. At `d = 32` that is `32 + 32 = 64`.

## Part 2 of a token: a fingerprint of its wavelength

Now the token must say *which* wavelength it is. Part 12 noted that Indradhanu's spatial
attention needs no positional encoding, because a convolution supplies position. Here there
is no grid and no convolution: 202 band tokens are, to attention, an unordered crowd. Without
a wavelength tag, swapping two bands would change nothing. So the tag is required.

The tag is computed, not learned: sines and cosines of the wavelength at several fixed
**periods** (how far the wave travels before repeating). One period, two numbers:

```
angle = 2*pi * wavelength / period
tag   = [ sin(angle), cos(angle) ]
```

With period 1000 nm:

| wavelength | angle (rad) | sin | cos |
|---|---|---|---|
| 660 nm | 4.1469 | -0.8443 | -0.5358 |
| 670 nm | 4.2097 | -0.8763 | -0.4818 |
| 1650 nm | 10.3673 | -0.8090 | -0.5878 |

Nearby wavelengths (660, 670) get nearby tags. Good. But look at 1650: its tag is almost the
same as 660's. With one period, 1650 nm and 660 nm are nearly indistinguishable, because the
wave has gone round once and come back. **One period aliases; several periods do not.**

Add a long period, 10,000 nm:

| wavelength | sin | cos |
|---|---|---|
| 660 nm | +0.4029 | +0.9152 |
| 670 nm | +0.4086 | +0.9127 |
| 1650 nm | +0.8607 | +0.5090 |

Now 1650 is clearly different. The short period separates neighbours; the long one separates
regions. The design uses 16 periods spread from 10 nm to 10,000 nm, giving `16 x 2 = 32`
numbers - exactly `d`.

How similar do the full 32-number tags come out? Measured with cosine similarity (part 08,
section 3 - 1 means identical direction, 0 unrelated):

```
660 nm vs  670 nm:  0.671     neighbours, related
660 nm vs 1650 nm:  0.013     far apart, unrelated
```

## The token = value part + wavelength part

The two are **added**, element by element. With `d = 2`, the 0.3 band at 660 nm using the
period-1000 tag:

```
value part:      [ 0.25,    -0.10   ]
wavelength tag:  [-0.8443,  -0.5358 ]
                  -------   -------
token:           [-0.5943,  -0.6358 ]
```

Adding rather than placing side by side keeps the token at width `d`. It works because
training learns a value stretch and tags that occupy usefully different directions.

## Why wavelength beats position

| | position (today) | wavelength (next Indradhanu) |
|---|---|---|
| what slot 87 means | depends on the sensor | slot numbers are not used |
| the gap 1350-1450 nm | invisible | 1340 and 1460 get clearly different tags |
| a new sensor | resample onto 165 first | read its native bands directly |
| FWHM (band width) | ignored | can be encoded too (a refinement for later) |

## Common confusions

**"Is the wavelength tag learned?"** No. It is a formula of the wavelength, identical in every
run. Only the value stretch (64 parameters) is learned in this part. That is deliberate: a
learned table indexed by band number would bring back the position problem from part 31.

**"Why add instead of concatenate?"** Concatenating would double the width to 64 and every
later layer would grow with it. Adding is the standard choice and costs nothing.

**"Does the order of tokens matter?"** Not to attention. The tag carries the wavelength, so
the 202 tokens can arrive in any order - which also means EnMAP's interleaved VNIR/SWIR
overlap (bands near 900-1000 nm from both detectors) needs no special sorting.

**"What about the dropped water bands?"** They are absent: SpectralEarth sends 202 tokens,
not 224. Nothing marks the gap except the wavelengths themselves - which is enough.

## Check yourself

1. In this design, what is one token?
2. Using `w = [0.5, -1.0]`, `b = [0.1, 0.2]`, compute the value part for a band measuring
   `0.6`.
3. Using period 1000 nm, compute the tag `[sin, cos]` for 500 nm. (Angle `= 2*pi*0.5 = pi`.)
4. Why does one period alone fail to tell 660 nm from 1650 nm?
5. How many learned parameters does the tokeniser have at `d = 32`?

<details>
<summary>Answers</summary>

1. One band of one pixel, as a `d`-number vector: stretched value plus wavelength tag.
2. `0.6*0.5 + 0.1 = 0.30 + 0.10 = 0.40`; `0.6*-1.0 + 0.2 = -0.60 + 0.20 = -0.40`. So
   `[0.40, -0.40]`.
3. `sin(pi) = 0`, `cos(pi) = -1`, so `[0, -1]`.
4. 1650 is 990 nm past 660, almost exactly one full period, so the wave has nearly returned
   to the same point and the sin/cos pair nearly repeats.
5. `32 + 32 = 64` (the `Linear(1, 32)`); the tag has none.

</details>

---

**Next:** part 34 - 32 learned questions, put to those tokens, and the parameter count.
