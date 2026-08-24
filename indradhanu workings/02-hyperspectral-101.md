# 02 · Hyperspectral 101

> **The one thing this part teaches:** a hyperspectral image gives every pixel a
> *curve* instead of a colour, and that curve identifies what the pixel is made
> of.

---

## Start from a photo you understand

Your phone camera records **three numbers** per pixel: how much red, how much
green, how much blue.

A 1000 x 1000 phone photo is therefore three grids of numbers stacked on top of
each other. Programmers write that shape as:

```
(3, 1000, 1000)
 ^   ^     ^
 |   |     +-- columns (width)
 |   +-------- rows (height)
 +------------ the three colour layers
```

Each of those three layers, on its own, is a grey picture. Red is a grey picture
of "how red is it here". The colour you see is your brain combining the three.

---

## Now add more colours. Many more.

A **hyperspectral** sensor does the same thing, but instead of three broad
colour buckets it records a couple of hundred very narrow slivers of colour —
including slivers your eye cannot see, out into the infrared.

The satellites this project uses (PRISMA, EnMAP, AVIRIS-NG) each record 200 or
more of these slivers.

Each sliver is called a **band**. So the picture becomes:

```
(200-ish, 1000, 1000)
```

That stack of grids is called a **cube**. It is just a photograph with 200
layers instead of 3.

```
          200 layers, one per wavelength
        +---------------------------------+
        |  /                             /|
        | /   every layer is a full      / |    1000 rows
        |/    grey picture of the ground /  |
        +---------------------------------+
                    1000 columns
```

> **Common confusion.** "Cube" makes people imagine something 3-D on the ground,
> like a volume of air. It is not. All 200 layers are pictures of the *same flat
> ground*. The third dimension is **colour**, not depth.

---

## The spectrum — the reason all this exists

Pick one pixel. Say row 40, column 90.

Read its value in layer 1, then layer 2, then layer 3, all the way to layer 165.
You now have 165 numbers. Plot them, with wavelength along the bottom:

```
value
 0.4 |                        ___
     |                  _____/   \___
 0.2 |             ____/             \_____
     |        ____/                        \___
 0.0 |____ __/
     +--------------------------------------------  wavelength (nanometres)
     460         1000         1600          2450
```

That curve is called the pixel's **spectrum**.

And here is the payoff: **a spectrum is a material fingerprint.**

- Healthy vegetation has a famously sharp jump around 700 nm. Botanists call it
  the "red edge". Nothing else does that.
- Clay minerals have a distinctive dip near 2200 nm, where their chemistry
  absorbs light.
- Water absorbs almost everything past about 1200 nm, so it goes dark there.

Two materials can look **identical** in an ordinary RGB photo — same shade of
grey-green — and have completely different spectra.

That is why we bother with 165 numbers per pixel instead of 3. And it is the
reason the model in this course is named after a rainbow.

---

## What the numbers mean: reflectance

The values in the cube are **surface reflectance**: the fraction of the sunlight
that hit the ground and bounced back, at that particular wavelength.

Because it is a fraction, it has **no units** and normally sits between 0 and 1.

- 0.0 would mean "absorbed everything" (perfectly black).
- 1.0 would mean "reflected everything" (a perfect mirror).

Real ground is neither. Real scenes mostly sit between **0.02 and 0.30** —
that is, most surfaces bounce back between 2% and 30% of the light.

You can check this claim yourself right now. Open
[`app/constants/hyperspectral.json`](../app/constants/hyperspectral.json). It
contains the average value of every band across the whole training corpus. The
first few:

```json
"mean": [
  0.04301138692474853,
  0.043005908140582944,
  0.04554786179838881,
  ...
```

About 0.043. Exactly in the range described. Good — the documentation and the
data agree.

> **Why this matters later.** In part 09 the model divides by numbers of roughly
> this size. If someone feeds it data on a totally different scale, everything
> downstream is nonsense. Which brings us to...

### Not every sensor plays fair

One satellite in this system, HotSat-1, does **not** ship reflectance. It ships
raw sensor counts, called "digital numbers" or DN, sitting around 5000 with a
spread of about 400.

Feed a 5000-ish number into a model that expects 0.04-ish numbers and you get
garbage. There is a specific mechanism to handle this (`PixelStatsOverride`,
part 09), and it exists purely because of this mismatch.

---

## Thermal images, for contrast

Two of the satellites here (Landsat-9 and HotSat-1) are **thermal** sensors. They
record one thing only: how hot the ground is.

So a thermal cube is:

```
(1, 1000, 1000)
```

One layer. That is it.

This matters because Indradhanu has a **sibling** called Chakshu that handles
thermal data. The two share almost all their code. Everywhere this course says
"165 bands", Chakshu says "1 band".

If you open
[`app/foundation_models/components/`](../app/foundation_models/components/) you
will literally see the pair:

```
seg_former_mae.py                  <- Chakshu, thermal, 1 band
hyperspectral_seg_former_mae.py    <- Indradhanu, 165 bands
```

Reading the thermal one first is often easier, because it has one fewer thing
going on.

---

## Validity: not every pixel is real data

Satellite data is messy in ways ordinary photos are not:

- The sensor sweeps a strip, so the edges of a scene are ragged and partly
  empty.
- Individual detectors fail and produce dead lines.
- Clouds block the ground entirely.

So every cube travels with a companion of the **same shape**, called the
**validity cube**:

```
1  =  this is a real measurement
0  =  this is not
```

### A convenient simplification

For hyperspectral data in this repo, after the preprocessing pipeline has
finished, validity is **all-or-nothing per pixel**. A pixel is either good in
all 165 bands or bad in all 165 bands. There is no "good in band 3, bad in
band 90".

That means you can read band 0 alone and know the answer for every band. Which
is exactly what the training code does:

```python
validity = batch["validity_cube.npy"].to(self.device)
# Use band 0 as spatial validity proxy: (B, C, H, W) -> (B, 1, H, W)
return validity[:, 0:1, :, :].float()
```

(from
[`hyperspectral_segformer_mae_trainer.py`](../app/foundation_models/trainers/hyperspectral_segformer_mae_trainer.py))

Reading one band instead of 165 is 165 times less memory traffic, on an
operation that happens for every batch of every epoch. Small change, real saving.

---

## Vocabulary — get these straight now

You will trip over these for weeks if you do not pin them down today.

| Word | What it is | Shape |
|---|---|---|
| **band** | one wavelength layer; a grey picture | `(H, W)` |
| **spectrum** | one pixel's values across all bands; the fingerprint curve | `(C,)` |
| **cube** | the whole stack | `(C, H, W)` |
| **patch** | a small square cut out of the cube | `(165, 128, 128)` |
| **scene** | one complete satellite acquisition | `(165, ~1000, ~1000)` |
| **validity** | the 0/1 companion saying which pixels are real | `(C, H, W)` |

The two easiest to confuse are **band** and **spectrum**. Remember:

- A **band** is a horizontal slice: one colour, all pixels.
- A **spectrum** is a vertical skewer: one pixel, all colours.

---

## Common confusions

**"Is a hyperspectral image just a very colourful photo?"**
No. Most of its bands are invisible to the eye (infrared). You cannot display a
165-band cube directly; you have to pick three bands to show as red, green and
blue, and that display throws away 162 bands of information.

**"Do more bands always mean better?"**
Not automatically. Adjacent bands 10 nm apart are almost identical, so a lot of
those 165 numbers are near-duplicates. Part 10 is entirely about exploiting that
redundancy.

**"If validity is all-or-nothing, why is the validity cube 165 layers deep?"**
Because the type is shared with earlier pipeline stages where it genuinely
varied per band. By the time the model sees it, the layers agree.

---

## Check yourself

1. What shape is a hyperspectral cube, and what does each of the three numbers
   mean?
2. What is the difference between a band and a spectrum?
3. What range do reflectance values normally fall in, and what does 0.30 mean
   physically?
4. Why can the training code get away with reading only band 0 of the validity
   cube?
5. What is Chakshu, and how does its input differ?

<details>
<summary>Answers</summary>

1. `(C, H, W)`: number of wavelength bands, then rows, then columns.
2. A band is one wavelength across all pixels (a grey picture). A spectrum is
   one pixel across all wavelengths (a curve). Horizontal slice versus vertical
   skewer.
3. Roughly 0.02 to 0.30. A value of 0.30 means the ground bounced back 30% of
   the sunlight that hit it, at that wavelength.
4. Because after preprocessing, a pixel is either valid in all 165 bands or
   invalid in all 165, so band 0 tells you everything.
5. Chakshu is the thermal sibling model, sharing nearly all the same code, with
   1 input band (temperature) instead of 165.

</details>

---

**Next:** where the number 165 comes from, in
[03-the-165-band-grid.md](03-the-165-band-grid.md)
