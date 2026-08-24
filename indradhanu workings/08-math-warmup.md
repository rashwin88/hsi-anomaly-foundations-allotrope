# 08 · Math warm-up: five things, worked by hand

> **The one thing this part teaches:** every piece of maths in this model is one
> of five simple operations. None of them is harder than school arithmetic.

Do each worked example on paper. Reading them is not the same as doing them —
and each one takes under a minute.

---

## 1. Dot product

### What it is

Multiply two lists of numbers position by position, then add up the results.

```
a . b  =  a1*b1 + a2*b2 + ... + an*bn
```

The result is a **single number**, not a list. (That is why it is sometimes
called the "scalar product".)

### Worked

```
a = [0.1, 0.2, 0.3]
b = [0.3, 0.2, 0.1]
```

Step by step:

```
position 1:  0.1 * 0.3 = 0.03
position 2:  0.2 * 0.2 = 0.04
position 3:  0.3 * 0.1 = 0.03
                        ------
             add them:   0.10
```

So `a . b = 0.10`.

### What it means intuitively

The dot product is **big** when the two lists are both large *and* their large
values line up in the same positions. It is **small** when they disagree about
where the big values are.

### Where it appears

In [`sam_loss.py`](../app/foundation_models/components/sam_loss.py):

```python
dot = (xh_m * x_m).sum(dim=1, keepdim=True)
```

`dim=1` is the spectral axis, so this computes a dot product between two
165-band spectra — simultaneously, at every pixel in the batch.

---

## 2. Norm (the length of a vector)

### What it is

Square every entry, add them up, take the square root.

```
||a||  =  sqrt( a1^2 + a2^2 + ... + an^2 )
```

The double bars mean "length of". This is just Pythagoras extended past two
dimensions.

### Worked

```
a = [0.1, 0.2, 0.3]

squares:      0.1^2 = 0.01
              0.2^2 = 0.04
              0.3^2 = 0.09
                     ------
sum:                  0.14
square root:  sqrt(0.14) = 0.37417
```

And for `b = [0.3, 0.2, 0.1]` you get the same 0.37417 — same numbers, different
order, and order does not affect length.

### What it means for a spectrum

For a pixel's spectrum, the norm is essentially **overall brightness**. A bright
pixel has a long vector; a dark pixel has a short one, even if the shape of the
curve is identical.

---

## 3. The angle between two vectors

This is the one that matters most. Take your time.

### The formula

```
cos(theta)  =  (a . b) / ( ||a|| * ||b|| )

theta       =  arccos( cos(theta) )
```

In words: take the dot product, then divide out both lengths. What is left is
purely about **direction**, because the lengths have been cancelled.

`arccos` (also written `acos`, or cos-inverse on a calculator) is the operation
that turns a cosine value back into an angle.

### Worked example A — two different-shaped spectra

```
a = [0.1, 0.2, 0.3]
b = [0.3, 0.2, 0.1]
```

We already have every piece:

```
a . b   = 0.10           (from section 1)
||a||   = 0.37417        (from section 2)
||b||   = 0.37417

||a|| * ||b|| = 0.37417 * 0.37417 = 0.14

cos(theta) = 0.10 / 0.14 = 0.7143

theta = arccos(0.7143) = 0.7754 radians
```

Converting to degrees (multiply by 180, divide by pi ≈ 3.14159):

```
0.7754 * 180 / 3.14159 = 44.4 degrees
```

These two spectra point in noticeably different directions.

### Worked example B — the crucial one

Now take:

```
a = [0.1, 0.2, 0.3]
b = [0.2, 0.4, 0.6]        <- exactly twice a, every entry
```

```
a . b = 0.1*0.2 + 0.2*0.4 + 0.3*0.6
      = 0.02 + 0.08 + 0.18
      = 0.28

||a|| = sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14) = 0.37417
||b|| = sqrt(0.04 + 0.16 + 0.36) = sqrt(0.56) = 0.74833

||a|| * ||b|| = 0.37417 * 0.74833 = 0.28

cos(theta) = 0.28 / 0.28 = 1.0

theta = arccos(1.0) = 0 radians = 0 degrees
```

## **The angle is exactly zero, even though every single number is different.**

Sit with that for a moment. `b` is twice as bright as `a` in every band. The
angle does not notice at all.

Angle measures **shape**. It is completely blind to **brightness**.

This is not a defect. It is the single reason the model needs *two* different
error measurements (part 25). One of them notices brightness; the other notices
shape; you need both.

### Ranges to remember

| Angle | Meaning |
|---|---|
| 0 | identical shape (any brightness) |
| 0.01 – 0.10 radians | 0.6 to 5.7 degrees — a good reconstruction here |
| pi/2 ≈ 1.571 | at right angles, completely unrelated |
| pi ≈ 3.142 | exactly opposite |

### Radians versus degrees

Radians are the units the maths uses; degrees are what humans read. Conversion:

```
degrees = radians * 180 / pi
```

The training code literally does this when it logs:

```python
f"SAM: {avg_sam:.6f} rad ({avg_sam * 180 / 3.14159:.2f} deg)"
```

---

## 4. Matrix multiply, and the 1x1 convolution

### Matrix times vector

A matrix `W` with `D` rows and `C` columns, times a vector `x` of length `C`,
gives a vector of length `D`:

```
y_d  =  sum over c of  W[d, c] * x[c]        for each row d
```

In words: **each output number is a weighted mixture of all the input numbers**,
and each output has its own set of weights (one row of the matrix).

### Why this matters

A "1x1 convolution" — a phrase that appears constantly in this codebase —
**is exactly this, applied independently at every pixel.**

```python
nn.Conv2d(in_channels=165, out_channels=32, kernel_size=1)
```

means: at each of the H x W pixels, take that pixel's 165 numbers, multiply by a
32-by-165 matrix, add a 32-long bias, and write back 32 numbers.

Crucially, **there is no spatial mixing at all**. Pixel (3,7) never looks at
pixel (3,8). The word "convolution" makes people expect neighbourhood mixing;
with a 1x1 kernel there is none.

### Worked, tiny

`C = 3` inputs, `D = 2` outputs, one pixel.

```
W = [[1.0, 0.0, 0.0],        b = [0.0,
     [0.0, 0.5, 0.5]]              0.1]

x = [0.10, 0.20, 0.30]
```

Row 0 of `W` is `[1.0, 0.0, 0.0]`:

```
y0 = 1.0*0.10 + 0.0*0.20 + 0.0*0.30 + 0.0
   = 0.10 + 0 + 0 + 0
   = 0.10
```

Row 1 of `W` is `[0.0, 0.5, 0.5]`:

```
y1 = 0.0*0.10 + 0.5*0.20 + 0.5*0.30 + 0.1
   = 0 + 0.10 + 0.15 + 0.1
   = 0.35
```

```
y = [0.10, 0.35]
```

Three numbers became two. Output 0 just copied input 0; output 1 averaged inputs
1 and 2 and added a small offset. In the real model these weights are learned,
not chosen — but the arithmetic is identical.

### Counting parameters

A matrix of `D` rows and `C` columns has `D * C` entries, plus `D` bias values:

```
parameters = D * C + D
```

Our tiny example: `2 * 3 + 2 = 8`.

The real compressor: `32 * 165 + 32 = 5280 + 32 = 5312`.

Now go and look at the `torchinfo` table at the bottom of
[`research/model_break_down/05_hyperspectral_segformer_mae.md`](../research/model_break_down/05_hyperspectral_segformer_mae.md).
It says:

```
└─Conv2d: 2-1     [1, 165, 128, 128]    [1, 32, 128, 128]    5,312
```

**5,312.** You just predicted a real number from the real model with a formula
you can do in your head. That is the level of understanding this course is
aiming for.

---

## 5. Softmax

### What it is

Takes any list of numbers and turns it into a set of proportions that are all
positive and add up to exactly 1 — emphasising the larger ones.

```
softmax(s)_i  =  exp(s_i) / sum over j of exp(s_j)
```

`exp(x)` means `e` raised to the power `x`, where `e` ≈ 2.71828. The important
property is that `exp` is always positive and grows fast.

### Worked

```
s = [2, 1, 0]
```

Step 1, exponentiate each:

```
exp(2) = 7.389
exp(1) = 2.718
exp(0) = 1.000
```

Step 2, add them:

```
7.389 + 2.718 + 1.000 = 11.107
```

Step 3, divide each by that total:

```
7.389 / 11.107 = 0.665
2.718 / 11.107 = 0.245
1.000 / 11.107 = 0.090
```

Check: `0.665 + 0.245 + 0.090 = 1.000`. Good.

Notice what happened: the input gaps were 1 and 1 (2, 1, 0 — evenly spaced), but
the output gaps are much bigger. The largest input took 66% of the total. That
amplification is exactly what softmax is for.

### Where it appears

Attention (part 12) uses it to convert raw "how relevant is this?" scores into
weights that sum to 1, so the result is a proper weighted average.

---

## Bonus: z-score normalisation

Used constantly, and simple:

```
z = (x - mean) / std              forwards
x = z * std + mean                backwards
```

`std` is standard deviation — a measure of typical spread.

### Worked

Band 0 has an average of 0.0430 and, say, a spread of 0.0350. A pixel reads
0.0780:

```
z = (0.0780 - 0.0430) / 0.0350
  = 0.0350 / 0.0350
  = 1.0
```

That pixel is exactly one standard deviation above the average for its band.

And reversing it:

```
x = 1.0 * 0.0350 + 0.0430 = 0.0780
```

Back where we started. Part 09 is entirely about this operation.

---

## Common confusions

**"Is the dot product the same as multiplying two vectors?"**
No. Multiplying entry-by-entry gives you another *vector*. The dot product goes
one step further and adds those products into a single *number*.

**"Why does the angle ignore brightness?"**
Because dividing by both lengths cancels them out. Whatever scale the vectors
were on, that scale disappears in the division.

**"Is a 1x1 convolution really a convolution?"**
Technically yes, with a window of size one. Practically it does no neighbourhood
mixing at all — it is a per-pixel matrix multiply. Do not let the name mislead
you.

---

## Check yourself

1. Compute the dot product of `[1, 2]` and `[3, 4]`.
2. Compute the norm of `[3, 4]`.
3. What is the angle between `[1, 1]` and `[2, 2]`? Why, without calculating?
4. How many parameters does `Conv2d(32, 165, kernel_size=1)` have?
5. Compute softmax of `[0, 0]`.

<details>
<summary>Answers</summary>

1. `1*3 + 2*4 = 3 + 8 = 11`.
2. `sqrt(9 + 16) = sqrt(25) = 5`.
3. Zero. The second vector is exactly twice the first, so they point in the same
   direction; angle ignores length.
4. `32 * 165 + 165 = 5280 + 165 = 5445`. (This is the real decompressor —
   compare it against the `torchinfo` table.)
5. `exp(0) = 1` for both; total 2; so `[0.5, 0.5]`. Equal inputs give equal
   shares.

</details>

---

**Next:** the first real module of the model, in
[09-normalisation.md](09-normalisation.md)
