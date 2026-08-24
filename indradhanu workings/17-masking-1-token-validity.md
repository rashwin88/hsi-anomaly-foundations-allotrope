# 17 · Masking, step 1: which tokens are real data?

> **The one thing this part teaches:** before you can decide which tokens to
> hide, you must decide which tokens contain real measurements at all.

**Source:** `TokenMasking.pixel_mask_to_token_mask` in
[`app/foundation_models/components/token_masking.py`](../app/foundation_models/components/token_masking.py)

---

## Where we are

Masking takes three steps and they are easy to confuse. Here is the map:

| Step | Question it answers | Part |
|---|---|---|
| **1** | Which tokens contain real data? | **this part** |
| 2 | Of those, which shall we hide? | 18 |
| 3 | How do we physically delete and restore them? | 19 |

---

## The mismatch this step resolves

The validity information arrives per **pixel**: a `(B, 1, H, W)` array of ones
and zeros.

Masking operates on **tokens**: a `(B, N)` array.

One token covers 16 pixels (a 4x4 block). So what happens when a block is
partly valid? Nine good pixels and seven bad? Somebody has to make a ruling.

---

## The rule

**Average-pool the pixel mask with exactly the same geometry as the patch
embedding, then threshold at 0.5.**

```python
token_fractions = F.avg_pool2d(pixel_mask.float(),
                               kernel_size=kernel_size,   # 4
                               stride=stride,             # 4
                               padding=padding)           # 0
token_valid = (token_fractions > 0.5).float()
token_mask  = token_valid.squeeze(1).flatten(1)           # (B, N)
```

Why does average-pooling give a fraction? Because the mask contains only 0s and
1s. The average of sixteen 0/1 values **is** the proportion that were 1.

```
16 valid pixels out of 16  ->  average = 1.00  ->  100% valid
 8 valid pixels out of 16  ->  average = 0.50  ->   50% valid
 0 valid pixels out of 16  ->  average = 0.00  ->    0% valid
```

Then: more than half valid, and the token counts as valid.

### The last two lines

```python
token_mask = token_valid.squeeze(1).flatten(1)
```

- `squeeze(1)` removes the size-1 channel dimension: `(B, 1, 32, 32) -> (B, 32, 32)`
- `flatten(1)` flattens everything from dimension 1 onward: `-> (B, 1024)`

Result: one 0/1 value per token, in the same row-major order the tokens
themselves use.

---

## Why "exactly the same geometry" is non-negotiable

Look at those constants again: kernel 4, stride 4, padding 0. They are identical
to stage 1's patch embedding (part 11). That is not a coincidence, it is a
requirement.

Suppose the pooling used a different window. Then a token's *validity* would be
computed from a different set of pixels than the token's *value* was computed
from. You could end up declaring a token valid on the strength of pixels it
never saw, and grading the model on a token built entirely from nodata.

To prevent drift, both the trainer and the inferencer declare the constants at
the top of the file:

```python
STAGE1_KERNEL_SIZE = 4
STAGE1_STRIDE = 4
STAGE1_PADDING = 0  # Non-overlapping: kernel=stride, no padding
```

and pass them explicitly at every call site. Three files, one set of numbers.

---

## Worked example

An 8x8 pixel validity mask. Kernel 4, stride 4, padding 0, so the token grid is
2x2 and `N = 4`.

```
             columns 0-3     columns 4-7
   row 0     1  1  1  1      0  0  0  0
   row 1     1  1  1  1      0  0  0  0
   row 2     1  1  1  1      1  1  0  0
   row 3     1  1  1  1      1  1  0  0
             ------------------------------
   row 4     1  1  1  0      1  1  1  1
   row 5     1  1  1  0      1  1  1  1
   row 6     1  1  1  0      0  0  0  0
   row 7     1  1  1  0      0  0  0  0
```

Now count each 4x4 block.

**Token 0** — top-left, rows 0-3 and columns 0-3:

```
1 1 1 1
1 1 1 1     all sixteen are 1
1 1 1 1
1 1 1 1
fraction = 16/16 = 1.000    is 1.000 > 0.5?  YES  -> valid
```

**Token 1** — top-right, rows 0-3 and columns 4-7:

```
0 0 0 0
0 0 0 0     four 1s (rows 2 and 3, first two columns)
1 1 0 0
1 1 0 0
fraction = 4/16 = 0.250     is 0.250 > 0.5?  NO   -> invalid
```

**Token 2** — bottom-left, rows 4-7 and columns 0-3:

```
1 1 1 0
1 1 1 0     three 1s per row, four rows = 12
1 1 1 0
1 1 1 0
fraction = 12/16 = 0.750    is 0.750 > 0.5?  YES  -> valid
```

**Token 3** — bottom-right, rows 4-7 and columns 4-7:

```
1 1 1 1
1 1 1 1     eight 1s
0 0 0 0
0 0 0 0
fraction = 8/16 = 0.500     is 0.500 > 0.5?  NO   -> invalid
```

Flattened in row-major order:

```
token_mask = [1, 0, 1, 0]
```

### Note token 3 carefully

Exactly half valid. The test is a **strict greater-than**:

```python
token_valid = (token_fractions > 0.5).float()
```

`0.500 > 0.5` is false, so it is rejected. **Ties go to "invalid".**

That is the conservative choice, and the right one — a token built from half
nodata is not something you want to grade the model on.

---

## What happens to invalid tokens

Here is the part people get wrong. Invalid tokens are:

- **not** deleted from the sequence,
- **not** eligible to be hidden,
- **not** graded.

They stay in the encoder's input, carrying their zero-ish values. Remember from
part 09: invalid pixels were zeroed before normalisation, so they sit at about
-1.22 in z-score space — a value the model learns to recognise as "no data
here".

So the encoder is explicitly told where the holes are, rather than being asked
to hallucinate across them.

Part 18 shows the one-line trick that guarantees they are never selected for
hiding.

---

## Where this function is called

Three places, all with the same `(4, 4, 0)` constants:

| Caller | File | Purpose |
|---|---|---|
| `_build_prediction_mask` | the trainer | before choosing hiding targets |
| `_checkerboard_keep_mask` | the inferencer | checkerboard strategy |
| `_build_random_keep_mask` | the inferencer | random strategy |

Same helper, three different masking strategies on top.

---

## Common confusions

**"Is `token_mask` the same as `keep_mask`?"**
No, and this is the most common mix-up in the whole masking area.

- `token_mask`: **is this token real data?** (from the sensor)
- `keep_mask`: **shall we show this token to the encoder?** (our choice)

An invalid token has `token_mask = 0` and `keep_mask = 1` — it is not real data,
but we do keep it in the sequence.

**"Why threshold at 0.5 rather than requiring 100% valid?"**
Requiring all sixteen pixels would throw away enormous amounts of usable data
near every cloud and swath edge. Half is a reasonable compromise, and the
erosion step (part 20) provides a second layer of protection.

**"Does this run for stages 2 to 4 as well?"**
No. Only stage 1 hides anything, so only stage 1 needs a token validity mask.

---

## Check yourself

1. Why does average-pooling a 0/1 mask give you a fraction?
2. A token's block has 10 valid pixels out of 16. Valid or invalid?
3. A token's block has exactly 8 out of 16. Valid or invalid, and why?
4. Why must the pooling use the same kernel, stride and padding as the patch
   embedding?
5. What is the difference between `token_mask` and `keep_mask`?

<details>
<summary>Answers</summary>

1. Because the values are only 0 and 1, so their average is exactly the
   proportion of 1s.
2. `10/16 = 0.625 > 0.5` — valid.
3. `8/16 = 0.500`, and the test is strict `>`, so **invalid**. Ties go to
   invalid, which is the conservative choice.
4. Otherwise a token's validity would be judged from a different set of pixels
   than the ones its value was computed from.
5. `token_mask` says whether a token contains real sensor data. `keep_mask` says
   whether we choose to show it to the encoder. Invalid tokens are kept but
   never hidden and never graded.

</details>

---

**Next:** choosing which valid tokens to hide, in
[18-masking-2-prediction-targets.md](18-masking-2-prediction-targets.md)
