# 18 · Masking, step 2: choosing what to hide

> **The one thing this part teaches:** a neat vectorised trick — add random
> noise, push invalid tokens to the back by adding 2, sort, take the first few.

**Source:** `TokenMasking.generate_prediction_mask` in
[`token_masking.py`](../app/foundation_models/components/token_masking.py)

---

## The job

**Input:**
- `token_mask` — which tokens are real data (from part 17)
- `mask_ratio` — what fraction of the real ones to hide, e.g. 0.65

**Output:** two complementary masks.

```
pred_mask : 1 = hide this token, and grade the model on it
keep_mask : 1 = show this token to the encoder
```

with the invariant that always holds:

```
keep_mask = 1 - pred_mask
```

There is no third state. Every token is either hidden-and-graded or
shown-and-ignored.

**The hard requirement:** invalid tokens must never be selected. Grading the
model on a region that contains no data would be meaningless — and worse, it
would inject nonsense into the training signal.

---

## Why not just write a loop?

You could. For each batch item, find the valid indices, shuffle them, take the
first 65%. Three lines of Python.

But that runs on the CPU, one batch item at a time, with different-length lists
per item. On a GPU processing 128 patches at once, that is a disaster — you
would stall the GPU on Python loops for every single batch, forever.

So the whole thing is done with **array operations only**: no loops, no
branching, identical work for every batch item. That is why the code looks
indirect. It is optimised for the machine, not for the reader.

---

## The trick

```python
noise = torch.rand(B, N, device=device)          # random, in [0, 1)
noise = noise + (1 - token_mask) * 2             # invalid ones land in [2, 3)

num_valid   = token_mask.sum(dim=1)
num_to_mask = (num_valid * mask_ratio).long()

sorted_indices   = noise.argsort(dim=1)          # ascending
positions        = torch.arange(N).unsqueeze(0).float().expand(B, -1)
pred_mask_sorted = (positions < num_to_mask.unsqueeze(1)).float()

pred_mask = torch.zeros(B, N, device=device)
pred_mask.scatter_(1, sorted_indices, pred_mask_sorted)
keep_mask = 1.0 - pred_mask
```

In plain English:

1. Give every token a random priority number between 0 and 1.
2. **Add 2 to every invalid token's number**, so invalid ones are 2 to 3.
3. Sort everyone by priority, smallest first.
4. Take however many you need from the front of the queue.

Because valid numbers are always below 1 and invalid ones always above 2, **an
invalid token can never sort ahead of a valid one**.

> That single `+ 2` is the entire guarantee. No `if` statement, no boolean
> indexing, no variable-length arrays. Every batch item does identical work.

---

## Worked example, entirely by hand

Six tokens, `mask_ratio = 0.5`.

```
token_mask = [1, 1, 0, 1, 1, 0]
             ^t0 t1 t2 t3 t4 t5

Four valid (0, 1, 3, 4). Two invalid (2 and 5).
```

### Step 1 — random priorities

Say the random generator produced:

```
noise = [0.30, 0.70, 0.10, 0.50, 0.20, 0.80]
```

Note token 2 got 0.10 — the lowest of all. Without protection it would be first
in the queue, and it is invalid.

### Step 2 — add the bias

Add `2` wherever `token_mask` is 0, that is at positions 2 and 5:

```
before: [0.30, 0.70, 0.10, 0.50, 0.20, 0.80]
add:    [   0,    0,    2,    0,    0,    2]
        ---------------------------------------
after:  [0.30, 0.70, 2.10, 0.50, 0.20, 2.80]
                     ^^^^              ^^^^
                     now unreachable at the back
```

Token 2 went from most-likely-to-be-picked to last in the queue.

### Step 3 — how many to hide

```
num_valid   = 1+1+0+1+1+0 = 4
num_to_mask = int(4 * 0.5) = 2
```

Note it is a fraction of the **valid** tokens, not of all tokens.

### Step 4 — sort, and record where each value came from

Sort `[0.30, 0.70, 2.10, 0.50, 0.20, 2.80]` ascending:

```
value:  0.20   0.30   0.50   0.70   2.10   2.80
came
from:      4      0      3      1      2      5

sorted_indices = [4, 0, 3, 1, 2, 5]
```

`argsort` returns exactly that list of original positions.

### Step 5 — flag the first two places in the queue

```
positions        = [0, 1, 2, 3, 4, 5]
num_to_mask      = 2
pred_mask_sorted = positions < 2  ->  [1, 1, 0, 0, 0, 0]
```

The first two entries **in sorted order** are the targets.

### Step 6 — scatter back to original positions

Right now the flags are in sorted order. We need them in the original order.
`scatter_` performs `pred_mask[sorted_indices[i]] = pred_mask_sorted[i]`:

```
i=0:  sorted_indices[0] = 4,  value 1  ->  pred_mask[4] = 1
i=1:  sorted_indices[1] = 0,  value 1  ->  pred_mask[0] = 1
i=2:  sorted_indices[2] = 3,  value 0  ->  pred_mask[3] = 0
i=3:  sorted_indices[3] = 1,  value 0  ->  pred_mask[1] = 0
i=4:  sorted_indices[4] = 2,  value 0  ->  pred_mask[2] = 0
i=5:  sorted_indices[5] = 5,  value 0  ->  pred_mask[5] = 0
```

### The result

```
pred_mask = [1, 0, 0, 0, 1, 0]      hide tokens 0 and 4, grade the model there
keep_mask = [0, 1, 1, 1, 0, 1]      show tokens 1, 2, 3 and 5 to the encoder
```

Check it against the requirements:

- Exactly 2 of the 4 valid tokens hidden. Correct for `mask_ratio = 0.5`.
- Neither invalid token (2 or 5) was touched. Correct.
- Invalid tokens have `keep_mask = 1`, so they stay in the sequence. Correct.

**Do this again with different random numbers.** Give yourself
`noise = [0.9, 0.1, 0.05, 0.4, 0.6, 0.02]` and work it through. It takes two
minutes and it is the fastest way to make the trick stick.

---

## Choosing the mask ratio

| Configuration | `mask_ratio` |
|---|---|
| v0.1.0 (`hyperspectral_segformer_exp_1.json`) | 0.50 |
| v0.2.0 (`hyperspectral_segformer_exp_2.json`) | **0.65** |

The trade-off runs in both directions:

**Too low** — say 0.10. Nine tokens out of ten are visible, so filling in the
tenth is nearly trivial: just interpolate from the neighbours. The model learns
smoothing rather than understanding. Worse, at inference it will reconstruct
*anomalies* rather well too, since it can lean on their immediate surroundings.
Anomalies stop standing out.

**Too high** — say 0.90. Almost nothing is visible. There is not enough evidence
to reason from, so the best available strategy is to output a blurry average of
everything. Then *every* pixel has a large residual and nothing stands out
either.

The masked-autoencoder literature for natural photographs settled around 0.75.
This project settled on 0.65 for v0.2.0.

> **A useful way to think about it.** You are setting the difficulty of an exam.
> Too easy and everyone passes, so the exam tells you nothing. Too hard and
> everyone fails, so the exam tells you nothing. You want the difficulty where
> the score is most informative.

---

## Inference does not use this function

At inference we need a score at **every** pixel, so random selection is no good
— some tokens would never be hidden and would never get a score.

Instead the inferencer builds an exact complementary split: either a
checkerboard or a random half plus its exact opposite. Part 29.

The invalid-token protection is still there, though, expressed slightly
differently:

```python
pred_mask = token_validity * (1.0 - checker)
```

Multiplying by `token_validity` means only valid tokens can ever become targets.
Same guarantee, different mechanism.

---

## Common confusions

**"Is `mask_ratio` a fraction of all tokens or of the valid ones?"**
Of the valid ones. `num_to_mask = num_valid * mask_ratio`. In a patch that is
half nodata, fewer tokens are hidden in absolute terms.

**"Why add exactly 2? Why not 1 or 100?"**
Valid noise is in `[0, 1)`. Adding 1 would put invalid noise in `[1, 2)` — still
strictly above every valid value, so 1 would work. 2 is simply an
unambiguous margin. Any value of at least 1 is correct.

**"Does the same mask get used every batch?"**
No. `torch.rand` is called fresh every time, so every batch of every epoch sees a
different hiding pattern. Over a training run every token gets hidden many times
in many different combinations.

**"What if `num_to_mask` comes out as 0?"**
Then nothing is hidden, `loss_mask` is empty, and `compute_loss` returns zero
with `num_kept = 0`, which the training loop skips. There is an explicit guard
for this.

---

## Check yourself

1. What exactly does the `+ 2` guarantee, and why is it needed?
2. `token_mask = [1, 1, 1, 0]` with `mask_ratio = 0.65`. How many tokens are
   hidden?
3. Given `sorted_indices = [2, 0, 3, 1]` and `pred_mask_sorted = [1, 0, 0, 0]`,
   what is the final `pred_mask`?
4. What goes wrong if `mask_ratio` is 0.1? What about 0.9?
5. Why is this written with sorting instead of a simple loop?

<details>
<summary>Answers</summary>

1. It pushes every invalid token's priority above every valid token's, so
   invalid tokens can never be selected as hiding targets — without any branching
   or variable-length arrays.
2. `num_valid = 3`, and `int(3 * 0.65) = int(1.95) = 1`. One token is hidden.
3. `scatter_` writes value 1 at position `sorted_indices[0] = 2`, and 0
   everywhere else. So `pred_mask = [0, 0, 1, 0]`.
4. At 0.1 the task is too easy — the model learns interpolation and reconstructs
   anomalies too well, so nothing stands out. At 0.9 there is too little evidence
   — it outputs blurry averages and everything looks anomalous.
5. Because it must run on the GPU for a whole batch at once, with identical work
   per batch item. Python loops and variable-length index lists would stall the
   GPU on every batch.

</details>

---

**Next:** physically deleting and restoring the tokens, in
[19-masking-3-remove-and-restore.md](19-masking-3-remove-and-restore.md)
