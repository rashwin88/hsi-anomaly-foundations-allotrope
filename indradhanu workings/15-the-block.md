# 15 · The SegFormer block

> **The one thing this part teaches:** a block is attention and Mix-FFN, each
> wrapped in "normalise first, add the result back". Two lines of real code.

**Source:**
[`app/foundation_models/components/segformer_block.py`](../app/foundation_models/components/segformer_block.py)
— 107 lines, and now you can read all of it.

---

## The whole block, in two lines

```
x = x + Dropout( ESA(    LayerNorm(x), H, W ) )      the attention sublayer
x = x + Dropout( MixFFN( LayerNorm(x), H, W ) )      the feed-forward sublayer
```

That is the entire block. Everything below explains why each piece sits exactly
where it does.

Reading it aloud: *"Normalise, attend, maybe drop some of it, add it to what we
had. Then normalise, think, maybe drop some, add it to what we had."*

---

## Piece 1: the residual connection

Notice that we write `x = x + f(x)`, not `x = f(x)`.

That little `x +` is called a **residual connection** (or skip connection), and
it is one of the most important ideas in modern deep learning. Two ways to
understand why it matters:

### The gradient view

Training works by sending an error signal backwards through the network. Every
layer it passes through multiplies it by something.

Stack 20 layers, and the signal has been multiplied 20 times. If those factors
average slightly below 1, the signal shrinks toward nothing (**vanishing
gradients**) and early layers stop learning. Slightly above 1, and it explodes.

A residual connection creates a path from input to output that **skips the
layer entirely**. The error signal can travel along that path unchanged. It is
a motorway alongside the winding road.

### The "learn a correction" view

Often more intuitive. With `x = f(x)`, the layer must reproduce everything
useful about `x` *and* add its improvement. With `x = x + f(x)`, `x` is already
there — the layer only has to produce the **correction**.

And if the best thing to do is nothing, `f(x) = 0` is easy to learn. Without
residuals, "do nothing" means learning to be exactly the identity function,
which is surprisingly hard.

---

## Piece 2: LayerNorm

For each token independently, across its `C` features:

```
mu    = the mean of this token's features
sigma = the standard deviation of this token's features
y     = gamma * (x - mu) / sigma + beta
```

`gamma` and `beta` are learned — one scale and one shift per feature — which is
why `LayerNorm(C)` has exactly `2C` parameters.

### Worked example

Token `x = [1.0, 2.0, 3.0, 4.0]`, with `gamma = 1` and `beta = 0`.

**Mean:**

```
(1.0 + 2.0 + 3.0 + 4.0) / 4 = 10.0 / 4 = 2.5
```

**Variance** (average of squared distances from the mean):

```
(1.0 - 2.5)^2 = (-1.5)^2 = 2.25
(2.0 - 2.5)^2 = (-0.5)^2 = 0.25
(3.0 - 2.5)^2 = ( 0.5)^2 = 0.25
(4.0 - 2.5)^2 = ( 1.5)^2 = 2.25
                          ------
                   sum =    5.00
             sum / 4  =     1.25
```

**Standard deviation:**

```
sqrt(1.25) = 1.118
```

**Normalise each entry:**

```
(1.0 - 2.5) / 1.118 = -1.342
(2.0 - 2.5) / 1.118 = -0.447
(3.0 - 2.5) / 1.118 =  0.447
(4.0 - 2.5) / 1.118 =  1.342
```

```
y = [-1.342, -0.447, 0.447, 1.342]
```

Same pattern, rescaled to be centred on zero with a spread of one.

### LayerNorm versus BatchNorm — get this straight

| | LayerNorm | BatchNorm |
|---|---|---|
| Normalises across | this token's own features | one channel, across the whole batch |
| Depends on other examples? | **no** | **yes** |
| Behaves differently in train vs eval? | no | yes (running statistics) |
| Right for sequences? | yes | no |

BatchNorm makes a token's output depend on which other patches happened to share
its batch — which is strange for a sequence model, and makes small batches
unstable. LayerNorm keeps every token self-contained.

(This model does use one BatchNorm — in the spectral compressor, part 10, where
it operates on image channels and is entirely appropriate.)

---

## Piece 3: pre-norm, not post-norm

Where you put the LayerNorm matters more than you would expect.

| | post-norm (2017 original) | **pre-norm (used here)** |
|---|---|---|
| formula | `x = LayerNorm(x + f(x))` | `x = x + f(LayerNorm(x))` |
| the residual path | passes through a norm | **untouched** |
| deep training | needs careful learning-rate warmup | stable |

The distinction: in post-norm, the "motorway" from part 1 above runs through a
toll booth every block. The normalisation rescales the signal, so the clean
gradient path is no longer clean.

In pre-norm, the normalisation is *inside* the branch. The residual path is
pristine all the way from input to output.

Everything modern — GPT-2, ViT, SegFormer, this code — uses pre-norm. It is also
why this model trains fine without warmup, which turns out to be relevant
(part 28 has a surprise about warmup).

---

## Piece 4: dropout, and where it sits

**What dropout does.** During training, randomly set a fraction of the values to
zero, and scale the survivors up so the average is unchanged. At evaluation
time, do nothing at all.

**Why.** It stops the network leaning too hard on any single feature. If a
feature might vanish at any moment, the network is forced to spread its bets —
which usually generalises better.

**Where it sits here:**

```
x = x + Dropout(f(LayerNorm(x)))
        ^^^^^^^ only the sublayer's contribution
```

The dropout is **inside** the residual branch, so only the *new contribution*
can be dropped. The information already travelling along the skip connection is
never touched.

That is deliberate. Dropping the residual itself would destroy accumulated
information at random, which is not regularisation, it is damage.

### The dropout rates here

| Version | `drop_rate` |
|---|---|
| v0.1.0 | 0.3 |
| v0.2.0 (current) | **0.4** |

0.4 is aggressive — nearly half of every sublayer's output is discarded during
training. It was raised specifically to fight overfitting, which the
configuration file lists as a diagnosis to watch for:

```json
"train_loss_drops_but_val_doesnt": "Overfitting — increase dropout or reduce model size"
```

At inference it is forced off in two independent ways, belt and braces:

```python
drop_rate=0.0,  # No dropout at inference
```

and

```python
self.model.eval()
```

---

## How many blocks

`num_blocks = [2, 2, 2, 2]` — two blocks at every stage.

Within a stage, all blocks share the same `embed_dim`, `num_heads` and
`reduction_ratio`. Resolution and channel width change **only** at the patch
embedding between stages, never inside one.

---

## Verify the parameter count yourself

This is the moment the last three parts pay off. A block is: ESA + Mix-FFN + two
LayerNorms.

Stage 1:

```
ESA           (part 13)                     69,856
Mix-FFN       (part 14)                      9,632
2 LayerNorms  2 * (2 * 32)                     128
                                          --------
              one block                     79,616
              x 2 blocks per stage         159,232
```

Now open `research/model_break_down/05_hyperspectral_segformer_mae.md`, find the
`torchinfo` table, and look at stage 1's block list:

```
│    │    └─ModuleList: 3-2      --      --      159,232      True
```

**159,232.** Exact.

You just predicted a number from a real trained model by adding up components
you derived by hand in three separate parts. If that matches, you have
understood attention, Mix-FFN and the block. That is not a small thing.

---

## Common confusions

**"Why two LayerNorms per block?"**
One before attention, one before Mix-FFN. Each sublayer gets freshly normalised
input.

**"Is the `+` really just addition?"**
Yes. Element-by-element addition of two tensors of identical shape. Nothing
clever.

**"Would more blocks per stage be better?"**
Deeper is usually more capable and always slower. `[2, 2, 2, 2]` is the standard
SegFormer-B0 setting, chosen for a good size-to-quality ratio. Nobody has
retuned it here.

**"Does dropout happen at inference?"**
No. Both `drop_rate=0.0` and `model.eval()` ensure it does not. If you ever see
run-to-run variation at inference, dropout is not the cause — check the masking
strategy instead (part 29).

---

## Check yourself

1. Write the two-line block formula from memory.
2. Give both explanations for why residual connections help.
3. LayerNorm the token `[2.0, 4.0]` with gamma=1, beta=0.
4. Why is the norm *before* the sublayer and the dropout *inside* the branch?
5. Reconstruct stage 1's block parameter count from its three components.

<details>
<summary>Answers</summary>

1. `x = x + Dropout(ESA(LayerNorm(x)))`, then
   `x = x + Dropout(MixFFN(LayerNorm(x)))`.
2. Gradient view: they create an unobstructed path for the error signal, so it
   neither vanishes nor explodes through depth. Learning view: the sublayer only
   has to learn a correction, and "do nothing" becomes easy.
3. Mean 3.0; variance `((2-3)^2 + (4-3)^2)/2 = (1+1)/2 = 1`; std 1. Result
   `[-1.0, 1.0]`.
4. Pre-norm keeps the residual path free of any normalisation, so gradients flow
   cleanly. Dropout inside the branch means only the new contribution is
   dropped; the skip connection's information is preserved.
5. `69,856 (ESA) + 9,632 (Mix-FFN) + 128 (two LayerNorms) = 79,616` per block,
   times 2 blocks = **159,232**.

</details>

---

**Next:** stacking blocks into the four-stage encoder, in
[16-the-encoder.md](16-the-encoder.md)
