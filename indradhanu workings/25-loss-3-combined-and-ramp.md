# 25 · Loss, part 3: combining the two, gradually

> **The one thing this part teaches:** the two loss terms are added with a weight
> that starts at zero and ramps up — and after the ramp, the shape term
> dominates.

---

## The combined objective

```
L_total(t)  =  L1(x_hat, x; loss_mask)  +  lambda(t) * SAM(x_hat, x; loss_mask)
```

`t` is the epoch number. `lambda` (the Greek letter, written `sam_weight` in the
code) is **not constant** — it changes as training progresses.

```python
def _get_sam_weight(self, epoch: int) -> float:
    sam_max     = cfg.sam_weight
    ramp_epochs = cfg.sam_ramp_epochs
    if ramp_epochs <= 0:
        return sam_max
    return sam_max * min(1.0, epoch / ramp_epochs)
```

Read the last line carefully:

- `epoch / ramp_epochs` grows linearly from 0.
- `min(1.0, ...)` caps it at 1 once the ramp finishes.
- multiplying by `sam_max` scales the whole thing.

So: **a straight line from 0 up to `sam_weight` over `sam_ramp_epochs`, then
flat forever.**

---

## Why ramp instead of just picking a weight

At epoch 0 the network's weights are random. Its output is noise. The spectra it
produces have essentially random shape, so SAM is large — around 0.3 to 0.4
radians — and its gradient points in a direction that has very little to do with
the much more basic job of getting the brightness roughly right.

Optimising two hard objectives simultaneously from a random start is unstable.
One tends to sabotage the other.

The ramp says: **learn the easy thing first.**

- Early epochs: `lambda` near 0, so L1 dominates. Get the overall levels right.
- Later epochs: `lambda` at full strength, tightening the shape constraint on a
  model that already produces sensible magnitudes.

> **Analogy.** Teaching someone to draw. First get the proportions roughly
> right; only then start correcting the fine contours. Insisting on perfect
> contours in the first five minutes produces a worse drawing and a frustrated
> student.

The configuration file names the failure mode directly:

```json
"L1_increases_when_SAM_ramp_starts": "SAM weight too high (try sam_weight=0.5)"
```

If you turn the shape penalty up too fast, the model sacrifices brightness
accuracy to satisfy it, and L1 goes **up**. That is a real symptom somebody hit.

---

## The two shipped configurations

| Setting | v0.1.0 | v0.2.0 (current) |
|---|---|---|
| `sam_weight` | 1.0 | **0.5** |
| `sam_ramp_epochs` | 20 | **10** |
| `mask_ratio` | 0.50 | 0.65 |
| `compressed_channels` (D) | 24 | 32 |
| `drop_rate` | 0.3 | 0.4 |
| `learning_rate` | 1e-4 | 1e-3 |
| epochs | 500 | 200 |
| final validation loss | 0.07694 | **0.04349** |

Note that v0.2 both **halved** the final weight and **halved** the ramp length —
it reaches full strength twice as fast, but full strength is half as strong.

---

## The ramp, worked out epoch by epoch

For v0.2.0: `sam_max = 0.5`, `ramp = 10`.

```
lambda(t) = 0.5 * min(1, t / 10)
```

| epoch `t` | `t / 10` | `min(1, .)` | `lambda` |
|---|---|---|---|
| 0 | 0.0 | 0.0 | **0.00** |
| 1 | 0.1 | 0.1 | 0.05 |
| 2 | 0.2 | 0.2 | 0.10 |
| 3 | 0.3 | 0.3 | 0.15 |
| 5 | 0.5 | 0.5 | 0.25 |
| 8 | 0.8 | 0.8 | 0.40 |
| 9 | 0.9 | 0.9 | 0.45 |
| 10 | 1.0 | 1.0 | **0.50** |
| 50 | 5.0 | 1.0 | 0.50 |
| 200 | 20.0 | 1.0 | 0.50 |

Straight line for 10 epochs, then constant for the remaining 190.

---

## Worked example of the combined number

Take the configuration's own epoch-100 baseline: `L1 = 0.010`,
`SAM = 0.060` rad. At that point `lambda = 0.5`:

```
L_total = 0.010 + 0.5 * 0.060
        = 0.010 + 0.030
        = 0.040
```

Now look hard at those two contributions:

```
L1 contributes    0.010     25% of the total
SAM contributes   0.030     75% of the total
```

## **After the ramp, SAM is three times more influential than L1.**

This surprises people, so let us be explicit about why.

The two terms are in **completely different units**:

- L1 is in reflectance, and a good value is around 0.01.
- SAM is in radians, and a good value is around 0.06.

SAM's natural magnitude is about six times larger. Multiplying by
`lambda = 0.5` brings it to three times larger — not smaller.

So `lambda` is doing **unit conversion** as much as weighting. It is not "L1
plus a small correction". After the ramp, **spectral shape is the dominant
training signal**, which for a hyperspectral model is the intended design.

---

## How to read `val_loss`

The checkpoint manifest says:

```json
"val_loss": 0.04349
```

You now know exactly what that is: **L1 + 0.5 x SAM on the validation set**.

It is not an accuracy. It is not a percentage. It is not an L1. And critically:

> **You cannot compare it against a thermal model's `val_loss`.** Chakshu's loss
> is pure L1 on a single band in degrees Celsius. Different terms, different
> units, different scale. Comparing the two numbers is meaningless.

Model comparison has to happen on the *scores*, via ROC and AUC on annotated
scenes (part 30), not on training losses.

---

## What actually gets logged

The trainer keeps the two components separate so a human can inspect them:

```python
logger.info(
    f"  L1: {avg_l1:.6f} | SAM: {avg_sam:.6f} rad "
    f"({avg_sam * 180 / 3.14159:.2f} deg) | lambda: {sam_weight:.3f}"
)
```

and mirrors them to Weights and Biases as `train/l1_loss`, `train/sam_loss`,
`train/sam_loss_deg` and `train/sam_weight`.

### The practical advice

**When watching a run, watch the components, not the combined number.**

During the first ten epochs `lambda` is rising, so the combined loss can
legitimately go **up** while the model is genuinely improving. Look:

| epoch | L1 | SAM | lambda | combined |
|---|---|---|---|---|
| 0 | 0.090 | 0.35 | 0.00 | **0.090** |
| 10 | 0.040 | 0.20 | 0.50 | **0.140** |

L1 more than halved. SAM nearly halved. And the combined number went *up* by
50%, purely because the weighting changed. A newcomer watching only the combined
number would conclude training had failed.

---

## Validation does something training does not

`compute_validation_loss` runs **two complementary passes** so that every valid
pixel is graded exactly once.

```python
rand_mask   = (torch.rand(1, N, device=self.device) > 0.5).float()

pred_mask_1 = token_validity * (1.0 - rand_mask)   # hide where rand_mask = 0
keep_mask_1 = 1.0 - pred_mask_1

pred_mask_2 = token_validity * rand_mask           # hide the exact opposite set
keep_mask_2 = 1.0 - pred_mask_2

x_hat_1 = model(pixels, mask=mask, keep_mask=keep_mask_1)
x_hat_2 = model(pixels, mask=mask, keep_mask=keep_mask_2)

x_hat = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels
```

The two `pred_mask`s are exact complements: `(1 - rand_mask)` and `rand_mask`.
So every valid token is hidden in **exactly one** of the two passes.

Each pixel's final value is taken from the pass in which it was hidden. The
result is a full reconstruction in which nothing was self-predicted.

### Why bother

**Comparability.** Training masks randomly, so the training loss depends partly
on which tokens happened to be hidden. Validation covers every pixel exactly
once, every time, so the number is comparable across epochs and across models.

That is why checkpoint selection uses validation loss, not training loss.

The loss is then computed over `eroded_mask` alone — no `pred_mask` factor,
because by construction every valid pixel was a target in one of the two passes:

```python
l1 = ((x_hat - pixels).abs().mean(dim=1, keepdim=True) * eroded_mask).sum()
     / eroded_mask.sum().clamp(min=1)
```

(This is also exactly what inference does, on a larger scale — part 29.)

---

## Baselines to sanity-check a run

From `configs/hyperspectral_segformer_exp_2.json`:

| epoch | L1 | SAM | combined |
|---|---|---|---|
| 0 | 0.08 – 0.10 | 0.3 – 0.4 rad (20-25 deg) | 0.08 – 0.10 |
| 10 | 0.03 – 0.05 | 0.15 – 0.25 rad (10-15 deg) | 0.10 – 0.15 |
| 20 | 0.02 – 0.03 | 0.08 – 0.15 rad (5-8 deg) | 0.10 – 0.15 |
| 100 | 0.005 – 0.015 | 0.04 – 0.08 rad (2-5 deg) | 0.04 – 0.08 |
| 500 | 0.003 – 0.008 | 0.02 – 0.05 rad (1-3 deg) | 0.02 – 0.05 |

(The `lambda` column in the shipped JSON reflects v0.1.0's schedule. The L1 and
SAM columns are the ones to compare against.)

Notice the combined column rising from epoch 0 to epoch 10 and then falling —
exactly the ramp artefact described above, visible in the project's own
documented baselines.

### The red-flag table, straight from the config

| symptom | likely cause |
|---|---|
| L1 not below 0.05 by epoch 10 | learning rate too low, or a data-loading problem |
| SAM stuck above 0.3 after epoch 30 | bottleneck `D` too narrow — try 32 |
| L1 rises when the SAM ramp starts | `sam_weight` too high |
| train loss falls but validation does not | overfitting — more dropout or a smaller model |

Every one of those has a matching knob in the configuration, and at least two of
them were actually hit and fixed between v0.1 and v0.2 (`D` went 24 to 32;
`sam_weight` went 1.0 to 0.5).

---

## Common confusions

**"Is a lower combined loss always better?"**
Only after the ramp completes. During the ramp the weighting is changing, so the
combined number is not comparable epoch to epoch.

**"Could you just normalise both terms to the same scale?"**
You could, and it would make `lambda` a pure preference rather than a unit
conversion. Nobody has done it here; `lambda` currently plays both roles.

**"Why does validation use random masks rather than the checkerboard inference
uses?"**
Two complementary halves is the requirement; how you split them is a detail.
Random matches training's regime more closely.

---

## Check yourself

1. Compute `lambda` at epoch 4 for v0.2.0 (`sam_max = 0.5`, `ramp = 10`).
2. With `L1 = 0.02` and `SAM = 0.12` at epoch 50 of v0.2.0, what is the combined
   loss, and which term contributes more?
3. Why can the combined loss rise during the first ten epochs while the model
   improves?
4. What exactly is the `0.04349` in the checkpoint manifest, and why can it not
   be compared with a thermal model's number?
5. Why does validation run two passes?

<details>
<summary>Answers</summary>

1. `0.5 * min(1, 4/10) = 0.5 * 0.4 = 0.20`.
2. Epoch 50 is past the ramp, so `lambda = 0.5`.
   `0.02 + 0.5 * 0.12 = 0.02 + 0.06 = 0.08`. SAM contributes 0.06 versus L1's
   0.02 — three times more.
3. Because `lambda` is increasing, so the SAM term is being weighted more
   heavily each epoch. Both components can fall while their weighted sum rises.
4. It is `L1 + 0.5 * SAM` on the validation set, mixing reflectance units and
   radians. A thermal model's loss is pure L1 on one band in Celsius — a
   different quantity entirely.
5. So that every valid pixel is graded exactly once, from a prediction that
   never saw it. That makes the number comparable across epochs and models, which
   is what checkpoint selection needs.

</details>

---

**Next:** counting every parameter in the model, in
[26-parameter-budget.md](26-parameter-budget.md)
