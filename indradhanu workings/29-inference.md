# 29 · Inference: two passes and a sliding window

> **The one thing this part teaches:** to score a real scene, the model runs
> twice per tile (so no pixel predicts itself) over hundreds of overlapping tiles
> (because scenes are far bigger than 128 pixels).

**Sources:**
[`segformer_mae_inferencer.py`](../app/foundation_models/inferencers/segformer_mae_inferencer.py)
— the machinery, inherited — and
[`hyperspectral_segformer_mae_inferencer.py`](../app/foundation_models/inferencers/hyperspectral_segformer_mae_inferencer.py),
which is only 118 lines because it overrides just `build_model` and adds one
scoring method.

---

## Two problems inference has that training does not

**1. Every pixel needs a score.** Training hides a random 65% and grades only
there, which is fine because over thousands of patches everything gets hidden
eventually. But here we have one specific scene and need an answer everywhere.

**2. Scenes are enormous.** The model takes 128x128 patches. A PRISMA scene is
roughly 1000x1000. You cannot feed it in one go.

Solved by **two-pass masking** and a **sliding window** respectively.

---

## Loading the model

`FoundationInferencer.__init__` does three things:

```python
self.model = self.build_model().to(self.device)
self._load_weights(config.checkpoint_path)
self.model.eval()
```

and `_load_weights` is:

```python
ckpt = torch.load(path, map_location=self.device, weights_only=False)
self.model.load_state_dict(ckpt["model_state_dict"])
```

### Why the order matters

You must **build the model first, then load the weights into it**. PyTorch does
not reconstruct an architecture from a checkpoint; it copies numbers into a
structure you have already created.

And `load_state_dict` is **strict** by default: every entry in the saved file
must have a matching slot, with a matching shape.

That includes the 165-long normalisation buffers (part 09), and it includes the
spectral compressor sized by `D`. Which is why the resolver reconstructs `D`
from the manifest:

```python
spectral_dim = int(current.get("spectral_dim_D", 32))
```

Build with `D = 24` against a `D = 32` checkpoint and you get an immediate,
loud shape error — not a silent mis-scoring. That strictness is a feature.

`model.eval()` then disables dropout and switches BatchNorm to its stored
statistics (part 28).

---

## Two-pass masking

Split the tokens into two complementary halves. Run the model twice. Take each
pixel's value from the pass in which it was **hidden**.

```python
x_hat_1 = self.model(tensor, mask=mask, keep_mask=keep_mask_1)
x_hat_2 = self.model(tensor, mask=mask, keep_mask=keep_mask_2)
reconstruction = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels
reconstruction = reconstruction * mask
```

`pass1_pixels` and `pass2_pixels` are 0/1 masks that never overlap and together
cover everything. So that multiply-and-add is a **selection**, not a blend:

```
where pass1_pixels = 1:   1 * x_hat_1  +  0 * x_hat_2   =  x_hat_1
where pass2_pixels = 1:   0 * x_hat_1  +  1 * x_hat_2   =  x_hat_2
```

Every pixel ends up with a prediction made **without seeing itself**. That is
the rule from part 06, enforced for the whole scene.

---

## Strategy A: checkerboard (the default)

```python
rows = torch.arange(H_tokens) // cell_size
cols = torch.arange(W_tokens) // cell_size
grid = (rows[:, None] + cols[None, :]) % 2
```

### Worked, on a 4x4 token grid with `cell_size = 1`

Build a table of `row + col`:

```
        col 0  col 1  col 2  col 3
row 0     0      1      2      3
row 1     1      2      3      4
row 2     2      3      4      5
row 3     3      4      5      6
```

Now take each entry modulo 2 (0 if even, 1 if odd):

```
        0  1  0  1
        1  0  1  0
        0  1  0  1
        1  0  1  0
```

A checkerboard. The classic trick: on a chessboard, squares where row + column
is even are one colour and odd is the other.

`invert=True` flips it for the second pass. With `cell_size = 2`, the integer
division groups indices into pairs first, giving 2x2 blocks instead of single
tokens.

### The broadcasting

```python
rows[:, None] + cols[None, :]
```

`rows[:, None]` is a column of shape `(4, 1)`; `cols[None, :]` is a row of shape
`(1, 4)`. Adding them broadcasts to `(4, 4)` — the addition table above,
without any loop.

---

## Strategy B: random

```python
rand_mask   = (torch.rand(1, N, device=self.device) > 0.5).float()
pred_mask_1 = token_validity * (1.0 - rand_mask)
pred_mask_2 = token_validity * rand_mask          # the exact complement
```

Draw a random half; the other pass hides exactly the rest.

### Why offer this at all?

A checkerboard is a **regular, repeating** pattern. If the model is slightly
weaker at token boundaries — and it is — that weakness lands in exactly the same
relative positions every time.

The result is a faint **grid pattern** printed into the residual map. Not huge,
but visible, and an analyst may mistake it for structure on the ground.

Random masking scatters those artefacts differently in each tile. Since tiles
overlap heavily (below), the artefacts average away instead of reinforcing.

Set via `masking_strategy` on `InferenceConfig`. The default is
`"checkerboard"`, and the anomaly-scoring Action does not currently override it.

---

## Both strategies protect invalid tokens

```python
pred_mask = token_validity * (1.0 - checker)
keep_mask = 1.0 - pred_mask
```

Multiplying by `token_validity` means only tokens that are **both valid and
selected by the pattern** become targets. Invalid tokens are always kept —
exactly as in training (part 18), by a different mechanism.

---

## The sliding window

`predict_full_scene` handles a whole scene.

```python
ps     = self.config.patch_size          # 128
stride = self.config.stride or ps // 2   # 32 for Indradhanu (resolver default)
```

Note the resolver's default stride is **32**, not 64:

```python
default_patch_size=128,
default_stride=32,
```

A quarter of the patch size, giving heavy overlap. Training used a stride of 64;
inference uses 32 because the smoothing is worth the extra compute when you only
do it once.

```python
plan   = PatchPlanGenerator().generate_patching_plan(
             PatchRequest(input_cube=(c, h, w), width=ps, height=ps, stride=stride))
coords = plan.patch_coordinates

recon_sum = torch.zeros(c, h, w, device=self.device)
count     = torch.zeros(1, h, w, device=self.device)
```

Two accumulators, both the size of the scene: a running **sum** of predictions
and a running **count** of contributions. This is how you average overlapping
tiles without storing them all.

---

## Worked: how many tiles?

Scene 1000x1000, `ps = 128`, `stride = 32`. Along one axis:

```
0, 32, 64, ... how far can we go?

last valid start:  start + 128 <= 1000  ->  start <= 872
multiples of 32 up to 872:  872 / 32 = 27.25, so 864 is the largest

coords so far: 0, 32, 64, ..., 864
count = 864/32 + 1 = 27 + 1 = 28

next would be 896:  896 + 128 = 1024 > 1000, too far
                    snap back to 1000 - 128 = 872, keep it, stop

total: 29 coordinates
```

So `29 x 29 = 841` tiles.

At `inference_batch_size = 8` that is `841 / 8 = 106` batches. And since each
batch runs the model **twice**:

```
212 forward passes through the network
```

---

## Worked: how many times is each pixel covered?

An interior pixel appears in every tile that contains it. Along one axis, tiles
start every 32 pixels and are 128 long, so:

```
128 / 32 = 4 tiles cover it horizontally
128 / 32 = 4 tiles cover it vertically

4 x 4 = 16 tiles
```

**Sixteen independent reconstructions of every interior pixel**, each made from
a different offset and therefore a different set of visible context.

That is where the smoothness of the final residual map comes from — and it is
why a stride of 32 is worth sixteen times the compute of a stride of 128.

---

## Accumulate, then average

```python
for j, idx in enumerate(valid_indices):
    r, cs = batch_coords[idx]
    patch_eroded = eroded_mask[:, r:r+ps, cs:cs+ps]
    recon_sum[:, r:r+ps, cs:cs+ps] += recon[j] * patch_eroded
    count    [:, r:r+ps, cs:cs+ps] += patch_eroded

reconstruction = torch.where(count > 0, recon_sum / count, scene)
```

Three details worth pausing on.

**1. Contributions are weighted by the eroded mask.** Border pixels (part 20)
contribute 0 to the sum *and* 0 to the count. Not "a small amount" — nothing at
all. They cannot pull the average.

**2. `count` makes the average correct even at scene edges.** A pixel in the
middle has count 16; a pixel in the corner may have count 1. Dividing by the
actual count gives a true mean in both cases, with no special casing.

**3. The fallback is the input itself.**

```python
torch.where(count > 0, recon_sum / count, scene)
```

Where `count == 0` — no tile could contribute anything — the "reconstruction" is
set to the **original scene value**.

Think about what that does downstream. The score is
`|original - reconstruction|`, so:

```
score = |scene - scene| = 0
```

Uncovered pixels score exactly zero and therefore never appear as detections.
Elegant: rather than leaving garbage or NaN in unreachable corners, they are
made deliberately uninteresting.

---

## The tile validity filter

```python
MIN_VALID_FRACTION = 0.1
valid_frac = m.float().mean().item()
if valid_frac >= MIN_VALID_FRACTION:
    valid_indices.append(i)
```

Tiles that are more than 90% nodata are skipped entirely — the model never saw
such patches during training, so running it on them produces nonsense.

(As flagged in part 23: the comment above this line claims it "matches training",
where the threshold is actually 0.4. The numbers differ. The intent is the same.)

---

## Indradhanu's own addition: dual scoring

Everything above is inherited from the thermal model. This is the one method
Indradhanu adds:

```python
def compute_anomaly_scores(self, original, reconstruction, mask) -> dict:
    l1 = (original - reconstruction).abs().mean(dim=1)          # (B, H, W)
    ...
    cos_sim = (dot / (norm_o * norm_r + 1e-8)).clamp(-1.0, 1.0)
    sam = torch.acos(cos_sim).squeeze(1)                        # (B, H, W)
    return {"l1": l1, "sam": sam}
```

It returns **two maps in a dictionary**, deliberately not one fused number. The
docstring explains: *"A downstream consumer can fuse them however it likes (max,
product, normalised sum). The inferencer keeps them separate to give you
control."*

Note `torch.acos` here rather than the `atan2` formulation from part 24 — fine,
because inference takes no gradients.

> **Two gotchas.**
> **(a)** `predict_full_scene` returns a *tensor*; `compute_anomaly_scores`
> returns a *dictionary*. Do not index the dict as if it were a tensor.
> **(b)** In production, the Action does not call this method at all. It uses the
> numpy `compute_score` in `app/utils/anomaly_detection/scoring.py` instead
> (part 30). This method exists for notebooks and experiments.

---

## Memory, roughly

For a 1000x1000 scene at 165 bands, float32 (4 bytes):

```
one full cube :  165 * 1000 * 1000 * 4 = 660,000,000 bytes = 660 MB

scene tensor      660 MB
recon_sum         660 MB
reconstruction    660 MB
                 -------
                ~2 GB, before any activations
```

Which is why `inference_batch_size` is exposed as a per-Action override and
defaults to a conservative 8. The resolver's comment calls it *"the main memory
dial when running on a CPU-only Mac VM"*.

---

## Common confusions

**"Does the scene get resized to 128x128?"**
No. It is tiled into overlapping 128x128 pieces, each reconstructed
independently, and the pieces are averaged back into a full-size image.

**"Why not run the model on the whole scene at once?"**
Memory, and the model was trained at 128 — its four stages assume that scale.
A 1000x1000 input would give a wildly different token count at every stage.

**"Are the two passes twice as slow?"**
Yes, exactly twice. That is the price of the never-predict-yourself rule, and it
is not negotiable.

**"Why do reconstruction and score sometimes look smooth?"**
Sixteen overlapping predictions per pixel are averaged. Smoothness is expected.

---

## Check yourself

1. Why must inference run two passes rather than one?
2. Draw the checkerboard pattern for a 4x4 token grid using `(row + col) % 2`.
3. For a 600x600 scene with `ps = 128, stride = 32`, how many tiles are there?
4. What happens at pixels where `count == 0`, and why is that the right choice?
5. Why do border pixels contribute nothing to the average rather than a small
   amount?

<details>
<summary>Answers</summary>

1. Because every pixel needs a prediction made without seeing itself. One pass
   can only hide half the tokens; the second pass hides the complement.
2. Rows of `0 1 0 1 / 1 0 1 0 / 0 1 0 1 / 1 0 1 0`.
3. Largest start with `start + 128 <= 600` is 472; the largest multiple of 32 at
   or below that is 448, giving `448/32 + 1 = 15` coordinates; then 480 would
   overflow, so snap to `600 - 128 = 472`, making 16 per axis.
   `16 x 16 = 256` tiles.
4. The reconstruction is set to the original scene value, so the residual is
   exactly zero and the pixel can never be flagged. Better than leaving garbage
   in unreachable corners.
5. Because they are weighted by the eroded mask, which is 0 there — so they add
   0 to both the sum and the count. A small non-zero weight would still let
   contaminated predictions pull the average.

</details>

---

**Next:** turning residuals into an answer, in
[30-scoring-and-the-product.md](30-scoring-and-the-product.md)
