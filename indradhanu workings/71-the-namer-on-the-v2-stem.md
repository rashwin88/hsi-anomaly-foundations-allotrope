# 71 - A second consumer of V2's stem: the spectral namer

> **The one thing this part teaches:** the spectral namer built on 2026-10-31 lifts three things out
> of the shipped Indradhanu V2 checkpoint (FTH3 epoch 5) - the 17,825-parameter reduced stem, the
> post-stem BatchNorm and the normalisation curve - and runs them on single pixels to name materials;
> the stem transplants cleanly because nothing in it is tied to a band position, but the BatchNorm
> brought part 51's trap in a new form - nine different kinds of batch per step in training, one stored
> blend of them at evaluation - and the fix, made before N001 ran, was to freeze it at V2's own running
> statistics; the one thing the namer added on top of the transplant since then is a single number,
> `l1_bias`, when its level 1 moved from a linear head to matching (347,684 -> 347,685 parameters).

## Where we are

Part 70 shipped Indradhanu Heavy as V2. Until now every user of the spectral stem was an Indradhanu
model: the stem fed an encoder that reconstructed a scene. On 2026-10-31 a second, unrelated model
started reading it - the **spectral namer**, which names the material of a pixel that something else
has already flagged. Its design and its training are taught in the anomaly-detection course, parts 26
and 27; this part is the Indradhanu side only: what is taken from V2, why it fits, and what to watch.
No final namer result exists yet. Run N001 started on an H100 capacity block on 2026-10-31 at 11:37 UTC
and was stopped after epoch 1 because its level 1 (natural or man-made) sat at chance; N001b, the same
run with level 1 decided by matching, started at 11:59 UTC and had reported epoch 1 only when this was
written (`research/experiments/spectral-namer/N001/README.md`, `N001b/README.md`). The open-set run N002
has a config and no README yet.

| file | what it holds |
|---|---|
| `app/spectral_match/namer/model.py` | `SpectralNamer`, `encode`, `from_spectral_stem` |
| `app/spectral_match/namer/train.py` | `NamerRun`: loading `init_from`, the two learning rates, `decode` |
| `configs/namer/N001.json`, `N001b.json`, `N002.json` | all three: `"init_from": "/opt/dlami/nvme/models/FTH3_ep5.pt"`, `"stem_lr_scale": 0.3`; N001b and N002 add `"level1": "matching"` |
| `app/foundation_models/components/spectral_reduced.py` | `SpectralReducedCompressor` (part 40) |
| `app/foundation_models/components/spectral_stem_seg_former_mae.py` | V2's `spectral_in`, `norm = nn.BatchNorm2d(D)` |
| `research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/config.json` | `compressed_channels` 64, `token_dim` 32, `attention_heads` 4, `bin_nm` 40 |
| `research/textbook/tools/compute_namer_submission.py`, `compute_namer_full.py` (+ `_output.txt`) | the counts below |

> **Analogy.** An organ transplant. The donor (V2) gives one organ (the stem) plus the small piece of
> tissue that connects it (the BatchNorm). The organ works in any body because it was never shaped to
> one - it reads wavelengths, not band positions. The connecting tissue is the risk: it was tuned to the
> donor's circulation, and in the new body it is fed by several different vessels in turn.

## What is taken, and how

`SpectralNamer.from_spectral_stem(state_dict)` copies three things from a V2 checkpoint
(`model_state_dict`):

```python
m = cls(state_dict["stats.wl"].tolist(), state_dict["stats.mean"].tolist(), state_dict["stats.std"].tolist(), **kw)
for prefix, mod in (("spectral_in.", m.stem), ("norm.", m.norm)):
    mod.load_state_dict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})
```

1. **The normalisation curve** (`stats.wl / mean / std`, part 49): per-wavelength mean and spread, read
   from the checkpoint and "never recomputed" (`model.py` docstring). The same pixel is normalised the
   same way the reconstruction model normalised it - absolute reflectance kept, never per-scene min-max.
2. **The stem** (`spectral_in.*`): Heavy's `SpectralReducedCompressor`.
3. **The post-stem BatchNorm** (`norm.*`): V2 has `nn.BatchNorm2d(64)`; the namer has `nn.BatchNorm1d(64)`.
   Both store the same five tensors (`weight`, `bias`, `running_mean`, `running_var`,
   `num_batches_tracked`), so the state dict loads across. `load_state_dict` is strict by default: if
   the shapes did not match it would fail loudly rather than load half a stem.

The shapes must match Heavy's, and they do by default: `SpectralNamer` defaults to
`compressed_channels=64, token_dim=32, attention_heads=4, bin_nm=40.0, band_weighting="width"`, and
N001's `"model"` block sets only `hidden` and `embed`. Heavy's FTH3 config says 64, 32, 4 and 40.

**Predict the stem's size.** Part 40's pieces, with `token_dim = 32` and `D = 64` (an
`nn.Linear(i, o)` has `i x o + o` numbers; `_hyper` is Linear(32, 64) -> GELU -> Linear(64, out)):

```
band_weight  (k)     32*64+64 = 2,112   +  64*32+32 = 2,080    ->  4,192
skip_weight  (W)     2,112              +  64*64+64 = 4,160    ->  6,272
bin_bias             32*32+32                                  ->  1,056
queries              64 x 32                                   ->  2,048
q, k, v, o           4 x (32*32+32) = 4 x 1,056                ->  4,224
to_channel           32*1+1                                    ->     33
total                4,192 + 6,272 + 1,056 + 2,048 + 4,224 + 33 = 17,825
```

The namer reports `stem 17,825`. **No number in it depends on the band count**, which is why one stem
reads EnMAP's 224, PRISMA's list and AVIRIS-NG's 372 (`test_namer_reads_any_band_list_and_loads_a_stem_checkpoint`
runs it at 60 and 37 bands). Of the namer's 347,685 parameters it is `17,825 / 347,685 = 5.1%`; the rest
(pixel MLP, library MLP, three heads, the temperature `log_tau`, the level-1 bias `l1_bias`) are new.
They start from random, except `log_tau` (started at `log 0.07`) and `l1_bias` (started at 0).

## How the stem is called on one pixel

The stem expects images, `(B, N, H, W)` with one wavelength list `(N,)`. The namer has spectra, `(P, N)`:

```python
x = spectra[:, :, None, None]                            # (P, N, 1, 1): P one-pixel images
z = self.stats.normalize(x, wavelengths_nm)              # the checkpoint's curve
z = z.masked_fill(~bv, 0.0)                              # dead bands to 0, as V2 does
return self.norm(self.stem(z, wavelengths_nm, bv)[:, :, 0, 0])   # (P, 64)
```

Because the stem works band by band within each pixel (part 40), a one-pixel image loses nothing. Two
consequences of "one wavelength list per call" (the stem raises otherwise): the namer's `decode`
averages a batch's per-sample lists ("PRISMA's per-scene centres differ by ~1 nm"), and each sensor is
encoded in its own call - the same constraint part 52 met.

## Two learning rates

`NamerRun.__init__` gives the stem and its BatchNorm `lr x stem_lr_scale` and the rest `lr`:

```
rest:          3e-4
stem + norm:   3e-4 x 0.3 = 9e-5
```

The transplanted part is allowed to move, but slowly: it already encodes spectra well enough to
reconstruct three sensors, and the new heads start from random, so their early, large gradients would
otherwise rewrite it before they learn anything. This is the fine-tuning convention of a smaller rate for
pretrained layers; 0.3 is N001's choice, not a measured optimum.

## The BatchNorm, again: a risk found, then fixed

**The risk.** Part 51 found that one BatchNorm fed one sensor per batch normalises each sensor by its own statistics in
training, while evaluation uses one stored estimate weighted towards the last batches. The namer meets a
sharper version. Per step and per sensor, `forward` calls `encode` three times: the planted pixels `x`,
their ring means `r`, then the whole library `bank.refl`. With three sensors in a fixed order (EnMAP,
PRISMA, AVIRIS-NG), that is nine calls per step, each a different kind of batch - pixels, smoother ring
means, clean laboratory spectra.

In training, each call is normalised by **its own** batch statistics. The stored `running_mean` is an
exponential moving average with PyTorch's default momentum 0.1: after each call,
`running = 0.9 x running + 0.1 x batch`. In steady state, the call `k` places from the end of a step
carries weight `0.1 x 0.9^k / (1 - 0.9^9)`:

```
0.9^9 = 0.3874;   1 - 0.3874 = 0.6126
last call (AVIRIS-NG library):  0.1 / 0.6126           = 0.1632
next     (AVIRIS-NG ring):      0.1 x 0.9 / 0.6126     = 0.1469
...
first    (EnMAP pixels):        0.1 x 0.9^8 / 0.6126   = 0.0703
by kind:  pixels 0.299,  rings 0.332,  library 0.369        (a real BatchNorm1d agrees: 4.692 vs 4.692)
```

In evaluation (`evaluate` and `infer.py` call `.eval()`), pixels, rings and library spectra are all
normalised by that **one** blend, of which pixels are under a third. So the library embeddings the model
was trained against (library-batch statistics) are not quite the ones it scores against at evaluation
(blended statistics), and the same holds for the pixel side.

That was a reading of the first version of the code, written down while N001 was being prepared; it was
never measured.

**The fix.** Before N001's code bundle (`namer2`) went to the H100, `SpectralNamer` got its own `train()`:

```python
def train(self, mode: bool = True):
    # docstring: "The post-stem BatchNorm stays in eval mode: V2's running statistics (real pixels of
    # the three sensors) normalise pixels, ring means and library spectra alike. ..."
    super().train(mode)
    self.norm.eval()
    return self
```

PyTorch calls `train()` on every module whenever the model is switched to training, so overriding it is
the one place that catches every switch. In eval mode a BatchNorm uses its stored `running_mean` and
`running_var` and never updates them. So in training and evaluation alike, every call - pixels, rings,
library, any sensor - is normalised by the same numbers: V2's, estimated on real pixels of the three
sensors. The table above collapses to one row with weight 1.

**Why freezing, of the two remedies.** One call over concatenated pixels, rings and library would give
consistent statistics within a step, but they would still be a mix of very different inputs (clean
laboratory spectra shift them), they would drift step by step, and the call would cost memory. Freezing
keeps the statistics of the data the stem was trained on and the namer is applied to: real pixels.

**What still trains.** The norm's scale and shift (`weight`, `bias`, 128 numbers) are ordinary parameters.
Their names start with `norm.`, so `NamerRun` puts them in the stem's group, at `0.3 x` the learning rate.
Eval mode only stops the *statistics* from moving; gradients still reach the scale and shift. The model is
347,685 parameters (the next section explains the last one).

## Level 1 by matching: one new number, one idle head

The namer answers two questions in turn: level 1, natural or man-made; level 2, which of seven man-made
classes. Level 2 was always matching: each class's score is the logsumexp, over its library entries, of
`cos(z, e_j) / tau` (anomaly-detection part 27, Step 4). In N001, level 1 was not: it was `to_manmade`,
one `nn.Linear(256, 1)` on the pixel tower's `h`. N001's README: "Level 2 learned (class accuracy
0.60-0.68 at fill 0.25-1.0 on all three sensors) but **level 1 sat at chance**: accuracy 0.49-0.52 at every
fill, 0.50 even at fill 1.0, where ACE gets 0.87". Its laptop diagnosis (EnMAP smoke shards, 150 steps at
fill ~1): "the linear level-1 head reached BCE 0.64, accuracy 0.65, recall at ACE's FA 0.41; deciding level 1
by matching ... reached BCE 0.43, accuracy 0.80, recall 0.74."

**The rule** (`model.level1_logit`, used when the config says `"level1": "matching"`; `train.py`'s default
is `"head"`): the natural library spectra become one more class, column 7 of the score table, and

```
logit       = logsumexp(S_0, ..., S_6) - S_natural + l1_bias     # torch.logsumexp(scores[:, :-1], 1) - scores[:, -1] + self.l1_bias
P(man-made) = sigmoid(logit)
```

**A hand example** (the textbook's, chapter 42). Three entries, scores already divided by `tau`: plastic
6.0, metal 5.0 (two man-made classes, one entry each), soil 5.5 (natural); `l1_bias = 0`:

```
logsumexp(6.0, 5.0) = 6.0 + log(1 + e^-1) = 6.0 + log 1.3679 = 6.0 + 0.3133 = 6.3133
logit               = 6.3133 - 5.5 + 0                                    = 0.8133
e^-0.8133           = 0.4434
P(man-made)         = 1 / (1 + 0.4434) = 1 / 1.4434                      = 0.6928
```

The same figure is the man-made share of the softmax: `(e^6 + e^5) / (e^6 + e^5 + e^5.5) = (403.4288 +
148.4132) / (403.4288 + 148.4132 + 244.6919) = 551.8420 / 796.5339 = 0.6928`.

**Why it matters on the Indradhanu side.** Both levels now read the one cosine table that the stem's
features feed, so level 1's gradient reaches the stem by the same path as level 2's: through the library
tower as well as the pixel tower. And a new library entry, natural ones included, changes level 1
without retraining, because nature is described by prototypes rather than frozen in 256 weights.

**The count, again.** Matching added one parameter and stopped one head from training:

```
pixel MLP 192 -> 256 -> 256 -> 256     49,408 + 65,792 + 65,792     = 180,992
library MLP 64 -> 256 -> 256 -> 128    16,640 + 65,792 + 32,896     = 115,328
heads   to_embed 256*128+128 = 32,896;  to_manmade 257;  to_fill 257  =  33,410
BatchNorm scale + shift                 64 + 64                      =     128
scalars log_tau + l1_bias               1 + 1                        =       2
stem                                                                 =  17,825
total   180,992 + 115,328 = 296,320;  + 33,410 = 329,730;  + 128 = 329,858;  + 2 = 329,860;  + 17,825 = 347,685
```

`compute_namer_full_output.txt` (section 1) instantiates the model with N001b's settings and prints
347,685, "(347,684 before l1_bias was added for level 1 by matching)". **`to_manmade`'s 257 parameters
are still built and still counted, but under matching its output is overwritten before the loss, so
they receive no gradient.** It stays in the model so that N001's checkpoints load (`load_namer` refuses
unexpected keys, and tolerates a missing `l1_bias`).

## What the namer does *not* take

No encoder, no decoder, no segmentation head, no residual: the namer never reconstructs anything, and its
pixel side sees the pixel and a 15 x 15-minus-5 x 5 ring mean of raw neighbours (anomaly-detection part 27),
not V2's predicted background. So the leakage questions of parts 66 and 69 do not arise for it - and nor
does V2's knowledge of spatial context.

---

## Common confusions

**"The namer is a fine-tune of Indradhanu V2."** It shares 17,825 numbers with V2, 5% of the namer, and
nothing after the stem. It is a different model for a different job.

**"Changing V2's checkpoint changes the namer."** No: the namer copies the weights at the start of
training (`init_from`) and then trains them at 0.3x. A future V2 would need a new namer run to be shared.

**"BatchNorm2d and BatchNorm1d weights are not compatible."** They store the same tensors for the same
channel count; only the input shape differs.

**"The stem was built for 128 x 128 tiles, so it needs space."** It is per pixel along wavelength; the
1 x 1 images are only its calling convention.

**"A frozen BatchNorm does not learn."** Its running statistics are frozen; its scale and shift still train
at the stem's rate.

**"The namer's level 1 is a linear head on the pixel tower."** Only in N001. Since N001b it is
`logsumexp(man-made class scores) - natural score + l1_bias`; the head `to_manmade` is still built and
counted (257 of 347,685) but trains under no loss.

## Check yourself

1. Count the stem's parameters for `token_dim = 32` but `D = 32` (FT04's latent), showing which pieces
   change.
2. What learning rate does the stem get with `lr = 3e-4` and `stem_lr_scale = 0.3`, at the peak of the
   schedule?
3. In the unfixed code, if the order of sensors in a step were AVIRIS-NG, PRISMA, EnMAP, which call
   would have carried the largest weight in the running mean, and how large? What is it in the fixed
   code?
4. Why can the same stem read a 37-band and a 60-band spectrum, with no parameter added?
5. Name two ways the BatchNorm mismatch could be removed, what each costs, and which one the code
   chose.
6. In the hand example of "Level 1 by matching", add a second natural entry, also at 5.5. Compute the
   natural score, the logit and P(man-made). Then: how many of the namer's parameters receive a gradient
   under matching?

<details>
<summary>Answers</summary>

1. `band_weight` (`_hyper(32)`): unchanged, 4,192. `skip_weight` (`_hyper(D)`): `2,112 + (64 x 32 + 32 =
   2,080)` = 4,192. `bin_bias`: 1,056. `queries`: `32 x 32 = 1,024`. q, k, v, o: 4,224. `to_channel`: 33.
   Total `4,192 + 4,192 + 1,056 + 1,024 + 4,224 + 33 = 14,721` - part 40's figure for the reduced stem
   at `D = 32`.
2. `3e-4 x 0.3 = 9e-5` (after the 500 warm-up steps; the cosine then lowers it).
3. The last call of the step, EnMAP's library: `0.1 / (1 - 0.9^9) = 0.1 / 0.6126 = 0.1632`. In the fixed
   code no call updates the running mean at all: V2's stays, with weight 1.
4. Every weight is a function of the wavelength encoding (`band_weight(enc)`, `skip_weight(enc)`) or of the
   fixed 40 nm bins, never of a band's position, so a new band list only changes the inputs to those small
   networks.
5. Keep Heavy's running statistics frozen (no update in training; costs adaptation to planted and library
   spectra, and training and eval would then both use Heavy's statistics); or normalise pixels, rings and
   library in one call (consistent statistics per step, but the library's clean spectra would still shift
   the batch statistics, and it costs memory). The code froze them (`SpectralNamer.train()` keeps
   `self.norm` in eval mode); the scale and shift still train.
6. `S_natural = logsumexp(5.5, 5.5) = 5.5 + log 2 = 5.5 + 0.6931 = 6.1931`. Logit `6.3133 - 6.1931 + 0 =
   0.1202`. `e^-0.1202 = 0.8867`; `P = 1 / 1.8867 = 0.5300` (the textbook's tool prints logit 0.1201 and
   P 0.53, from unrounded inputs). Gradients: `347,685 - 257 = 347,428` parameters, all but
   `to_manmade`.

</details>

---

That is the course so far. Indradhanu V2 is Indradhanu Heavy (FTH3 epoch 5, `v2.1.0`), opt-in next to
v0.2.0 (still the default codename); since 2026-10-31 the band filter keeps native bands by default, and
V2's stem has a second user, the spectral namer, whose level 1 is now decided by matching (N001b,
training when this was written).
