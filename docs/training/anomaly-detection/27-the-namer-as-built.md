# Part 27 - The spectral namer as built

> **The one thing this part teaches:** the namer built on 2026-10-31 is a small dual encoder on
> Indradhanu V2's wavelength-aware stem, trained only on samples it makes itself - a real pixel with a
> library spectrum mixed in, `x = g (a t + (1 - a) b) + n` - and it scores a whole class at once
> (logsumexp of the cosines to that class's library spectra, with the planted spectrum left out), so
> it is rewarded for naming the material, not for recognising one laboratory sample; level 1 (natural
> or man-made) is read from the same score table, with the natural library spectra as one more class;
> whether it beats classical ACE is decided by one number, its man-made recall at ACE's own natural
> false-alarm rate, and the run that will give it was still training when this was written.

## Where we are

Part 26 explained *why* naming was redesigned: use-based families, two questions in turn, a matching
model, and a classical baseline (the separability check). The same day the design became
`docs/proposals/spectral-namer.md` and then code. This part reads that code. **No final result exists.**
Run N001 started on an H100 capacity block on 2026-10-31 at 11:37 UTC (block until 2026-11-02 11:30
UTC) and was stopped after epoch 1, because its level 1 sat at chance (Step 5). Run N001b, the same
configuration with level 1 decided by matching, started at 11:59 UTC; its README reports epoch 1 only.
The open-set run N002 has a config and no README yet. Every number below is either a setting, a count
you can reproduce on a laptop, the CPU smoke test, or a figure quoted from those two READMEs.

| file | what it holds |
|---|---|
| `app/spectral_match/namer/classes.py` | the seven v1 man-made classes and `class_of` (family-library entry -> class) |
| `app/spectral_match/namer/bank.py` | `LibraryBank`, `build_bank`: the library on one band list, the 20% hash split |
| `app/spectral_match/namer/model.py` | `SpectralNamer`: stem, post-stem BatchNorm, pixel and library MLPs, `class_scores`, `level1_logit` |
| `app/spectral_match/namer/generator.py` | `ring_and_noise`, `smooth_gain`, `plant`: the planted samples, on the GPU |
| `app/spectral_match/namer/train.py` | `NamerRun`: the loss, the loop, `evaluate`, the ACE comparison |
| `app/spectral_match/namer/infer.py` | `load_namer`, `name_pixels`: naming the pixels of a committed mask |
| `scripts/train_namer.py`, `configs/namer/N001.json`, `N001b.json`, `N002.json` | the entry point and the three runs (N001 linear level 1; N001b and N002 `"level1": "matching"`) |
| `tests/test_spectral_match/test_namer.py` | eight tests (class map, ring, split and planting, full fill, leave-out, any band list, ring parity, level 1 by matching); the N001 README says "8 unit tests" |
| `research/experiments/spectral-namer/N001/README.md` | what N001 trains, the pass rule, the CPU smoke, why it was stopped |
| `research/experiments/spectral-namer/N001b/README.md` | N001b: level 1 by matching, ACE re-scored on the v1 classes, epoch 1 |
| `research/textbook/tools/compute_namer_full.py` (+ `_output.txt`) | the parameter count and the level-1 numbers; the textbook's chapter 42 is the long form of this part |
| `docs/proposals/spectral-namer.md`, section "As built" | where the code departs from the design |
| `research/textbook/tools/compute_namer_submission.py` (+ `_output.txt`) | the other worked numbers in this part (Step 5's six-entry example is worked, and checked two ways, in the text) |

> **Analogy.** A wine-tasting course that cannot get enough labelled wines. The teacher pours a glass of
> ordinary table wine, adds a measured splash of one reference wine from the cellar, and asks "which
> grape did I add?". The splash size varies, the glasses are slightly different (gain), and the room is
> noisy. The student is marked right for naming the *grape* - not for recognising the exact bottle -
> and the bottle used for the splash is never one of the answers on the card. Some bottles are kept
> locked away for the exam.

## Step 1 - The model: two encoders, one stem

Two words first. An **encoder** is a network that turns an input into a short vector, its
**embedding**. A **dual encoder** has two of them whose embeddings are compared by cosine (part 26,
Step 5). Here (`model.py`, docstring):

```
pixel side:   x, r (ring mean) -> stem -> [z_x, z_r, z_x - z_r] -> MLP -> h
              h -> embedding (unit, 128) | fill in [0, 1] | (level-1 linear head, unused with level1="matching")
library side: t -> stem -> MLP -> embedding (unit, 128)
class score:  logsumexp over the class's library embeddings of  cos(z, e_j) / tau
level 1:      "matching" (N001b on) - natural entries as one more class, level1_logit; or the linear head (N001)
```

(The textbook's chapter 42 still quotes the older docstring, whose second line read "level-1 logit
(man-made)"; the source above is the current one.)

Both sides share one **stem**: the wavelength-aware reduced-attention compressor of Indradhanu V2
(`SpectralReducedCompressor`, the Indradhanu course's part 40). It reads a spectrum at *any* band list
with its wavelengths, and outputs 64 numbers per pixel. That is what lets one namer serve PRISMA,
EnMAP and AVIRIS-NG without resampling. The stem works on each pixel's spectrum on its own and takes
images shaped (B, N, H, W), so a batch of P spectra is passed as P images of 1 x 1 pixel:

```python
x = spectra[:, :, None, None]                       # (P, N) -> (P, N, 1, 1)
z = self.stats.normalize(x, wavelengths_nm)         # Heavy's per-wavelength mean / std curve
...
return self.norm(self.stem(z, wavelengths_nm, bv)[:, :, 0, 0])   # (P, 64)
```

`self.norm` is a `BatchNorm1d(64)`: Heavy's own post-stem BatchNorm, copied with the stem. A
**BatchNorm** layer rescales each channel with a mean and spread; normally it uses the current
batch's statistics in training and the running averages it stored in evaluation. The namer's does not:
`SpectralNamer.train()` keeps it in eval mode, so V2's stored statistics normalise pixels, rings and
library spectra alike, in training and evaluation (its scale and shift still train, at the stem's
rate). The Indradhanu course's part 71 explains the mismatch this removes. The stem, the norm and the normalisation curve come from the released
V2 checkpoint, FTH3 epoch 5 (`SpectralNamer.from_spectral_stem`, which loads the `spectral_in.` and
`norm.` keys and `stats.wl / mean / std`). The curve is never recomputed.

The pixel side encodes the pixel `x` and its ring mean `r` separately and concatenates
`[z_x, z_r, z_x - z_r]`, 3 x 64 = 192 numbers. The difference is part 15's "subtract the ring" idea,
done in feature space. Three linear heads read `h`: `to_embed`, a 128-number embedding (made length
1); `to_fill`, a fill estimate squeezed into [0, 1]; and `to_manmade`, one **logit** (a raw score;
`sigmoid(logit)` is a probability). `to_manmade` was level 1 in N001. Under `"level1": "matching"`
(N001b, N002) it is still built and still computed, but its output is replaced before the loss sees it,
so it receives no gradient: level 1 comes from the score table instead (Step 5).

**Predict the parameter count.** An `nn.Linear(i, o)` has `i x o` weights and `o` biases. The pixel MLP
is 192 -> 256 -> 256 -> 256, the library MLP 64 -> 256 -> 256 -> 128:

```
pixel MLP     192*256+256 = 49,408;  256*256+256 = 65,792;  256*256+256 = 65,792    -> 180,992
library MLP    64*256+256 = 16,640;  65,792;                256*128+128 = 32,896    -> 115,328
to_embed      256*128+128 = 32,896
to_manmade, to_fill        256+1 = 257 each                                        ->     514
heads         32,896 + 514                                                           ->  33,410
BatchNorm     64 scales + 64 shifts                                                  ->     128
log_tau, l1_bias           1 + 1                                                     ->       2
stem (SpectralReducedCompressor, measured)                                           ->  17,825
total         180,992 + 115,328 + 33,410 + 128 + 2 + 17,825                          = 347,685
```

Add it in steps: `180,992 + 115,328 = 296,320`; `+ 33,410 = 329,730`; `+ 128 = 329,858`; `+ 2 =
329,860`; `+ 17,825 = 347,685`. Instantiating `SpectralNamer` with N001b's settings reports 347,685
(`compute_namer_full_output.txt`, section 1, and `compute_namer_submission_output.txt`). **It was 347,684
until 2026-10-31: the extra one is `l1_bias`, the single learned number that level 1 by matching
added (Step 5).** The BatchNorm's 128 are its scale and shift; its running mean and variance are
buffers, not parameters, so they are not counted. Of the 347,685, the 257 of `to_manmade` receive no
gradient under matching. The proposal's "Size: small (~1-3 M parameters)" is three to nine times the
built model: the code has 0.35 M. The stem, the only part taken from Heavy, is `17,825 / 347,685 =
5.1%` of it.

## Step 2 - The ring mean, 3-7 px

The pixel's surroundings are summarised by the **ring mean** `r`: the mean of the valid pixels in a
15 x 15 box minus the central 5 x 5 box (`generator.py`, `INNER, OUTER = 5, 15`). A box sum is computed
for every pixel at once by `avg_pool2d` times the box area:

```python
s = box(v, OUTER) - box(v, INNER)     # sum of valid values in the ring, (B, N, H, W)
c = box(m, OUTER) - box(m, INNER)     # number of valid pixels in the ring
ring = s / c.clamp(min=1)
```

In the interior, `c = 15 x 15 - 5 x 5 = 225 - 25 = 200` pixels. **Which distances are in the ring?** The
5 x 5 box covers offsets -2..2, so it removes every pixel up to 2 steps away (counting diagonal steps
as one); the ring is what is 3 to 7 steps away - in the generator's docstring, "3 to 7 px away in
Chebyshev distance, 200 of them". (`infer.py`'s docstrings still say "2-7 px", meaning "beyond 2 px, up
to 7"; same ring.) The gap matters: a target's halo (part 18) mostly sits within 2 px and must not leak into its own
background.

Do one on paper. A flat 0.20 tile; the target pixel 0.50; its eight neighbours 0.35 (halo):

```
15x15 sum = 216 x 0.20 + 8 x 0.35 + 0.50 = 43.2 + 2.8 + 0.5 = 46.5     (225 = 216 + 8 + 1)
 5x5 sum  =  16 x 0.20 + 8 x 0.35 + 0.50 =  3.2 + 2.8 + 0.5 =  6.5     ( 25 =  16 + 8 + 1)
ring      = (46.5 - 6.5) / 200 = 40.0 / 200 = 0.200
```

The halo and the target cancel out exactly; the code gives 0.2000 with count 200. Samples are only
taken where every band is valid, the ring has at least 20 valid pixels, and the pixel is at least 7 px
from the tile edge (`plant`: `count[:, 0] >= 20`, `ok[:, :OUTER // 2] = False`). Inference computes
the same ring band by band with `scipy.ndimage.uniform_filter` (`infer.ring_means`); a test pins the
two to 1e-5 (`test_ring_means_match_the_generator_on_a_valid_tile`).

## Step 3 - The generator: making a training sample

There are almost no real pixels with a known material, so the namer is trained on samples it makes
itself from real tiles of the three sensors' fine-tune shards (`train._loader`, the fine-tuner's shard
pipeline). For each sampled pixel `b` with ring mean `r` (`generator.py`, docstring):

```
kind 0 man-made     t a man-made library spectrum (train split), fill a log-uniform in [a_min, 1]
kind 1 natural-mix  t a natural library spectrum, same fill
kind 2 plain        a = 0
x  = g (a t + (1 - a) b) + n,   r' = g r,   g = 1 + gain * smooth curve in [-1, 1],   n ~ N(0, sigma)
```

N001's `kinds` are `[0.5, 0.25, 0.25]`: half the samples man-made. Kinds 1 and 2 are level-1
"natural". Four ingredients, each with a reason.

**The fill `a`, log-uniform.** `fill = exp(log(a_min) * (1 - u))` with `u` uniform in [0, 1], which is
`a_min ** (1 - u)`. With `a_min = 0.05`:

```
u = 0     a = 0.05 ** 1    = 0.0500
u = 0.5   a = 0.05 ** 0.5  = 0.2236        (the median)
u = 1     a = 0.05 ** 0    = 1.0000
P(a <= 0.25) = ln(0.25 / 0.05) / ln(1 / 0.05) = ln 5 / ln 20 = 1.6094 / 2.9957 = 0.5372
```

More than half the man-made samples fill a quarter of the pixel or less. A uniform fill would put only
`(0.25 - 0.05) / 0.95 = 21%` there. Part 26 showed that small fills are where ACE fails and where most
30 m targets live; log-uniform spends the training there.

**The gain `g`, smooth.** Three random coefficients `c ~ N(0, 0.5^2)` make a parabola over the band
index `t` in [-1, 1], clipped to [-1, 1], and `g = 1 + 0.1 x curve` (`smooth_gain`). With
`c = (0.2, -0.4, 0.6)`:

```
t = -1:  0.2 + 0.4 + 0.6 = 1.2  -> clip 1.0  -> g = 1 + 0.1 x 1.0 = 1.10
t =  0:  0.2                    -> 0.2       -> g = 1.02
t = +1:  0.2 - 0.4 + 0.6 = 0.4  -> 0.4       -> g = 1.04
```

It stands for an atmospheric-correction error, within +-10%. The same `g` multiplies the ring mean,
because an atmosphere error hits a pixel and its neighbours alike.

**The noise `n`, from the tile itself.** Per tile and band, `sigma` is the root mean square of the
differences between horizontal neighbours, divided by `sqrt(2)` (`ring_and_noise`). Why divide? If two
neighbours carry independent noise of spread `s`, their difference has variance `s^2 + s^2 = 2 s^2`, so
its spread is `s x sqrt(2)`. On a row 0.20, 0.23, 0.21, 0.22:

```
differences   0.03, -0.02, 0.01
squares       0.0009, 0.0004, 0.0001        sum 0.0014
mean square   0.0014 / 3 = 0.000467
RMS           sqrt(0.000467) = 0.0216
sigma         0.0216 / 1.4142 = 0.01528      (the code: 0.01528)
```

The generator's docstring says so: "RMS (uncentred) ... - texture counts too". The mean difference is
not subtracted first, so real texture (a field edge, a slope) counts as noise too. Sigma is therefore an *over*-estimate on
textured tiles, which errs on the hard side.

**Planting one pixel by hand.** Three bands, `b = (0.10, 0.30, 0.25)`, `t = (0.40, 0.35, 0.05)`,
`a = 0.25`, `g = (1.02, 1.00, 0.98)`, `n = (0.003, -0.002, 0.001)`:

```
a t           = (0.25 x 0.40, 0.25 x 0.35, 0.25 x 0.05)  = (0.1000, 0.0875, 0.0125)
(1 - a) b     = (0.75 x 0.10, 0.75 x 0.30, 0.75 x 0.25)  = (0.0750, 0.2250, 0.1875)
sum           = (0.1750, 0.3125, 0.2000)
g x sum       = (1.02 x 0.1750, 1.00 x 0.3125, 0.98 x 0.2000) = (0.1785, 0.3125, 0.1960)
x = ... + n   = (0.1815, 0.3105, 0.1970)
```

Notice what is *not* modelled: the planting is a single pixel - no point-spread function, no halo, no
adjacency - and the ring is untouched by it. The proposal's Risks section says so ("linear mixing,
smooth gain and Gaussian noise are simplifications").

**Bands.** Per batch, only bands valid in more than half the batch's pixels are kept (`forward`:
`keep = valid.float().mean(dim=(0, 2, 3)) > 0.5`), so water-vapour bands drop out, and the library is
cut to the same bands (`bank.bands(keep)`). If fewer than 20 bands survive, the batch is skipped.

## Step 4 - The class score, and the planted spectrum left out

Part 26 Step 5 matched a pixel to *single* spectra. The built model scores *classes*. For a pixel
embedding `z` and library embeddings `e_j`, the scaled score of entry `j` is `s_j = cos(z, e_j) / tau`;
the score of class `k` is

```
score_k = log( sum over j in class k of exp(s_j) )          ("logsumexp")
```

and `P(class | man-made)` is the softmax of the seven class scores (`class_scores`, then `.softmax(1)`
in `infer.py`). **Logsumexp** is a smooth maximum: `log(e^7 + e^6) = 7 + log(1 + e^-1) = 7.3133`, a
little above the larger term.

`tau`, the **temperature**, divides every cosine; a small `tau` makes the softmax decisive. It is
learned, stored as `log_tau` and started at `log(0.07) = -2.6593` (so `1 / tau = 14.286` at the start).

Work it with `tau = 0.1` (round numbers). The pixel was planted with metal spectrum e1. Cosines:
metal e1 0.95, e2 0.70, e3 0.60; plastic e4 0.75; oil e5 0.20. Divide by 0.1: 9.5, 7.0, 6.0, 7.5, 2.0.

```
with e1 in the library:
  metal   = log(e^9.5 + e^7 + e^6) = 9.6064;   plastic = 7.5;   oil = 2.0
  P = softmax(9.6064, 7.5, 2.0) = (0.8911, 0.1084, 0.0004)       CE = -ln 0.8911 = 0.1153
with e1 left out (exclude = 0):
  metal   = log(e^7 + e^6) = 7.3133;   plastic = 7.5;   oil = 2.0
  P = (0.4524, 0.5453, 0.0022)                                    CE = -ln 0.4524 = 0.7931
```

With its own spectrum present, the pixel is "metal" at 0.89 because it recognises e1, the sample it was
made from. With e1 left out, it must rely on the *other* metals, and here plastic wins. **That is the
point of leaving the planted spectrum out: the only way to get a low loss is to put all metals near each
other, which is learning the material, not the sample.** It is the separability check's
leave-one-spectrum-out (`ace[e] = -1`), used as a training rule. In code, `exclude=s["entry"]` sets
that entry's score to minus infinity before the logsumexp; a test checks that a class whose only entry
is excluded scores minus infinity (`test_class_scores_leave_out_the_planted_entry`).

A second way to see the class score: the softmax over classes equals the per-entry softmax summed over
each class's entries. Over e2..e5 the per-entry softmax is (0.3308, 0.1217, 0.5453, 0.0022);
`0.3308 + 0.1217 = 0.4525`, the metal 0.4524 above (rounding). So the loss counts *any* metal as the
right answer - exactly the remedy the step-by-step book proposed for "false negatives" (two spectra of one
material pushed apart).

Which library spectra take part? In the seven class scores, only **train-split man-made** entries
(`train_lib = bank.manmade & ~bank.test`, minus an open-set class). Under `"level1": "matching"` the
natural spectra take part too, as an eighth column that only level 1 reads (Step 5).

## Step 5 - Level 1 by matching

**The problem.** Level 1 asks "natural or man-made?". In N001 it was answered by `to_manmade`, one
linear layer on `h`: a weight vector of 256 numbers and a bias, `logit = w . h + b0`. That is one flat
cut through a 256-number space, with all of nature on one side. N001's README records what happened
on the H100 after epoch 1: "Level 2 learned (class accuracy 0.60-0.68 at fill 0.25-1.0 on all three
sensors) but **level 1 sat at chance**: accuracy 0.49-0.52 at every fill, 0.50 even at fill 1.0, where
ACE gets 0.87; man-made recall at ACE's false-alarm rate 0.21-0.30 (random = the FA rate, ~0.15-0.2).
Train BCE 0.684." At fill 1.0 a planted man-made pixel *is* a library spectrum, and level 2 named its
class two times in three, so the embedding carried the information. The linear head could not read
"man-made" out of it.

The README's diagnosis compares the two forms on a laptop (EnMAP smoke shards, 150 steps at fill ~1):

| level 1 | BCE | accuracy | man-made recall at ACE's FA |
|---|---|---|---|
| chance | `ln 2 = 0.6931` | 0.50 | - |
| linear head on `h` | 0.64 | 0.65 | 0.41 |
| by matching | 0.43 | 0.80 | 0.74 |

Its explanation: "The class scores have library prototypes; the bare head had to learn all of nature in
one linear layer." N001 was stopped and rerun as N001b with one change, `"level1": "matching"`
(`configs/namer/N001b.json`; `configs/namer/N002.json` sets it too; `N001.json` has no `level1` key, so
`train.py`'s default `"head"` applies, the linear head).

**The rule.** In `train.py` (`forward`), the natural library entries become one more class, index
`C = 7`, and the eight-column score table is split into level 1 and level 2:

```python
lib_cls = torch.where(bank.manmade, lib_cls, torch.full_like(lib_cls, C))     # natural -> class 7
scores = self.model.class_scores(z.float(), e.float(), lib_cls, C + 1, exclude=s["entry"])   # (P, 8)
logit, scores = self.model.level1_logit(scores), scores[:, :C]               # level 1, level 2
```

and `model.level1_logit` is one line:

```python
return torch.logsumexp(scores[:, :-1], dim=1) - scores[:, -1] + self.l1_bias
```

Each column of `scores` is already a class score from Step 4: the logsumexp, over that class's entries,
of `cos(z, e_j) / tau`. So

```
logit       = logsumexp(S_0, ..., S_6) - S_natural + l1_bias
P(man-made) = sigmoid(logit) = 1 / (1 + e^-logit)
```

`l1_bias` is one learned number, started at 0 (`nn.Parameter(torch.zeros(()))`). Level 2 is the first
seven columns, exactly as before. The planted spectrum is left out of the table whether it is man-made or
natural, so a natural mix cannot be recognised as its own spectrum either. `infer.py` does the same
when the checkpoint's config says `"level1": "matching"` (`load_namer` reads it, default `"head"`, and
accepts an older checkpoint with no `l1_bias`).

**Work one by hand.** Six library entries, scores `s_j` already divided by `tau`; `l1_bias = 0`. Take a
pen; it takes three minutes.

| entry | class | s_j | e^s_j |
|---|---|---|---|
| e1 | plastic | 6.0 | 403.4288 |
| e2 | plastic | 5.0 | 148.4132 |
| e3 | metal | 4.0 | 54.5982 |
| e4 | metal | 3.0 | 20.0855 |
| e5 | natural (soil) | 5.5 | 244.6919 |
| e6 | natural (grass) | 4.5 | 90.0171 |

```
S_plastic = log(e^6.0 + e^5.0) = 6.0 + log(1 + e^-1) = 6.0 + log 1.3679 = 6.0 + 0.3133 = 6.3133
S_metal   = log(e^4.0 + e^3.0) = 4.0 + log(1 + e^-1) = 4.0 + 0.3133          = 4.3133
S_natural = log(e^5.5 + e^4.5) = 5.5 + log(1 + e^-1) = 5.5 + 0.3133          = 5.8133

man-made  = log(e^6.3133 + e^4.3133) = 6.3133 + log(1 + e^-2)
          = 6.3133 + log 1.1353 = 6.3133 + 0.1269                             = 6.4402
logit     = 6.4402 - 5.8133 + 0                                               = 0.6269
e^-0.6269 = 0.5342
P(man-made) = 1 / (1 + 0.5342) = 1 / 1.5342                                   = 0.6518
```

Check it a second way. A logsumexp of logsumexps is the logsumexp of all the entries, and the sigmoid of
a log-ratio is a share, so P(man-made) is the share of the pixel's softmax weight that falls on man-made
entries:

```
man-made weight  403.4288 + 148.4132 + 54.5982 + 20.0855 = 626.5257
natural weight   244.6919 + 90.0171                       = 334.7090
share            626.5257 / (626.5257 + 334.7090) = 626.5257 / 961.2347 = 0.6518
```

If the truth is man-made, the level-1 BCE is `-ln 0.6518 = 0.4280`.

**Why it helps.** Three reasons; the table above is consistent with them but does not prove them.

1. **One cut cannot hold all of nature on one side.** Vegetation, soils, rocks, water and snow sit in
   different places, with man-made materials between them (a green tarpaulin near vegetation, concrete
   near rock). Matching is a nearest-shelf rule with one natural prototype per natural spectrum (219 on
   EnMAP's bank, `compute_namer_full_output.txt` section 16) instead of one half-space.
2. **Both levels read the one cosine table.** Whatever arrangement of embeddings helps level 2 now helps
   level 1, and the natural entries get a training signal they never had: in N001 nothing pulled a pixel
   towards a natural spectrum.
3. **A new library entry changes level 1 without retraining.** Add a natural spectrum and it becomes a
   new natural prototype the moment it is encoded, exactly as a new man-made one joins its class. The
   linear head's idea of nature was frozen in its 256 weights.

**One built-in offset, and why `l1_bias` exists.** The logsumexp of `m` equal scores is `s + ln m`, so
the side with more entries wins when nothing distinguishes them. EnMAP's training bank has 219 natural
entries and `159 - 34 = 125` train man-made: `ln 219 - ln 125 = 5.3891 - 4.8283 = 0.5608` towards
"natural". `l1_bias` can learn it away.

**Does it work?** N001b's README, epoch 1, fill 0.25: "level-1 accuracy 0.69 / 0.71 / 0.68 (EnMAP /
PRISMA / AVIRIS-NG; N001 0.52 / 0.51 / 0.50), man-made recall at ACE's false-alarm rate 0.57 / 0.57 /
0.55 (ACE 0.66 / 0.77 / 0.76)". Off chance, then, but below ACE after one epoch of forty; "Running; 40
epochs" is where the README stops.

**What is left unused.** `to_manmade` (257 parameters) is still built and `forward_pixel` still returns
its output, which `forward` then overwrites. Do not delete it to "tidy up": N001's checkpoints contain
it, and `load_namer` would refuse a checkpoint with an unexpected key. The proposal's design body
("Level 1 does not use the library at all") and the publication schematic in
`research/experiments/spectral-namer/figures/` still describe the old head; the proposal's "As built"
list records the change ("Level 1 by matching ... Runs N001b, N002"). The code and
`configs/namer/N001b.json` are what runs.

## Step 6 - The loss

```
loss = BCE(level-1 logit, man-made)                         every sample
     + CE(class scores, class)                              man-made samples, own spectrum left out
     + fill_weight * L1(fill, a)                            every sample
```

(`train.py`, docstring and `loss`; `fill_weight = 0.5` in N001, N001b and N002.) The level-1 logit is
whichever `forward` produced: `level1_logit` under matching (Step 5), `to_manmade`'s output under the
old `"head"` setting. The loss line is the same for both. **BCE**, binary cross-entropy, is the
loss for a yes/no answer; **CE**, cross-entropy, the one for a choice among several (minus the log of
the right answer's probability); **L1** is the mean absolute difference.

The **fill L1 is on every sample**, natural mixes and plain pixels included (plain target 0). The
proposal said "man-made samples only"; its As-built list records the change. The effect: the fill head
also learns to say "nothing was added" (a = 0) for a plain pixel, which is level-1 evidence.

Two chance levels to keep: a level 1 that knows nothing has BCE `ln 2 = 0.6931`; a class score
that knows nothing, over 7 classes, has CE `ln 7 = 1.9459`. The CPU smoke test (N001 README: EnMAP +
SpectralEarth smoke shards, 120 steps, linear head) moved class CE from 1.81 to 1.02 and left BCE "still
~0.68 (too short to judge)". In hindsight that was the first sign of Step 5's problem: N001's train BCE
after a full epoch on the H100 was 0.684, still at chance.

The optimiser is AdamW at 3e-4, with the stem and its BatchNorm at `stem_lr_scale = 0.3` of that
(9e-5), 500 warm-up steps and a cosine decay over `40 x 500 = 20,000` steps; gradients are clipped to
norm 1. Each step takes one batch of 16 tiles per sensor and 512 pixels per tile, so up to
`3 x 16 x 512 = 24,576` samples per step and about 491.5 million over the run.

## Step 7 - The splits: held-out spectra and a held-out class

**The 20% hash split** (`bank._is_test`): a man-made entry is "test" if the first 8 hex digits of the
SHA-1 hash of its name, read as a number and divided by `0xFFFFFFFF` (4,294,967,295), fall below
`TEST_SHARE = 0.2`. Test spectra are never planted in training and never in the training class scores.

```
"Copper Metal"     sha1[:8] = 1600007b = 369,098,875    / 4,294,967,295 = 0.0859  -> test
"Aluminum Metal"   sha1[:8] = d13908a4 = 3,510,175,908  / 4,294,967,295 = 0.8173  -> train
```

A hash, not a random draw, so every band list (EnMAP, PRISMA, AVIRIS-NG) agrees on the split without
sharing a file.

**Repeated names, found and fixed.** Some library names repeat ("Copper Metal" three times, "Galvanized
Steel Metal" twice - the same name from different source files). The hash sent repeats to the same side,
so the test split was clean; but in training, leaving out one "Galvanized Steel Metal" left its namesake in
the class score, a near-copy of the planted spectrum that defeats leaving it out. Since the N001 code
bundle (`namer2`), `build_bank` keeps only the first entry per name (`bank.py`: "Repeated names keep their
first entry only"). On the EnMAP band list the bank now holds **378** entries, **159** man-made, **34**
held out (it was 408 / 174 / 41 before the fix, the figures in the N001 README); on AVIRIS-NG 380 / 160
/ 34. Reproduced with `build_bank` on `enmap_224.json`.

**The share is about 20% overall, not per class.** The hash ignores the class. On EnMAP: oil 8 of 40
(20%), plastic 8 of 46 (17%), textile 3 of 14 (21%), wood/paper 2 of 11 (18%), paint 1 of 4 (25%), metal 3
of 10 (30%), concrete/asphalt 9 of 34 (26%); overall 34 / 159 = 21.4%. `bank.py`'s docstring now says this,
with "17-38% per class on EnMAP" - the range before de-duplication; with one entry per name it is 17-30%.
Notice how small some classes became: paint has 4 spectra, which is why the open-set run below holds
out textile instead.

**The held-out class (N002).** `configs/namer/N002.json` sets `"holdout_class": "textile_fabric"` and
`"level1": "matching"`; its description reads "Textile replaced paint (4 spectra after de-duplication),
user 2026-10-31", and textile has 14. (An earlier version of this part said `paint_coatings`; the config
wins.) On EnMAP's band list that leaves a train man-made pool of 114 and a test pool of 45, textile's
14 included (`compute_namer_full_output.txt`, section 16). The whole class is never planted in training and is not a candidate; at evaluation it is planted and the
trainer records how often level 1 still says man-made (`open_mm`) and how often it gets a confident
(P > 0.5) known class (`open_wrong`). Pass (proposal): man-made at least 70%, confident wrong at most 20%.

**The classes.** `classes.py` maps every family-library entry: natural chapters -> `natural`; building and
pavement -> `concrete_asphalt`; roofing dissolved by name (fiberglass / rubber / clear roof -> plastic;
cedar / shake / wood -> wood_paper; metal / tin / steel / aluminium -> metal; tar / felt / asphalt /
shingle / tile -> concrete_asphalt; anything else -> concrete_asphalt); glass left out ("cannot be
tested"). On the shipped library no roofing entry maps to metal: there is no metal roofing spectrum.

## Step 8 - The test it must pass

Each epoch, `evaluate` plants **test-split** spectra into **test** shards at four fixed fills (0.1,
0.25, 0.5, 1.0), half man-made and half natural mix (`kinds=(0.5, 0.5, 0.0)`), with a fixed seed. It
records level-1 accuracy, level-2 accuracy, and the pass number: **man-made recall at ACE's natural
false-alarm rate**.

Why that number? A level-1 score can buy recall by calling more things man-made. To compare fairly with
ACE, hold the false alarms equal: read ACE's natural false-alarm rate from
`separability_<sensor>.json` (`1 - level1_natural_called_natural`), set the namer's cut so it calls the
same share of natural plantings man-made, then count man-made plantings above the cut:

```python
cut = torch.quantile(torch.cat(nat_logits), 1 - fa)        # fa: ACE's natural false-alarm rate
recall = (torch.cat(mm_logits) > cut).float().mean()
```

A toy. Ten natural logits, sorted: -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2; `fa = 0.2`. The 0.8
quantile sits at position `0.8 x (10 - 1) = 7.2`, between 0.5 and 1.0: `0.5 + 0.2 x (1.0 - 0.5) = 0.6`.
Five man-made logits -1, 0.4, 0.7, 1.5, 3: above 0.6 are 0.7, 1.5, 3, so recall is `3 / 5 = 0.6`.

The real bars, at fill 0.25 (from the JSONs; the pass rule is "ACE + 0.10"):

| sensor | ACE natural right | ACE false alarms | ACE man-made recall | namer needs |
|---|---|---|---|---|
| PRISMA | 0.8342 | 0.1658 | 0.7724 | >= 0.8724 |
| EnMAP | 0.7851 | 0.2149 | 0.6635 | >= 0.7635 |
| AVIRIS-NG | 0.8134 | 0.1866 | 0.7603 | >= 0.8603 |

Level 2 must also beat ACE by 0.10. **But not against these JSONs' level-2 numbers**: the N001 README
notes that ACE's level-2 numbers were measured on the 17 old families and "must be re-scored on the v1
classes ... before comparing". N001b's README has that re-score ("ACE baseline on the v1 classes"): at
fill 0.25, EnMAP 0.47 (0.70), PRISMA 0.60 (0.78), AVIRIS-NG 0.58 (0.76), the figure in brackets counting
only spectra ACE also called man-made. The README says the fair comparison "lies between the two ACE
figures". `best.pt` is chosen on mean level-2 accuracy at
fill 0.25, and every epoch also writes `ep<n>.pt`.

Three things the trainer does **not** do, though the proposal lists them: it computes no bootstrap
interval ("interval above zero" is a later step), it has no "unknown" output (level 2 always returns
its best class with `p_cls`; "class unknown" is a threshold still to be set), and natural spectra are
planted in both splits (As built: "level-1 natural recall is not an unseen-spectrum test").

## Step 9 - Using a checkpoint

`infer.name_pixels(model, cube, wavelengths, pixels, library_root, cache_dir)` names the pixels of a mask
(for example a committed `anomaly_mask.tif`): it drops bands invalid at more than half of them, computes
their ring means, encodes the **whole** library at the scene's bands - test spectra included, since at
inference every spectrum is fair game - and returns per pixel `p_manmade`, the best class, `p_cls` and
`fill`. Level 1 follows the checkpoint: `load_namer` reads `config["level1"]` (default `"head"`), and for a
matching checkpoint `name_pixels` gives the natural entries class 7 and calls `level1_logit`, exactly as
in training. So a new library, natural spectra included, changes both levels: it is encoded, not trained. The AIGC object table (part 28) is the first caller.

---

## Step 10 - The namer in the product

Since 2026-11-01 the namer is the first stage of the `spectral_library_match` action (code:
`backend/allotrope/action_types/_spectral_namer_run.py`, `app/spectral_match/namer/objects.py`; design:
`docs/lld/spectral-namer-in-product.md`). It changes what the user sees and what the classical matchers do.

1. **Level 1 on every anomaly pixel**, then pixels are grouped into **objects** (8-connected clusters of the committed
   mask). An object's P(man-made) is the mean over its pixels; it is called man-made when that mean is above 0.5.
2. **Level 2 only for man-made objects:** the class with the highest mean P(class | man-made) over the object's
   pixels, shown as **"closest known class"**, and flagged *confident* when that mean is at least 0.55, otherwise
   *tentative*. The wording is deliberate: run N002 showed an unseen material gets a confident wrong class about half
   the time, so a class name is a best match, not an identification.
3. **SAM and the family match run only on man-made objects' pixels.** Natural pixels are labelled natural (code -3
   in the match rasters) and are not matched.

Work one object by hand. An object of 3 pixels has P(man-made) 0.9, 0.7 and 0.5, so its mean is
(0.9 + 0.7 + 0.5) / 3 = 2.1 / 3 = 0.70: man-made. Its pixels' P(plastic | man-made) are 0.6, 0.5 and 0.7, mean
1.8 / 3 = 0.60, the highest class: closest known class plastic, 0.60 >= 0.55, confident. Had they been 0.5, 0.5 and
0.6 (mean 0.53), it would read "plastic, tentative".

**Why 0.55:** at 90% coverage of the known classes, the P(class) threshold on held-out spectra was 0.49-0.61 at
fills 0.1-0.25 (`research/experiments/spectral-namer/N002/README.md`, calibration table). Larger fills give peakier
probabilities, so one value is a compromise aimed at the small fills the detector mostly finds.

**What it did on two real scenes** (local stack, 2026-11-01):
- PRISMA, p95 mask: 3,518 objects, 292 man-made (mostly concrete/asphalt over the town, oil 62), 3,226 natural
  (shoreline and terrain false alarms).
- Tanager (Permian Basin): 304 objects, 82 man-made (well pads: plastic 51, oil 21), 222 natural (the large ponds).
- On PRISMA, skipping natural pixels cut the action from 24.4 s to 18.9 s. On a small mask the namer's ~8 s costs
  more than it saves.

**In the UI:**
- The match card shows the man-made (orange) / natural (teal) map.
- The detail view opens on it, with a Materials toggle and an objects list (click to zoom).
- The matching export bundle adds the level-1 GeoTIFF, the objects table and `nm_*` attributes.
- The boolean submission mask is unchanged.

**Swapping the model** is a manifest edit: `/models/spectral_namer/current.json` points at the checkpoint (N001b
epoch 40 today).

## Common confusions

**"The namer reuses Indradhanu V2, so it detects anomalies."** It reuses V2's *stem* (17,825 of its
347,685 parameters) as a spectral feature extractor. It names pixels something else has already flagged.

**"Logsumexp is the same as taking the best cosine."** It is a smooth maximum: `log(e^7 + e^6) = 7.31`,
not 7. Every member of the class contributes, the best ones most.

**"Leaving the planted spectrum out is only for evaluation."** In this code it is in training too
(`exclude=s["entry"]` in `forward`, used by both `train` and `evaluate`). That is the main departure from
the proposal's per-spectrum InfoNCE.

**"The ring is 2 to 7 pixels away."** Beyond 2: the ring is 3 to 7 steps away (15 x 15 minus 5 x 5, 200
pixels in the interior), as the generator's docstring now says.

**"Level 1 is a model of nature."** That was the design (part 26). The built level 1 is a comparison of
two shelves of library spectra, man-made against natural, trained by BCE on planted samples. N002
measures how open it still is.

**"Level 1 is a separate head on the pixel tower."** It was, in N001 (`to_manmade`). Since N001b it is
`logsumexp(man-made class scores) - natural score + l1_bias`, read from the same score table as level
2. `to_manmade` is still in the model and still in the parameter count, but under matching nothing
trains it and nothing reads it.

**"Matching level 1 is the max of the man-made cosines against the max of the natural ones."** Smooth
maxima, not maxima, and over every entry: in Step 5, man-made pools to 6.4402, not 6.0, and the result
equals the man-made share of the whole softmax, 0.6518. More natural entries raise the natural side by
themselves (`s + ln m` for `m` equal scores), which is one reason `l1_bias` exists.

**"347,684 parameters."** That was the count before 2026-10-31. `l1_bias` adds one: 347,685.

**"20% of every class is held out."** About 20% of the man-made spectra overall (34 of 159 on EnMAP); per
class it ranges from 17% to 30%.

**"N001 passed / failed."** Neither: it was stopped after epoch 1 because level 1 sat at chance (accuracy
0.49-0.52), which is a failure of the linear head, not a verdict on the namer. N001b, its replacement, had
reported epoch 1 only when this was written.

## Check yourself

1. In the ring of Step 2, replace the halo value 0.35 by 0.60 and the target by 0.90. What is the ring
   mean now? Why?
2. With `a_min = 0.05`, what share of man-made samples has fill at most 0.1? Show the logarithms.
3. Plant `t = (0.30, 0.10)` into `b = (0.20, 0.40)` at `a = 0.5` with `g = (1, 1)` and no noise. Then do it
   with `g = (1.1, 0.9)`.
4. In Step 4, suppose metal had only e1 and e2. With e1 left out, what are metal's score and P(metal)?
   (Plastic 7.5, oil 2.0.)
5. A horizontal row reads 0.30, 0.30, 0.30, 0.34. Compute sigma as the code does. Is the 0.34 noise?
6. At PRISMA fill 0.25, the namer calls 16.58% of natural plantings man-made at its cut and 85% of
   man-made plantings man-made. Does level 1 pass?
7. Why is the hash split computed from the spectrum's name rather than drawn at random?
8. Take Step 5's six-entry library and add a seventh entry, a natural spectrum (dry grass) with
   `s = 6.0`, with no retraining. Compute the new natural score, the logit and P(man-made) by hand
   (`l1_bias = 0`). What does this show about level 1 by matching that the linear head could not do?
9. In N001, level 2 named the class correctly 0.60-0.68 of the time while level 1 sat at 0.49-0.52.
   Why does that pairing say the information was in the embedding, and which single setting changed
   in N001b?
10. How many parameters would the namer have if `to_manmade` were deleted? Why is it not deleted?

<details>
<summary>Answers</summary>

1. Still 0.20. The halo and the target lie within 2 steps of the centre, so they are in both the 15 x 15
   and the 5 x 5 box sums and cancel in the difference; the 200 ring pixels are all 0.20.
2. `P(a <= 0.1) = ln(0.1 / 0.05) / ln(1 / 0.05) = ln 2 / ln 20 = 0.6931 / 2.9957 = 0.2314`, about 23%.
3. `a t = (0.15, 0.05)`; `(1 - a) b = (0.10, 0.20)`; `x = (0.25, 0.25)`. With the gain:
   `(1.1 x 0.25, 0.9 x 0.25) = (0.275, 0.225)`.
4. Metal's only remaining entry is e2: score 7.0 (logsumexp of one term is the term). Softmax of
   (7.0, 7.5, 2.0): `e^7 = 1096.63`, `e^7.5 = 1808.04`, `e^2 = 7.39`; sum `2912.06`; P(metal) =
   `1096.63 / 2912.06 = 0.3766`.
5. Differences 0, 0, 0.04; squares 0, 0, 0.0016; mean `0.0016 / 3 = 0.000533`; RMS `0.02309`; sigma
   `0.02309 / 1.4142 = 0.01633`. The 0.34 may be a real edge, not noise - the RMS cannot tell, which is
   why it over-estimates noise on textured tiles.
6. Its false alarms equal ACE's (0.1658), so the comparison is fair. It needs 0.7724 + 0.10 = 0.8724;
   0.85 is below, so level 1 does not pass on PRISMA.
7. So that every band list - and every run, on any machine - holds out the same spectra without sharing a
   file; and repeated names landed on the same side, so a copy of a test spectrum could not sit in
   training (since the `namer2` fix only the first entry of a name is kept at all).
8. The man-made side is unchanged, 6.4402. The natural side:
   `e^5.5 + e^4.5 + e^6.0 = 244.6919 + 90.0171 + 403.4288 = 738.1378`;
   `S_natural = log 738.1378 = 6.6041`. Logit `6.4402 - 6.6041 + 0 = -0.1639`.
   `e^0.1639 = 1.1781`; `P(man-made) = 1 / (1 + 1.1781) = 1 / 2.1781 = 0.4591`. Check as a share:
   `626.5257 / (626.5257 + 738.1378) = 626.5257 / 1364.6635 = 0.4591`. The pixel flipped from man-made
   (0.6518) to natural because one library spectrum was added: level 1 by matching changes with the
   library, without training. The linear head's answer depended only on its 256 weights and bias.
9. At fill 1.0 a planted man-made pixel is (nearly) a library spectrum, and level 2 could name its class
   from the embedding two times in three, so the embedding separated man-made spectra well; one linear
   cut on `h` still could not separate them from all of nature. N001b changed only `"level1": "head"`
   (the default, which N001 used by leaving the key out) to `"level1": "matching"`.
10. `347,685 - 257 = 347,428`. It is kept because N001's checkpoints (and N001b's, which also contain it)
    would then carry an unexpected key, and `load_namer` raises on unexpected keys; the `"head"` setting
    also still uses it.

</details>

---

Next: part 28, producing the AIGC Stage-III submission - a boolean map and a shapefile per result, the
PRISMA grid fixed, native bands by default, and the per-object table.
