# 67 - The progressive decoder (FT09)

> **The one thing this part teaches:** FT09 replaced FT04's decoder with one that upsamples in two x2
> stages and is fed a full-resolution copy of the stem output with every hidden pixel zeroed; trained
> in two phases (decoder alone, then joint), it raised hidden-pixel contrast on all four PRISMA scenes
> and gave the first pAUC gain since cloud exclusion whose bootstrap interval clears zero - by a hair,
> with a small unresolved loss at the top 0.1%, so it is a candidate, not a release.

## Where we are

**Status (2026-10-22): FT09b done 16:06 UTC, best checkpoint = its last epoch (5). Criteria 1, 3 and
4 met; 2 fails at the top 0.1% by 4 objects (unresolved); 5 failed as written. Not adopted.
Indradhanu V2 (FT04, part 61) is unchanged. FT09c - 5 more joint epochs from FT09b - is running as
this is written; it has no results yet.**

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-finetune/FT09-decoder/README.md` | question, setup, pass criteria, all four runs' results, observations |
| `research/experiments/indradhanu-finetune/FT09a-decoder-warmup/`, `FT09b-decoder-joint/`, `FT09c-decoder-joint-more/` | each run's `config.json`; FT09a and FT09b also `finetune_metrics.csv`; FT09b `eval_report.md`, `eval_per_scene.csv` |
| `app/foundation_models/components/progressive_decoder.py` | `ProgressiveDecoder` |
| `app/foundation_models/components/spectral_stem_seg_former_mae.py` | `decoder_type`, and `_decode`, which zeroes hidden pixels in the skip |
| `app/foundation_models/trainers/finetune_trainer.py` | the warm start that drops `refine.*`, and `mode: "decoder"` |
| `tests/test_foundation_models/test_progressive_decoder.py` | the no-leak test and four others |

> **Analogy.** A muralist works from a sketch drawn at quarter size. FT04's decoder hands the sketch to
> an assistant who blows up each square of it into a 4 x 4 patch of wall, in one go. FT09's decoder
> enlarges the sketch to half size and touches it up with a half-size photo of the wall, then to full
> size and touches it up again with the full-size photo. The parts of the wall the muralist is meant
> to paint from imagination are taped over in both photos.

## The architecture

Part 66 asked for two things: upsampling in stages, so detail is rendered gradually, and a
**full-resolution skip** - a second input that bypasses the encoder and arrives at the decoder at full
size. Note the word: this is not part 15's skip (residual) connection inside a block. It is a U-Net-style
skip, from the start of the network to the end.

The skip is `z`, the stem output after BatchNorm: 32 numbers per pixel. `ProgressiveDecoder.forward`
for a 128 px tile, batch `B`, with FT09's widths (`decoder_dim = 256`, `skip_channels = 32`,
`stage_dims = (192, 96)`). Shapes were checked by building the model on 2026-10-22:

| step | operation | shape |
|---|---|---|
| in | encoder stages F1..F4 | (B, 32, 32, 32), (B, 64, 16, 16), (B, 160, 8, 8), (B, 256, 4, 4) |
| in | skip `z`, hidden pixels zeroed | (B, 32, 128, 128) |
| 1/4 | 1x1 projections to 256, bilinear up to 32 x 32, concat | (B, 1024, 32, 32) |
| 1/4 | 1x1 fuse + GELU (SegFormer's, warm-started) | (B, 256, 32, 32) |
| 1/2 | bilinear x2 | (B, 256, 64, 64) |
| 1/2 | concat `avg_pool2d(skip, 2)` | (B, 256 + 32 = 288, 64, 64) |
| 1/2 | `up_half`: 3x3 conv, GELU, 3x3 conv, GELU | (B, 192, 64, 64) |
| 1 | bilinear x2 | (B, 192, 128, 128) |
| 1 | concat skip | (B, 192 + 32 = 224, 128, 128) |
| 1 | `up_full`: 3x3 conv, GELU, 3x3 conv, GELU | (B, 96, 128, 128) |
| 1 | `head`: 1x1 conv | (B, 32, 128, 128) |
| out | `spectral_out` (part 40), unchanged | (B, N, 128, 128) |

Compare FT04's: after the fuse, one 3x3 conv, a 3x3 conv to `32 x 16 = 512` channels, and
PixelShuffle(4). Nothing at full resolution enters.

### By hand: the parameter count

Every convolution has `in x out x k x k + out` parameters (part 08).

```
projections   (32 + 64 + 160 + 256) x 256 + 4 x 256 = 131,072 + 1,024        =   132,096
fuse          1,024 x 256 + 256                                              =   262,400
up_half 1     288 x 192 x 9 + 192 = 497,664 + 192                            =   497,856
up_half 2     192 x 192 x 9 + 192 = 331,776 + 192                            =   331,968
up_full 1     224 x  96 x 9 +  96 = 193,536 +  96                            =   193,632
up_full 2      96 x  96 x 9 +  96 =  82,944 +  96                            =    83,040
head           96 x  32     +  32 =   3,072 +  32                            =     3,104
                                                                     total     1,504,096
```

Building the model and counting gives 1,504,096, and the source comment says "1.50 M with FT04 widths
(FT04's decoder: 2.16 M)". **The new decoder has fewer parameters than the old one.** It is not bigger;
it is wired differently. It does far more arithmetic, though: `up_full`'s first convolution alone runs
its 193,536 multiply-adds at each of `128 x 128 = 16,384` pixels, about 3.2 billion per tile, more than
FT04's two refine convolutions together at 32 x 32 (`(589,824 + 1,179,648) x 1,024`, about 1.8 billion).
Those two figures are hand counts, not timings.

## The skip must not carry hidden pixels

This is the trap in the whole design. The masked autoencoder works because a hidden pixel's value
never reaches its own reconstruction (part 06). The encoder guarantees that by removing hidden tokens
(part 19). A skip that bypasses the encoder would undo it: hand the decoder the real pixel and it will
copy it, the residual vanishes, and every target with it.

So `_decode` (`spectral_stem_seg_former_mae.py`) zeroes them before the decoder sees `z`:

```python
visible = keep_mask.reshape(B, 1, H // 4, W // 4).to(z.dtype)    # (B, 1, 32, 32) for 128 px: 1 = visible token
z = z * F.interpolate(visible, scale_factor=4, mode="nearest")   # (B, 32, 128, 128): each token's 4 x 4 pixels
return self.decoder(features, z)
```

Nearest-neighbour interpolation by 4 copies each token's 0 or 1 to its 16 pixels. The half-resolution
skip is `avg_pool2d(skip, 2)`, a 2 x 2 average; since 2 divides 4 and the blocks line up with token
borders, each average stays inside one token, so a hidden token is 0 at both resolutions. The module
docstring says what *does* reach a hidden pixel: "Neighbouring visible pixels still reach a hidden pixel
through the 3 x 3 convolutions - that is context, as in the encoder."

Without a keep mask (`keep_mask is None`, an unmasked forward) nothing is zeroed.

### How the test proves it

`test_hidden_pixels_never_reach_the_output` builds a small model and a 64 px tile:
`64 / 4 = 16` tokens per side, `16 x 16 = 256` tokens. It hides tokens 0, 17, 100 and 255, then makes a
second input `y` in which **only the pixels of those four tokens** are replaced with fresh random
values. It asserts that the reconstruction of `x` and of `y` are equal everywhere, to `1e-6` - through
`forward` and through `forward_mixed` (part 52).

The argument: if any hidden value leaked anywhere - through the skip, or through the encoder - some
output pixel would differ between `x` and `y`. None does. Its partner, `test_visible_pixels_do_reach_the_output`,
shows the skip is live: add 0.5 to one visible pixel and that pixel's own reconstruction moves by more
than `1e-4`. Together they say: visible values reach the output, hidden ones never do.

Where are the hidden tokens? Token `t` is at row `t // 16`, column `t % 16` (`divmod(t, 16)`), and covers
pixels `4 x row` to `4 x row + 3`:

```
token  17   divmod(17, 16) = (1, 1)     rows 4-7,   columns 4-7
token 100   divmod(100, 16) = (6, 4)    rows 24-27, columns 16-19
```

## The warm start

FT09 starts from FT04's `best.pt`. The new decoder kept the old one's layer names, `linear_projections`
and `fuse_conv`, on purpose, so those load directly. `build_model` in `finetune_trainer.py` detects a
SegFormer-decoder checkpoint (no `decoder.up_` key) and drops what does not fit:

```python
# from a SegFormer-decoder checkpoint (FT04): keep its projections and fuse, drop its PixelShuffle refine
state = {k: v for k, v in state.items() if not k.startswith("decoder.refine.")}
new_heads += ("decoder.up_", "decoder.head.")
```

The encoder, stem, `spectral_out` and segmentation head load whole. `test_warm_start_from_a_segformer_decoder_checkpoint`
checks that the only missing tensors start with `decoder.up_` or `decoder.head.` and that `fuse_conv`'s
weights arrive unchanged.

How much of the decoder is warm? From the count above, `132,096 + 262,400 = 394,496` of 1,504,096, that
is `394,496 / 1,504,096 = 0.26`. **Three quarters of the new decoder starts random.**

## Why joint-from-scratch lagged, and the two-phase fix

The first attempt trained jointly from the start (encoder and decoder together, FT04's mix, lr 1e-4
cosine, planned 8 epochs) and was stopped after epoch 3:

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.97 / 6.37 / 8.29 | 0.841 / 0.817 |
| 2 | 4.69 / 5.11 / 6.45 | 0.840 / 0.823 |
| 3 | 4.43 / 4.97 / 6.47 | 0.843 / 0.826 |

The README's reading: a mostly random decoder had `3 x 30,000 = 90,000` samples to learn in, against
the millions behind FT04's, on a learning rate already decaying; and while it was still bad its
gradients flowed into the encoder and pushed a good encoder around to suit a bad decoder. AVIRIS-NG was
flat from epoch 2. (The masks improved - cloud 0.826 - since the segmentation head shares the encoder
but not the decoder.)

The fix splits training in two:

| run | `mode` | what trains | lr | epochs | starts from |
|---|---|---|---|---|---|
| FT09a | `"decoder"` | `decoder.*` only; everything else frozen in eval mode | 1e-3 | 3 | FT04 `best.pt` |
| FT09b | `"joint"` | everything | 5e-5 | 5 | FT09a `best.pt` |
| FT09c (running) | `"joint"` | everything | 3e-5 | 5 | FT09b `best.pt` |

`mode: "decoder"` reuses the frozen-backbone machinery of parts 49 and 63: `requires_grad` is set only
for parameters whose names start with `decoder.`, and `self.model.eval()` is followed by
`self.model.decoder.train()`, so the frozen parts keep FT04's BatchNorm statistics (part 51). With
`spectral_out` frozen, the decoder is in effect learning to emit FT04's own 32-number code `z` for each
pixel. The segmentation loss is off in this mode (`_seg_batch` runs only for `probe` and `joint`), which
is why FT09a's IoUs never move.

FT09a, from `FT09a-decoder-warmup/finetune_metrics.csv`:

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.46 / 4.83 / 6.00 | 0.839 / 0.810 (frozen) |
| 2 | 4.17 / 4.54 / 5.83 | 0.839 / 0.810 |
| 3 | 3.98 / 4.14 / 5.50 | 0.839 / 0.810 |

**The decoder alone reached FT04's reconstruction - within 0.1 deg on all three sensors - in 3 epochs,**
where the joint attempt was still 0.5-1 deg behind after 3.

## Did it do what it was built for? The contrast table

Part 66's `visible_vs_hidden.py`, rerun with `--checkpoint` on FT09a. Mean absolute error per band,
normalised:

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| hidden background FT04 -> FT09a | 0.057 -> **0.051** | 0.137 -> **0.125** | 0.129 -> **0.117** | 0.139 -> **0.110** |
| hidden targets FT04 -> FT09a | 0.218 -> 0.204 | 0.220 -> 0.203 | 0.179 -> 0.171 | 0.191 -> 0.162 |
| target / background, FT04 -> FT09a | 3.82 -> **4.00** | 1.61 -> **1.62** | 1.39 -> **1.46** | 1.37 -> **1.47** |
| visible background (no mask) FT04 -> FT09a | 0.056 -> 0.176 | 0.110 -> 0.120 | 0.159 -> 0.120 | 0.135 -> 0.115 |

Work set1's background change by hand: `0.057 - 0.051 = 0.006`, and `0.006 / 0.057 = 0.105`, a 10.5%
fall. Its targets: `0.218 - 0.204 = 0.014`, `0.014 / 0.218 = 0.064`, 6.4%. Background fell more than
targets, so the ratio rose: `0.204 / 0.051 = 4.00`.

**Hidden-pixel background error fell more than target error on all four scenes, so contrast rose on
all four** - the intended effect, and the first model change in this series to move it.

**The README and its own table disagree slightly, and the table wins.** The README says hidden
background error "falls 6-21%". From the rounded table the falls are set1 10.5%, set2
`0.012 / 0.137 = 8.8%`, set3 `0.012 / 0.129 = 9.3%`, set4 `0.029 / 0.139 = 20.9%`: 9-21%. The 6% matches
set1's *target* fall; it may come from unrounded values or from the wrong row. The conclusion does not
change.

### Criterion 5, stated honestly

Before launch the README fixed five pass criteria. Number 5 was the mechanism check: "visible-pixel
background error clearly below FT04's on all four scenes - otherwise the decoder did not do what it was
built for, whatever detection says." The last row of the table above is that test. **It failed**: set1
got much worse (0.056 -> 0.176), set2 slightly worse, set3 and set4 better.

After seeing that, the README recorded a caveat: the progressive decoder only ever trained with about
65% of its skip zeroed (`mask_ratio: 0.65`), so a fully unmasked input - no zeros anywhere - is outside
anything it has seen. FT04's decoder has no skip, so the same test was fair for FT04. That argument is
reasonable. But be clear about its order: **the yardstick was judged wrong after the result came in.**
That is exactly what fixing criteria before launch is meant to prevent, so the README keeps criterion 5
marked as failed and leans on the contrast row - also a measure of the mechanism, but not the one
chosen in advance. Treat the mechanism as supported, not proven. The README reports criterion 5 on
FT09a only; it does not say the visible-pixel test was rerun on FT09b.

## FT09b - joint after the warm-up

Validation, from `FT09b-decoder-joint/finetune_metrics.csv`:

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.04 / 4.29 / 5.58 | 0.837 / 0.809 |
| 3 | 3.98 / 4.18 / 6.09 | 0.835 / 0.820 |
| **5** | **3.97 / 4.17 / 5.41** | **0.841 / 0.811** |

Detection, `configs/eval/ft09b.json`, cloud excluded on FT04's masks (`eval_report.md`):

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | real ROC-AUC | v2.1 caught top 1% |
|---|---|---|---|---|---|
| FT04 | 0.170 | 46.0% / **15.8%** | **0.065** | 0.852 | **36.2%** |
| FT09a (decoder only) | 0.169 | 46.1% / 15.6% | 0.061 | 0.856 | 35.8% |
| **FT09b** | **0.176** | **47.1%** / 15.4% | 0.063 | **0.854** | 36.1% |

Paired object bootstrap (part 64), from the README:

| measure | difference [95% interval] | verdict |
|---|---|---|
| FT09b - FT04, pAUC <= 1% | **+0.0060 [+0.0002, +0.0118]**, P(B > A) 0.976 | a real gain, narrowly |
| FT09b - FT04, caught top 1% | +1.1 pts [-0.3, +2.4] | tie |
| FT09b - FT04, caught top 0.1% | -0.4 pts [-1.4, +0.6] | tie |
| FT09a - FT04, pAUC <= 1% | -0.0006 [-0.0056, +0.0043] | tie |

### Per scene and per tier

pAUC per scene, from `eval_per_scene.csv` (cloud excluded):

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| FT04 | 0.246 | 0.128 | 0.084 | 0.220 |
| FT09b | **0.260** | **0.131** | 0.083 | **0.229** |
| change | +0.014 | +0.002 | -0.001 | +0.009 |

The bench pAUC is the mean over scenes. Check it by hand from the CSV's unrounded values:

```
FT04    0.24603 + 0.12817 + 0.08428 + 0.21986 = 0.67834     / 4 = 0.16959
FT09b   0.26010 + 0.13061 + 0.08301 + 0.22873 = 0.70245     / 4 = 0.17561
change  0.17561 - 0.16959 = 0.00602
```

which is the bootstrap's +0.0060 point value.

Objects caught at the top 0.1%, by scene (README): FT04 57 / 33 / 13 / 41 = 144, FT09b 56 / 34 / 9 / 41 =
140. **The whole loss is set3**, 13 -> 9; the others move by one or not at all. Across the bench's 909
objects, 4 objects is `4 / 909 = 0.44%`, the -0.4 points in the table.

Caught at the top 1% by tier (easy / medium / subtle): FT04 58.2 / 39.3 / 40.3%, FT09b 57.2 / **42.6 /
41.3**%. The gain sits in the medium and subtle tiers, where the README says the design aimed; the easy
tier falls a point. (Why the low-contrast tiers should gain most - a quieter background matters most to
a faint object - is this course's reading, not something the run measured.)

### The criteria

| # | criterion | FT09b | result |
|---|---|---|---|
| 1 | pAUC +0.005 with interval above zero | +0.0060 [+0.0002, +0.0118] | met, by +0.0002 |
| 2 | caught top 1% and top 0.1% not lower | +1.1 / -0.4 pts | **not met** at the top 0.1% (unresolved) |
| 3 | v2.1 caught top 1% within -0.5 pts | 36.2% -> 36.1%, -0.1 | met |
| 4 | SAM within 0.1 deg, IoU within 0.01 | 3.97 / 4.17 / 5.41; 0.841 / 0.811 | met (AVIRIS-NG better than FT04) |
| 5 | visible-pixel background clearly below FT04's | measured on FT09a | failed as written (caveat above) |

### The joint phase did the work

FT09a alone - higher hidden contrast already - **ties** FT04 on detection (-0.0006). FT09b, with the
encoder allowed to adapt to the new decoder, gains. And FT09b's best epoch was its last, which is why
FT09c continues it at a lower rate (3e-5, 5 epochs). FT09c is running; nothing in this part depends on
its outcome, and nothing about it is measured yet.

## A candidate, not a release

Why not ship FT09b as Indradhanu V3?

- **The interval clears zero by 0.0002.** Part 64's rule is met, but one more unlucky draw of objects
  and it would not be.
- **The bootstrap resamples objects, not scenes.** It does not cover scene-to-scene variation, and with
  four scenes the gain is carried mostly by set1 and set4.
- **Criterion 2 fails at the top 0.1%**, the operating point the product's analysts see first: 4 fewer
  objects, all in set3. Unresolved, but in the wrong direction.
- **Criterion 5 failed as written**, and its replacement was chosen after the fact.

**The sentence to remember: FT09 is the first change since cloud exclusion to clear the bench's
noise, and it cleared it by the narrowest margin the rule allows - worth continuing, not yet worth
shipping.**

The cumulative table from part 65 gains a row:

| change tried on FT04 | taught in | pAUC change [95% interval] | verdict |
|---|---|---|---|
| cloud exclusion | parts 57, 61, 64 | **+0.0074 [+0.0047, +0.0101]** | real gain, adopted |
| FT07 gated loss | part 65 | -0.0047 [-0.0082, -0.0020] | real loss, small |
| FT08 gated loss + blocks | part 65 | -0.0058 [-0.0106, -0.0010] | real loss, small |
| FT09a progressive decoder, decoder only | this part | -0.0006 [-0.0056, +0.0043] | tie |
| **FT09b progressive decoder, joint** | this part | **+0.0060 [+0.0002, +0.0118]** | real gain, small; candidate |

(The other rows of part 65's table are unchanged.)

## Common confusions

**"Part 21 said bilinear upsampling smears point anomalies, and this decoder uses bilinear."**
Part 21 compared bilinear upsampling *alone* with PixelShuffle. Here each bilinear x2 is followed by
3 x 3 convolutions that also see the skip at that resolution, so a visible pixel's own detail comes back
in at full size. For a hidden pixel there is nothing of its own to smear: its skip value is 0.

**"The skip is like the residual connections in part 15."**
Same word, different thing. Part 15's skip adds a block's input to its output inside the encoder. This
one carries the stem output from the start of the network to the decoder, past the encoder entirely.

**"The decoder got bigger, so the gain comes from more parameters."**
It got smaller: 1.50 M against 2.16 M. It does more arithmetic, because it works at full resolution.

**"FT09a reconstructs as well as FT04 and has higher contrast, so FT09a should detect better."**
It tied. Contrast in `visible_vs_hidden.py` is a mean over 24 tiles per scene; detection depends on the
ranking of the highest scores across whole scenes. The encoder had to adapt (FT09b) before the
ranking moved.

**"The skip lets hidden pixels see their neighbours, so it leaks like FT06's halo did."**
Neighbouring *visible* pixels do reach a hidden pixel through the 3 x 3 convolutions, and the docstring
calls that context. Whether that context includes a target's PSF halo, and so lets the decoder
partly extrapolate a target, was the risk the README wrote down before launch. The rise in contrast
says it did not dominate; it does not say it never happens.

**"Criterion 5 failed, so the result does not count."**
Criterion 5 is about the mechanism, not detection. Detection was judged on criteria 1-4. The failure is
recorded and carried as a caveat; it does not undo criterion 1.

## Check yourself

1. Count the parameters of `up_half`'s first convolution with `stage_dims = (160, 96)` instead of
   `(192, 96)`. (The input is still 256 + 32 channels.)
2. A 256 px tile goes through the progressive decoder. What are the shapes after the fuse, after
   `up_half`, and after `head`?
3. In `test_hidden_pixels_never_reach_the_output`, which pixel rows and columns does token 255 cover?
   What would the test catch if `_decode` forgot to zero the skip?
4. Why does `avg_pool2d(skip, 2)` keep hidden tokens at zero, and what would break if the half
   resolution skip used a 3 x 3 average instead?
5. Set4's hidden background fell 0.139 -> 0.110 and its targets 0.191 -> 0.162. Compute each fall as a
   percentage and the new target / background ratio.
6. Why was FT09 trained as FT09a then FT09b, rather than jointly from the start?
7. Name two reasons FT09b is a candidate and not a release.

<details>
<summary>Answers</summary>

1. `288 x 160 x 9 + 160 = 414,720 + 160 = 414,880`.
2. Fuse at 1/4: `(B, 256, 64, 64)`. `up_half` at 1/2: `(B, 192, 128, 128)`. `head` at full size:
   `(B, 32, 256, 256)`.
3. `divmod(255, 16) = (15, 15)`: rows 60-63, columns 60-63. Without zeroing, the random values put into
   `y` at the hidden tokens would reach the decoder through the skip, the two reconstructions would
   differ, and `allclose` would fail.
4. The 2 x 2 blocks line up with the 4 x 4 token borders, so each average uses pixels of one token only;
   four zeros average to zero. A 3 x 3 average would straddle token borders and mix visible values into
   hidden pixels' positions, and the result would no longer be exactly zero.
5. Background: `0.139 - 0.110 = 0.029`, `0.029 / 0.139 = 20.9%`. Targets: `0.191 - 0.162 = 0.029`,
   `0.029 / 0.191 = 15.2%`. Ratio `0.162 / 0.110 = 1.47`.
6. Three quarters of the new decoder starts random. Trained jointly, its early, poor gradients reach the
   encoder and disturb it, and the first attempt was still 0.5-1 deg SAM behind FT04 after 3 epochs.
   Training the decoder alone first (FT09a) brought it to FT04's reconstruction in 3 epochs without
   touching the encoder; then the joint phase could adapt both.
7. Any two of: the interval clears zero by only 0.0002; the bootstrap does not cover scene-level
   variation; the top-0.1% object count falls by 4 (criterion 2); criterion 5 failed as written and
   its replacement was chosen after the result.

</details>

---

That is the course so far. Indradhanu V2 (part 61) stands; FT09b is the first candidate since cloud
exclusion to clear the bench's noise, and FT09c, still running, continues it.
