# Part 29 - What HyperSL is, and what it does well

> **The one thing this part teaches:** HyperSL is a 9.03-million-parameter masked autoencoder that reads one pixel's
> spectrum at a time, after every band has been min-max scaled over the scene it came from; we checked it against its
> own code and checkpoint and reproduced its paper's clustering result almost to the decimal, so what parts 30-34 find
> is not a broken download but a good model built for a different job.

## Where we are

Parts 1-28 built the product's own models. People keep asking a fair question: there is a published spectral
foundation model, HyperSL (Kong et al., IEEE TGRS 63, 2025, article 5513119), with open code and weights - why not use
it? Parts 29-34 answer it. This part says what HyperSL is and what it is good at, read from its source rather than its
abstract. Parts 30-33 try it on each of our jobs (library matching, anomaly detection, unmixing, cloud masking).
Part 34 puts it all side by side.

A **foundation model** is a network pretrained on a large, broad, unlabelled data set so that its internal
representation can be reused for many tasks with little task-specific training.

| file | what it holds |
|---|---|
| `C:/Users/FITT/Documents/hypersl_match_data/hypersl_model/` | HyperSL's code (`model.py`, `train.py`, `mutihsi.py`, `loss.py`, ...) and `ckpt.pt`, outside the repo; byte-identical to upstream commit `e56de7e` |
| `git show 2d893e6:research/experiments/hypersl-validation/summary.md` | the validation: provenance, settings, paper reproduction |
| `research/textbook/tools/compute_hypersl.py` | the book's script: parameter count, the position code, every [C] number |
| `research/textbook/figdata/hypersl_ckpt_shapes.json` | every tensor in the released checkpoint, with its shape |

> **Analogy.** A concert pianist who learned every piece from recordings a studio had normalised so the loudest note
> of each was as loud as every other recording's loudest note. She plays beautifully and can finish any phrase you hum.
> Ask her whether this hall is louder than last night's and she cannot say: the one thing every recording removed was
> the absolute level. HyperSL is the pianist; the level is reflectance.

## Step 1 - one pixel in, one pixel out

HyperSL's input has shape `(B, 1, L)`: B pixels, one channel, L bands. **There is no height and no width; no pixel
ever sees another.** One PRISMA pixel on the 149-band axis our experiments used goes through:

```
(1, 1, 149)  min-max scaled spectrum                       input, scaled by the caller
(1, 256, 149) x3  Conv1d kernels 5, 9, 13 over BAND INDEX   embedding_layer_1..3 (stride 1, padding 'same')
(1, 256, 149)  concat -> 1x1 conv 768 -> 256                embedding_layer_4
(1, 149, 256)  one token per band, + sin/cos of (lambda - 400) / 10
(1, 7, 256)    keep a random 5%: int(149 * 0.05) = 7         random_masking, ratio 0.95
(1, 8, 256)    + a global token; 7 blocks + 1 layer          encoder
(1, 1, 256)    z = global token + mean of the 7              the embedding
(1, 7, 256)    cross-attention from z back onto the 7
(1, 149, 256)  + 142 mask tokens, unshuffle, + position code again
(1, 149, 256)  3 decoder blocks, LayerNorm
(1, 1, 149)    Conv1d 256 -> 1, kernel 3, then a SIGMOID     every output in (0, 1)
```

Three facts to carry forward:

- **It hides bands, not pixels.** Indradhanu hides whole pixels and rebuilds them from neighbours (part 11). HyperSL
  hides 142 of a pixel's own bands and rebuilds them from the other 7 of the same pixel. Part 31 is about what that
  does to anomalies.
- **The tokenizer convolves over band index**, whatever the wavelengths. Part 30 measures what that means on a band
  list with gaps.
- **The output is a sigmoid**, so it only works if every input is in [0, 1] - which is what per-scene min-max
  guarantees.

The **position code** adds sin and cos of `p = (lambda - 400) / 10` at 128 frequencies. The code computes them as
`exp(arange(0, 256, 2) * -(log(100.0) / 256))`, i.e. `100 ** (-2j / 256)`: 1.000000, 0.964662, ..., 0.010366. At
450 nm, `p = 5.0`: entry 0 is `sin(5.0) = -0.9589`, entry 1 is `cos(5.0) = +0.2837`, entry 254 is
`sin(5.0 x 0.010366) = sin(0.0518) = +0.0518`.

**Source check:** the paper's equations (5)-(6) use base 10,000; the code uses 100. The weights were trained with the
code. And the wavelengths must be in nm: fed micrometres, the EnMAP test loss rose from 0.0024 to 0.173.

**Source check:** `TransformerEncoderLayer.forward` does `x, k, v = self.attn(self.norm1(x))` then
`x = x + self.drop_path(x)` - the second `x` is the attention output, so the last encoder layer has no skip connection.
The weights were trained that way; do not "fix" it.

## Step 2 - predict the parameter count

A linear layer `i -> o` has `i x o + o` parameters; a norm over D has `2D`. With D = 256, feed-forward 1024:

```
one block   = 2 LayerNorms 1,024 + qkv (256 x 768 + 768) 197,376
            + projection (256 x 256 + 256) 65,792
            + feed-forward (256 x 1024 + 1024) + (1024 x 256 + 256) = 263,168 + 262,400 = 525,568
            = 1,024 + 197,376 + 65,792 + 525,568 = 789,760
encoder     = 8 x 789,760 = 6,318,080
decoder     = 3 x 789,760 = 2,369,280
cross-attn  = 2 x (256 x 256 + 256) = 131,584
tokenizer   = (5x256+256) + (9x256+256) + (13x256+256) + (768x256+256) = 1,536 + 2,560 + 3,584 + 196,864 = 204,544
BatchNorms  = 4 x 512 = 2,048;  tokens 512;  final norms 1,024;  output conv 256 x 3 + 1 = 769
total       = 9,027,841
```

The checkpoint holds 9,158,021 numbers. The difference, 130,180, is exactly its buffers: the position table `pe`
(500 x 256 = 128,000), `div_term` (128), the BatchNorm running statistics (2,048) and four step counters. The
"~9.16 M parameters" in team notes counts the buffers.

**Source check:** the class defaults (1024 / 32 / 16) do not match the checkpoint (256 / 8 / 4, 8 heads). And every
upstream downstream script ships with its checkpoint-load lines commented out - run unmodified, it evaluates a random
network.

## Step 3 - what it was trained on

`mutihsi.py` reads EnMAP (drops band indices 128-134) and DESIS (drops 0-5) L2A, multiplies by 1e-4, and calls
`sklearn.preprocessing.minmax_scale` on the pixels-by-bands table - **per band, over the scene**:

```
x'_b = (x_b - min_b) / (max_b - min_b)      min and max over the scene's pixels in band b
```

HyperSL has never seen a reflectance. Water bands were kept.

| claim | what the code and files show |
|---|---|
| sensors | EnMAP and DESIS only (`train.py`); a GF5 reader exists and is unused; the paper's 439-band airborne set is not in `train.py` |
| size | "over 300 million spectral instances" (abstract), "approximately 200 million" (section IV-A) |
| clouds | "cloud cover near zero", "below 10%" - part 33 needs this |
| schedule | 10 epochs, AdamW 5e-4, batch 256 per sensor, mask 0.95; released checkpoint is epoch 10 |
| loss | MSE + (1 - cosine) over the **whole** spectrum, visible bands included |
| data | ~265 GB corpus on Baidu Netdisk only; Hugging Face has the 10.5 GB test split |
| licence | no LICENSE on GitHub; Hugging Face card says MIT |

**Source check:** the paper calls the scaling "layer-by-layer normalization". It does not say min-max or per scene;
only the code does.

## Step 4 - did we run it right?

From `2d893e6:hypersl-validation/summary.md`:

- **Provenance:** checkpoint sha256 `3d5f1914...a74a68c` equals Hugging Face's `10_base_mask95`; code byte-identical
  to `e56de7e`.
- **Settings by elimination** (EnMAP test loss at mask 0.95): ours 0.0024; epoch 5 0.0032; epoch 0 0.0053; untrained
  0.218; 4 heads 0.087; 16 heads 0.008; micrometres 0.173; +20 nm 0.0073; per-spectrum scaling 0.0063. Ours wins every
  row. At mask 0 the loss is *worse* (0.0073) - it only trained at 0.95.
- **Zero-shot reconstruction:** Indian Pines equal or better than the paper (SAM 7.86 deg vs 8.43); Botswana MSE 10x
  better but SAM 8.84 vs 5.67 deg (unpublished wavelength file).
- **Chikusei clustering** (K-means, 19 clusters, 77,592 pixels):

| features | OA ours / paper | ARI ours / paper |
|---|---|---|
| raw spectra | 39.60 / 42.69 | 0.268 / 0.289 |
| PCA | 37.68 / 41.11 | 0.259 / 0.276 |
| **HyperSL, band 0-1 input** | **59.60 / 60.23** | **0.5404 / 0.5405** |
| HyperSL, raw input | 48.19 / - | 0.418 / - |

**Overall accuracy (OA)** is the share of pixels whose mapped cluster equals their class; the **adjusted Rand index
(ARI)** compares two groupings pair by pair, 0 for random and 1 for identical. HyperSL matches the paper to 0.63 OA
points and 0.0001 ARI. On raw reflectance it loses 59.60 - 48.19 = 11.41 points.

- **Fine-tuning** (10 seeds, paper protocol): Botswana pretrained 99.01 vs random 97.55 (+1.46; paper +2.61); Indian
  Pines 97.73 vs 94.94 (+2.79; paper +3.44). Pretraining helps, by 56% and 81% of the paper's gain.

## Step 5 - what it is good at

- It learns the shape of spectra: 95% hidden bands rebuilt at 7.9 deg on a sensor it never saw.
- Its embedding groups land cover without labels, +20 OA points on Chikusei.
- It is a good start for supervised classification, +1.5 to +2.8 OA points.
- It accepts any band list (the position code is a function of wavelength). Our stem shares that idea.

All four live in one setting: **one scene at a time, classes that cover many pixels, and only relative differences
inside the scene matter.** Each of our jobs asks for something else:

| our job | needs | HyperSL | part |
|---|---|---|---|
| library matching | same spectrum, same input in every scene; library has an input | input depends on the scene's extremes; library has no scene | 30 |
| anomaly detection | a pixel must not help predict itself | predicts a pixel's bands from its own bands | 31 |
| unmixing | per-pixel abundances that keep a 0.1-pixel target | 3x3 head + TV smooth it away | 32 |
| cloud masking | clouds in training, brightness kept | trained near cloud-free, brightness rescaled away | 33 |

## Common confusions

**"HyperSL looks at patches like Indradhanu."** One pixel's spectrum. Its tokens are bands. (Its classification fine-tune
uses 5x5 patches, but that is a downstream head.)

**"It has 9.16 million parameters."** 9,027,841 trainable; 9,158,021 with buffers.

**"The position code makes the tokenizer wavelength-aware."** The code is added after the convolutions, which have
already mixed neighbouring band indices.

**"Mask 0 gives the best embedding."** It never trained at mask 0; its loss is worse there.

**"The reproduction proves it suits the product."** It proves we ran the model the authors trained, as they ran it.

## Check yourself

1. Count one transformer block's parameters for D = 128 (feed-forward 512).
2. Compute entries 0 and 1 of the position code at 1000 nm. What is `p` if the wavelength is passed as 1.0?
3. EnMAP has 224 bands; HyperSL drops 7. How many bands does the encoder see at mask 0.95? For 202 bands?
4. On Chikusei, how much does HyperSL beat raw spectra with band 0-1 input, and with raw input? What share survives?
5. List the checkpoint's four kinds of buffer and check they sum to 130,180.

<details>
<summary>Answers</summary>

1. Norms `2 x (2 x 128) = 512`; qkv `128 x 384 + 384 = 49,536`; projection `128 x 128 + 128 = 16,512`; feed-forward
   `(128 x 512 + 512) + (512 x 128 + 128) = 66,048 + 65,664 = 131,712`; block `512 + 49,536 + 16,512 + 131,712 = 198,272`.
2. `p = (1000 - 400) / 10 = 60.0`; `sin(60.0) = -0.3048`, `cos(60.0) = -0.9524`. Passed as 1.0: `p = -39.9`.
3. 217 bands, `int(217 x 0.05) = int(10.85) = 10`. For 202: `int(10.1) = 10`.
4. `59.60 - 39.60 = 20.00`; `48.19 - 39.60 = 8.59`; `8.59 / 20.00 = 0.43`, 43%.
5. `pe` 128,000 + `div_term` 128 + running stats 2,048 + counters 4 = 130,180 = 9,158,021 - 9,027,841.

</details>

---

Next: part 30, per-scene min-max and a library with nowhere to stand.
