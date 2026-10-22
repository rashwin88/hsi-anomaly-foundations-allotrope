# 66 - Is the decoder the bottleneck?

> **The one thing this part teaches:** two measurements made without any training - how much of FT04's
> error lies outside what its spectral output layer can ever produce (2-10%), and whether FT04 copies
> pixels it can *see* better than pixels it must *predict* (barely) - say that FT04's error is mostly
> what its decoder cannot render, not what the context cannot predict, and point at the 4 x 4-token
> bottleneck with no full-resolution path back.

## Where we are

**Status: a diagnosis, done 2026-10-21 on the laptop, no training.** Part 65 ended with every change
since FT04 inside bench noise or a small cost. All of those changes were to the loss, the masking or
the score. None touched the shape of the network. The user's prompt for this part was: "Maybe a
bigger model is the answer", "Or maybe more layers of decoding", "I feel the decoder maybe too
un-expressive given the sophistication of the encoder."

Before building anything, you want to know *where* the error comes from. This part is the two tests
that answered that. Part 67 is what was built because of them.

| file | what it holds |
|---|---|
| `research/experiments/decoder-capacity/README.md` | the question, both tables, the consequence |
| `research/experiments/decoder-capacity/spectral_floor.py` | test 1: the spectral floor |
| `research/experiments/decoder-capacity/visible_vs_hidden.py` | test 2: visible vs hidden pixels |
| `app/foundation_models/components/seg_former_decoder.py` | FT04's decoder: fuse at 1/4, then PixelShuffle(4) (part 21) |
| `app/foundation_models/components/spectral_reduced.py` | `WavelengthLinearDecoder`, the spectral output (part 40) |

> **Analogy.** A police sketch artist draws faces from a witness's description, and the sketches come
> out poor. Is the witness's memory the problem, or the artist's pencil? Test it: put the real face in
> front of the artist and ask for a copy. If the copy is barely better than the sketch from memory,
> the memory was never the limit. The pencil is.

## Where the parameters are

You met FT04's parameter count in parts 26 and 61. Split by component (built from FT04's config and
counted with `sum(p.numel())` on 2026-10-22; the README rounds the same figures):

| component | parameters | what it does |
|---|---|---|
| encoder | 3,331,072 (3.33 M) | the four SegFormer stages (parts 11-16) |
| decoder (reconstruction) | 2,164,736 (2.16 M) | fuse four scales at 1/4, then PixelShuffle (part 21) |
| segmentation head | 371,536 (0.37 M) | water / cloud masks (part 47) |
| `spectral_in` | 14,721 (14.7 k) | many bands -> 32 numbers per pixel (part 40) |
| `spectral_out` | 4,257 (4.3 k) | 32 numbers per pixel -> any band list (part 40) |

The decoder is not small: it is 39% of the model, as part 21 counted. So "make the decoder bigger" is
not the obvious fix. The question is not how many parameters it has but **what it is able to draw**.

Two parts of the output path could be limiting:

```
encoder features --(decoder)--> z_hat (B, 32, H, W) --(spectral_out)--> bands (B, N, H, W)
                   test 2 asks about this                 test 1 asks about this
```

## Test 1 - the spectral floor

### The idea

`WavelengthLinearDecoder` (`spectral_reduced.py`) makes band `n` of a pixel as

```
out_n = U(lambda_n) . z + c(lambda_n)          U(lambda_n): 32 numbers, c(lambda_n): one number
```

Read that as: every output spectrum is the offset curve `c` plus some mixture of **32 fixed curves**
(column `d` of `U` across all bands is curve `d`). The mixture weights are the 32 numbers in `z`. Whatever
`z` the decoder predicts, the output can only be a spectrum built that way. The set of all spectra
you can build from those 32 curves is called their **span**.

A real spectrum usually has a small part that no mixture of those curves can match. That part is
the **spectral floor**: error that no change upstream of `spectral_out` can remove, however good the
decoder or encoder become. If the floor were most of FT04's error, the fix would be a richer spectral
output. If it is a small part, the fix is elsewhere.

### By hand: 3 bands, 1 curve

Do this on paper. Make the model tiny: 3 bands, one curve `u = (1, 1, 1)`, no offset. Every
reconstruction is `a x (1, 1, 1)` for some number `a`: a flat spectrum.

The real pixel is `x = (2, 1, 3)`.

The closest flat spectrum is the **projection** of `x` onto the span. For one curve:

```
a* = (x . u) / (u . u)
x . u = 2 x 1 + 1 x 1 + 3 x 1 = 2 + 1 + 3 = 6
u . u = 1 + 1 + 1             = 3
a*    = 6 / 3                 = 2
projection = 2 x (1, 1, 1)    = (2, 2, 2)
```

The floor is what is left over:

```
floor = x - projection = (2 - 2, 1 - 2, 3 - 2) = (0, -1, 1)
floor energy = 0^2 + (-1)^2 + 1^2 = 0 + 1 + 1 = 2
```

Now suppose the model predicted `a = 1.5`, so `recon = (1.5, 1.5, 1.5)`:

```
residual = recon - x = (1.5 - 2, 1.5 - 1, 1.5 - 3) = (-0.5, 0.5, -1.5)
energy   = 0.25 + 0.25 + 2.25 = 2.75
floor share = 2 / 2.75 = 0.73
```

73% of this residual's energy is floor. The other 27% is the model choosing `1.5` instead of `2`,
which a better decoder could fix. If the model had predicted the best `a = 2`, the residual would be
the floor itself and the share would be 1.00. **The share can never exceed 1**, because every
reconstruction lies in the span, so the out-of-span part of the residual is exactly the out-of-span
part of the data.

### What the script does

`spectral_floor.py` does the same thing with 32 curves instead of one. It reads `U` and `c` from FT04's
`spectral_out` at PRISMA's wavelengths, normalises both the real cube and the reconstruction (part 09's
units), subtracts `c`, and projects onto the span with a QR decomposition (`Q, _ = np.linalg.qr(U)`; `Q`
is an orthonormal basis of the same span, so the projection is `Q @ (Q.T @ x)`). It does this for every
target pixel and for 200,000 randomly chosen background pixels per scene.

| scene | background: actual / floor mean abs | floor share | targets: actual / floor | floor share |
|---|---|---|---|---|
| set1 | 0.048 / 0.013 | 0.07 | 0.189 / 0.022 | 0.02 |
| set2 | 0.122 / 0.017 | 0.02 | 0.204 / 0.025 | 0.02 |
| set3 | 0.121 / 0.033 | 0.09 | 0.149 / 0.043 | 0.10 |
| set4 | 0.115 / 0.015 | 0.03 | 0.177 / 0.025 | 0.03 |

The floor is 2-10% of the residual energy. **Over 90% of FT04's error is in which `z` the decoder
predicts, not in what `spectral_out` can express.** Test 1 clears `spectral_out`.

## Test 2 - visible vs hidden pixels

### Why this test separates two causes

FT04's background error could come from either of two places:

- **Unpredictability.** A hidden pixel's true value is not fully fixed by its neighbours. A field
  edge, a car, a puddle: the context does not say it is there. No decoder can fix this, and it is
  also exactly what makes targets detectable.
- **Capacity.** The network cannot *render* the detail even when it has it. This one a better
  architecture can fix.

Run the model with **no mask at all** - every token visible, pure autoencoding, the copy from the
analogy. Unpredictability is then gone, because nothing has to be predicted. Whatever error is left is
capacity. Compare it with the error on hidden pixels (the product's two complementary checkerboard
passes, part 29, so every pixel is hidden once):

```
visible error  much lower than hidden error   ->  the limit is what context cannot predict
visible error  about equal to hidden error    ->  the limit is what the network can render
```

### What the script does

`visible_vs_hidden.py` cuts 24 random 128 px tiles per PRISMA scene, keeping only tiles at least 95%
valid (`v.mean() < 0.95` is skipped). On each it runs FT04 twice - `net(...)` with no keep mask, and
`inf.predict(...)`, the product's two-pass inference - and takes the mean absolute error per band in
normalised units, separately for background and target pixels.

| scene | background visible (no mask) | background hidden (two-pass) | targets visible | targets hidden |
|---|---|---|---|---|
| set1 | 0.056 | 0.057 | 0.194 | 0.218 |
| set2 | 0.110 | 0.137 | 0.161 | 0.220 |
| set3 | 0.159 | 0.129 | 0.176 | 0.179 |
| set4 | 0.135 | 0.139 | 0.169 | 0.191 |

On set1 FT04 copies the background it can see (0.056) no better than the background it must predict
(0.057). On set2 and set4 visible is only modestly better. On set3 visible is *worse* than hidden; the
README does not explain that row, and neither can this part. A model limited by unpredictability would
autoencode visible pixels almost exactly. **This one is limited by what its architecture can render.**

A second reading: background error is largely detail FT04 cannot render, while targets stay 1.2-3.8x
above the background (for example set1 hidden, `0.218 / 0.057 = 3.82`). The background error that
residual Mahalanobis (part 45) fights is, in part, an artefact of the decoder.

## Why: the 4 x 4-token bottleneck

Where could the detail be lost? Follow one stage-1 token through FT04.

The stage-1 patch embedding is a convolution with kernel 4 and stride 4 (`seg_former_encoder.py`,
`patch_size = 4 if i == 0`), so each token covers a 4 x 4 block of pixels and reads all 32 channels of
the stem output for each. Count the numbers going in and coming out:

```
pixels per token          4 x 4 = 16
numbers per pixel         D = 32               (the stem's output)
numbers into the token    16 x 32 = 512
stage-1 width             embed_dims[0] = 32
compression               512 / 32 = 16 to 1
```

Every detail of those 16 pixels has to survive being squeezed into 32 numbers (plus whatever the coarser
stages add, which is coarser still). Then the decoder (part 21) fuses the four stages at 1/4 resolution
and emits each 4 x 4 block **at once** from one 256-wide vector: a 3 x 3 convolution to `32 x 16 = 512`
channels, and PixelShuffle(4) lays them out as 16 pixels. **Nothing at full resolution ever comes back
in.** Sub-token detail that did not fit through the 32 numbers is gone, for visible pixels as much as
hidden ones - which is what test 2 measured.

The README's consequence, and part 67's design: a decoder that upsamples in stages, and gets a
**full-resolution skip** - a copy of the stem output fed straight into the decoder at full size - with
every hidden pixel zeroed so nothing it must predict leaks in. The prediction, written before any
training: background error falls (the decoder can now render what it sees), unpredictable targets stay
high, contrast rises.

## Common confusions

**"The decoder has 2.16 M parameters, so it cannot be under-powered."**
Parameters are not the issue; paths are. Its parameters all work at 1/4 resolution on token-level
vectors. No count of them can recover a detail that never reached them.

**"A floor share of 0.07 means the floor is 7% of the error in the table's mean-abs column."**
No. The share is a ratio of *squared* error summed over pixels (energy). The table's other columns are
mean *absolute* errors. Set1 background: `0.013 / 0.048 = 0.27` in mean-abs terms, 0.07 in energy terms.
Squaring shrinks small parts relative to large ones.

**"Visible-pixel error should be zero - the answer is in the input."**
Only for a network that can copy. FT04 was never trained to copy: its loss is on hidden pixels only
(part 23), and its architecture squeezes each token 16 to 1. The point of the test is to see how far
from zero it is.

**"The floor is the part of the data the model got wrong."**
It is the part of the data the model *could not* get right, whatever it did. The rest of the residual
is what it chose badly.

**"PixelShuffle gives every pixel its own value, so part 21 said there is no smearing."**
Part 21 is right that every pixel gets its own output channel and its own weights. But all 16 channels
are computed from the same token-level vector. Own weights, shared input: the pixel's own detail is
not in that input unless it survived the 512-to-32 squeeze.

## Check yourself

1. FT04 has 3,331,072 encoder and 2,164,736 decoder parameters. What share of the two together is the
   decoder? (Two decimal places is enough.)
2. With one curve `u = (1, 1, 1)` and pixel `x = (4, 1, 1)`: what is the projection, the floor, and the
   floor energy? If the model predicts `(3, 3, 3)`, what is the floor share?
3. Why can the floor share never be above 1?
4. A token covers 4 x 4 pixels of 32 channels. A colleague proposes stage-1 width 64. What would the
   compression ratio become?
5. On set2, FT04's background error is 0.110 visible and 0.137 hidden. A different model scores 0.020
   visible and 0.130 hidden. Which one is limited by capacity, and which by unpredictability?
6. Why does test 2 use the product's two-pass inference for "hidden", rather than one random mask?

<details>
<summary>Answers</summary>

1. Sum `3,331,072 + 2,164,736 = 5,495,808`. Share `2,164,736 / 5,495,808 = 0.39`.
2. `x . u = 4 + 1 + 1 = 6`, `u . u = 3`, `a* = 6 / 3 = 2`, projection `(2, 2, 2)`. Floor
   `(4 - 2, 1 - 2, 1 - 2) = (2, -1, -1)`, energy `4 + 1 + 1 = 6`. Residual for `(3, 3, 3)`:
   `(3 - 4, 3 - 1, 3 - 1) = (-1, 2, 2)`, energy `1 + 4 + 4 = 9`. Share `6 / 9 = 0.67`.
3. Every reconstruction lies in the span, so the residual's out-of-span part equals the data's
   out-of-span part - the floor. The residual's energy is its in-span energy plus that floor energy
   (the two parts are at right angles), so it is at least the floor energy.
4. `16 x 32 = 512` numbers in, 64 out: `512 / 64 = 8` to 1.
5. FT04 is limited by capacity (visible barely below hidden). The other model copies what it sees
   almost exactly, so its hidden error is what the context cannot predict: unpredictability.
6. The two complementary passes hide every pixel exactly once, which is how the product scores a scene
   (part 29). One random mask would hide only about 65% of the pixels, and only those could be scored.

</details>

---

Next: part 67 builds the decoder this diagnosis asks for, and tests it on detection.
