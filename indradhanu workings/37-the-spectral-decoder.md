# 37 - The way back out: a decoder that answers for any wavelength

> **The one thing this part teaches:** the decoder turns each wavelength you want back into a
> question and puts it to the pixel's 32 latent channels, so one set of weights reconstructs
> 202, 224, 239 or 425 bands - whatever the sensor had.

## What has to come back out

The model's job (part 06) is to redraw the input and compare. So whatever went in must come
back out at the *same wavelengths*: a SpectralEarth pixel's 202 bands, a PRISMA pixel's 239.

Today's `SpectralDecompressor` (part 10) is `Conv2d(32, 165, kernel_size=1)`: one fixed recipe
per output band, and exactly 165 of them. It has the same three problems part 31 found in the
compressor, mirrored. It cannot produce a 202nd band, or any band not on the grid.

## Turn it around: the wavelength asks the question

In the compressor, 32 learned questions read many band tokens. In the decoder:

```
compressor:  32 learned questions  ->  read N band tokens   ->  32 channels
decoder:     N wavelength questions ->  read 32 channel tokens -> N reflectances
```

Each output band is one question, built from its wavelength alone: "given everything this
pixel's 32 channels say, what is the reflectance at 1650 nm?" Ask 202 questions, get 202
answers; ask 239, get 239. Part 32 called this kind of query "computed from something else".

> **Analogy.** The compressor is a reporter who interviews 202 witnesses and files a fixed
> 32-point summary. The decoder is a reader of that summary who can ask it any question that
> has a date on it - and gets an answer for any date, not only the dates the original
> witnesses covered.

## The pieces

Per pixel, after the SegFormer encoder and decoder (parts 16 and 21) hand back 32 channels:

| piece | what it does | parameters |
|---|---|---|
| channel tokens | each of the 32 channel values stretched (`Linear(1, 32)`) **plus** a learned identity vector for that channel, so the tokens know which channel they are | `64` + `32*32 = 1,024` |
| wavelength queries | the part 33 tag of each requested wavelength, through `Linear(32, 32)` | `1,056` |
| attention | Q, K, V and output projections, 4 heads | `4 * 1,056 = 4,224` |
| to reflectance | `Linear(32, 1)` per answer | `33` |

Why do the channel tokens need a learned identity, when band tokens used a wavelength tag?
Channels have no wavelength. Channel 7 is "whatever the compressor's 7th question measured",
so its tag must be learned.

## Predict the parameter count

```
query from wavelength   Linear(32, 32)      32*32 + 32   = 1,056
channel value stretch   Linear(1, 32)        1*32 + 32   =    64
channel identities      table (32, 32)      32*32        = 1,024
Q, K, V, O projections  4 x Linear(32, 32)  4 * 1,056    = 4,224
to reflectance          Linear(32, 1)       32 + 1       =    33
                                                          -----
total                                                     6,401
```

The prototype (`SpectralCrossAttentionDecoder`, written to check this course) reports:

```
decoder params: 6401
  query_from_wavelength 1056, channel_value_embed 64, channel_identity 1024,
  q 1056, k 1056, v 1056, o 1056, to_value 33
```

Today's decompressor is `32*165 + 165 = 5,445`. Comparable again - and again no band count in
the formula.

## One set of weights, any sensor

The same trained decoder, asked for four different wavelength lists:

```
SpectralEarth  (202 wavelengths) -> output shape (pixels, 202)
EnMAP          (224 wavelengths) -> output shape (pixels, 224)
PRISMA         (239 wavelengths) -> output shape (pixels, 239)
AVIRIS-NG      (425 wavelengths) -> output shape (pixels, 425)
```

**This is the property that makes "pretrain on SpectralEarth, fine-tune on PRISMA" work
without resampling.** The model never has a weight tied to a band position, on the way in
or the way out.

## The loss, on native bands

Parts 23 to 25 built the loss as `L1 + lambda * SAM`, computed per pixel over the bands. With
native bands nothing about that changes, except which bands count:

- compare each reconstructed band with the measured band at the **same wavelength** - there is
  no grid in between;
- **skip dead and padded bands** (part 35's mask): there is no truth to compare against;
- SAM's angle (part 24) is computed over the bands that remain.

The spatial masking of parts 17 to 19 is untouched: which *pixels* are hidden and must be
predicted from neighbours is the encoder's business, and still decides where the loss is
taken.

## Common confusions

**"Can the decoder predict wavelengths the sensor never measured?"** It will produce numbers
for them, because any wavelength can be turned into a question. Whether those numbers mean
anything depends on whether training ever saw that region. Asking EnMAP-trained weights for
2600 nm is extrapolation, not measurement.

**"Why not reuse the compressor's 32 queries in the decoder?"** They are questions about
spectra; the decoder needs questions about wavelengths. Different jobs, different queries.

**"Is the decoder's cost like the compressor's?"** Roughly, with roles swapped: `NB` queries
against 32 keys instead of 32 queries against `NB` keys. The score matrix is the same size,
`32 x NB` per head. Keys and values are 32 per pixel instead of `NB`, which is much lighter.

**"Does each output band get its own weights?"** No - that is exactly what was removed. Every
band uses the same projections; only its wavelength tag differs.

## Check yourself

1. In the decoder, where do the queries come from, and where do the keys and values come
   from?
2. Recompute the decoder's parameters for `d = 16`, `D = 32`. Show each line.
3. Why do the channel tokens need a learned identity table while band tokens did not?
4. A batch mixes SpectralEarth and PRISMA pixels. Which bands enter the loss for a
   SpectralEarth pixel padded to 239?
5. Today's decompressor has 5,445 parameters. Why can't it reconstruct a PRISMA pixel's 239
   bands even if you retrain it?

<details>
<summary>Answers</summary>

1. Queries from the requested wavelengths (their tags through a linear layer); keys and values
   from the pixel's 32 latent channel tokens.
2. Query-from-wavelength `16*16 + 16 = 272`; value stretch `1*16 + 16 = 32`; identities
   `32*16 = 512`; Q,K,V,O `4 * 272 = 1,088`; to-reflectance `16 + 1 = 17`. Total
   `272 + 32 + 512 + 1,088 + 17 = 1,921`.
3. A band has a physical wavelength to encode; a latent channel has none - its meaning is
   whatever training made it - so its tag must be learned.
4. Only its 202 real bands; the 37 padding bands are masked out of the loss.
5. Its output layer has exactly 165 output positions, fixed by its shape `(165, 32)`; it has
   no way to produce a 166th output or to place outputs at PRISMA's wavelengths.

</details>

---

**Next:** part 38 - one pixel, start to finish, and the three traps to design around.
