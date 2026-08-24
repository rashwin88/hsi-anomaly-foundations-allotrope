# 06 · The core idea: redraw it, then subtract

> **The one thing this part teaches:** the model predicts each pixel *without
> ever seeing that pixel*, and the size of its mistake is the anomaly score.

This is the most important part of the course. If you only properly understand
one, make it this one.

---

## The idea in one line

```
score(pixel) = how badly the model's redrawing disagrees with reality, at that pixel
```

Written with symbols, where `x` is what the satellite actually recorded and
`x_hat` (pronounced "x hat") is what the model predicted:

```
score(i, j) = disagreement( x(:, i, j) , x_hat(:, i, j) )
```

The `(:, i, j)` means "all 165 bands at row i, column j" — one pixel's whole
spectrum.

> **Notation note.** A "hat" over a letter is the standard mathematical way of
> writing "our estimate of this". You will see `x_hat` everywhere in the code
> and in this course. It always means the model's output.

---

## Why this finds anomalies

The model has been trained on tens of thousands of ordinary patches: fields,
forest, water, roads, bare soil, cloud shadow, urban rooftops. It has become
extremely good at ordinary things, and it has had no reason whatsoever to become
good at extraordinary things.

So when you cover up part of a picture and ask it to fill in the blank:

| What is really under the cover | What the model predicts | Disagreement |
|---|---|---|
| grass, surrounded by grass | grass | tiny |
| road, in the middle of a road | road | tiny |
| water, in the middle of a lake | water | tiny |
| **an unusual mineral, in a field** | **field** | **large** |

The disagreement is the score. That is the entire detection principle, and it
does not require a single label.

---

## The fatal flaw in the naive version

Here is the version a beginner would build, and why it fails completely.

**Naive plan:** show the model a picture, ask it to output the same picture,
train it to minimise the difference.

**What the model learns:** copy the input to the output.

That is a perfectly valid solution to the task as stated. Error zero, everywhere,
including on anomalies. The score map would be uniformly zero and would detect
nothing at all.

> **This is not a subtle failure mode.** It is the *optimal* solution to the
> naive task. Any model that can copy will copy, because copying is easier than
> understanding.

So the task must be changed so that copying is impossible.

---

## The fix: hide things (this is what "MAE" means)

The technique is called a **masked autoencoder**, abbreviated **MAE**. It is the
"mae" at the end of `hyperspectral_segformer_mae`.

The procedure:

```
1. Chop the picture into small square blocks (called tokens).
2. Randomly hide about half to two-thirds of them. Actually DELETE them.
3. Show the model only the survivors.
4. Ask it to produce the WHOLE picture, hidden parts included.
5. Grade it ONLY on the hidden parts.
```

Step 5 is what makes the exam honest. The model is being marked exclusively on
things it could not see.

Copying is now useless. There is nothing to copy at the hidden positions. The
only way to score well is to genuinely understand what usually goes there.

### The exam analogy

Think of a cloze test — the exercise where words are blanked out of a passage
and you fill them in:

```
The cat sat on the ____ , licking its ____ .
```

You can only fill those in if you understand English. And if the original text
said something bizarre — "The cat sat on the *helicopter*" — you will confidently
write "mat", and be wrong. Your wrongness is precisely the signal that something
unusual was there.

---

## What is hidden, exactly

Not individual pixels. **Tokens** — small square blocks, 4 pixels by 4 pixels in
this model.

Why blocks rather than single pixels? Because hiding one pixel is far too easy.
Its eight immediate neighbours give it away almost completely; a model could
interpolate and never learn anything. Hiding a whole 4x4 block forces genuine
reasoning from further away.

Part 11 explains tokens properly. For now: **a token is a 4x4 pixel block,
summarised as a list of numbers.**

---

## Extending the trick to inference

At training time we hide a random subset and grade only there. Fine — over
thousands of patches, everything gets hidden eventually.

But at inference we need a score at **every single pixel** of one specific
scene, right now. Hiding a random half would leave the other half unscored.

The solution is to run the model **twice**, on two complementary halves:

```
Pass 1:  hide set A, show set B    ->  keep the predictions at A
Pass 2:  hide set B, show set A    ->  keep the predictions at B
         ---------------------------------------------------
         every pixel now has a prediction made without seeing itself
```

Combining them is one line of code:

```python
reconstruction = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels
```

Since `pass1_pixels` and `pass2_pixels` are 0/1 masks that never overlap, that
multiply-and-add is a **selection**, not an average. Each pixel takes its value
from exactly one of the two passes.

(That line is in
[`segformer_mae_inferencer.py`](../app/foundation_models/inferencers/segformer_mae_inferencer.py),
method `infer`. Part 29 covers the whole procedure.)

---

## What the "SegFormer" part contributes

`SegFormer` is the architecture doing the redrawing. It is a **hierarchical
vision transformer**, which unpacks to:

- **transformer** — the family of neural network built around attention, the
  same family behind language models. Part 12 explains it from scratch.
- **vision** — adapted for images rather than text.
- **hierarchical** — it looks at the image at four different zoom levels
  (1/4, 1/8, 1/16 and 1/32 scale), not just one.

Plus a lightweight decoder that fuses all four zoom levels back into a
full-resolution picture.

### Why a transformer instead of a plain convolutional network?

Two concrete reasons.

**1. Attention can look anywhere.** A convolution only sees a small neighbourhood
at a time. To work out what belongs at a hidden block, attention can consult
tokens on the other side of the patch. That matters when a whole region is
hidden and the nearest surviving evidence is far away.

**2. Multi-scale features.** Anomalies come in wildly different sizes. A
single-pixel oddity needs fine detail. A 40-pixel unexplained structure needs
broad context. Four zoom levels give the decoder both.

This repo does contain simpler convolutional autoencoders — Pratibimba,
Antardhana and friends. The SegFormer-based Chakshu and Indradhanu are their
successors.

---

## What the "Hyperspectral" part contributes

Exactly two things, and they are the *only* real differences from the thermal
Chakshu:

1. **A learned spectral compressor.** It squeezes 165 bands down to 32 before
   the transformer, and expands them back afterwards. Part 10.
2. **A second scoring term called SAM.** It grades the *shape* of the predicted
   spectrum, not just its brightness. Part 24.

Everything else — token masking, the encoder, the decoder, the two-pass
inference — is shared code, literally inherited.

---

## The whole model on one page

```
x  (B, 165, H, W)      the observed reflectance cube
   |
   |  x = x * validity_mask        zero out non-measurements
   v
PixelNormalize                     per-band z-score            part 09
   |
   v
SpectralCompressor  165 -> 32      1x1 conv + BatchNorm        part 10
   |
   v
SegFormer Encoder                  4 stages, MAE hiding        parts 11-19
   |    produces F1 F2 F3 F4 at four scales
   v
SegFormer Decoder                  fuse + PixelShuffle         part 21
   |
   v
SpectralDecompressor 32 -> 165     1x1 conv, no norm           part 10
   |
   v
PixelDenormalize                   back to reflectance         part 09
   |
   v
x_hat  (B, 165, H, W)              the reconstruction
```

That diagram is a faithful transcription of the `forward()` method in
[`hyperspectral_seg_former_mae.py`](../app/foundation_models/components/hyperspectral_seg_former_mae.py).
Open it now. You will not understand every box yet, but you should recognise the
shape of it. Everything from here on is a zoom-in on one box.

---

## Where the "never see yourself" rule is enforced

This rule is so central that it is enforced in **three separate places**. Learn
all three; questions about them come up constantly.

| Where | Mechanism | Part |
|---|---|---|
| Encoder input | hidden tokens are physically deleted from the sequence | 19 |
| Inference | two complementary passes, so no pixel is ever self-predicted | 29 |
| The grade | the loss is computed only at hidden positions | 23 |

---

## Common confusions

**"Does the model know which pixels were hidden?"**
It knows *that* something is missing — the hidden slots are zeros after the
tokens are put back. It does not know what was there. That is the whole point.

**"Why not just train it to output an anomaly map directly?"**
Because that would need labelled anomalies, which do not exist. Part 01.

**"If the reconstruction looks blurry, is the model broken?"**
No — a blurry reconstruction is *expected and healthy*. It is a prediction made
with two-thirds of the evidence removed. A crisp, perfect reconstruction would
mean the model found a way to copy, which is the failure we designed against.

**"Is this the same as an autoencoder?"**
Nearly. A plain autoencoder squeezes the input through a narrow bottleneck and
rebuilds it. A *masked* autoencoder additionally deletes part of the input. The
mask is what stops it degenerating into a copier.

---

## Check yourself

1. Why can the naive "reproduce your input" training scheme never detect
   anything?
2. What does MAE stand for, and what are its five steps?
3. Why hide 4x4 blocks rather than individual pixels?
4. Why does inference run the model twice?
5. Name the three places the "never see yourself" rule is enforced.

<details>
<summary>Answers</summary>

1. Because copying the input is the optimal solution: error zero everywhere,
   including at anomalies.
2. Masked AutoEncoder. Chop into tokens; hide most of them; show the survivors;
   reconstruct the whole thing; grade only at the hidden positions.
3. A single hidden pixel is trivially recoverable from its immediate
   neighbours, so the model would learn interpolation instead of understanding.
4. So that every pixel is hidden in one of the two passes, and therefore every
   pixel gets a prediction made without seeing itself.
5. Token deletion in the encoder; two-pass complementary masking at inference;
   the loss mask restricting grading to hidden positions.

</details>

---

**Next:** notation and every shape in one place, in
[07-shapes-cheatsheet.md](07-shapes-cheatsheet.md)
