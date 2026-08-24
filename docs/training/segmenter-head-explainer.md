# The new segmentation head - a four-part explainer

Study material. Written 2026-08-24, for someone who has not worked on this model before.

**This describes a design that has not been built.** There is no segmenter in the repo
today. This module explains the shape we intend to give it and why, so the decisions are
reviewable before any code exists. It is subordinate to `docs/01-orientation.md` ..
`docs/09-known-issues.md` and to the source itself.

Related: `docs/proposals/spectral-attention-bottleneck.md` covers a separate, undecided
upgrade to the same model's spectral stem. The two are independent.

## Index

| Part | What it teaches |
|---|---|
| 1 | Why we reuse Indradhanu instead of training something new |
| 2 | How the model turns a small feature map back into a full-size picture |
| 3 | Why six separate yes/no answers, not "pick one" |
| 4 | The trap: teaching it to find rare things breaks its sense of certainty |

## Values that will drift - re-verify these

- **Part 1** quotes Indradhanu's parameter count (5,507,354) and normalisation mode from
  `allotrope_models/hyperspectral_segformer_mae/current.json`. A retrain changes both.
- **Part 4** quotes pixel counts from a cloud-mask experiment dated 2026-08-12 that lives
  outside the repo. Treat the ratios as illustrative, not as a fixed property of the data.
- **Parts 3 and 4** quote label statistics measured from a 212-scene EnMAP screen on
  2026-08-24. More scenes were being acquired at the time of writing; the balance will
  change.

---

## Part 1 - Reusing Indradhanu

> **The one thing this part teaches:** we are not building a new model. We are replacing
> the last layer of one you already trained.

**Some words first.**

A **layer** is one step of maths inside a model. A **parameter** (or weight) is a single
number the model learned during training - a model "is" its parameters. **Fine-tuning**
means starting from an already-trained model instead of from random numbers.

Indradhanu today does this:

```
picture in  ->  [encoder]  ->  [decoder]  ->  picture out again
```

The **encoder** squeezes the picture down into a compact summary. The **decoder** expands
that summary back out to full size. It was trained so the output matches the input, and
the leftover difference is the anomaly score.

The encoder is the part worth keeping. It has seen a great deal of satellite imagery and
has learned what scenes look like - texture, spectral shape, structure.

**The analogy.** Think of a chef who has spent years learning to taste. They can tell you
what is in a dish - the salt, the acid, the herb. If you now want them to fill in a
different form, you do not retrain their palate. You hand them a new form. The tasting
stays; only the last step changes.

**How much actually changes.** The decoder ends in
`app/foundation_models/components/seg_former_decoder.py:113-117`:

```python
self.refine = nn.Sequential(
    nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),            # transfers
    nn.GELU(),                                                                # no parameters
    nn.Conv2d(decoder_dim, out_channels * 4 * 4, kernel_size=3, padding=1),   # NEW
)
```

Only that last line changes shape, because its width depends on `out_channels`. Everything
before it - the encoder, the band compressor, and three of the decoder's four layers - is
the same shape and transfers untouched.

**Let us count it, because the number is the proof.**

A `Conv2d(in, out, 3x3)` has `in x out x 3 x 3` weights plus `out` biases.

The old final layer, reconstructing 32 channels, so `out_channels x 16 = 512`:

```
256 x 512      = 131,072
131,072 x 9    = 1,179,648   weights
             +       512     biases
               = 1,180,160   parameters
```

The new final layer, 6 classes, so `6 x 16 = 96`:

```
256 x 96       =  24,576
24,576 x 9     = 221,184     weights
             +      96       biases
               = 221,280     parameters
```

Indradhanu has **5,507,354** parameters (from its `current.json`). We also drop the
decompressor, `Conv2d(32, 165, 1x1)`, which is `32 x 165 = 5,280` weights plus 165 biases
= **5,445**.

```
 5,507,354   Indradhanu
-1,180,160   old final layer
-    5,445   decompressor
= 4,321,749   parameters carried over

 4,321,749
+  221,280   new head
= 4,543,029   the segmenter
```

**About 78% of the model is inherited, and roughly 5% of it is new.**

### Common confusions

**"Head" and "decoder" - are they the same thing?**
No, and this trips people. The **decoder** is the whole expanding half. The **head** is
only the final layer that produces the answers. We keep the decoder and replace its head.

**Does the model need the pixel-statistics file we worried about earlier?**
No. Indradhanu's `current.json` says `"mode": "baked_in"` - the per-band averages live
inside the `.pt` file itself as saved buffers. They travel with the weights.

**Is 78% reuse guaranteed to help?**
No. It is a strong bet, not a certainty. If the encoder's summary happens not to contain
the information that distinguishes cloud from snow, starting from it saves nothing.

### Check yourself

1. What does the encoder do, in one sentence?
2. Which single layer changes shape, and why that one?
3. Work out by hand: if we used **10** classes instead of 6, how many parameters would the
   new final layer have?
4. Why must we not swap in a different pixel-statistics file?
5. Roughly what fraction of the model is newly initialised?

<details>
<summary>Answers</summary>

1. It squeezes the picture into a compact summary that captures what is in it.
2. The last conv in `refine`, because its output width is `out_channels x 16`, and
   `out_channels` is the number of things we predict.
3. `out_channels x 16 = 160`. Weights: `256 x 160 = 40,960`, then `40,960 x 9 = 368,640`.
   Biases: `160`. Total **368,800**.
4. The statistics are baked into the checkpoint and the transferred weights were tuned
   against them. Different numbers would silently mean different inputs - and nothing
   would error.
5. 221,280 of 4,543,029, which is about **4.9%**.

</details>

**Next:** how the model gets from a small feature map back to a full-size picture.

---

## Part 2 - Getting back to full size

> **The one thing this part teaches:** the model works out its answer on a grid four times
> smaller than the image, then expands it - so the finest real detail it has is four
> pixels wide.

**Some words first.**

The encoder does not keep the picture at full size. It shrinks it in stages. By the time
the decoder has fused everything, it is working on a grid **a quarter of the width and
height** of the input - written `H/4`.

For a 128 x 128 training patch, that grid is 32 x 32.

**Expanding it back.** The current code uses a trick called **PixelShuffle**. Instead of
one value per grid position, it produces **16** values, then unfolds them into a 4 x 4
block:

```
32 x 32 positions, 16 values each   ->   128 x 128 pixels
        32 x 4 = 128
```

**The analogy.** Picture a mosaic tiler. They glance at one small patch of the reference
photo, then place sixteen tiles at once from that single glance. They never look closer.
The sixteen tiles can be different from each other - the tiler has learned what patterns
tend to appear - but all sixteen came from one look.

**Why it was built this way.** The source says so plainly, in
`app/foundation_models/components/seg_former_decoder.py:108-110`:

> "This is critical for point anomaly detection: a 1-pixel anomaly at 45C surrounded by
> 30C background gets its own predicted value, not a smoothed average of the 4x4 block."

That is exactly right for finding anomalies. **For drawing cloud boundaries the trade-off
flips**, because real information still only exists at 4-pixel granularity. EnMAP pixels
are 30 m, so:

```
4 pixels x 30 m = 120 m
```

Cloud edges can only be placed to within about 120 m, and blocky 4 x 4 steps may show
along them.

**What we are doing about it.** Keeping PixelShuffle for the first version. It costs
nothing, and the layer before it transfers from Indradhanu. Then we look at an actual
output. If the edges are visibly blocky we switch to smooth (bilinear) expansion, which is
what the standard version of this architecture uses. This is a genuine trade-off, not a
clear win either way, and one picture settles it faster than any argument.

### Common confusions

**Does PixelShuffle invent detail?**
No. It rearranges 16 learned values into 16 positions. It cannot know anything the 32 x 32
grid did not carry.

**So is the output 32 x 32 or 128 x 128?**
The output is 128 x 128 - one number per real pixel. But neighbouring pixels inside a
4 x 4 block came from the same source, so they are not sixteen independent observations.

**Is 120 m accuracy a problem?**
For cloud, probably not - edges are fuzzy anyway. For a thin cloud filament narrower than
120 m, yes.

### Check yourself

1. What size grid does the decoder do its thinking on, for a 128 x 128 patch?
2. Why 16 values per position and not 4?
3. If we trained on 256 x 256 patches instead, what would the decoder's grid be?
4. In metres, how precisely can a cloud edge be placed?
5. Why is PixelShuffle right for anomaly detection but questionable for segmentation?

<details>
<summary>Answers</summary>

1. 32 x 32, because `128 / 4 = 32`.
2. Because we are expanding by 4 in both directions, and `4 x 4 = 16`.
3. `256 / 4 = 64`, so 64 x 64.
4. About 120 m - `4 pixels x 30 m`.
5. Anomalies are single pixels that must not be averaged away, so independent per-pixel
   prediction helps. Class boundaries want smoothness, so independent prediction can
   produce blocky steps.

</details>

**Next:** why the head gives six separate answers rather than one.

---

## Part 3 - Six yes/no answers, not one choice

> **The one thing this part teaches:** each pixel gets six independent verdicts, because
> the real world lets several of them be true at once.

**Two ways to build the output.**

**One choice ("softmax"):** the model must pick exactly one label per pixel. The six
numbers are forced to add up to 1 - more cloud means less water, automatically.

**Six checkboxes ("sigmoid"):** each class gets its own independent 0-to-1 score. All six
can be high. All six can be low.

**The analogy.** A form where you must tick exactly one box - "cloud OR water OR snow" -
versus a form with six separate yes/no checkboxes. Thin cirrus drifting over a lake is
genuinely both. The first form makes you lie.

**We are using six checkboxes.** Two reasons, and the second is the stronger:

1. Reality allows overlap - cirrus over water, haze over snow.
2. **Our labels are already built that way.** EnMAP ships five separate mask files, one
   per condition, each independent - see
   `app/utils/dataset_builder/enmap_dataset_builder.py:187-200`. If we forced a single
   choice we would have to invent a priority order and throw information away.

So the six are: **cloud, cirrus, haze, cloud shadow, snow, water.**

**A free win on cirrus.** Reading the actual scene files, the cirrus layer is not yes/no.
It carries **0, 1, 2, 3** by thickness. In one scene the values split 50.8% / 39.9% /
9.3%. The other four layers were plain 0/1.

Rather than flatten that to yes/no, use the level as a **partial answer**:

```
level 0  ->  target 0.00
level 1  ->  target 0.33
level 2  ->  target 0.67
level 3  ->  target 1.00
```

The model then learns thin versus thick cirrus rather than discarding the distinction, and
this needs **no change to the head at all** - same six outputs, same loss, just a more
informative target. Cirrus is our weakest class on physical grounds, so taking extra
information for free is worth having.

**One thing that is not a class.** About 25% of every EnMAP scene is empty corner - the
satellite's swath is tilted relative to the rectangular file. That is not a seventh class,
it is "no data", and it gets excluded from training entirely. The `QUALITY_CLASSES` file
marks it for us.

### Common confusions

**If cirrus can be 0.67, is it still a checkbox?**
Yes. The output is still one independent number for cirrus. We are only being more honest
about what the correct answer is.

**Won't six independent outputs produce contradictions - "definitely cloud and definitely
clear"?**
There is no "clear" output. Clear means all six are low. The model can produce odd
combinations, and unlike the single-choice version nothing stops it. In practice it learns
the real co-occurrences from the data.

**Is 25% background wasted?**
It is excluded, not wasted. Training on it would teach the model that black corners are
"clear", which is true and useless.

### Check yourself

1. Why can we not use "pick exactly one label" here?
2. What does "all six outputs are low" mean?
3. A cirrus pixel has level 2. What target do we train against?
4. Why is background excluded rather than made a seventh class?
5. Which single fact about the label files most strongly favours checkboxes?

<details>
<summary>Answers</summary>

1. Because conditions genuinely overlap - cirrus over water - and forcing one choice would
   need an invented priority order.
2. The pixel is clear.
3. `2 / 3 = 0.67`.
4. It is missing data, not a surface condition. Training on it teaches nothing useful.
5. EnMAP ships five separate independent mask files, not one combined label image.

</details>

**Next:** the trap that would otherwise bite us months later.

---

## Part 4 - The rare-class trap

> **The one thing this part teaches:** the fix for "cloud is rare" damages the model's
> sense of certainty, and we need the certainty, so we must undo the damage on purpose.

**The problem, in numbers.** From a cloud-mask experiment dated 2026-08-12, the validation
pixel counts were:

```
clear    12,501,566
cloud        92,994
shadow       10,641
             ----------
total    12,605,201
```

Cloud's share:

```
92,994 / 12,605,201 = 0.00738  =  0.738%
```

So a model that answers **"not cloud" to every single pixel** scores:

```
100% - 0.738% = 99.26% correct
```

**That is the trap.** Left alone, the model learns that answering "no" is nearly always
right, and never learns cloud at all.

**The standard fix.** Tell the loss to care more about the rare class - roughly in
proportion to how rare it is:

```
(100 - 0.738) / 0.738  =  99.262 / 0.738  =  134.5
```

So getting a cloud pixel wrong counts about **135 times** as much as getting a clear pixel
wrong. Now it pays attention.

**The damage this does.** The model is no longer answering "how likely is cloud". It is
answering "how likely is cloud, given that being wrong about cloud is 135 times worse".
Its numbers come out inflated. A pixel it reports as 0.8 is not 80% likely to be cloud.

**The analogy.** A smoke alarm turned up so sensitive it shrieks at toast. It will still
catch every real fire - that is why you turned it up. But "loud" no longer tells you "big
fire". You have traded away the meaning of the reading to gain the sensitivity.

**Why that matters here specifically.** We chose six soft outputs precisely so the numbers
would be usable:

- fade thin cirrus out gradually instead of cutting it in or out
- treat the fuzzy band around a cloud edge - the pixels between about 0.2 and 0.8 - as a
  **learned buffer zone**, instead of shrinking a hard mask blindly with erosion the way
  the pipeline does today (`keep_mask_erosion_kernel_size` in
  `backend/allotrope/action_types/anomaly_scoring.py`)

Both of those need the numbers to mean something. **If we weight the loss and stop there,
we get a model that segments well and whose probabilities lie.**

**So the head has a two-part contract:**

1. Train with the rare-class weighting, so it learns cloud at all.
2. **Recalibrate afterwards** on held-out data, per class, so the outputs read as honest
   probabilities again.

Step 2 is small - it fits one number per class - but skipping it produces a failure nobody
notices for weeks, because the segmentation looks fine and only the downstream soft mask
behaves oddly.

### Common confusions

**Could we skip the weighting and avoid the whole problem?**
You could. You would very likely get a model that never predicts cloud. The 99.26% figure
above is why.

**Is recalibration the same as retraining?**
No. The model is frozen. You are fitting a small correction to its outputs, on data it
never trained on.

**Is 99.26% accuracy a good score?**
It is what "always say no" scores. Whenever you see accuracy quoted on data this
imbalanced, ask what the do-nothing baseline gets first.

### Check yourself

1. What accuracy does a model that always answers "not cloud" achieve here?
2. Show the arithmetic for the weighting factor.
3. Name the two downstream uses that need honest probabilities.
4. Why is the smoke alarm turned up despite the cost?
5. If shadow is 10,641 of 12,605,201 pixels, what is its share, and roughly what weighting
   would it need?

<details>
<summary>Answers</summary>

1. `100% - 0.738% = 99.26%`.
2. `(100 - 0.738) / 0.738 = 99.262 / 0.738 = 134.5`, about 135.
3. Fading out thin cirrus gradually, and using the 0.2-to-0.8 band as a learned edge
   buffer instead of blind erosion.
4. Because missing a real fire costs far more than a false alarm - the same reason we
   accept distorted probabilities to catch rare clouds.
5. `10,641 / 12,605,201 = 0.000844 = 0.0844%`. Weighting:
   `(100 - 0.0844) / 0.0844 = 99.9156 / 0.0844 = 1,184`. Over a thousand - which tells you
   shadow is not really trainable on that data.

</details>

---

## Decisions still open at the time of writing

None of these are blocked by acquiring more scenes.

- **Cirrus and the 1380 nm band.** More scenes will not fix this. The common wavelength
  grid excludes 1350-1450 nm as a water-vapour window (`exclusion_ranges` in
  `app/models/dataset/vendables.py`), and 1380 nm is the band that makes thin cirrus
  obvious. Either accept a weak cirrus class, add that one band back as a side channel, or
  give up reusing Indradhanu's encoder - which expects exactly 165 channels in that order.
- **Freeze the encoder for the first few epochs?** Recommendation: yes, three epochs, so a
  randomly initialised head does not push large gradients into pretrained weights.
- **Confirm six classes** including water.
