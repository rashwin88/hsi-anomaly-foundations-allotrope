# Part 11 - Masking

> **The one thing this part teaches:** a pixel that can see itself will be copied, not
> predicted - so every pixel must be reconstructed while hidden, which takes two passes.

## The failure this prevents

Part 6 left a hole in the painter analogy: the painter must not see the aeroplane while
painting it. Here is what happens if they do.

Give an autoencoder an image and ask it to reproduce it. The shortest path to low loss is to
learn the identity function - copy input to output. Reconstruction is then perfect
everywhere, error is zero everywhere, and **the detector finds nothing while reporting no
error at all**.

The bottleneck makes copying hard, not impossible. Masking makes it impossible.

## Masked reconstruction

Hide part of the input. Ask the network to predict the hidden part from what remains.

Now it cannot copy - the answer is not in front of it. To do well it must learn that
hedgerows continue, that a field has consistent texture, that a river bends smoothly. It
must learn what this terrain *is like*.

This is the cloze test from language teaching: "the cat sat on the ___". You fill it from
context. If the word were visible, the exercise would measure nothing.

## Three channels, not one

The masked models feed the encoder **three** channels, not one:

```
channel 0:  pixels          (zeroed where hidden)
channel 1:  validity_mask   1 = physically valid, 0 = invalid
channel 2:  input_mask      1 = visible, 0 = hidden or invalid
```

The reason repays attention. A zeroed pixel in channel 0 is **ambiguous**. It could be:

- off-swath, where the sensor never looked
- cloud-masked, where the ground is hidden
- a deliberately hidden prediction target

These demand different responses. For the first two, "there is nothing here" is the right
answer and no prediction is wanted. For the third, a prediction is exactly what is wanted.

Comparing channels 1 and 2 disambiguates:

| validity | input_mask | meaning |
|---|---|---|
| 1 | 1 | valid and visible - use it |
| 1 | 0 | valid but hidden - **predict this** |
| 0 | 0 | invalid - ignore, no prediction expected |
| 0 | 1 | cannot occur |

**That distinction is the whole basis of masked reconstruction.** Without it the model cannot
tell a prediction target from a hole in the data.

## Two passes at inference

Training hides a random subset. Inference has a harder requirement: **every** pixel needs a
score, and every pixel must be predicted while hidden.

One pass cannot do it. Hide half and you score half.

So run twice with complementary masks:

```
pass 1:  hide the black squares   ->  take predictions at black positions
pass 2:  hide the white squares   ->  take predictions at white positions
combine: each pixel from the pass where it was HIDDEN
```

A checkerboard, then its inverse. Every pixel is predicted from context exactly once, and no
pixel ever contributes to its own reconstruction.

Work a 4x4 corner by hand. `1` = visible, `0` = hidden:

```
pass 1 mask          pass 2 mask
1 0 1 0              0 1 0 1
0 1 0 1              1 0 1 0
1 0 1 0              0 1 0 1
0 1 0 1              1 0 1 0
```

Pixel (0,0) is visible in pass 1, hidden in pass 2 - so its final value comes from pass 2.
Pixel (0,1) is hidden in pass 1 - its value comes from pass 1. Every position is hidden in
exactly one pass. **Trace two more positions yourself.**

## Token removal, which is stronger

The SegFormer models go further. Rather than zeroing pixels, they **delete tokens** from the
sequence before the encoder.

A token is a small patch of the image after the embedding step. Zeroing leaves a token that
still occupies a slot and still influences attention. Removing it means the encoder never
processes it at all - no compute spent, and no information leaked.

This forces one design choice worth knowing. `SegFormerEncoder` stage 1 uses
**non-overlapping** patch embedding (`patch_size=4, stride=4, padding=0`), while stages 2 to
4 overlap (`k=3, s=2`).

Overlapping patches share pixels. If a removed token's pixels also sit inside a neighbouring
token, the information survives removal and the model can cheat. Stage 1 must not overlap.
Later stages may, because by then the removal has already happened.

## The traps in this area

**Inference default does not match training.** `InferenceConfig.masking_strategy` defaults to
`"checkerboard"`, while training and validation use random two-pass masking. Recorded in
`docs/09-known-issues.md`.

**A comment that lies.** `segformer_mae_inferencer.py` sets `MIN_VALID_FRACTION = 0.1` under a
comment claiming it "matches training". Training uses **0.4**. The source wins; the comment
is wrong.

**Uncovered pixels differ by model family.** In full-scene inference, a pixel covered by no
patch is filled differently: SegFormer falls back to the original scene, so the residual is
exactly zero and nothing is flagged. The convolutional models fall back to zeros, producing a
large spurious residual. Also in the known-issues register.

## Common confusions

**"Does masking hide whole regions or scattered pixels?"**
Scattered, in a checkerboard or a random pattern. Hiding a contiguous block would make the
task inpainting - a different and much harder problem.

**"Why not one pass hiding 10%, repeated ten times?"**
You could, at ten times the cost. Two complementary passes cover everything in two.

**"If channel 0 is zeroed where hidden, the model still sees a zero. Is that a leak?"**
The zero carries no information about the true value - it is a constant. Channel 2 tells the
model that position is a prediction target. What matters is that the true value is absent.

**"Is masking used at training only?"**
Both, but differently. Training hides a random subset each step; inference runs two
complementary passes so every pixel is covered exactly once.

## Check yourself

<details>
<summary>1. Why does an unmasked autoencoder fail as an anomaly detector, and why is the failure dangerous?</summary>

It learns the identity function, reconstructing everything perfectly including anomalies, so
error is near zero everywhere and nothing is flagged. It is dangerous because it fails
silently - no error, no warning, just an empty result.
</details>

<details>
<summary>2. What do validity=1, input_mask=0 mean together, and why can't one channel express it?</summary>

A valid pixel deliberately hidden as a prediction target. One channel cannot distinguish it
from an invalid pixel, since both appear as zero in the pixel channel - and they demand
opposite responses.
</details>

<details>
<summary>3. For the 4x4 masks above, which pass supplies the final value at (1,1) and at (2,3)?</summary>

Position (1,1): pass 1 mask is `1` (visible), pass 2 is `0` (hidden). Its value comes from
**pass 2**.

Position (2,3): pass 1 is `0` (hidden), pass 2 is `1`. Its value comes from **pass 1**.
</details>

<details>
<summary>4. Why must SegFormer stage 1 use non-overlapping patch embedding?</summary>

Overlapping patches share pixels. If a removed token's pixels also appear in a neighbouring
token, the information survives removal and the model can reconstruct from leaked data.
Later stages may overlap because removal has already happened.
</details>

<details>
<summary>5. A colleague says inference uses the same masking as training. Are they right?</summary>

No. `InferenceConfig.masking_strategy` defaults to `"checkerboard"` while training and
validation use random two-pass masking. There is also a comment claiming
`MIN_VALID_FRACTION = 0.1` matches training when training uses 0.4. Both are in the
known-issues register.
</details>

---

Next: [part 12](12-residual-to-score.md) - turning reconstruction error into a decision.
