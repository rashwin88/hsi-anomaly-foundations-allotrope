# Part 6 - Anomaly as surprise

> **The one thing this part teaches:** every detector here answers one question - how badly
> does this pixel fit the pattern - and the two families differ only in how they define the
> pattern.

## The move

Part 1 reframed "is this a gas leak?" into "is this pixel unlike its surroundings?". This
part makes that computable.

Three steps, and every detector in this repository is an instance of them:

1. Build a model of what **normal** looks like.
2. Measure how badly each pixel **fits** that model.
3. The worst-fitting pixels are the candidates.

The families differ only in step 1.

## Family one: statistical

Describe normal with statistics computed from the scene itself. The mean spectrum, and how
bands vary together. Then measure distance from that description.

**No training. No checkpoint. No labels.** Feed it a scene and it computes what it needs from
that scene, then scores it. Part 8 covers these.

## Family two: reconstruction

Train a neural network to **rebuild ordinary imagery**. Show it a scene and it redraws it.
Wherever its redrawing is badly wrong, something unusual is there.

The training needs no anomaly labels - the target is the input itself. Part 10 covers these.

## Why reconstruction error is a sensible score

A painter who has spent thirty years painting the same countryside can reproduce it from
memory: the hedgerows, the light on the fields. Ask them to paint the same scene with a
crashed aeroplane in the middle and the landscape still comes out right - the aeroplane
comes out wrong, because they have never painted one.

Compare their painting to the photograph. The difference is near zero across the fields and
large exactly where the aeroplane is. **You have located the aeroplane without ever telling
them what one looks like.**

The network is the painter. Reconstruction error is the difference. That is the whole idea.

## Where the analogy stops

Two places, both of which matter.

**The painter must not see the aeroplane while painting it.** If they copy directly, the
reproduction is perfect everywhere and the error is zero everywhere. Part 11 is entirely
about preventing this, and it is subtler than it sounds.

**A good enough painter reproduces the aeroplane too.** A network with enough capacity
learns to copy rather than to understand, and copies anomalies faithfully. Detection then
fails silently - low error everywhere, nothing flagged, no error message. This is why the
models here are deliberately small: Chakshu has about 406,000 parameters, which for a modern
network is tiny.

Retire the analogy there rather than stretching it.

## Both families, both worth having

They fail differently, which is the argument for keeping both.

| | statistical | reconstruction |
|---|---|---|
| needs training | no | yes, hours on a GPU |
| needs a checkpoint | no | yes |
| new sensor | works immediately | needs retraining or transfer |
| finds | statistically unusual spectra | patterns unlike its training data |
| misses | anomalies close to the scene mean | anomalies resembling its training set |
| interpretable | yes - a distance | not really |

The product runs both and shows both. Where they agree, confidence is high.

## Scoring produces a map, not a decision

Every detector outputs a **score map** - one number per pixel, same height and width as the
scene, higher meaning more anomalous.

It does not output a yes/no mask. That comes later, when a human picks a threshold, and it
is a separate Action for the reasons in part 3.

Invalid pixels are excluded throughout - either `NaN` or zero depending on the detector, and
always excluded from statistics computed over the map.

## Common confusions

**"Is this unsupervised learning?"**
The statistical family is not learning at all - it is descriptive statistics. The
reconstruction family is self-supervised: there are labels, but they are generated from the
input rather than provided by a human. "Unsupervised" gets used loosely for both.

**"Reconstruction error - error against what?"**
Against the input itself. The network is given a scene and asked to reproduce it; the error
is per-pixel difference between input and output. There is no external ground truth.

**"If the network reconstructs perfectly, that is good, surely?"**
Perfect reconstruction everywhere means zero error everywhere, which detects nothing. You
want a model good enough to rebuild ordinary terrain and bad enough to fail on things it has
never seen. That balance is why capacity is constrained deliberately.

**"Two families - which is better?"**
Wrong question. They fail differently. Read the table again.

## Check yourself

<details>
<summary>1. State the three steps every detector here performs.</summary>

Model what normal looks like; measure how badly each pixel fits; rank the worst fits as
candidates. The families differ only in the first step.
</details>

<details>
<summary>2. What does a reconstruction model train on, given there are no labels?</summary>

The input itself. It is shown imagery and asked to reproduce it, so the target is generated
from the data rather than supplied by a human - self-supervised rather than unsupervised.
</details>

<details>
<summary>3. Why is a model with more capacity not automatically better here?</summary>

Enough capacity lets it learn to copy rather than to understand, so it reproduces anomalies
faithfully and the error goes to zero everywhere. Detection then fails with no error
message. Hence small models - Chakshu is about 406,000 parameters.
</details>

<details>
<summary>4. Three pixels have reconstruction errors 0.02, 0.31, 0.04, and a fourth is invalid. Which is most anomalous, and what happens to the fourth?</summary>

The 0.31 pixel - roughly ten times the others. The invalid pixel is excluded entirely: not
scored, and not counted in any statistic computed over the map. Include it and whatever fill
value it holds may well outrank 0.31.
</details>

<details>
<summary>5. In the painter analogy, what corresponds to the training set, and what would the analogy fail to capture?</summary>

Thirty years of painting that countryside is the training set. It fails to capture that the
painter must not see the aeroplane while painting - part 11 - and that a sufficiently
skilled painter would reproduce it anyway.
</details>

---

Next: [part 7](07-the-band-pipeline.md) - making five sensors comparable.
