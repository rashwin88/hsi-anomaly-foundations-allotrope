# 01 · The problem: finding things nobody has labelled

> **The one thing this part teaches:** we never teach the computer what an
> anomaly looks like. We teach it what *normal* looks like, and call whatever is
> left over the anomaly.

---

## The situation

A satellite flies over a patch of ground and takes a picture. Somebody at a
mining ministry, an environmental agency or an insurance company wants to know:

**"Is there anything unusual down there?"**

Unusual might mean an illegal mine. A leaking pipeline. A chemical dumped in a
field. A vehicle parked where no vehicle should be. A mineral outcrop that could
be worth money.

---

## The obvious approach, and why it fails

If you have done any machine learning before, your instinct is:

1. Collect 10,000 pictures of illegal mines.
2. Label them "illegal mine".
3. Train a classifier.
4. Show it new pictures.

That approach is completely standard and it does not work here. Three reasons,
and each one alone would sink it.

### Reason 1 — There are no labels

To label a satellite pixel you have to know what is on the ground under it. That
means somebody physically walking to that spot with a GPS.

Nobody has done that for the hundred thousand pixels in one scene. We have
essentially **zero** labelled anomalies. A classifier with no training labels is
not a classifier, it is a blank slate.

### Reason 2 — You do not know what you are looking for

A classifier can only find the things you named in advance. If you train it on
"illegal mine" and "oil spill", it will find illegal mines and oil spills.

But the valuable discovery is almost always the one nobody anticipated. The
whole point of surveying is to find the thing you were not expecting.

### Reason 3 — The interesting thing is vanishingly rare

Suppose 0.001% of pixels are genuinely anomalous. In a million-pixel scene, that
is ten pixels.

Now consider a classifier that ignores its input entirely and always answers
"normal". Its accuracy is **99.999%**. It is also completely useless. Rarity
breaks the ordinary way of measuring success.

---

## The reframe

Stop asking "is this pixel a mine?" That question needs labels.

Ask instead:

> **"Given everything else in this picture, how surprising is this pixel?"**

That question needs no labels at all. It needs only a model of what normal looks
like. Anything the model cannot account for is, by definition, a surprise.

Here is the sentence to memorise:

> ### We never model anomalies. We model normality, and call the residue anomaly.

Read it again. Everything in the next 29 parts is machinery for building "a
model of normality" and machinery for measuring "the residue".

---

## An analogy: the proofreader

You are proofreading a book written in a language you speak fluently. You do not
have a list of possible typos. You have never memorised "common typos in this
book".

What you have is a deep sense of what correct sentences look like. When you hit
"the cat sat on the mta", your eye snags. You did not recognise a known error —
you failed to recognise a normal word.

That snag is the anomaly score.

---

## Two ways this codebase models normality

There are exactly two families, and Indradhanu belongs to the second.

### Family 1 — Classical detectors

These compute statistics of the scene itself: the average pixel, and how much
pixels typically vary. Then they measure how far each pixel is from that
average, in units of "typical variation".

- No training. No neural network. No saved file of learned weights.
- They are fitted fresh on every single scene, from that scene's own pixels.
- Example in this repo: **MNF-RX**, in `app/detectors/`.

The classic version is called **RX**, and it measures something called
Mahalanobis distance — a distance that accounts for the fact that some
directions in the data vary more than others.

> **Historical note you will hear referenced:** plain RX on 165-band data was
> deliberately abandoned in this project on 2026-05-11. With that many bands the
> statistics become numerically unstable and the distances blow up to absurd
> values (around 1e11 — that is 100 billion). **MNF-RX** replaced it: same idea,
> but the statistics are computed in a much smaller, better-behaved space. Do
> not try to reintroduce plain RX; it has been tried.

### Family 2 — Foundation models

These are neural networks trained, once and offline, on tens of thousands of
ordinary satellite images. They learn to redraw imagery. The error in their
redrawing is the anomaly score.

- Training happens once, on a big machine, and produces a file of weights.
- Running it on a new scene just loads that file.
- Examples in this repo: **Indradhanu** (hyperspectral) and **Chakshu**
  (thermal).

> **Careful with the words "foundation model".** In the wider industry that
> phrase usually means an enormous language model like Claude or GPT. In *this*
> repo it means something much more specific and much smaller: *a network
> trained to reconstruct imagery, whose reconstruction error is used as the
> anomaly score*. Indradhanu has about 5.5 million parameters. A large language
> model has hundreds of billions. Different world entirely.

---

## What Indradhanu actually outputs

Not a yes/no. Not a list of anomalies. Not a box drawn around anything.

It outputs a **score map**: one number per pixel, never negative, where higher
means more surprising. A picture-sized grid of numbers.

A human then looks at that map, drags a threshold slider, and decides where to
cut. That human step is deliberate and happens in a completely separate step of
the pipeline (`anomaly_detection_prep`). Part 30 closes that loop.

---

## One rule that will bite you on day one

**Thresholds in this system are always percentile-based, never absolute.**

Here is why, concretely. Suppose you decide "anything scoring above 0.05 is an
anomaly".

- Scene A is a calm desert. Its typical score is 0.002 and its worst pixel is
  0.04. Your rule flags **nothing**, including the real anomaly.
- Scene B is a cloudy coastal city. Its typical score is 0.08. Your rule flags
  **the entire image**.

The absolute numbers differ by an order of magnitude between scenes because the
imagery differs. So the rule is always relative: "flag the top 0.1% of pixels in
*this* scene".

---

## Common confusions

**"So the model is trained to detect anomalies?"**
No. It is trained to *redraw ordinary imagery*. Detection is something we do
afterwards, by subtracting. The model has no concept of an anomaly.

**"Does it need examples of anomalies to work?"**
No, and that is the entire point. It has never been shown one on purpose.

**"Is a high score a guarantee of something interesting?"**
No. A high score means "the model could not explain this". Sensor glitches,
cloud edges and unusual-but-boring terrain also score high. Turning a high score
into a real finding is the job of the later steps in the chain.

---

## Check yourself

1. Give the three reasons a supervised classifier is the wrong tool here.
2. Fill in the blank: "We never model ______, we model ______."
3. Which of the two detector families needs a training run, and which is fitted
   fresh on every scene?
4. Why is a fixed score threshold like "0.05" a bad idea?

<details>
<summary>Answers</summary>

1. No labels exist; you cannot name in advance what you are looking for; the
   target is so rare that a "always say normal" classifier scores 99.999%.
2. "We never model **anomalies**, we model **normality**." (And call the residue
   the anomaly.)
3. Foundation models (Indradhanu, Chakshu) are trained once, offline. Classical
   detectors (MNF-RX) are fitted fresh on each scene and have no learned
   weights.
4. Typical score magnitudes differ by roughly 10x between scenes, so one fixed
   cut flags everything in one scene and nothing in another. Thresholds are
   percentiles of the current scene.

</details>

---

**Next:** what a satellite picture actually contains, in
[02-hyperspectral-101.md](02-hyperspectral-101.md)
