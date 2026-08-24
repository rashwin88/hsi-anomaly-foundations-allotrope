# Part 1 - The problem

> **The one thing this part teaches:** we are looking for things nobody has ever labelled,
> which rules out almost every technique you already know.

## The situation

A satellite photographs a patch of ground. Somewhere in that image is something that should
not be there: a gas leak, an illegal dump site, an unusually hot piece of machinery, a
mineral deposit, a chemical spill.

Nobody has marked it. Nobody has marked one in any other image either. And nobody can tell
you in advance what it will look like, because the interesting cases are the ones nobody
anticipated.

## Why your first instinct will not work

The obvious approach is a classifier. Collect examples of gas leaks, train a model to
recognise them, run it over new imagery.

That fails for three separate reasons, and it is worth being clear about all three because
each one closes off a different escape route.

**There is no training set.** Not "a small one" - none. Labelling requires someone to have
visited the site and confirmed what was there, for thousands of examples, across every
terrain type and season.

**The target class is open-ended.** Even with a thousand labelled gas leaks, you would have
a gas-leak detector. The next real find might be a chemical you have never seen. A
classifier trained on known classes is blind to the unknown ones by construction, and the
unknown ones are the point.

**The interesting things are rare.** A scene has tens of millions of pixels and perhaps a
few dozen anomalous ones. Train a classifier on that and it learns to answer "normal" every
time, scoring 99.9999% accuracy while being completely useless.

## The reframing

Stop asking "is this a gas leak?" and start asking **"is this pixel unlike its
surroundings?"**

That question needs no labels. It can be answered from the image alone, because every scene
carries its own definition of normal. And it catches things nobody anticipated, because
"unlike everything else here" does not require knowing what the thing is.

The cost is honesty about what you get back. The system does not say "gas leak". It says
"these 40 pixels are the least ordinary in this scene, ranked". A human decides what to do
with that. **The output is a shortlist for an analyst, not a verdict.**

## An analogy that will keep working

A proofreader who does not speak the language can still find the typo. They do not know what
the words mean, but they know what letter patterns look normal in this text, and `teh`
does not. They cannot tell you it should be `the` - only that something here is off.

That is exactly the trade. You give up saying *what* in exchange for not needing to be told
*what to look for*.

## Why satellites make this harder than it sounds

Two complications you will meet constantly.

**Scale.** A single PRISMA scene is roughly 1,210 by 1,219 pixels with 239 measurements per
pixel. That is about 352 million numbers, per scene, and there are many scenes.

**Not every pixel is real.** Sensors have dead detectors. Clouds block the ground. Scenes
have ragged edges where the satellite's swath did not cover the rectangle. A large fraction
of any scene is data you must actively ignore, and forgetting to ignore it is the single
most common source of false detections in this codebase. You will meet **validity masks** in
part 5 and they will follow you to the end.

## Common confusions

**"Anomaly detection means outlier detection, right?"**
Related, but the spatial part matters. A pixel can hold a perfectly ordinary temperature and
still be anomalous because everything around it is 20 degrees colder. Several detectors here
score a pixel against its immediate neighbours rather than against the whole scene - part 9.

**"So this replaces the analyst?"**
No, and designing as if it did would break it. The system narrows tens of millions of pixels
to a few dozen candidates. The threshold that decides how many is chosen by a human looking
at the score map, which is why there is an interactive step in the product at all - part 3.

**"If there are no labels, how is anything trained?"**
Two families. The classical detectors are not trained at all; they compute statistics from
the scene in front of them. The neural models are trained, but on a task that needs no
labels - reconstructing ordinary imagery. Part 6 sets this up and part 10 does it properly.

**"Anomaly and outlier and novelty all mean the same thing here?"**
In this codebase, effectively yes, and the code says "anomaly" throughout. Do not read
significance into the choice.

## Check yourself

<details>
<summary>1. Why can you not solve this with a supervised classifier, in one sentence per reason?</summary>

No labelled examples exist; the target class is open-ended so a classifier is blind to
unknown categories; and the extreme class imbalance means "always say normal" scores
near-perfect accuracy.
</details>

<details>
<summary>2. A scene has 40 anomalous pixels out of 20 million. What accuracy does a model achieve by answering "normal" for every pixel? Work it out.</summary>

```
normal pixels    = 20,000,000 - 40 = 19,999,960
correct answers  = 19,999,960
accuracy         = 19,999,960 / 20,000,000
                 = 0.999998
                 = 99.9998%
```

Which is why accuracy is a meaningless metric here. Part 12 covers what is used instead.
</details>

<details>
<summary>3. What does the system output, and what does it deliberately not output?</summary>

A ranked shortlist of candidate pixels with coordinates, and for hyperspectral data a
likely material. It does not output a classification such as "gas leak", and it does not
decide the threshold - a human does.
</details>

<details>
<summary>4. Why is a large fraction of a scene unusable, and why does that matter more than it sounds?</summary>

Dead detectors, cloud cover, and ragged swath edges. It matters because invalid pixels
often hold extreme or zero values, and a detector that forgets to exclude them will rank
those extremes as the most anomalous thing in the scene.
</details>

<details>
<summary>5. In the proofreader analogy, what corresponds to the scene, and what corresponds to the anomaly score?</summary>

The text is the scene; each letter pattern is a pixel. The score is how unusual a pattern
looks against the rest of the text. The proofreader's inability to say what the word should
be corresponds to the system's inability to name the anomaly's cause.
</details>

---

Next: [part 2](02-light-bands-cubes.md) - what a hyperspectral cube actually is.
