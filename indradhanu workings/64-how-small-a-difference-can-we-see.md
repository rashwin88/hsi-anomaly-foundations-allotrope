# 64 - How small a difference can we see? The paired object bootstrap

> **The one thing this part teaches:** the real PRISMA bench is 4 scenes and 909 objects, so a
> difference of 0.005 pAUC between two scorers can be luck in which objects happen to be on it; the
> paired object bootstrap redraws the objects, the same draw for both scorers, thousands of times, and
> from 2026-10-21 a gain counts only if the 95% interval of that difference clears zero.

## Where we are

Parts 60 to 63 measured FT04 against one idea after another, and every verdict leaned on gaps like
0.170 against 0.163. Nobody had asked how big a gap the bench can tell apart from noise. On
2026-10-21 that question got its own small experiment and tool:

| file | what it holds |
|---|---|
| `research/experiments/bench-resolution/README.md` | the question, the method, the results table, the new rule |
| `app/evaluation/significance.py` | `object_pauc_parts` and `paired_bootstrap` - the whole method in about 50 lines |
| `scripts/compare_scorers.py` | the command line: two (model, scorer) pairs from a finished evaluation run |
| `docs/lld/evaluation.md` | the run table, now with the `compare_scorers.py` row |

You need two measures from the anomaly-detection course, part 16: **pAUC <= 1%** (the area under
the ROC curve between 0 and 1% false alarms, divided by 0.01, so random scores about 0.005 and
perfect scores 1), and **objects caught at the top 0.1%** (the share of labelled objects with a pixel
within 1 px of the highest-scoring 0.1% of pixels).

> **Analogy.** Two teachers each mark the same class of 30 essays, and teacher B's average is 0.4
> marks higher. Is B more generous, or did a few essays happen to suit B? You cannot hire a new class,
> but you can ask "if the class had been a slightly different 30 essays, drawn from the same kind of
> students, would B still come out ahead?" You redraw the class from the 30 you have, many times, and
> look at B minus A each time. Crucially, both teachers mark the *same* redrawn class each time -
> otherwise you would be comparing one teacher's easy class with the other's hard one.

## The problem: 909 objects is not many

The gains chased since part 60 are about 0.005 to 0.01 pAUC. FT04's real-bench pAUC with cloud
excluded is 0.170. A gain of 0.005 is a change of 3%:

```
0.005 / 0.170 = 0.029     about 3%
```

pAUC averages over every target pixel, but target pixels are not independent pieces of evidence.
The pixels of one object rise and fall together: if the model reconstructs a rooftop well, all of its
pixels score low at once. **The unit of evidence is the object, and there are 909 of them**, split
across 4 scenes. Swap a dozen hard objects for a dozen easy ones and pAUC moves by more than 0.005.

## Step 1 - what a bootstrap is

A **bootstrap** estimates how much a number would wobble if you had collected different data, using
only the data you have. You draw a new sample of the same size **with replacement** - some items
appear twice, some not at all - recompute the number, and repeat. The spread of the recomputed
numbers stands in for the spread you would have seen across fresh data.

A **95% interval** from a bootstrap is the 2.5th and 97.5th percentiles of those recomputed numbers.
With 1000 draws, sorted from lowest to highest, that is roughly the 25th and the 975th: the middle
950 draws sit between them. `paired_bootstrap` uses `np.percentile(..., [2.5, 97.5])`, which
interpolates between neighbouring draws.

## Step 2 - why resample objects, not pixels

`significance.py` says it in its module docstring: "draw each scene's objects with replacement (the
same draw for both scorers - paired), recompute both measures, repeat". Objects, not pixels, because
of the dependence above. Resampling pixels would pretend the bench had thousands of independent target
pixels, and the interval would come out far too narrow - you would call noise a gain.

The background is **not** resampled. The docstring's reason: "there are millions; their sampling
error is small next to the objects'". Each scene's background pixels are held fixed, and only the
target side of the measure is redrawn.

## Step 3 - pAUC rebuilt from objects

To redraw objects you need pAUC as a sum over objects. The docstring gives the trick:

```
c(p) = max(0, max_fpr - FPR(p)) / max_fpr          max_fpr = 0.01
FPR(p) = share of background pixels scoring at or above target pixel p
scene pAUC = (sum of c over its target pixels) / (number of target pixels)
```

Read `c(p)` as "how much of the first 1% of false alarms this pixel is caught for". A target pixel
that beats every background pixel has `FPR = 0` and is caught over the whole 0-to-1% range: `c = 1`.
A pixel that only beats the bottom 99.5% of background has `FPR = 0.005` and is caught over the
second half: `c = 0.5`. A pixel with `FPR` at or above 0.01 is never caught in that range: `c = 0`.

So `object_pauc_parts` hands back two numbers per object: the **sum of c over its pixels**
and its **pixel count**. A bootstrap draw adds up those pairs for the drawn objects and divides.
Nothing about the score maps is touched again, which is why 1000 draws are cheap.

### By hand: a tiny scene

Do this on paper. One scene, 1,000 background pixels, three objects, four target pixels. Each FPR is
how many background pixels score at or above that target pixel, divided by 1,000:

```
object 1, pixel a    2 background pixels above    FPR 0.002    c = (0.01 - 0.002) / 0.01 = 0.008 / 0.01 = 0.8
object 1, pixel b    6 background pixels above    FPR 0.006    c = (0.01 - 0.006) / 0.01 = 0.004 / 0.01 = 0.4
object 2, pixel c    0 background pixels above    FPR 0.000    c = (0.01 - 0.000) / 0.01 = 1.0
object 3, pixel d   30 background pixels above    FPR 0.030    c = max(0, 0.01 - 0.030) / 0.01 = 0

object parts         object 1 (0.8 + 0.4 = 1.2, 2)    object 2 (1.0, 1)    object 3 (0.0, 1)
scene pAUC           (1.2 + 1.0 + 0.0) / (2 + 1 + 1) = 2.2 / 4 = 0.55
```

Now check it against the ROC area of part 16. Sort the four target pixels by FPR. The true-positive
rate (TPR) is 1/4 from FPR 0, 2/4 from FPR 0.002, 3/4 from 0.006, and pixel d never arrives before
0.01. The area from 0 to 0.01 is three rectangles:

```
FPR 0     to 0.002     width 0.002 x height 0.25 = 0.0005
FPR 0.002 to 0.006     width 0.004 x height 0.50 = 0.0020
FPR 0.006 to 0.010     width 0.004 x height 0.75 = 0.0030
area                   0.0005 + 0.0020 + 0.0030  = 0.0055
pAUC                   0.0055 / 0.01             = 0.55
```

The same 0.55. The docstring says the two agree "up to ties": `detection_measures` puts one ROC point
at the end of each run of tied scores and interpolates the segment that crosses 0.01, while `c(p)`
treats every pixel as its own step. With millions of distinct float scores the gap is negligible -
and the README reports the bootstrap's point values reproducing the evaluator's (0.1621 / 0.1695).

### By hand: one paired draw

Give scorer B its own parts for the same objects: object 1 (1.0, 2), object 2 (0.9, 1), object 3
(0.2, 1).

```
point values    A = 2.2 / 4 = 0.550     B = (1.0 + 0.9 + 0.2) / 4 = 2.1 / 4 = 0.525     B - A = -0.025
```

One bootstrap draw picks three objects with replacement - say objects 1, 1, 3. Both scorers use that
same draw:

```
A    (1.2 + 1.2 + 0.0) / (2 + 2 + 1) = 2.4 / 5 = 0.48
B    (1.0 + 1.0 + 0.2) / (2 + 2 + 1) = 2.2 / 5 = 0.44       B - A = -0.04
```

Another draw, objects 2, 2, 2:

```
A    (1.0 + 1.0 + 1.0) / 3 = 1.00
B    (0.9 + 0.9 + 0.9) / 3 = 0.90                            B - A = -0.10
```

Look at what moved. A swung from 0.48 to 1.00 between the two draws; B - A moved only from -0.04 to
-0.10. That is why the draw is **paired**: which objects are on the bench moves both scorers together,
and the difference cancels most of it. Draw A and B independently and you would mostly be measuring
the luck of A's draw against the luck of B's.

On the real bench each scene's objects are drawn separately (`rng.integers(len(ids[s]), size=len(ids[s]))`
per scene), each scene's pAUC is computed as above, and the four are averaged - the same "mean over
scenes" as the evaluator.

## Step 4 - objects caught, the second measure

If you give `paired_bootstrap` per-object hits (1 if caught at the cut-off, 0 if not), it also
resamples **objects caught**. `compare_scorers.py` reads them from the run's `per_object.csv`, column
`hit_0.1` by default (`--top 0.1`). One detail differs from pAUC: the hits are **pooled across the
four scenes** (`ha / m`, all caught objects over all drawn objects), not averaged per scene.

## What it does not cover: only 4 scenes

The bench has 4 scenes. The bootstrap redraws objects *inside* each scene and never redraws the
scenes. A model can be lucky in the scenes too - one scene that suits it - and this interval cannot
see that. The README says it plainly: "Scene-level uncertainty (4 scenes) is not covered and is
larger: this interval is a lower bound on the real uncertainty."

Why not bootstrap the scenes as well? Count the distinct ways to draw 4 scenes from 4 with
replacement (order not mattering):

```
choose(4 + 4 - 1, 4) = choose(7, 4) = (7 x 6 x 5 x 4) / (4 x 3 x 2 x 1) = 840 / 24 = 35
```

Thirty-five possible draws, one of which is "scene 2 four times". That is too few, and too lumpy, to
give an interval worth trusting. **So read every interval below as the narrowest the truth could be,
not the actual width.**

## Running it

```
python scripts/compare_scorers.py configs/eval/scene_adapt.json "FT04_ep10|RM, cloud excluded" \
    "adapt50|RM, cloud excluded" [--top 0.1] [--boot 2000]
```

Each side is `<model name>|<scorer label>`, exactly as in that run's `per_scene.csv` (`-` as the model
name for a model-free scorer). It reads the cached score maps and `per_object.csv` of a **finished**
run, only for `prisma_bench` scenes, and recomputes nothing. It prints both point values, B - A, the
95% interval and `P(B>A)`, the share of draws in which B beat A.

## The results (README, 1000 draws)

| comparison (A -> B) | pAUC <= 1%: B - A [95% interval] | caught top 0.1%: B - A [95% interval] |
|---|---|---|
| FT04 RM -> + cloud excluded | 0.162 -> 0.170, **+0.0074 [+0.0047, +0.0101]** | 14.9 -> 15.8%, **+1.0 pts [+0.4, +1.7]** |
| FT04 (cloud excl.) -> gated adaptation, N = 50 | 0.170 -> 0.160, **-0.0091 [-0.0176, -0.0028]** | -0.7 pts [-1.7, +0.2] |
| FT04 (cloud excl.) -> FT06 residual / sqrt b | 0.170 -> 0.163, -0.0067 [-0.0174, +0.0037] | +0.8 pts [-0.4, +2.0] |

Read it row by row:

- **Cloud exclusion is a real gain.** Both intervals sit wholly above zero. It is the one change
  adopted since FT04 (parts 57 and 61).
- **Gated per-scene adaptation's loss is real** on pAUC - the interval sits wholly below zero (anomaly
  course, part 25). On objects caught the interval straddles zero.
- **FT06's sqrt variant is a tie** (part 63). Its pAUC loss and its top-0.1% gain both have intervals
  that cross zero.

The README's summary of resolution: interval half-widths of about **+-0.005 to +-0.01 pAUC and +-1
to +-1.5 points of objects caught at the top 0.1%**. Cloud exclusion's own interval is narrower
(half-width `(0.0101 - 0.0047) / 2 = 0.0027`). A likely reason, from the method rather than the README:
excluding cloud leaves most objects' scores untouched, so most per-object differences are zero and the
paired difference barely wobbles.

The scoring-gains comparisons of the anomaly course's part 24 could not be rerun. Their cached score
maps are keyed by the `ScorerSpec` JSON of the time, and the fields added since (`scale_power`,
`scale_floor_pct`) change every key (`docs/lld/evaluation.md`).

## The new rule

**A gain counts only if its paired-bootstrap 95% interval clears zero.** From now on every run
reports the interval against FT04 next to its point values. Note what that does to an old pass
criterion: "+0.005 pAUC" sits at the edge of what this bench can resolve, so a run that clears
+0.005 on its point value may still be a tie. Part 65's runs are the first judged this way.

## Common confusions

**"The interval says there is a 95% chance the true gain lies inside it."**
It says how far the difference wobbles when the *objects* are redrawn, scenes held fixed. Scene luck
is extra, so the real uncertainty is wider than this.

**"Resampling pixels would be more precise - there are far more of them."**
More numbers, not more evidence. An object's pixels move together; counting them separately
manufactures confidence the bench does not have.

**"Why not bootstrap A and B separately and see if their intervals overlap?"**
Because the luck of the draw moves both together. The by-hand draws show it: A swung by 0.52 while
B - A moved by 0.06. Separate intervals would overlap almost always, and you would call real
differences ties.

**"pAUC from `c(p)` is an approximation of the real thing."**
It is the same area, rearranged so each target pixel's share can be added up. It differs from the
evaluator only at tied scores.

**"Objects caught is averaged per scene, like pAUC."**
Not in the bootstrap. pAUC is a mean over the four scenes; objects caught is pooled over all drawn
objects.

**"The gated adaptation loss straddles zero, so it is a tie."**
On objects caught, yes. On pAUC its interval [-0.0176, -0.0028] is wholly below zero, so the pAUC loss
is real. Read each measure's interval on its own.

## Check yourself

1. In the tiny scene, pixel d's score rises until 5 background pixels score at or above it. What is
   its `c`, and the scene's pAUC for A?
2. With 2000 bootstrap draws, roughly which sorted draws mark the 95% interval?
3. A result reads `+0.006 [-0.001, +0.013]`. Does it count as a gain under the new rule?
4. Why is the background not resampled, and what does that leave out?
5. In the paired example, draw objects 1, 2, 3 (each once). Compute A, B and B - A.
6. Why can the bootstrap not cover scene-level luck on this bench?

<details>
<summary>Answers</summary>

1. `FPR = 5 / 1000 = 0.005`, so `c = (0.01 - 0.005) / 0.01 = 0.005 / 0.01 = 0.5`. Object 3's parts
   become (0.5, 1). pAUC = `(1.2 + 1.0 + 0.5) / 4 = 2.7 / 4 = 0.675`.
2. The 2.5th percentile of 2000 is about the 50th lowest (`2000 x 0.025 = 50`), and the 97.5th about
   the 1950th (`2000 x 0.975 = 1950`), with `np.percentile` interpolating between neighbours.
3. No. The interval includes zero, so it is a tie, whatever the point value says.
4. There are millions of background pixels, so their sampling error is small next to the objects'
   (the module docstring). It leaves out whatever uncertainty the background does carry - and, more
   importantly, scene-level luck.
5. That is the original bench, so the point values: A `(1.2 + 1.0 + 0.0) / (2 + 1 + 1) = 2.2 / 4 = 0.55`,
   B `(1.0 + 0.9 + 0.2) / 4 = 2.1 / 4 = 0.525`, B - A `= -0.025`.
6. Only 4 scenes: there are `choose(7, 4) = 35` distinct ways to redraw them, too few and too lumpy
   for a trustworthy interval. So the object interval is a lower bound on the real uncertainty.

</details>

---

Next, part 65: two fine-tuning runs, FT07 and FT08, that tried to stop the model learning the
targets - the first runs judged by this rule.
