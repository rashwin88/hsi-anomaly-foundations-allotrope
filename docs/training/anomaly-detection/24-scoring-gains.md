# Part 24 - Three ways to improve the score, and why none was kept

> **The one thing this part teaches:** on 2026-10-19 three changes to the *score* were tried on FT04's
> fixed reconstruction - blurring the residual (a spatial matched filter), fusing residual Mahalanobis
> with PCA-RX by rank, and whitening each background type separately - and every one lost real objects
> to plain residual Mahalanobis, even the one that raised pixel pAUC and formally passed its criteria;
> a pixel measure counts a big object's pixels many times over and a 1-pixel target once, so pass
> criteria must count objects caught, and residual Mahalanobis stays the score.

## Where we are

Part 23 put FT04 into the product as Indradhanu V2. Its detection case was thin: on the real PRISMA
bench it ties v0.2.0 (pAUC 0.170 against v0.2.0's 0.165, both with cloud excluded). The same pattern holds for
every recent model - v0.2.0, FT03 and FT04 all tie once you score them with residual Mahalanobis (RM,
part 12). The experiment record puts it this way: RM "absorbs model gains".

So the question on 2026-10-19 was the other half: **keep the reconstruction fixed and change only the
score.** Can a better score find more?

| file | what it holds |
|---|---|
| `research/experiments/scoring-gains/README.md` | the question, the pass criteria fixed before the run, results, observations, verdict |
| `research/experiments/scoring-gains/report.md` | the evaluator's full tables, by tier and by sensor |
| `configs/eval/scoring_gains.json` | the run: one model (FT04 epoch 10), two datasets, nine scorers |
| `app/evaluation/scorers.py` | `blur`, `blurred_mahalanobis`, `rank_map`, `background_types`, and the `fuse` / `mixture` branches of `score` |
| `app/evaluation/config.py` | `ScorerSpec`'s new fields `spatial_sigma`, `parts`, `rule`, `k` |
| `app/evaluation/runner.py` | the score-map cache, and a float16 fix this run forced |
| `docs/09-known-issues.md` | the new entry "bench-v2.1 places implants where real targets do not occur" |
| `docs/lld/evaluation.md` | the scorer list and the run table, now including this run |

These are evaluation scorers, not product settings. Nothing in `anomaly_scoring` changed; none of the
three was adopted.

> **Analogy.** A school wants better exam results without changing the teaching, so it tries
> changing the marking. Rule one: give credit for answers that are nearly right. The class average
> rises - but the students who got one hard question exactly right and nothing else now look like
> everyone else. Rule two: average two examiners' rankings. Rule three: grade each class against its
> own class average. Rule three makes the mock exam look great, because the mock spread its hard
> questions evenly over every class, while the real exam's hard cases all sat in one small class -
> who, graded against themselves, now look ordinary. Before trusting any new marking, count how many
> individual students it identifies correctly, not only the average mark.

## Step 1 - the three ideas, and what "a gain" had to mean

| | idea | why it might help | `ScorerSpec` |
|---|---|---|---|
| A1 | spatial matched filter: blur the residual (sigma 1.0 or 1.4 px) before whitening | a target spreads over a point-spread footprint (part 20, sigma about 1.4); its residual adds up across the footprint while uncorrelated noise averages down | `spatial_sigma` |
| A1+ | rank mean of plain RM and RM blurred at 1.4 | keeps the 1-pixel targets that blurring dilutes | `fuse` |
| A2 | RM + PCA-RX (part 19), fused by rank mean or rank max | PCA-RX wins where a target is foreign to the scene, RM where it is foreign to the model; the hope was that they fail on different objects | `fuse`, `rule` |
| A3 | mixture of backgrounds: k-means into k = 6 or 10 types, then RM whitened per type | one covariance mixes land, water and roofs; each type's own normal is tighter | `mixture`, `k` |

Two words. A **matched filter** is a filter shaped like the signal you are looking for: if a target
always arrives smeared into a known footprint, weighting the neighbours by that same footprint adds
the target's signal up faster than it adds the noise up. **k-means** is the standard clustering
recipe: pick k centres, give each pixel to its nearest centre, move each centre to the mean of its
pixels, repeat.

The README fixed the pass criteria **before** the run. A scorer counted as a gain over RM on FT04 if:

1. real PRISMA bench: pAUC up to 1% false alarms rose by at least 0.005 **and** TPR at 0.1% false
   alarms did not fall;
2. bench-v2.1: objects caught at the top 1% fell by no more than 0.5 points.

Criterion 2 is only a guard. bench-v2.1's implants are blurred with sigma 1.4 by construction (part 20),
so a sigma-1.4 blur would partly be measuring its own reflection there. The real bench is the judge.

One detail: the baseline here is FT04 with plain RM and **no** cloud exclusion - pAUC 0.162, not the
0.170 that part 23 quoted for V2 with its own cloud mask. Every row in this part is on the same footing.

## Step 2 - the results

From the README (real PRISMA bench: 4 scenes, 909 objects; bench-v2.1: 18 hosts):

| scorer | real pAUC | real TPR @ 0.1% | real caught top 1% / 0.1% | v2.1 caught top 1% | criteria |
|---|---|---|---|---|---|
| RM (baseline) | 0.162 | 0.061 | **45.7% / 14.9%** | 35.1% | - |
| A1 RM blur 1.0 | 0.177 | 0.064 | 40.7% / 12.0% | 33.4% | 1 met, 2 failed |
| A1 RM blur 1.4 | 0.177 | 0.069 | 36.3% / 10.2% | 29.4% | 1 met, 2 failed |
| A1+ RM + RM blur 1.4 | 0.180 | 0.064 | 43.0% / 12.8% | 35.0% | both met |
| A2 rank mean | 0.162 | 0.057 | 43.8% / 14.0% | 34.9% | failed |
| A2 rank max | 0.138 | 0.046 | 42.7% / 11.4% | 34.8% | failed |
| A3 mixture k=6 | 0.111 | 0.045 | 39.4% / 11.4% | 41.2% | failed |
| A3 mixture k=10 | 0.096 | 0.034 | 34.4% / 8.1% | 41.9% | failed |
| PCA-RX (no model) | 0.128 | 0.029 | 39.7% / 8.1% | 33.2% | - |

Read the bold column first. **No scorer catches as many real objects as plain RM, at either budget.**
Now look at the A1 rows: pixel pAUC goes up, objects caught go down. That is the puzzle the next step
works by hand.

## Step 3 - A1: why blurring raised pixel measures and lost objects

### The promise, by hand

Take one band, already in units of the background's spread, background mean 0. Use the 3-tap kernel
`[0.25, 0.5, 0.25]` as a stand-in for the Gaussian. Pure noise first: three independent pixels, each
with spread 1. The blurred value is a weighted sum, and the variance of a weighted sum of independent
values is the sum of squared weights times each variance:

```
noise variance after blur = 0.25^2 + 0.5^2 + 0.25^2
                          = 0.0625 + 0.25 + 0.0625
                          = 0.375
noise spread after blur   = sqrt(0.375) = 0.612
```

Now a target that arrives already smeared into the same footprint, values `[1, 2, 1]`:

```
blurred centre = 0.25 x 1 + 0.5 x 2 + 0.25 x 1
               = 0.25 + 1 + 0.25
               = 1.5
signal-to-noise, raw     = 2 / 1     = 2.00
signal-to-noise, blurred = 1.5 / 0.612 = 2.45
```

That is the matched-filter promise: 2.45 against 2.00. Now a target sharper than the kernel, one
pixel wide, values `[0, 2, 0]`:

```
blurred centre = 0.25 x 0 + 0.5 x 2 + 0.25 x 0 = 1.0
signal-to-noise, blurred = 1.0 / 0.612 = 1.63      (raw was 2.00)
```

The same filter that helps a footprint-shaped target hurts a 1-pixel one. In the real code
(`blurred_mahalanobis`) the whitening is refitted on the blurred residual, which is the "divide by the
new spread" step above. `blur` uses normalised convolution - it divides by the blurred validity mask -
so nodata does not bleed into the edge of the scene.

### What a pixel measure rewards, by hand

You need a pen for this one; it takes about five minutes. A strip of 12 pixels, one band, background
mean 0, score = residual squared (a 1-band global RM). Pixels 5-8 are a big object, pixel 11 is a
1-pixel target, pixels 2 and 3 are noise that flips sign from one pixel to the next:

```
pixel      1    2     3    4   5   6   7   8   9   10   11   12
residual   0   4.5  -4.2   0   3   4   5   3   0    0   4.8   0
           .  noise noise  .  [ big object  ]  .    .  target .
score r^2  0  20.25 17.64  0   9  16  25   9   0    0  23.04  0
```

The analyst's budget is the top 4 pixels. Plain RM's top 4: pixel 7 (25), pixel 11 (23.04), pixel 2
(20.25), pixel 3 (17.64).

Now blur each pixel: `b_i = 0.25 x r_(i-1) + 0.5 x r_i + 0.25 x r_(i+1)`, zero outside the strip.

```
b2  = 0.25 x 0    + 0.5 x 4.5  + 0.25 x -4.2 = 0     + 2.25 - 1.05  =  1.2    -> 1.44
b3  = 0.25 x 4.5  + 0.5 x -4.2 + 0.25 x 0    = 1.125 - 2.1  + 0     = -0.975  -> 0.95
b5  = 0.25 x 0    + 0.5 x 3    + 0.25 x 4    = 0     + 1.5  + 1     =  2.5    -> 6.25
b6  = 0.25 x 3    + 0.5 x 4    + 0.25 x 5    = 0.75  + 2    + 1.25  =  4.0    -> 16
b7  = 0.25 x 4    + 0.5 x 5    + 0.25 x 3    = 1     + 2.5  + 0.75  =  4.25   -> 18.06
b8  = 0.25 x 5    + 0.5 x 3    + 0.25 x 0    = 1.25  + 1.5  + 0     =  2.75   -> 7.56
b11 = 0.25 x 0    + 0.5 x 4.8  + 0.25 x 0    = 0     + 2.4  + 0     =  2.4    -> 5.76
```

(Pixels 1, 4, 9, 10 and 12 come out at 1.27, 0.09, 0.56, 1.44 and 1.44 - all small.) Blurred top 4:
pixel 7 (18.06), 6 (16), 8 (7.56), 5 (6.25). The target, at 5.76, is fifth.

| | object pixels in top 4 (of 5) | false alarms in top 4 | objects caught (of 2) |
|---|---|---|---|
| plain | 2 (pixels 7, 11) | 2 (pixels 2, 3) | 2 |
| blurred | 4 (pixels 5-8) | 0 | 1 |

A pixel measure such as TPR doubles, 2/5 = 0.4 to 4/5 = 0.8, and the false alarms vanish. Objects
caught halves. **A pixel measure counts each pixel of an object, so filling in more of an object you
had already found looks like progress, and losing a 1-pixel target costs one pixel's worth.** (Three
simplifications: one band, a global mean instead of part 12's ring, and a 3-tap kernel. The real
scorer has 30 whitened components and the 21/61 ring; the mechanism is the same.)

### The real bench, by size

The README breaks objects caught at the top 1% down by object size (561 / 170 / 158 / 20 objects):

| scorer | 1-2 px | 3-5 px | 6-20 px | > 20 px |
|---|---|---|---|---|
| RM (baseline) | 36.5% | 52.4% | 66.5% | 80.0% |
| A1 RM blur 1.0 | 33.3% | 45.3% | 57.6% | 75.0% |
| A1+ RM + RM blur 1.4 | 35.1% | 48.2% | 60.8% | 80.0% |
| A3 mixture k=6 | 32.6% | 43.5% | 54.4% | 75.0% |

Blur loses objects at **every** size, not only the small ones. The README's explanation is that
blurring "pulls isolated small targets and target edges down into their neighbours". Turned back into
counts (derived from the percentages, not printed in the report): `0.365 x 561 = 205` small objects
for RM against `0.333 x 561 = 187` blurred, 18 lost. In the largest class the loss is one object -
`0.80 x 20 = 16` against `0.75 x 20 = 15` - which on 20 objects is too few to read anything into.

Note also that 561 of 909 real objects, 62%, are 1-2 pixels. A filter that favours footprint-sized
targets is working against most of the bench.

### A1+ passed its criteria and was still not adopted

A1+ (rank mean of RM and RM blurred) met both criteria: pAUC `0.180 - 0.162 = +0.018`, at least
+0.005; TPR 0.064, not below 0.061; bench-v2.1 `35.0 - 35.1 = -0.1` points, within 0.5. It also lost
`45.7 - 43.0 = 2.7` points of real objects at the top 1% and `14.9 - 12.8 = 2.1` at the top 0.1%. The
README: "The criteria were pixel-level and missed this; not adopted." That is a criterion that was
right to fix in advance and wrong in what it measured.

## Step 4 - A2: rank fusion, and why rank max adds false alarms

RM and PCA-RX give scores on different scales, so they cannot be averaged directly. `rank_map` turns
each map into "the fraction of valid pixels at or below this one" (ties averaged, 0 outside valid).
`fuse` stacks those maps and takes the mean or the maximum.

Six pixels. P1 is a real target. P2 is RM's false alarm (a field edge, say); P4 is PCA-RX's (a roof
that is spectrally unusual but ordinary to the model).

```
              P1    P2    P3    P4    P5    P6
RM score       9    12     3     2     1     4
PCA-RX score   8     1     2    10     3     4

RM rank       5/6   6/6   3/6   2/6   1/6   4/6
PCA-RX rank   5/6   1/6   2/6   6/6   3/6   4/6

rank mean    10/12  7/12  5/12  8/12  4/12  8/12   = 0.83 0.58 0.42 0.67 0.33 0.67
rank max      5/6   6/6   3/6   6/6   3/6   4/6    = 0.83 1.00 0.50 1.00 0.50 0.67
```

With a budget of one pixel: RM alone picks P2, PCA-RX alone picks P4 - both miss. Rank mean picks P1:
a pixel both scorers think odd beats a pixel only one of them thinks odd. That is the best case for
fusion. Rank max puts P2 and P4 at the top together, and P1 is third: **the maximum lets either
scorer's false alarm through**, so it inherits both sets. The README predicted exactly that, and on the
real bench rank max fell to pAUC 0.138.

Rank mean needs the two scorers to fail on *different* objects. On the real bench they did not: rank
mean tied pAUC (0.162) and lost objects (43.8% against 45.7%). PCA-RX alone catches 39.7%; its wins on
bench-v1 (part 19) "do not carry to real targets", so averaging in its ranks mostly dilutes RM.

## Step 5 - A3: mixture of backgrounds, a bench that disagrees with reality

`background_types` runs k-means (fixed seed, 25 iterations, fitted on a 50,000-pixel sample) on the
**cube's** whitened top 10 principal components - so the types are surfaces, not residual patterns.
`score` then hands those types to `class_aware_residual_mahalanobis` in
`app/utils/anomaly_detection/residual_mahalanobis.py` - RM run once per class, written for a segmentation
head's classes: each type gets its own whitening and a ring of same-type neighbours. Types
under 5,000 valid pixels join type 0 (`MIN_CLASS_PX`).

It won bench-v2.1 by 6 points (35.1% to 41.2%) and cut the real bench's pAUC by a third (0.162 to
0.111). Here is why, by hand. One band, every type's mean 0, score = residual squared / variance.

```
field type    (16 px):  seven at +1, seven at -1, one edge at +2.5, one at -2.5
built-up type  (6 px):  -2, -2, +2, +2 (ordinary roofs), and two real targets at -6 and +6
```

One global variance, all 22 pixels:

```
sum of squares = 14 x 1 + 2 x 6.25 + 4 x 4 + 2 x 36
               = 14 + 12.5 + 16 + 72
               = 114.5
variance       = 114.5 / 22 = 5.20
target score   = 36 / 5.20   = 6.92
edge score     = 6.25 / 5.20 = 1.20
```

The targets top the ranking. Now each type on its own:

```
field:     variance = (14 + 12.5) / 16 = 26.5 / 16 = 1.656
           edge score   = 6.25 / 1.656 = 3.77
built-up:  variance = (16 + 72) / 6    = 88 / 6    = 14.67
           target score = 36 / 14.67   = 2.45
```

Now the two field edges (3.77) outrank both targets (2.45). The targets are two of the built-up type's
six pixels, so they **are** a large share of what that type's "normal" is fitted on. Graded against
themselves, they look ordinary.

That is what the README found on the real scenes: 60-70% of target pixels sit in two or three small
background types of about 5% of the scene each - the built-up surfaces where targets live. In set4,
one type holds 60% of the target pixels on 5.2% of the scene.

And the implants? bench-v2.1 spreads them over every surface. Put an implant with residual 4 into the
field type above: globally `16 / 5.20 = 3.07`, per type `16 / 1.656 = 9.66`. A tighter normal makes an
implant on an ordinary surface stand out more - so per-type whitening only sharpens the implants.

This is **a second bias of bench-v2.1** (the first, part 19, was bench-v1's cube-level oddity):
implant placement is not conditioned on where real targets occur. `docs/09-known-issues.md` now says
so, and says the fix (placement on target-bearing surfaces) is not built. Until it is, judge scorers on
the real bench and use bench-v2.1 only to catch regressions.

## Step 6 - expectations, a cache bug, and the verdict

The README wrote its expectations down before the run. Scored afterwards:

| idea | expected | what happened |
|---|---|---|
| A1 | a small gain on the real bench | right on pixels, wrong on objects |
| A2 | rank mean a small gain, rank max a loss | rank max right; rank mean tied pAUC and lost objects |
| A3 | a gain on multi-surface scenes (set2 / set4) | wrong in direction |

The run also exposed a bug in the cache (part 19: score maps are cached, reconstructions are not).
Maps were stored as float16, whose largest finite value is 65,504. Blurred and mixture squared
distances went past it, so a reloaded map had `inf` at its top. `runner.py` now keeps a map float32 if
any value is non-finite or at least 6e4 in size; the 13 overflowed cache files were deleted. The README
notes this run's metrics were computed before the cast and are unaffected. Because a `ScorerSpec`
field was added, every cache key changed (the key is the spec's JSON), so old caches recompute.

**Verdict: RM stays the score.** The lesson is carried forward in writing: the next attempt's pass
criteria include real objects caught at the top 1% and 0.1%. The next attempt is on the model side -
an uncertainty head, `docs/proposals/uncertainty-head.md`, proposed and not yet run (the Indradhanu
course's part 62 outlines it).

## Common confusions

**"pAUC went up, so A1 is better."**
pAUC and TPR count pixels. A1 filled in more pixels of objects that were already found and lost
objects at every size. The analyst reviews objects, not pixels.

**"The blur in A1 is the same as bench-v2.1's PSF blur."**
Same shape, different job. bench-v2.1 blurs the implant before scoring to imitate a real target; A1
blurs the residual of the whole scene. That overlap is why A1's bench-v2.1 numbers were treated as
partly circular.

**"A1+ met the criteria, so it should have shipped."**
The criteria were fixed in advance, which is right, but they only checked pixel measures. A1+ lost 2.7
points of real objects. The record changes the criteria rather than shipping a regression.

**"Rank max is the generous choice - a pixel only needs one scorer to like it."**
That generosity is the problem: any pixel either scorer wrongly likes reaches the top too.

**"A3 won a benchmark, so it is a better scorer that the real bench is too small to show."**
The real bench lost a third of its pAUC, and the mechanism is understood: real targets cluster in
small built-up types and become their types' normal. The benchmark is what is wrong here.

**"`mixture` is a new scorer from scratch."**
It is `class_aware_residual_mahalanobis` with k-means types in place of a segmentation head's classes.

**"V2's 0.170 and this part's 0.162 disagree."**
0.170 is FT04 with its own cloud mask (part 23). This part's runs did not exclude cloud; its baseline
is 0.162.

## Check yourself

1. Redo the strip from Step 3 with the target at pixel 11 raised from 4.8 to 5.2. Compute its blurred
   score and say whether it makes the blurred top 4. Then say what that does to objects caught.
2. A 2-pixel target has residuals `[3, 3]` with zeros either side. Blur both pixels with
   `[0.25, 0.5, 0.25]` and compare each pixel's signal-to-noise (blurred noise spread 0.612) with the
   raw 3.0.
3. In the rank example of Step 4, change P1's PCA-RX score from 8 to 11. Recompute P1's rank mean and
   rank max. Which pixel does each rule put first?
4. In Step 5's toy, the built-up type gains two more ordinary roofs at -2 and +2 (8 pixels, still two
   targets). Recompute the built-up variance and the target score. Do the targets now beat the field
   edges' 3.77?
5. Using the README's size table, how many 3-5 px objects did RM catch at the top 1%, and how many did
   blur 1.0 catch? Show the multiplication.
6. Why is bench-v2.1 a fair guard against regressions but not a fair judge for A3?

<details>
<summary>Answers</summary>

1. `b11 = 0.5 x 5.2 = 2.6`, score `2.6^2 = 6.76`. The blurred ranking becomes 18.06, 16, 7.56, 6.76,
   6.25: the target is fourth and pixel 5 drops to fifth. Both objects are now caught (the big one via
   pixels 6-8). One pixel of extra residual decided whether the 1-pixel target survived blurring.
2. Each pixel: `0.25 x 0 + 0.5 x 3 + 0.25 x 3 = 0 + 1.5 + 0.75 = 2.25`. Signal-to-noise
   `2.25 / 0.612 = 3.68` against the raw `3 / 1 = 3.0`. A 2-pixel target already gains; the 1-pixel
   target (1.63 against 2.00) lost.
3. PCA-RX order becomes P2 1, P3 2, P5 3, P6 4, P4 10, P1 11, so P1's PCA-RX rank is 6/6 and P4's is
   5/6. P1 rank mean `(5 + 6) / 12 = 11/12 = 0.92`, now first. Rank max: P1 `max(5/6, 6/6) = 1.00`, P2
   `max(6/6, 1/6) = 1.00` - a tie for first between the target and RM's false alarm.
4. Sum of squares `6 x 4 + 2 x 36 = 24 + 72 = 96`; variance `96 / 8 = 12`; target score
   `36 / 12 = 3.0`. Still below 3.77 - the targets lose. Diluting them with more ordinary roofs helps,
   but not enough; on set4 one type holds 60% of target pixels, far more concentrated than this.
5. RM: `0.524 x 170 = 89.08`, so 89 objects. Blur 1.0: `0.453 x 170 = 77.01`, so 77. Twelve lost.
6. bench-v2.1 places implants over every surface type; real targets cluster in a few small built-up
   types. A scorer that changes nothing about where targets live (a new model with RM) moves on both
   benches alike, so a drop on bench-v2.1 is a real warning. A3's mechanism depends on exactly the
   placement bench-v2.1 gets wrong, so its bench-v2.1 gain says nothing about real targets.

</details>

**Update, 2026-10-20 - the model-side follow-up failed too.** The uncertainty head this part's verdict
pointed to was trained as FT06 and is not adopted: it learned to predict its own error, but it expects large
errors on targets themselves (their halo is visible in neighbouring tokens), so dividing the residual by that
expectation hides the anomalies - pAUC 0.170 fell to 0.125 as designed. The Indradhanu course's part 63
(`indradhanu workings/63-the-uncertainty-head-run.md`) teaches the run.

Next: [part 25](25-gated-scene-adaptation.md) - GT-HAD's gating and early stopping, transplanted onto FT04 as
per-scene adaptation, and why neither helps a pretrained model.
