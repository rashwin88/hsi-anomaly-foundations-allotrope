# Part 25 - Gated scene adaptation: borrowing GT-HAD's trick, and why it found nothing

> **The one thing this part teaches:** on 2026-10-20 FT04 was briefly retrained on each scene with the
> pixels that already looked anomalous left out of the loss - GT-HAD's gating and early stopping,
> transplanted onto a pretrained model - and it lost a little at every setting, gated or not; the gate
> changed nothing because adaptation never learned the targets in the first place (the damage was to
> every pixel's residual), and there was no early window to stop in because a pretrained model starts
> past the point that GT-HAD's early stopping is looking for.

## Where we are

Part 24 kept FT04's reconstruction fixed and changed the score; nothing beat residual Mahalanobis (RM).
Its update note recorded the model-side follow-up, the uncertainty head (FT06), failing too. Both
failures, and the mixture of backgrounds before them, have the same shape: **every attempt to learn
"normal" from a scene's own context also learned that scene's targets.**

One method in the record looked as if it had an answer to that. GT-HAD, a published detector
reproduced on our four PRISMA scenes, trains on the scene itself and has two defences against learning
the targets. This part is the attempt to borrow them.

| file | what it holds |
|---|---|
| `research/experiments/scene-adaptation/README.md` | the question, method, learning-rate probe, pass criteria fixed before the run, results, observations |
| `research/experiments/scene-adaptation/eval_report.md` | the evaluator's tables, overall and by tier |
| `research/experiments/scene-adaptation/eval_per_scene.csv` | every measure per scene - the source of the hand calculation in Step 5 |
| `research/experiments/scene-adaptation/adapt_probe.py` | the learning-rate probe (no ground truth) |
| `app/foundation_models/scene_adaptation.py` | `gate_mask` and `adapt` |
| `app/evaluation/models.py` | `_adapted_reconstruction` - gate, adapt, reconstruct, restore |
| `app/evaluation/config.py` | `ModelSpec`'s fields `adapt_steps`, `adapt_lr`, `adapt_gate_pct`, `adapt_halo` |
| `configs/eval/scene_adapt.json` | the run: FT04 plus four adapted variants, real PRISMA bench |
| `research/experiments/gt-had-prisma/summary.md` | the GT-HAD reproduction: 5 against 150 epochs, PD at 1e-3 |

This is an evaluation experiment, not a product setting. Nothing in `anomaly_scoring` changed, and
nothing was adopted.

> **Analogy.** A newcomer moves to a town and starts learning faces. In the first week they learn the
> faces everyone sees every day - the baker, the postman. The handful of odd visitors they have not
> learned yet, so in week one a stranger stands out. Give them a year and they know the visitors too,
> and nobody stands out. GT-HAD is the newcomer, and stopping early is testing them in week one. FT04
> is a detective who has worked a hundred towns. Sending the detective to spend a week in this town
> does not give them a week-one advantage - they are already past week one, and the week only nudges
> their sense of "ordinary" slightly off. Telling them to avoid the people already flagged as
> suspicious (the gate) changes nothing, because they were never going to memorise a few visitors in
> a week anyway.

## Step 1 - what GT-HAD's gating and early stopping are for

Two words first. **Training** a network means repeatedly nudging its parameters (the numbers inside it)
in the direction that lowers the loss; each nudge is a **step**, and the **learning rate** (lr) sets how
big each nudge is. An **epoch** is one pass over all the training data.

GT-HAD is a reconstruction detector (part 10) with one unusual feature: it has no pretraining. It
trains from scratch on the very scene it is scoring, cut into 9x9 pixel blocks, then scores each pixel
by how badly it reconstructs it. That creates an obvious danger. Train long enough and the network
learns to reconstruct *everything* in the scene, targets included - and a target it reconstructs well
has a small residual and a low score.

It has two defences:

| defence | what it does | why it helps |
|---|---|---|
| **gating** | a periodic search (the paper calls it CMM) decides which blocks look like background; those are routed to a background branch that learns from them | the background branch's idea of normal is built without the likely anomalies |
| **early stopping** | stop training before the network has had time to learn the rare things | a network learns the common background before the rare targets |

The second defence is visible in our own reproduction. From `research/experiments/gt-had-prisma/summary.md`,
valid-pixel AUC on the four PRISMA scenes:

| | 5 epochs | 25 epochs | 150 epochs (the paper's setting) |
|---|---|---|---|
| mean valid AUC | **0.873** | 0.780 | 0.843 |
| mean PD at 1e-3 false alarms | **5.5%** | - | 2.0% |

(PD is the detection rate, the same idea as part 16's TPR.) Shorter training did better. The summary
is careful about what that means: "choosing the epoch count from these GT AUCs would leak labels, so 5
epochs is a diagnostic, not a recommendation."

**Both defences exist for one purpose: stop the network learning to reconstruct the anomaly.** That
was exactly the failure the last three experiments kept hitting, which is why the user's comment after
the GT-HAD discussion - "Gating is something we can use" - started this one.

## Step 2 - the transplant

GT-HAD starts from nothing. Our version starts from FT04, already trained on many scenes, and only
adapts it. The README's method, per scene, starting from FT04's weights each time:

```
1. score the scene with FT04 + RM                         (the unadapted score)
2. gate = top 1% of that score, dilated by 2 px           (gate_mask)
3. adapt N steps on the scene's own tiles, gated pixels
   left out of the loss                                   (adapt)
4. reconstruct, score RM, cloud excluded on unadapted
   FT04's masks; then restore FT04's weights              (_adapted_reconstruction)
```

`gate_mask` in `app/foundation_models/scene_adaptation.py` is four lines: take the pixels at or above
the 99th percentile of the score over valid pixels, then `ndimage.binary_dilation(gate, iterations=halo)`.
Why a halo at all? FT06 (part 24's update note) showed that a target leaks into its neighbours - the
point-spread blur of part 20 puts some of its light into the next pixels over. Gate only the target's
core and the network still learns from its edges.

`adapt` uses the fine-tuning loss (L1 plus 0.5 x SAM, on hidden pixels, with 65% of tokens hidden - the
masking of part 11) on batches of 8 random 128 px tiles from the scene. Three details matter:

- **The loss mask is `hidden x valid x not-gated`.** In the code: `learn = valid & ~gate`, then
  `loss_mask = hidden * l[:, None].float()`. A gated pixel can still be *seen* as context; it only
  never counts towards the loss.
- **Eval mode during training.** `model.eval()` is called before the loop, so BatchNorm keeps the
  statistics it learned on many scenes (and dropout is off). Gradients still flow; only those stored
  averages are frozen. Batches of 8 tiles from one scene would otherwise overwrite them with that scene's.
- **The segmentation head is outside the loss.** The parameter list skips names starting `seg_head.`
  and `scale_head.`. Only the reconstruction path moves - which is also why cloud is excluded on the
  *unadapted* FT04's masks.

`_adapted_reconstruction` in `app/evaluation/models.py` copies the full `state_dict` before adapting and
reloads it in a `finally` block, so a crash mid-scene cannot leave the next scene starting from adapted
weights.

### How far does a 1% gate reach? By hand

The probe printed `gated px 4.55%` on set1 (`adapt_probe.py` divides by valid pixels). A 1% gate became
4.55%. Why that much, and why not more?

`binary_dilation` with no structuring element uses a cross: each pass adds the four direct neighbours
(up, down, left, right) of every gated pixel. Do it on paper for one isolated pixel:

```
start              1 pixel
pass 1 adds        4  (its four neighbours)
pass 2 adds        8  (the next ring of a diamond: 4 straight out, 4 diagonal)
total              1 + 4 + 8 = 13 pixels
```

A 2x2 clump of gated pixels: pass 1 adds 2 neighbours per side x 4 sides = 8, pass 2 adds 12.

```
total              4 + 8 + 12 = 24 pixels        growth 24 / 4 = 6.0x
```

A 3x3 clump: pass 1 adds 12, pass 2 adds 16.

```
total              9 + 12 + 16 = 37 pixels       growth 37 / 9 = 4.1x
```

(Those three totals, 13 / 24 / 37, were checked against scipy itself.) If set1's top 1% were all
isolated pixels, the gate would cover about 13%. It covers 4.55%, a growth of 4.55x - between the 3x3
and the 2x2 cases. **The top of the score comes in clumps, so a halo costs far less than it would for
scattered single pixels.** Even so, more than 95% of the scene still trains the network.

## Step 3 - choosing settings without ground truth

There are two settings to choose: the learning rate and N, the number of steps. Choosing either by
looking at which one detects the labelled targets best would be cheating - the same leak the GT-HAD
summary warned about. A setting tuned on the answers looks better than it will be on a new scene.

### The learning rate, from the loss alone

`adapt_probe.py` ran 300 steps on set1 at three learning rates and recorded only the adaptation loss -
no labels. From the README (1.6 steps per second on the laptop GPU):

| lr | loss, first 50 steps | steps 100-150 | last 50 steps |
|---|---|---|---|
| 1e-5 | 0.0495 | 0.0409 | 0.0389 |
| 3e-5 | 0.0459 | 0.0382 | 0.0368 |
| 1e-4 | 0.0461 | 0.0365 | 0.0354 |

The choice was 3e-5: "as fast as 1e-4 early, a third of the step size - the aim is to learn background
without reaching the targets." That is a judgement, not a measurement. The loss can tell you how fast
the background is being fitted; it cannot tell you whether the targets are being learned, because they
are a tiny share of it (Step 6 works that number).

### N, by leave-one-scene-out

Four variants were run: N = 50, 200 and 1000 gated, and N = 200 with the gate set to 0% to isolate the
gate's effect. **Leave-one-scene-out** picks N for each scene using only the other three: for set1,
take the N with the best mean pAUC over set2, set3 and set4, and report set1's result at that N. Repeat
for every scene; average the four results.

Why bother? Pick N by the best mean over all four and the scene you report has helped choose its own
setting. Leave-one-scene-out means **no scene's number comes from a setting tuned on that scene**. It is
not label-free - the other three scenes' labels are used - but it is honest about how a setting chosen
on old scenes would do on a new one, which is the product's situation.

Step 5 works it by hand on the real numbers.

## Step 4 - the results

The pass criteria were fixed before the run: against FT04 + RM with cloud excluded (pAUC 0.170), a
leave-one-scene-out pAUC gain of at least +0.005, objects caught at the top 1% and 0.1% not lower, and a
bench-v2.1 guard. Part 24's lesson is visible: objects caught are in the criteria now.

The README also wrote down what it expected: a gain peaking at small N and decaying as N grows
(GT-HAD's curve), and ungated worse than gated at the same N, because ungated would learn the targets.

Real PRISMA bench, cloud excluded on unadapted FT04's masks (`eval_report.md`):

| | pAUC <= 1% | caught top 1% | caught top 0.1% | TPR @ 0.1% |
|---|---|---|---|---|
| FT04 (the bar) | **0.170** | 46.0% | **15.8%** | **0.065** |
| gated, N = 50 | 0.160 | 46.8% | 15.2% | 0.059 |
| gated, N = 200 | 0.159 | 46.9% | 14.9% | 0.059 |
| gated, N = 1000 | 0.161 | 46.5% | 15.0% | 0.057 |
| ungated, N = 200 | 0.159 | 46.9% | 14.9% | 0.059 |

Read it in three passes:

1. **Every N is below FT04** on pAUC (about -0.01) and on objects caught at the top 0.1% (-0.6 to -0.9
   points). Criteria 1 and 2 fail.
2. **Objects caught at the top 1% rose** by 0.5 to 0.9 points. With the top 0.1% falling, the README
   reads this as "a shift of hits from the very top into the top 1%, not more finds."
3. **Gated and ungated N = 200 are the same row.** To three decimals, on every measure.

Both written expectations were wrong. There is no peak at small N - the loss is already there at
N = 50 and does not grow with N - and ungated is not worse. bench-v2.1 (criterion 3) was not run; with
criteria 1 and 2 failed, it could not change the verdict.

One scene behaves differently: set2 gains at every N (pAUC 0.128 to 0.139-0.142). set1 and set4 lose.
The README's verdict on that: "Four scenes cannot say whether that is signal."

## Step 5 - leave-one-scene-out, by hand

The README calls this step "moot": every N is below FT04 on the mean, so whichever N is chosen the
headline is below 0.170. Work it anyway - it is the procedure you will reuse, and the numbers are small.
You need a pen; it takes about ten minutes.

Per-scene pAUC, cloud excluded, from `eval_per_scene.csv` (rounded to three places):

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| FT04 | 0.246 | 0.128 | 0.084 | 0.220 |
| N = 50 | 0.223 | 0.142 | 0.084 | 0.192 |
| N = 200 | 0.228 | 0.142 | 0.081 | 0.185 |
| N = 1000 | 0.240 | 0.139 | 0.083 | 0.181 |

First the baseline, so you know what the target is:

```
FT04 mean = (0.246 + 0.128 + 0.084 + 0.220) / 4
          = 0.678 / 4
          = 0.1695   -> 0.170
```

**Hold out set1.** Mean over set2, set3, set4 for each N:

```
N = 50:    (0.142 + 0.084 + 0.192) / 3 = 0.418 / 3 = 0.1393
N = 200:   (0.142 + 0.081 + 0.185) / 3 = 0.408 / 3 = 0.1360
N = 1000:  (0.139 + 0.083 + 0.181) / 3 = 0.403 / 3 = 0.1343
```

Best is N = 50, so set1 reports its N = 50 value: **0.223**.

**Hold out set2.** Mean over set1, set3, set4:

```
N = 50:    (0.223 + 0.084 + 0.192) / 3 = 0.499 / 3 = 0.1663
N = 200:   (0.228 + 0.081 + 0.185) / 3 = 0.494 / 3 = 0.1647
N = 1000:  (0.240 + 0.083 + 0.181) / 3 = 0.504 / 3 = 0.1680
```

Best is N = 1000; set2 reports **0.139**.

**Hold out set3.** Mean over set1, set2, set4:

```
N = 50:    (0.223 + 0.142 + 0.192) / 3 = 0.557 / 3 = 0.1857
N = 200:   (0.228 + 0.142 + 0.185) / 3 = 0.555 / 3 = 0.1850
N = 1000:  (0.240 + 0.139 + 0.181) / 3 = 0.560 / 3 = 0.1867
```

Best is N = 1000; set3 reports **0.083**.

**Hold out set4.** Mean over set1, set2, set3:

```
N = 50:    (0.223 + 0.142 + 0.084) / 3 = 0.449 / 3 = 0.1497
N = 200:   (0.228 + 0.142 + 0.081) / 3 = 0.451 / 3 = 0.1503
N = 1000:  (0.240 + 0.139 + 0.083) / 3 = 0.462 / 3 = 0.1540
```

Best is N = 1000; set4 reports **0.181**.

The headline:

```
leave-one-scene-out = (0.223 + 0.139 + 0.083 + 0.181) / 4
                    = 0.626 / 4
                    = 0.1565
gain over FT04      = 0.1565 - 0.1695 = -0.013
```

The criterion asked for +0.005; this is -0.013. (The same procedure on the unrounded CSV gives the same
picks and 0.1567. That computation is this course's, not the README's - the README stopped at "moot".)

Notice two things. Set1 picked a different N from the others: the choice is not stable, because three
scenes is a small jury. And the honest number, 0.1565, is a little worse than the best single row in
the table (0.161 for N = 1000). **That gap is the size of the flattery you would have reported by
choosing N on all four scenes.**

## Step 6 - two lessons

### Lesson 1: the gate changed nothing, because adaptation never learned the targets

The gate exists to stop the network learning targets. If gated and ungated agree to three decimals,
then without the gate the network was not learning the targets either - so the gate had nothing to
prevent. The README: "Adaptation is not learning the targets ... it changes the residual for
everything. Expectation 2 (ungated worse) was wrong because the failure mode it guards against never
happened."

Why would it not learn them? Look at their weight in the loss. GT-HAD's summary gives the scale: scenes
of about 1200 x 1200 with 600 to 1200 anomaly pixels.

```
scene pixels        1,200 x 1,200 = 1,440,000
anomaly pixels      about 1,000
share               1,000 / 1,440,000 = 0.0007 = 0.07%
```

In a loss that is a mean over pixels, 0.07% of the pixels contribute about 0.07% of the gradient. The
README puts it as the targets being "< 0.1% of pixels" that "1000 steps of a 5.9 M-parameter model
barely see". Be precise about "barely see": it is the *weight* that is small, not the number of
visits. 1000 steps x 8 tiles x 128 x 128 px is about 131 million pixel-visits, roughly 90 per pixel of
a 1.44-million-pixel scene - each target is in a tile many times, but each time it is one voice in
16,384.

So what did go wrong? Something that affects every pixel. **The failure was global, not target-specific:**
nudging the network towards this scene's background moved the residual of every pixel a little, and the
careful ranking at the very top - where the top 0.1% lives - got slightly shuffled. Hits slid from the
top 0.1% into the rest of the top 1%. No gate on 4.55% of pixels can protect against a change to the
other 95%.

### Lesson 2: a pretrained model has no early window

GT-HAD's curve - better at 5 epochs than 150 - is a statement about a network that *started from
nothing on this scene*. Early on it has learned the common background and not the targets; that moment
is what early stopping finds. Here is where that moment sits next to FT04:

| | detection at 1e-3 false alarms |
|---|---|
| GT-HAD, 150 epochs (the paper's setting) | 2.0% |
| GT-HAD, 5 epochs (its best, a diagnostic) | 5.5% |
| FT04, no adaptation | 6.5% |

(GT-HAD figures are PD on valid pixels from `summary.md`; FT04 is TPR at 0.1% with cloud excluded, the
0.065 above. Close enough in definition to compare, not identical: different band preparation, and GT-HAD
did not exclude cloud. Without cloud exclusion FT04's figure is 6.1%, still ahead.)

FT04 already knows what ordinary land, water and roofs look like - it learned that from many scenes
before it saw this one - and it was never trained on this scene's targets. It is already where GT-HAD's
early stopping is trying to get to, and beyond it. **A pretrained model starts past the point early
stopping finds, so there is no early window to stop in** - and that is why the damage is the same at
N = 50 as at N = 1000. Any adaptation moves it away from a good starting point, not towards one.

## Common confusions

**"The gate made no difference, so gating is useless."**
The gate made no difference *here*, because the thing it guards against - the network memorising the
targets - did not happen in 1000 steps. In a from-scratch method like GT-HAD, trained for 150 epochs,
it does happen, and gating has a job.

**"`adapt` here is the same as the FT02 'adapt' fine-tuning mode."**
Two things called the same word. FT02's `adapt` mode (`docs/lld/finetune.md`) is a fine-tuning run on
many scenes that produces a new checkpoint. Per-scene adaptation runs inside the evaluator for one
scene, then throws the weights away.

**"Eval mode means the model is not training."**
`model.eval()` only switches BatchNorm to its stored statistics and turns dropout off. The optimiser
still updates the weights every step. The name is a PyTorch convention, not a promise.

**"Objects caught at the top 1% went up, so adaptation found more."**
The top 0.1% went down at the same time. Hits moved down the list from the very top into the top 1%;
the analyst looks at the very top first.

**"set2 improved at every N, so adapt only scenes like set2."**
Deciding which scenes are "like set2" needs a rule that does not peek at the labels, and one scene out
of four cannot be told apart from chance.

**"The learning rate was tuned on set1, so set1's result is contaminated."**
It was chosen from set1's adaptation *loss*, which uses no labels. That is a weaker leak than choosing
from detection numbers - a small one remains, since set1 is also scored - and it is why N, the setting
most likely to matter, went through leave-one-scene-out.

**"GT-HAD's 5 epochs is a setting we could have copied."**
Its own summary says 5 epochs was found by looking at the ground truth and is "a diagnostic, not a
recommendation". And it describes from-scratch training; it does not translate into a step count for a
pretrained model.

## Check yourself

1. A single gated pixel sits in a corner of the valid area, so only its right and lower neighbours are
   inside the image. Dilate it twice by hand with the cross and count how many pixels inside the image
   the gate now covers.
2. A scene's top 1% is made entirely of isolated 2x2 clumps. Using Step 2's numbers, what share of the
   scene does the gate cover? What if it were all isolated single pixels?
3. Redo Step 5's leave-one-scene-out with only N = 50 and N = 1000 as candidates. Which N does each
   scene pick, and what is the headline?
4. From the learning-rate table, compute how much the loss fell between the first 50 steps and the last
   50 steps at each learning rate, as a fraction of the first-50 value. Does the chosen 3e-5 fall
   fastest? Why does the first-50 value of 1e-5 sit highest when every run started from the same
   weights?
5. At 1.6 steps per second, how long does N = 1000 take per scene? How long for all four variants
   across the four scenes (N = 50, 200, 1000 and the ungated 200)?
6. In one or two sentences each: why did the gate change nothing, and why is there no early window for
   FT04?

<details>
<summary>Answers</summary>

1. Start 1. Pass 1 adds the right and lower neighbours: 2. Pass 2 adds the pixel two to the right, two
   below, and the diagonal one right-and-down: 3. Total `1 + 2 + 3 = 6` pixels, against 13 in the
   open. Image edges and nodata borders clip the halo, one more reason the real gate grows by less than
   13x.
2. 2x2 clumps grow 6.0x: `1% x 6.0 = 6%`. Single pixels grow 13x: `1% x 13 = 13%`. Set1's 4.55% is less
   than either, so its top 1% includes clumps of 3x3 and larger.
3. Hold out set1: N = 50 `0.418 / 3 = 0.1393`, N = 1000 `0.403 / 3 = 0.1343` - picks 50, reports 0.223.
   Set2: `0.1663` against `0.1680` - picks 1000, 0.139. Set3: `0.1857` against `0.1867` - picks 1000,
   0.083. Set4: `0.1497` against `0.1540` - picks 1000, 0.181. Same picks as before, because N = 200
   never won; headline `0.626 / 4 = 0.1565`.
4. 1e-5: `0.0495 - 0.0389 = 0.0106`, `0.0106 / 0.0495 = 0.214`, 21.4%. 3e-5: `0.0459 - 0.0368 = 0.0091`,
   `0.0091 / 0.0459 = 0.198`, 19.8%. 1e-4: `0.0461 - 0.0354 = 0.0107`, `0.0107 / 0.0461 = 0.232`, 23.2%.
   3e-5 has the smallest relative fall here. The first-50 value is an *average* over 50 steps: 1e-5
   descends slowest, so more of its first 50 steps are spent near the starting loss. Its 21.4% looks
   large only because it starts from a higher average. The real case for 3e-5 is the README's: it
   matches 1e-4 in the first 50 steps (0.0459 against 0.0461) with a third of the step size.
5. `1000 / 1.6 = 625` seconds, about 10.4 minutes per scene. Steps per scene for all variants:
   `50 + 200 + 1000 + 200 = 1,450`; four scenes: `1,450 x 4 = 5,800` steps; `5,800 / 1.6 = 3,625`
   seconds, about an hour of adaptation. The README's ~1.5 h includes the reconstructions and scoring.
6. The targets are under 0.1% of the pixels in the loss, so a few hundred or thousand steps do not learn
   them, gated or not; the harm was a small change to every pixel's residual, which a gate on the top
   4.55% cannot touch. GT-HAD's early window is the moment a from-scratch network has learned the
   background but not yet the targets; FT04 arrives already knowing the background and never trained on
   these targets, so it is past that moment before the first step.

</details>

---

## You have finished

You can now read the source. Where to go next:

- **`docs/01-orientation.md`** through **`docs/10-code-style.md`** - the reference docs
- **`docs/lld/`** - low-level designs for individual subsystems; for the last ten parts,
  `docs/lld/evaluation.md`, `docs/lld/implant-bench.md` and `docs/lld/spectral-stem.md` ("Serving")
- **`docs/09-known-issues.md`** - what is currently broken, and what to check before trusting
  a green test run

One habit worth keeping from this course: **when a comment and the code disagree, the code
wins**, and when a summary and the data disagree, the data wins. You met several such disagreements -
the regularisation default, the `MIN_VALID_FRACTION` comment and the discarded `units` field in parts 1
to 15; `compute_roc`'s threshold docstring in part 16; the LLD's old `strength_for_z` and "10 px apart" in
part 18; the cache key and `dirty` flag in part 19, both fixed since; bench-v2's per-tier foreignness in
part 20; "no real object sits on water" and "set4 is bright desert" in part 21; a "close to, not
identical" docstring that was hiding a factor of nearly three (15.8% against 44.2%), and a cross-check
whose result was misread, in part 22; a band filter description that still said the common grid was
"Required for cross-sensor models like Indradhanu" - true of v0.2.0, wrong for V2 - in part 23; a scorer
that met every pass criterion while finding fewer real objects in part 24; and, in this part, two
expectations written down in advance and both wrong, and "barely see" meaning small weight rather than
few visits. There will be more.
