# Residual-RX experiments, explained with pictures

Companion to the run on 2026-10-21 (Drive: `MyDrive/PS_11/claude_experiments/residual_rx_2026-10-21/`,
survey page: https://claude.ai/artifact/VtKtnWfVpY4RYDFJ3d3o9m). One section per question. All numbers are from
that run: Indradhanu v0.2.0, 4 Stage II PRISMA scenes (909 objects), and HAD100 (100 crops of 64x64).

Contents:
1. [What is pAUC, and why it matters](#1-what-is-pauc-and-why-it-matters)
2. [Does today's baseline use a guard window?](#2-does-todays-baseline-use-a-guard-window)
3. ["Keeping only the top 30 components halved pAUC"](#3-keeping-only-the-top-30-components-halved-pauc)
4. [Why anomalies still get reconstructed](#4-why-anomalies-still-get-reconstructed)
5. [RM and d](#5-rm-and-d)
6. [Ledoit-Wolf shrinkage](#6-ledoit-wolf-shrinkage)
7. [Phase B, and B2 in particular](#7-phase-b-and-b2-in-particular)
8. [Did Phases C and D run?](#8-did-phases-c-and-d-run)
9. [The robust fit](#9-the-robust-fit)
10. [The IJDE benchmark](#10-the-ijde-benchmark)
11. [The overall conclusion](#11-the-overall-conclusion)
12. [How can CFAR on RM d20 be better when ROC-AUC drops?](#12-how-can-cfar-on-rm-d20-be-better-when-roc-auc-drops)

---

## 1. What is pAUC, and why it matters

**Short answer.** pAUC is the area under the ROC curve over the first 1% of false alarms only, rescaled so that
1 is perfect. It measures what happens at the top of the ranking, which is the only part an analyst ever looks at.

### The ROC curve, and the part pAUC looks at

Sort every valid pixel by score, highest first. Walk down the list. At each step:
- **TPR** (true positive rate) is the share of target pixels you have passed.
- **FPR** (false positive rate) is the share of background pixels you have passed.

```
 TPR
 1.0 |                                   _______________----------
     |                         ______----
     |                ____-----
     |          __----                      ROC-AUC = area under the WHOLE curve
     |      _---                            (FPR from 0 to 1)
     |   _-'
     |  /
     | /
     |/
 0.0 +--+---------------------------------------------------------  FPR
     0  0.01                                                     1.0
     |<>|
      pAUC = area under the curve for FPR 0 .. 0.01 only, divided by 0.01
```

Zoom into that sliver:

```
 TPR (zoom: FPR 0 to 1%)
 0.30 |                              .........
      |                    ..........
      |            ........                 area of this strip / 0.01
 0.15 |      ......                         = AVERAGE detection rate while
      |   ...                                 false alarms stay under 1%
      | ..
 0.00 +.----------------------------------  FPR
      0                               0.01
```

**pAUC = the average detection rate while false alarms stay under 1%.** RM's 0.159 means that over that range,
on average 15.9% of target pixels are already found. A random score gets 0.005 (half the strip); a perfect one
gets 1.

### Why not just use ROC-AUC?

Worked numbers for Set 1:
- 1186 x 1196 pixels, 70% valid: about **993,000 pixels**.
- 1% FPR is about **9,900 background pixels** flagged; 0.1% is about **990**.
- An analyst reviews a few hundred candidates, not 500,000. Pixels at FPR 30% are never looked at.

ROC-AUC asks: "pick a random target pixel and a random background pixel; how often does the target score
higher?" About 99.9% of the pixels are background. That makes ROC-AUC mostly about how the huge middle of the
background is ordered.

Two detectors can disagree:

```
 TPR
     |              A ........................
     |          B ____----------------------    A: higher ROC-AUC (better in the middle and the tail)
     |       _--'.....                          B: higher pAUC (better in the first 1%)
     |    _-'.:
     |  /.:'                                    Analysts use B.
     | /:'
     |/'
     +--+-------------------------------------  FPR
     0  0.01
```

That is why every pass rule in this run was written on pAUC and **objects caught** (does any pixel of an object
land in the top 1% / 0.1%?). ROC-AUC was reported, not judged. Section 12 shows a real case where they disagree.

---

## 2. Does today's baseline use a guard window?

**Short answer: yes for the mean, no for the covariance.** RM uses a 21x21 guard inside a 61x61 window to
compute the local mean. The covariance (the "shape" of normal residuals) is global: one per scene.

```
 ┌──────────────────────────── 61 px ────────────────────────────┐
 │                                                               │
 │   RING: about 3,280 pixels (61² − 21² = 3721 − 441).          │
 │   Their average residual (whitened) is the LOCAL MEAN.        │
 │                                                               │
 │                ┌────────── 21 px ──────────┐                  │
 │                │                           │                  │
 │                │   GUARD: 441 pixels,      │                  │
 │                │   ignored                 │                  │
 │                │                           │                  │
 │                │             ●  test pixel │                  │
 │                │                           │                  │
 │                │                           │                  │
 │                └───────────────────────────┘                  │
 │                                                               │
 └───────────────────────────────────────────────────────────────┘

 score(●) = || whitened residual(●) − ring mean ||²
            (covariance used for whitening: the WHOLE scene, top 30 PCs)
```

Why the guard: a target is a few pixels wide, and its blurred halo reaches a few pixels further. Without a
guard, the target and its halo would be inside its own "background" mean and would cancel themselves. A guard of
21 px means nothing within 10 px of the test pixel counts as background.

Fallback: if the ring has fewer than 2d = 60 valid pixels (scene edge, next to nodata), the pixel is scored
against the global mean instead.

Code: `app/utils/anomaly_detection/residual_mahalanobis.py`, `LOCAL_WINDOW = (21, 61)`.

---

## 3. "Keeping only the top 30 components halved pAUC"

That sentence was about **HyperSL's embeddings**, before we ran anything on Indradhanu. It was a hypothesis, and
Phase A tested it on Indradhanu's residuals. **It turned out false for Indradhanu.** Here is the whole story.

### The picture: what PCA keeps and drops

Every pixel's residual has 165 numbers. PCA rotates them into directions sorted by how much the background varies
along each:

```
 variance
   |█
   |█
   |█ █
   |█ █ █
   |█ █ █ █ ▆ ▄ ▃ ▂ ▂ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ...   ▁ (165 directions)
   +--------------------------------------------------------
    PC1 ............. PC30 | PC31 ........................ PC165
    <---- kept by RM ---->   <-------- dropped ------------>
    big background swings      tiny variance: noise ... or anomalies?
```

RX divides each direction by its spread. A direction with tiny spread gets a huge multiplier, which raises two
possibilities:

| If the tail directions hold... | then dropping them... | and the fix is... |
|---|---|---|
| anomalies (things the background never does) | throws away the signal | keep them, but condition the covariance (Ledoit-Wolf, section 6) |
| sensor noise | removes noise that would be blown up | keep dropping them |

### What we saw in HyperSL (the source of the sentence)

| HyperSL embedding score | pAUC, HyperSL's own protocol | pAUC, our evaluator |
|---|---|---|
| RX on all 256 dims, Ledoit-Wolf | 0.134 | 0.137 |
| RX on top 30 PCs | 0.070 | 0.077 |

For HyperSL, truncating to 30 roughly halved pAUC, so its tail directions held signal.

### What Phase A found for Indradhanu's residual (A4)

```
 pAUC
 0.165 |  ●  d20
 0.159 |     ●  d30 (today)
 0.140 |          ●  d50
 0.122 |               ●  d80
 0.094 |                      ●  all 165 (Ledoit-Wolf)
       +---------------------------------> components kept
```

The opposite: each extra direction makes it worse. Why the difference:
- **HyperSL's embedding** is a learned feature space. Its small-variance directions are features that rarely
  fire, and a target firing one is real signal.
- **Indradhanu's residual** is what is left after the model explained the scene. The background structure is
  already gone, and its small-variance directions are mostly sensor noise. Amplifying them adds noise.

### "The protocols differ"

The HyperSL numbers came from the HyperSL run's own scoring script (valid-pixel AUC, its own pAUC code). RM's
numbers came from the product evaluator (`app/evaluation/metrics.py`). The two can differ slightly, so they
could not be compared directly. Experiment 0 put the saved HyperSL maps through our evaluator. On one protocol,
HyperSL's best is pAUC 0.137 against RM's 0.159. RM was ahead.

---

## 4. Why anomalies still get reconstructed

Indradhanu works on **tokens**: 4x4-pixel squares. To predict a token, the model hides it and fills it in from
the visible tokens around it. A pixel's residual is large only if the model **could not** guess it from context.
Three things let it guess targets anyway.

### 4a. The checkerboard leaks across token edges

Inference runs two passes. Pass 1 hides the tokens where (row + col) is even; pass 2 hides the odd ones.

```
 PASS 1    H = hidden (predicted)   v = visible (used as context)

            col 0   col 1   col 2   col 3
 row 0    [  H  ] [  v  ] [  H  ] [  v  ]
 row 1    [  v  ] [  H  ] [  v  ] [  H  ]
 row 2    [  H  ] [  v  ] [  H  ] [  v  ]
 row 3    [  v  ] [  H  ] [  v  ] [  H  ]
```

Every hidden token's four edge neighbours (up, down, left, right) are **visible**. Now place a 6-pixel-wide target
across the boundary between token (1,1) and token (1,2):

```
          token (1,1)  HIDDEN          token (1,2)  VISIBLE
        ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
        │ . │ . │ . │ . │            │ . │ . │ . │ . │
        ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
        │ . │ . │ X │ X │ ◀── copy ──│ X │ X │ X │ X │    X = target pixel
        ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
        │ . │ . │ X │ X │ ◀── copy ──│ X │ X │ X │ X │
        ├───┼───┼───┼───┤            ├───┼───┼───┼───┤
        │ . │ . │ . │ . │            │ . │ . │ . │ . │
        └───┴───┴───┴───┘            └───┴───┴───┴───┘

 Pass 1: the model predicts the left half while LOOKING AT the right half.
 Pass 2: roles swap; it predicts the right half while looking at the left half.
 Both halves get "explained", so the residual is small.
```

### 4b. The halo leaks even for a 1-pixel target

At PRISMA's 30 m pixels the sensor blurs (its point-spread function, PSF). A 1-pixel target spills part of its
spectrum into its neighbours:

```
 target signal strength around a 1-px target (illustrative)

     2%   5%   2%
     5%  100%  5%       the 5% ring falls in the NEXT token,
     2%   5%   2%       which is visible in the same pass
```

The model sees a faint copy of the target's spectrum in the visible neighbour, and nudges its prediction toward
it. FT06 measured the consequence. Its uncertainty head learned to expect large errors on targets: b on a target
was 1.04-1.19x the b of its own 3-8 px ring, and targets sat at the 87th-90th percentile of b.

### 4c. Training and inference hide different patterns

```
 TRAINING (v0.2.0: hyperspectral_segformer_exp_2.json, mask_ratio 0.65)   INFERENCE (checkerboard, 50%)

   H H v H H H v H                                                       H v H v H v H v
   v H H v H H H v                                                       v H v H v H v H
   H H H v v H H H        random, 65% hidden, clumpy:                    H v H v H v H v
   H v H H H v H H        large hidden areas, few visible                v H v H v H v H
                          tokens                                         regular, 50%, every hidden token
                                                                          has 4 visible neighbours
```

The model was trained on clumpy 65% masks. At inference it gets a regular 50% grid, with more and closer
context than it ever had in training.

### The measured symptom

Median residual of each target divided by the median residual of its own 3-8 px ring (1.0 = rebuilt as well as
its surroundings):

| Set 1 | Set 2 | Set 3 | Set 4 |
|---|---|---|---|
| 1.43 | 1.12 | **0.98** | **0.90** |

On Sets 3 and 4, targets are reconstructed as well as the background around them.

---

## 5. RM and d

**Yes to both.** RM = **residual Mahalanobis**. **d = the number of principal components kept.**

```
 cube x  (165 bands)                   one pixel
 reconstruction x̂ (165 bands)
          │
          ▼
 residual r = x − x̂                    165 numbers
          │
          ▼  rotate onto the top-d principal directions of r (covariance over the scene)
 d numbers
          │
          ▼  divide each by its standard deviation ("whiten")
 d numbers, each ~ unit spread for normal pixels
          │
          ▼  subtract the ring mean (section 2)
          ▼  square and add
 score = Σ (whitened component − ring mean)²     ≈ d for an average background pixel
                                                 ≫ d for an anomaly
```

So "RM d30" is today's score with 30 components, and "RM d20" keeps 20.

---

## 6. Ledoit-Wolf shrinkage

**The problem.** RX divides by the covariance. Across 165 neighbouring bands the sample covariance S has some
eigenvalues near zero (adjacent bands are almost copies of each other). Dividing by about 0 sends scores to about
1e11. That is why plain RX was dropped in 2026-05.

**Two fixes.**
- **Truncation (what RM does):** keep only the top d directions and throw the rest away.
- **Shrinkage (Ledoit-Wolf):** keep every direction, but blend S with a perfectly round covariance:

```
 Σ_LW = (1 − δ) · S  +  δ · m · I         m = average eigenvalue of S,  I = identity
```

δ (between 0 and 1) is not tuned by hand. Ledoit and Wolf (2004) give a formula that picks the δ minimising the
expected estimation error from the data itself. The effect on the eigenvalues is easy to see:

```
 λ_new = (1 − δ) · λ_old + δ · m

 worked example: eigenvalues 100, 10, 1, 0.001   →  m = 27.75,  say δ = 0.1

   before:   100     10      1       0.001     →  RX weight 1/λ:  0.01  0.1  1   1000  ← blow-up
   after:     92.8   11.8    3.7     2.8       →  RX weight 1/λ:  0.011 0.085 0.27 0.36  ← tame

 picture:
   before  █████████████████ █▊ ▏ ·
   after   ████████████████  ██ ▋ ▌        small eigenvalues lifted off the floor,
                                           big ones pulled down a little
```

Big directions barely change. The near-zero ones are lifted to a safe floor, so nothing explodes and no direction
is discarded.

In A4, "RX full LW" kept all 165 residual directions this way and scored pAUC 0.094, the worst of the family. The
shrinkage worked numerically, but the extra directions were noise (section 3).

---

## 7. Phase B, and B2 in particular

Phase B changed **which tokens are hidden** at inference. The weights were not touched. Each variant is a list of
passes. In each pass some tokens are hidden, and some of the hidden ones are **scored**: their prediction is kept.
Every token is scored in exactly one pass, so every pixel's reconstruction comes from a pass where it was hidden.

### B0: today's checkerboard (2 passes), the reference

```
 pass 1            pass 2
 H v H v H v       v H v H v H
 v H v H v H       H v H v H v
 H v H v H v       v H v H v H
```

### B1: 8 random masks at 0.65 (matching training)

```
 pass 1            pass 2            ...  pass 8
 H H v H H v       v H H H v H
 v H H H v H       H H v H H H
 H v H H H H       H v H v H H
 each pixel's reconstruction = average over the passes where its token was hidden
```

### B2a: blind block, 3x3 tokens, scored at the centre (16 passes)

One pass. **C** = scored centre, **H** = hidden guard around it, **.** = visible context:

```
 token grid (each cell = 4x4 px)

   0 1 2 3 4 5 6 7 8 9 10 11
 0 . . . . . . . . . . .  .
 1 . H H H . H H H . H H  H        centres sit on a lattice
 2 . H C H . H C H . H C  H        4 tokens apart
 3 . H H H . H H H . H H  H
 4 . . . . . . . . . . .  .
 5 . H H H . H H H . H H  H
 6 . H C H . H C H . H C  H
 7 . H H H . H H H . H H  H
 8 . . . . . . . . . . .  .

 hidden share: 9 of every 16 tokens = 56%
```

The next pass shifts the whole lattice by one token, and so on over 4 x 4 = **16 shifts**, so every token is a
centre exactly once.

What the scored centre "sees", in pixels:

```
 ┌───────────────────────────────────────┐
 │ visible context (beyond the guard)    │
 │   ┌───────────────────────────────┐   │
 │   │ guard: hidden, 4 px all round │   │
 │   │       ┌───────────────┐       │   │
 │   │       │ centre: 4x4 px│       │   │
 │   │       │   SCORED      │       │   │
 │   │       └───────────────┘       │   │
 │   │                               │   │
 │   └───────────────────────────────┘   │
 │          12 x 12 px hidden            │
 └───────────────────────────────────────┘

 nearest visible pixel is at least 5 px from any scored pixel
 → a target up to ~8 px plus its halo cannot be copied from context
```

This is your idea 3 (rebuild the inner box from the outer box), and it closes both leaks in section 4.

### B2b: blind block, 5x5 tokens (36 passes)

The same idea, bigger: a 20x20 px hidden block, an **8 px guard**, centres 6 tokens apart. That is 6 x 6 =
36 passes, with 25 of every 36 tokens (69%) hidden.

### B3: checkerboard, with suspects never used as context

```
 pass 1 of the checkerboard, but tokens touching the top 1% of RM (+1 token) are
 hidden in BOTH passes (S):

 H v H v H v
 v H S S v H        S: hidden in pass 1 AND pass 2, so a suspected target
 H v S S H v           never serves as context for its neighbours
 v H v H v H
```

### Why B2 did not win: it closes the leak, but the background pays

```
 What the mechanism check showed (target ÷ own ring, Set 3):   0.98 → 1.18 (B2a), 1.22 (B2b)   ✔ leak closed
 What happened to ordinary background (median residual, Set 3): 0.0116 → 0.0179 (B2a), 0.0219 (B2b)  ✘ 1.5-1.9x worse
```

The model was trained to fill single scattered 4x4 holes. A 12-20 px hole is a different task, and it is
hardest on edges:

```
 a field boundary running through a hidden block

 visible   │ hidden block │  visible
  crop     │   ? ? ? ?    │  bare soil
  crop     │   ? ? ? ?    │  bare soil        the model must guess WHERE the edge
  crop     │   ? ? ? ?    │  bare soil        runs inside the block; guess wrong
                                             by 2 px → a stripe of big residuals
```

Those edge errors land in the top 1% of scores, the only part pAUC looks at, and push targets down. So B2a
scored pAUC 0.148 and B2b 0.153, against 0.159. B2 did catch more large objects (>20 px: 75% to 80%) and fewer
1-2 px ones (36.5% to 32-35%).

---

## 8. Did Phases C and D run?

| Phase | What | Ran? |
|---|---|---|
| C1 | fine-tune with block masks | **No.** You decided no fine-tuning. |
| C2 | uncertainty head with a 3x3 hidden context | **No** (also training). |
| D1 | RX / kNN on Indradhanu's own features | **Yes, as F5.** It failed badly (pAUC 0.100-0.108). A reconstruction model's features describe a pixel's surroundings, not the pixel. |
| D2 | fuse RM with HyperSL maps | **Yes, as F3.** "RM d20 + HyperSL embedding RX" passed on PRISMA (0.168). It was not checked on HAD100, because no HyperSL maps exist for HAD100. |

So C did not run, and D ran inside Phase F.

---

## 9. The robust fit

**The problem.** RM learns "normal" from the scene itself. The scene includes the targets, which contaminate
normal in two places.

### Contamination 1: the ring mean

```
 ┌──────────── 61 px ring ────────────┐
 │                                    │
 │      ▲▲  ← another target (or the  │      the ring average is pulled toward
 │      ▲▲    same target's halo)     │      target-like spectra, so:
 │         ┌──── guard ────┐          │        - the test target looks LESS unusual
 │         │      ●        │          │        - nearby background looks MORE unusual
 │         └───────────────┘          │
 └────────────────────────────────────┘
```

### Contamination 2: the covariance

```
 2 of the 30 whitened components, every pixel a dot

   fit on ALL pixels                    fit WITHOUT the suspects (x)

        x                                    x
          .:::::.     x                        .:::.       x
      .:::::::::::::.                       .:::::::.
    x  ':::::::::::'   x                 x   ':::::'     x
          '::::'                                         ← ellipse fits the background tightly
   ellipse stretched toward the x's;     x's now sit far outside it:
   they look only mildly unusual         they look very unusual
```

### The fix: two passes

```
 pass 1:  score every pixel with RM, as today
            │
            ▼
          suspects = top 1% of scores, grown by a 2-px halo
            │
            ▼
 pass 2:  refit the covariance AND the ring mean using only NON-suspect pixels;
          score EVERY pixel (suspects included) against that clean background
```

This is the "background purification" idea from the literature (local-density refinement, LOF dictionaries).
Results:

| | PRISMA pAUC | HAD100 pAUC | HAD100 crops won / lost |
|---|---|---|---|
| RM d30 (today) | 0.159 | 0.662 | - |
| robust RM, top 1% removed | 0.166 | **0.738** | 78 / 20 |

The gain is much bigger on HAD100. In a 64x64 crop (4,096 px), a target can be 1-5% of all pixels, so it
contaminates the fit much more than in a million-pixel PRISMA scene.

---

## 10. The IJDE benchmark

The **International Journal of Digital Earth**, 2022: a comparison of **22 hyperspectral anomaly detectors on 17
public datasets** (https://doaj.org/article/67a835161e174c44bbb0f16d1cf772df). Per its abstract, AED (attribute
and edge-preserving filters) ranked among the best by mean and median ROC-AUC. Reducing the data to about **20
principal components** improved results, and reducing further hurt.

Two cautions:
- I read the abstract and summaries, not the full paper.
- Its "20 PCs" is PCA on the **image cube** for classical detectors, not on a model's residual. It agreed with
  our PRISMA A4 result (d20 > d30), but HAD100 did not confirm d20 for RM. The match is suggestive, not proof.

---

## 11. The overall conclusion

Your summary is right about reconstruction and needs two corrections on scoring.

**Reconstruction: no better technique found, without training.** Blind blocks (B2), random masks (B1), context
removal (B3) and shift averaging (F4) all failed the bar. The blind block does close the leak, but this model
can't fill large holes well. That would need the block-mask fine-tune you ruled out.

**Scoring:**

| Method | PRISMA pAUC | HAD100 pAUC | Status |
|---|---|---|---|
| RM d30 (today) | 0.159 | 0.662 | baseline |
| **CFAR on kNN (10 nearest neighbours, whitened top-20 residual)** | **0.185** | **0.742** | **confirmed on both; best** |
| kNN alone | 0.176 | 0.723 | confirmed on both |
| robust RM (top 1% removed from the fit) | 0.166 | 0.738 | confirmed on both |
| RM d20 | 0.165 | 0.666 | **not confirmed** (HAD100 53 wins / 42 losses, p = 0.11): PRISMA only |
| CFAR on RM d20 | 0.173 | 0.647 | **not confirmed** (lost on HAD100: 34 / 63) |

The two corrections:
1. **RM d20 on its own is not a confirmed win.** Top-20 does live on inside the kNN method, which works on the
   whitened top-20 residual.
2. **d20 does not lower ROC-AUC** (0.851 vs 0.854, about equal). What lowers ROC-AUC is **CFAR** (section 12).

So the stack from best to simplest is CFAR on kNN, then kNN alone, then the robust fit.

---

## 12. How can CFAR on RM d20 be better when ROC-AUC drops?

**Short answer.** CFAR reorders the very top of the ranking for the better and the middle for the worse. pAUC
only sees the top; ROC-AUC mostly sees the middle. And on HAD100 CFAR on RM was **not** better at all: it lost
63 of 100 crops. The version that survives is CFAR on kNN.

### What CFAR does

```
 CFAR score = score / (median score in the ring around the pixel)
```

It asks "how unusual is this pixel **for its neighbourhood**", not "how unusual overall".

### A five-pixel worked example

| Pixel | Raw score | Local median | CFAR score |
|---|---|---|---|
| edge of a field (busy area), background | 30 | 20 | 1.5 |
| target in a busy town | 30 | 20 | 1.5 |
| target in a quiet field | 20 | 0.5 | **40** |
| quiet water, ordinary noise | 0.5 | 0.1 | **5** |
| quiet field, ordinary pixel | 0.4 | 0.5 | 0.8 |

```
 ranking BEFORE CFAR (raw)            ranking AFTER CFAR

 1. field edge (bkg)    30   ✘        1. target, quiet field   40   ✔  ← top improves:
 1. target, busy town   30   ✔        2. water noise            5   ✘     edge false alarm gone
 3. target, quiet field 20   ✔        3. field edge (bkg)      1.5  ✘
 4. water noise          0.5          3. target, busy town     1.5  ✔  ← busy-area target sinks
 5. quiet field pixel    0.4          5. quiet field pixel     0.8
```

Three effects at once:
1. **Busy-area false alarms drop out of the top.** The field edges and urban texture that filled the top 1% are
   divided by large local medians. This is why **pAUC rises** (0.159 to 0.173 on PRISMA).
2. **Quiet-area noise gets inflated.** Tiny fluctuations over water or uniform fields are divided by tiny
   medians. They move from the bottom into the **middle** of the ranking, mixing with many targets there. This
   is why **ROC-AUC falls** (0.854 to 0.808).
3. **Targets inside busy areas sink.** Set 3 lost pAUC in every CFAR variant.

### The two ROC curves cross

```
 TPR
     |                        RM d20 (no CFAR)  ___-------------
     |               CFAR ...........________---
     |          .....'      ___----              in the middle, CFAR is BELOW
     |      ..'     ___---'                      → lower ROC-AUC (0.808 vs 0.851)
     |   .:'   __--'
     |  .' _--'
     | :_-'     in the first 1%, CFAR is ABOVE
     |/'        → higher pAUC (0.173 vs 0.165)
     +--+--------------------------------------  FPR
     0  0.01
```

### Why CFAR on kNN survives where CFAR on RM fails

The likely reason (an explanation, not something measured): kNN already handles "this neighbourhood is busy"
partly by itself. A field-edge residual has close neighbours among the many other edge residuals in the bank, so
it scores low. CFAR on top of kNN then only has to fix what
is left, without such strong side effects. On HAD100, CFAR on kNN **raised** ROC-AUC (0.9838 to 0.9868), while
CFAR on RM lowered it (0.9772 to 0.9688). On PRISMA, CFAR on kNN still lowers ROC-AUC (0.864 to 0.827). That is
acceptable because the product thresholds at the top of the ranking, by percentile, not at FPR 30%.
