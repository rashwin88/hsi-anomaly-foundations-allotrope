# Part 30 - Per-scene min-max, and a library with nowhere to stand

> **The one thing this part teaches:** per-scene min-max scaling makes a pixel's input depend on the brightest and
> darkest pixels of its scene, so the same material reads differently in every scene, one bright anomaly changes every
> other pixel, and a library spectrum - which has no scene - has no defined input at all; measured on four PRISMA
> scenes, HyperSL's embeddings did not name man-made targets better than plain spectral angle, and one sage-leaf
> spectrum absorbed 60% of a scene's anomalies.

## Where we are

Part 29 showed that HyperSL scales every band by its scene's minimum and maximum before it sees anything. Library
matching (parts 14, 15, 27) compares a pixel with a laboratory spectrum. That has a precondition: **the two must be put
on the same footing first.** This part shows, on a three-pixel scene, why HyperSL's scaling cannot meet it, and then
what happened when the team tried anyway.

| file | what it holds |
|---|---|
| `C:/Users/FITT/Documents/hypersl_match_data/results/summary.json` | the 2026-09-23 four-scene matching run's numbers (outside the repo) |
| `git show 4bbf91d:research/experiments/hypersl-splib-match/run_match.py` | how that run matched, and what it measured |
| `git show 4bbf91d:research/experiments/hypersl-splib-match/hypersl_model.py` | the scalings it tried, and the 41% clipping note |
| `docs/proposals/spectral-matching-foundation-model.md` | the 28% band-sampling result (line 63) and the conclusion |
| `research/textbook/tools/compute_hypersl.py` | every worked number below |

> **Analogy.** Marks graded on a curve. In each class the weakest student gets 0 and the strongest 100. A 75 means
> different things in different classes; one brilliant newcomer pushes everyone else down; and an external candidate who
> sat the exam alone has no curved mark at all. The library spectrum is the external candidate.

## Step 1 - the formula, rearranged

```
x'_b = (x_b - m_b) / (M_b - m_b)  =  s_b x_b + c_b,     s_b = 1 / (M_b - m_b),  c_b = -m_b / (M_b - m_b)
```

A **per-band affine map**: each band multiplied by its own number and shifted by its own number. Our fixed z-score
(`WavelengthStats`, part 23) is the same form, `z_b = (x_b - mu(lambda_b)) / sigma(lambda_b)`. **The difference is where
`s_b` and `c_b` come from: the scene, or a fixed curve in the checkpoint.** Everything else follows from that.

## Step 2 - a three-pixel scene by hand

Scene S1: `p1 = (0.10, 0.30)`, `p2 = (0.20, 0.50)`, `p3 = (0.30, 0.40)`. Band 1: min 0.10, max 0.30, span 0.20. Band 2:
min 0.30, max 0.50, span 0.20.

```
p1' = ((0.10-0.10)/0.20, (0.30-0.30)/0.20) = (0, 0)
p2' = (0.10/0.20, 0.20/0.20)             = (0.5, 1.0)
p3' = (0.20/0.20, 0.10/0.20)             = (1.0, 0.5)
```

**(a) Same spectrum, other scene.** S2 holds `(0.05, 0.20)`, `(0.20, 0.50)`, `(0.40, 0.60)`: spans 0.35 and 0.40.
`(0.20, 0.50)` becomes `(0.15/0.35, 0.30/0.40) = (0.4286, 0.75)`. In S1 it was `(0.5, 1.0)`. One material, two inputs.

**(b) One bright anomaly moves everybody.** Add `a = (0.60, 0.90)` to S1. Maxima become 0.60 and 0.90, spans 0.50 and 0.60.

```
p2' = (0.10/0.50, 0.20/0.60) = (0.2, 0.3333)     was (0.5, 1.0)
p3' = (0.20/0.50, 0.10/0.60) = (0.4, 0.1667)     was (1.0, 0.5)
a'  = (1, 1)
```

No reflectance changed. The anomaly rewrote every ordinary pixel's input.

**(c) A library spectrum has no frame.** `t = (0.70, 0.20)` was measured alone. Three ways to force it in, all bad:

- *Borrow S1's frame:* `((0.70-0.10)/0.20, (0.20-0.30)/0.20) = (3.0, -0.5)` - outside [0, 1], where the sigmoid output
  cannot reach. Clip to `(1, 0)`. On real data the first attempt clipped **41% of all ECOSTRESS values**, mostly natural
  entries.
- *Put `t` into the scene:* new minima `(0.10, 0.20)`, maxima `(0.70, 0.50)`; `t' = (1, 0)` and `p1` moves from `(0, 0)` to
  `(0, 0.3333)`. 2,000 library entries means 2,000 frames.
- *Scale each spectrum by itself ("per-curve"):* fair to both sides but off the training distribution - the loss roughly
  doubles (part 29, Step 4).

Borrowing the scene's frame is **circular**: a scene's extremes are usually set by its unusual pixels, the very ones being
named, so the frame depends on what the target is.

**(d) Spectral angle does not survive.** Before, `p2 . p3 = 0.06 + 0.20 = 0.26`; `|p2| = sqrt(0.29) = 0.5385`;
`|p3| = sqrt(0.25) = 0.5000`; `cos = 0.26 / 0.26926 = 0.9656`, **15.07 deg**. After, `(0.5, 1.0) . (1.0, 0.5) = 1.0`; both
lengths `sqrt(1.25) = 1.1180`; `cos = 1.0 / 1.25 = 0.8`, **36.87 deg**. And `p1` became `(0, 0)`, for which no angle exists.

Why: with `x'_b = s_b x_b`, `cos' = sum s_b^2 u_b v_b / (sqrt(sum s_b^2 u_b^2) sqrt(sum s_b^2 v_b^2))`. Equal `s_b` cancel;
unequal ones re-weight the bands. With `s = (5, 10)`, the 15.07 deg pair becomes 9.25 deg. Offsets change it again.

**The fixed z-score changes angles too - but identically in every scene and for every library entry.** Scene-dependence,
not affinity, is the problem.

## Step 3 - a convolution over band index

A 1-D convolution with kernel `k` makes band `i` from bands `i-h ... i+h`, `h = (k-1)/2`, with the same weights everywhere. It
knows list positions, not wavelengths. The **span** of a window is the last band's wavelength minus the first's.

PRISMA's 149-band list (all our HyperSL runs) drops 1325-1495 nm. Around the gap:

```
below: ..., 1250.98, 1262.53, 1273.50, 1284.49, 1295.42, 1306.22, 1317.26
above: 1502.02, 1512.63, 1523.22, ...
window centred on 1273.50: 1295.42 - 1250.98 = 44.44 nm
window centred on 1317.26: steps 10.80, 11.04, 184.76, 10.61;  span 1512.63 - 1295.42 = 217.21 nm
```

| band list | bands | k=5 median / max (nm) | k=13 median / max (nm) |
|---|---|---|---|
| PRISMA, all 234 good bands (water kept) | 234 | 36.4 / 45.0 | 108.9 / 132.7 |
| PRISMA 149 (water, noisy cut) | 149 | 39.7 / 280.4 | 122.5 / 358.9 |
| EnMAP via SpectralEarth (water cut) | 202 | 31.0 / 291.6 | 93.1 / 379.2 |
| product 165-band grid | 165 | 40.0 / 200.0 | 120.0 / 280.0 |

With water bands kept - how HyperSL trained on EnMAP - the spans are even and the design is consistent. Every list this
product uses cuts the water windows, and the largest span grows six- to sevenfold. Our stem groups bands into 40 nm
wavelength bins instead (part 23): an absent band is absent.

## Step 4 - what happened when it was tried (2026-09-23)

`hypersl-splib-match` matched every ground-truth anomaly pixel of the four PRISMA scenes against ECOSTRESS (1,947
entries, 45 man-made) and USGS splib07 (518), with HyperSL per-curve (mask 0), HyperSL raw, and SAM.

Share of pixels whose best ECOSTRESS match is man-made (anomaly / brightness-matched background):

| scene (anomaly px) | HyperSL per-curve | HyperSL raw | SAM |
|---|---|---|---|
| set1 (638) | 7.2% / 2.7% | 0.16% / 0.0% | 9.6% / 1.9% |
| set2 (992) | 0.40% / 0.0% | 0.30% / 0.0% | 0.81% / 0.20% |
| set3 (778) | 0.26% / 0.51% | 0.39% / 0.13% | 2.7% / 2.1% |
| set4 (1,165) | 8.3% / 33.0% | 0.0% / 0.0% | 6.4% / 24.3% |

HyperSL is below SAM on three of four scenes. **Attractors:** with HyperSL per-curve, *Salvia leucophylla 3* (a sage leaf)
was the best match for **59.6%** of set2's anomaly pixels; *Dolomitic Marble* for 31.2% of set4's. SAM's largest attractor
took 19.9%.

**To be fair:** HyperSL's space groups library classes a little better than raw angle (5-nearest-neighbour same-class share
on ECOSTRESS 0.965 vs SAM 0.949; man-made 0.124 vs 0.093). But 0.124 means seven of eight neighbours of a man-made entry
are natural.

**Source check:** the run's man-made test is `e.chapter == "manmade"`; USGS calls its chapter "artificial", so every USGS
man-made share is 0 by construction. Read the USGS purity, ignore its man-made shares.

**Band sampling.** An ECOSTRESS entry at its lab sampling found its own PRISMA-resampled copy as nearest neighbour only
**28%** of the time (proposal, line 63). **Source check:** that number has no output file - the committed scripts do not
save it. Treat it as recorded by the people who ran it.

## Common confusions

**"Min-max only changes brightness, and angle ignores brightness."** It changes each band by its own scale and offset:
15.07 deg became 36.87 deg.

**"Then our z-score is just as bad."** It is also affine, but the same in every scene and for every library entry.

**"Per-curve scaling fixes it."** Fair to both sides, off-distribution, and it erases brightness.

**"The position code tells the tokenizer about the gap."** It is added after the convolutions.

**"HyperSL never matched USGS man-made entries."** The 0% is a label-name mismatch, not a result.

## Check yourself

1. Scene S3: `(0.15, 0.25)`, `(0.20, 0.50)`, `(0.25, 0.55)`. Compute the input of `(0.20, 0.50)`.
2. In S1 plus `a`, what is `p1`'s input? Why did it not move?
3. Compute the angle between `(0.20, 0.50)` and `(0.30, 0.40)` after scaling by `(5, 10)`.
4. On the 165-band grid, find the span of the 5-band window centred at 1780 nm (gap 1790 -> 1960).
5. Write the fixed z-score of `t = (0.70, 0.20)` with mean `(0.25, 0.35)` and spread `(0.10, 0.12)`. Does it depend on the
   scene?

<details>
<summary>Answers</summary>

1. Band 1 span 0.10: `0.05 / 0.10 = 0.5`. Band 2 span 0.30: `0.25 / 0.30 = 0.8333`. A third input for one spectrum.
2. `(0, 0)`: it holds both minima, and the bright pixel changed only the maxima.
3. `(1, 5)` and `(1.5, 4)`: dot `1.5 + 20 = 21.5`; lengths `sqrt(26) = 5.0990`, `sqrt(18.25) = 4.2720`;
   `cos = 21.5 / 21.783 = 0.9870`, 9.25 deg.
4. 1760, 1770, 1780, 1790, 1960: `1960 - 1760 = 200` nm.
5. `(0.45 / 0.10, -0.15 / 0.12) = (4.5, -1.25)`. No: the curve is fixed in the checkpoint.

</details>

---

Next: part 31, anomaly detection - a model that completes the anomaly.
