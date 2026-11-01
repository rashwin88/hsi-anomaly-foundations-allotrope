# Part 32 - Unmixing and sub-pixel targets with HyperSL

> **The one thing this part teaches:** a published "endmember-free" unmixing method built on HyperSL lost to classical
> VCA + FCLS on every data set with exact truth, needed VCA to start in the authors' own code, gained nothing from
> pretraining, took 5-13 minutes per scene on a 96 GB GPU, and - because its abundances pass through a 3x3 head and a
> total-variation penalty - smoothed a target a seventh of a pixel wide down to an object AUC of 0.68, where per-pixel
> ACE scored 0.97.

## Where we are

Part 30 asked whether HyperSL's embedding can name a material. A second route: use HyperSL as the backbone of an
**unmixing** network that splits each pixel into pure materials and their shares (Liu et al., "Endmember-Free
Hyperspectral Unmixing", IEEE JSTARS 18, 2025). If it worked it would answer "which material" and "how much of the
pixel" at once. It was tested on 2026-10-13/14 on a Colab RTX PRO 6000 (96 GB).

| file | what it holds |
|---|---|
| `git show 4d292ac:research/experiments/hypersl-unmixing/README.md` | the whole experiment: presets, tables, findings, limitations |
| `git show 4d292ac:research/experiments/manmade-library-unmixing/README.md` | the summary added to our own unmixing record |
| `docs/lld/spectral-family-match.md` | the rejected-techniques table, HyperSL row |
| `research/textbook/tools/compute_hypersl.py` | the 3x3 and TV worked numbers |

> **Analogy.** A newspaper photo printed through a coarse halftone screen. A crowd reproduces well; a person far off,
> smaller than one dot, is averaged into the grey and vanishes. A sub-pixel target is that person; HyperSL's unmixing head
> is the screen.

## Step 1 - the words

- **Linear mixing model:** `x = E a + n`, the pixel as a weighted sum of K pure spectra (**endmembers**, the columns of
  E) with weights `a_k >= 0`, `sum a_k = 1` (**abundances**). With K = 2 it is part 27's fill model `x = (1-a) b + a t`.
- **Blind unmixing:** find E and the abundances from the image alone. **VCA** finds endmembers as the most extreme pixels;
  **FCLS** fits the abundances per pixel under both constraints. Seconds per scene.
- **mSAD:** mean spectral angle (radians) between matched found and true endmembers; **mRMSE:** mean RMS abundance error.
  Lower is better. 0.1 rad = 5.7 deg.
- **Object AUC:** the share of background pixels an object's best pixel outranks.

Data: DC1-3 and EGU-Net (synthetic, exact truth, "grade B"; DC3 has no pure pixels and EGU has spectral variability - the
two cases built to favour a learned method); Samson, Jasper, Urban4 (real, "truth" is FCLS from hand-picked endmembers,
"grade D"); DLR HySU (5 materials, squares 0.25-3 m at 0.7 m, VNIR); Gudalur AVIRIS-NG at 32 m (16 m sheets, fill <= 0.25).

| preset | endmember start | R_theta | source |
|---|---|---|---|
| `paper` | random | yes | the paper |
| `paper_noR` | random | no | the paper without R_theta |
| `upstream` | **VCA** | no | the authors' `downstream_unmixing.py` |
| `scratch` | random | yes | `paper` with a random backbone |

**Source check:** the authors' released script overwrites the decoder with VCA endmembers and prints "Endmember
initialized by VCA". It is not endmember-free. Only it came close to a published number (Urban 0.262 vs 0.213); the
endmember-free version reached 0.323 on Jasper against a reported 0.045.

## Step 2 - blind unmixing results

mSAD (rad), mean +- SD over seeds; best per row in bold:

| set (grade) | VCA+FCLS (10) | upstream (2) | paper_noR (2) | paper (1) | scratch (1) |
|---|---|---|---|---|---|
| DC1 (B) | **0.000 +- 0.000** | 0.092 +- 0.000 | 0.084 +- 0.003 | 0.173 | 0.173 |
| DC2 (B) | **0.000 +- 0.000** | 0.085 +- 0.001 | 0.082 +- 0.029 | 0.182 | 0.175 |
| DC3 (B) | **0.069 +- 0.008** | 0.124 +- 0.000 | 0.164 +- 0.043 | 0.199 | 0.153 |
| EGU (B) | **0.004 +- 0.001** | 0.106 +- 0.001 | 0.083 +- 0.005 | 0.264 | 0.249 |
| Samson (D) | 0.085 +- 0.062 | 0.278 +- 0.108 | **0.074 +- 0.025** | 0.531 | 0.493 |
| Jasper (D) | 0.397 +- 0.038 | 0.405 +- 0.011 | **0.323 +- 0.039** | 0.472 | 0.472 |
| Urban4 (D) | 0.418 +- 0.015 | **0.262 +- 0.050** | 0.322 +- 0.084 | 0.419 | 0.413 |

mRMSE (abundance):

| set (grade) | VCA+FCLS | upstream | paper_noR | paper | scratch |
|---|---|---|---|---|---|
| DC1 (B) | **0.000 +- 0.000** | 0.332 +- 0.025 | 0.333 +- 0.001 | 0.380 | 0.388 |
| DC2 (B) | **0.000 +- 0.000** | 0.220 +- 0.010 | 0.209 +- 0.005 | 0.387 | 0.296 |
| DC3 (B) | **0.103 +- 0.026** | 0.324 +- 0.033 | 0.300 +- 0.009 | 0.357 | 0.347 |
| EGU (B) | **0.081 +- 0.005** | 0.229 +- 0.001 | 0.242 +- 0.004 | 0.453 | 0.438 |
| Samson (D) | 0.282 +- 0.021 | 0.301 +- 0.040 | 0.236 +- 0.000 | 0.308 | **0.225** |
| Jasper (D) | 0.317 +- 0.044 | 0.298 +- 0.026 | **0.242 +- 0.113** | 0.472 | 0.461 |
| Urban4 (D) | 0.380 +- 0.033 | 0.282 +- 0.119 | **0.281 +- 0.021** | 0.517 | 0.569 |

1. **Exact truth: VCA + FCLS wins every set on both measures.** On DC3, `0.069 x 57.296 = 3.95 deg` against `paper`'s
   `0.199 x 57.296 = 11.40 deg`.
2. **HyperSL wins only on grade D**, whose truth is itself an unmixing - agreement, not accuracy. Our VCA + FCLS matches
   the paper's own VCA rows (Jasper 0.397 vs 0.406; Urban 0.418 vs 0.473), so the chain is calibrated.
3. **Pretraining adds nothing in the paper's configuration:** `scratch` = `paper` (DC1 0.173 vs 0.173; EGU 0.249 vs
   0.264).
4. **R_theta**, an unconstrained CNN path in `x_hat = E a + R_theta(...)`, carries **29-56% of the reconstruction energy**,
   so the abundances need not explain the pixel; on target scenes it drove target abundance to ~1 even on empty grass.

**Not run:** `paper_noR` with a random backbone, so pretraining's value without R_theta is untested.

## Step 3 - sub-pixel targets

HySU object AUC:

| target side (true area, px) | 0.25 m (0.13) | 0.5 m (0.51) | 1 m (2.0) | 2 m (8.2) | 3 m (18.4) |
|---|---|---|---|---|---|
| ACE, our library + field spectra | **0.97** | **1.00** | 1.00 | 1.00 | 1.00 |
| step 5 gain, our library + field | **0.97** | **1.00** | 1.00 | 1.00 | 1.00 |
| ACE, our library only | 0.88 | 0.99 | 1.00 | 1.00 | 1.00 |
| HyperSL anchored (`paper_noR`) | 0.68 | 0.77 | 1.00 | 1.00 | 1.00 |
| HyperSL blind (`paper_noR`) | 0.72 | 0.62 | 0.80 | 0.88 | 0.92 |

Gudalur (32 m, fill <= 0.25): ACE 0.99, step 5 0.98, anchored HyperSL 1.00, blind HyperSL 0.59. Blind HyperSL never learned
a target as an endmember (nearest 18 deg on HySU, 10 deg at Gudalur; 3-4 of 5 materials share one).

## Step 4 - why, by hand

**3x3 averaging.** One target pixel at fill `a` among zeros; the 3x3 average at the target is `(a + 8 x 0) / 9 = a / 9`, and
each of its 8 neighbours also gets `a / 9`. HySU's smallest: `0.13 / 9 = 0.0144`; `0.51 / 9 = 0.0567`. The peak falls
ninefold. (A learned head is not exactly an average; any positive weights summing to 1 smooth the same way - an assumption.)

**Total variation (TV)** is the sum over side-by-side pixel pairs of `|a_i - a_j|`; a TV penalty adds `lambda x TV` to the
loss. For one target pixel reported at `a'`: the reconstruction misses by `(a - a')(t - b)`, squared length
`(a - a')^2 c` with `c = |t - b|^2`; its TV is `4 a'` (four neighbours).

```
J(a')  = c (a - a')^2 + 4 lambda a'
dJ/da' = -2c (a - a') + 4 lambda = 0   ->   a' = a - 2 lambda / c,   floored at 0
c = 0.20, lambda = 0.013 (assumed):  2 x 0.013 / 0.20 = 0.13
a = 0.13 -> 0;   a = 0.51 -> 0.38;   a = 1 -> 0.87
```

For a k x k block the TV is `4k a'` and the data term `k^2 c (a - a')^2`, giving `a' = a - 2 lambda / (k c)`: shrinkage 0.13,
0.065, 0.043, 0.026 for k = 1, 2, 3, 5. **TV charges by perimeter and the data pay by area, so the smallest targets lose
most.** ACE and the step-5 gain score each pixel alone and keep them.

Fill (median relative area error on HySU): HyperSL anchored 0.97 / 0.99 at 0.25 / 0.5 m - the TV result, measured; step 5
with the right spectrum 0.30-0.35 / 0.51-0.68.

**Cost:** VCA + FCLS seconds; HyperSL 5 min (`upstream`) to 13 min (`paper_noR`) per scene on 96 GB. Rejected
(`docs/lld/spectral-family-match.md`).

## Common confusions

**"HyperSL won on Jasper and Urban."** Against references that are themselves FCLS. On exact truth it lost every set.

**"Endmember-free means no classical step."** The released script starts from VCA.

**"Pretraining is what makes it work."** With R_theta, a random backbone matched it.

**"A 3x3 head only blurs edges."** For a one-pixel target the whole object is edge.

**"TV removes noise, not targets."** For a one-pixel target it subtracts the same `2 lambda / c` from both.

## Check yourself

1. Target at fill 0.25 among zeros, 3x3 average: value at the target, at a diagonal neighbour, two steps away?
2. Convert EGU's mSAD for `upstream` and VCA + FCLS to degrees.
3. With `c = 0.20`, what is the largest `lambda` that keeps a single 0.13 target above 0?
4. Repeat the TV shrinkage for a 2x2 target at fill 0.13 (`c = 0.20`, `lambda = 0.013`).
5. HyperSL takes 13 min per scene. How long for the 4 PRISMA bench scenes, and 18 implant hosts?

<details>
<summary>Answers</summary>

1. `0.25 / 9 = 0.0278` at the target and at the diagonal neighbour; 0 two steps away (its window misses the target).
2. `0.106 x 57.296 = 6.07 deg`; `0.004 x 57.296 = 0.23 deg`.
3. Need `0.13 > 2 lambda / 0.20`, so `lambda < 0.013`: the worked example sits exactly on the boundary.
4. `2 x 0.013 / (2 x 0.20) = 0.065`; `a' = 0.13 - 0.065 = 0.065`. It survives at half its fill.
5. `4 x 13 = 52` min; `18 x 13 = 234` min = 3.9 h.

</details>

---

Next: part 33, cloud masking with HyperSL.
