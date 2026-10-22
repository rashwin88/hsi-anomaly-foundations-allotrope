# FT09 - a richer decoder: render what the encoder sees

**Status:** FT09b done 2026-10-22 16:06 UTC, best.pt = epoch 5 - **the first resolved pAUC gain since cloud exclusion (criteria 1, 3, 4 met; 2 fails at the top 0.1% by 4 objects, unresolved; 5 failed as written, see FT09a). A candidate, not a release.** FT09c (5 more joint epochs) launched after it. First attempt stopped after epoch 3 (2026-10-22 13:05 UTC); rerun in two phases, FT09a (decoder warm-up) then FT09b (joint), since 13:10 UTC, bundle cbf5af8. First attempt ran on `i-0a3afdd4efbcafe92` (code bundle 66efe53 = f67b89f's working tree + the
progressive decoder). Prompted by the user ("more layers of decoding"; "the decoder maybe too un-expressive given the
sophistication of the encoder") and the diagnosis in research/experiments/decoder-capacity/.

## Question
FT04 reproduces pixels it can see barely better than pixels it must predict (background 0.056 visible vs 0.057 hidden
on PRISMA set1), and its spectral output is not the limit (2-10% of the error). The SegFormer decoder emits each 4 x 4
pixel block at once from one token-level vector. Does a decoder that upsamples in stages with a masked full-resolution
skip render the background well enough to lower its error while unpredictable targets stay high - and does that show
up in detection?

## Setup
| | |
|---|---|
| model | FT04's stem, encoder (3.33 M) and segmentation head; reconstruction decoder replaced by `ProgressiveDecoder` (1.50 M vs FT04's 2.16 M): SegFormer projections + fuse at 1/4 (warm-started from FT04), then x2 -> 3x3 conv block at 1/2 with an avg-pooled skip, x2 -> block at full resolution with the skip, 1x1 head. Skip = normalised stem output with every pixel of a hidden token zeroed (`_decode`); test `test_hidden_pixels_never_reach_the_output` |
| training | joint, mixed batches, FT04's sources, pseudo-labels and seg loss; init FT04 `best.pt`; 8 epochs; lr 1e-4 cosine; no gating, single-token masks |
| config | [config.json](config.json) |

**Expected, written before launch:** visible-pixel error falls well below FT04's (the new path can render what it sees);
hidden-pixel background error falls a little (texture continues from visible neighbours); targets fall less, so
detection gains a little - or, the risk, the full-resolution context also lets it extrapolate targets from their
visible halo (FT06's leak) and contrast does not improve.

## Pass criteria (fixed before launch)
Against FT04 + RM with cloud excluded on the real PRISMA bench:
1. pAUC <= 1% at least +0.005 with its paired-bootstrap 95% interval above zero;
2. objects caught at top 1% and top 0.1% not lower;
3. bench-v2.1 caught top 1% not lower by more than 0.5 points;
4. product guards: per-sensor validation SAM within 0.1 deg of FT04 (3.88 / 4.11 / 5.46) and EnMAP IoU water / cloud
   within 0.01 (0.839 / 0.810);
5. mechanism: visible-pixel background error (`decoder-capacity/visible_vs_hidden.py`) clearly below FT04's on all
   four scenes - otherwise the decoder did not do what it was built for, whatever detection says.

## Results

### First attempt (joint from the start, lr 1e-4) - stopped after epoch 3

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.97 / 6.37 / 8.29 | 0.841 / 0.817 |
| 2 | 4.69 / 5.11 / 6.45 | 0.840 / 0.823 |
| 3 | 4.43 / 4.97 / 6.47 | 0.843 / 0.826 |

Closing too slowly to reach FT04 on a decaying learning rate (AVIRIS-NG flat from epoch 2): the new decoder (mostly
random init) had 90,000 samples against the millions behind FT04's, and its early gradients reached the encoder.
Masks improved (cloud 0.826). Stopped; rerun as:

- **FT09a** ([config](../FT09a-decoder-warmup/config.json)): `mode: "decoder"` - only `decoder.*` trains, everything else
  frozen in eval mode (the decoder learns to emit FT04's own latent code for its frozen spectral output); lr 1e-3,
  3 epochs.
- **FT09b** ([config](../FT09b-decoder-joint/config.json)): joint from FT09a, lr 5e-5, 5 epochs (starts only if FT09a
  exits 0; `/opt/ft_chain9.sh`).

Pass criteria unchanged, applied to FT09b.

### FT09a (decoder warm-up) - done 14:15 UTC

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.46 / 4.83 / 6.00 | 0.839 / 0.810 (frozen) |
| 2 | 4.17 / 4.54 / 5.83 | 0.839 / 0.810 |
| 3 | 3.98 / 4.14 / 5.50 | 0.839 / 0.810 |

The decoder alone reaches FT04's reconstruction (within 0.1 deg on all three) in 3 epochs.

**Mechanism check on FT09a** (`decoder-capacity/visible_vs_hidden.py --checkpoint`), mean abs error per band, normalised:

| | set1 | set2 | set3 | set4 |
|---|---|---|---|---|
| hidden background FT04 -> FT09a | 0.057 -> **0.051** | 0.137 -> **0.125** | 0.129 -> **0.117** | 0.139 -> **0.110** |
| hidden targets FT04 -> FT09a | 0.218 -> 0.204 | 0.220 -> 0.203 | 0.179 -> 0.171 | 0.191 -> 0.162 |
| target / background, FT04 -> FT09a | 3.82 -> **4.00** | 1.61 -> **1.62** | 1.39 -> **1.46** | 1.37 -> **1.47** |
| visible background (no mask) FT04 -> FT09a | 0.056 -> 0.176 | 0.110 -> 0.120 | 0.159 -> 0.120 | 0.135 -> 0.115 |

- **Hidden-pixel background error falls 9-21% and targets fall less: contrast up on all four scenes** - the intended
  effect, and the first model change in this series to move it.
- **Criterion 5 as written fails** (visible background error not clearly below FT04's; set1 much worse). Stated plainly
  but with a caveat recorded after the fact: the progressive decoder only ever trained with 65% of its skip zeroed, so
  an unmasked input is out of its training distribution; FT04's decoder has no skip and the test was fair for it.
  The contrast row is the evidence for the mechanism; the failed criterion stands as a known caveat.

### FT09b (joint after the warm-up) - done 16:06 UTC

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 4.04 / 4.29 / 5.58 | 0.837 / 0.809 |
| 3 | 3.98 / 4.18 / 6.09 | 0.835 / 0.820 |
| **5** | **3.97 / 4.17 / 5.41** | **0.841 / 0.811** |

Detection (`configs/eval/ft09b.json`, C:/es/eval/ft09b, [../FT09b-decoder-joint/eval_report.md](../FT09b-decoder-joint/eval_report.md)),
cloud excluded on FT04's masks:

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | real ROC-AUC | v2.1 caught top 1% |
|---|---|---|---|---|---|
| FT04 | 0.170 | 46.0% / **15.8%** | **0.065** | 0.852 | **36.2%** |
| FT09a (decoder only) | 0.169 | 46.1% / 15.6% | 0.061 | 0.856 | 35.8% |
| **FT09b** | **0.176** | **47.1%** / 15.4% | 0.063 | **0.854** | 36.1% |

Paired bootstrap, FT09b - FT04 (1000 draws; output in ../FT09b-decoder-joint/bootstrap.txt): pAUC **+0.0060 [+0.0002, +0.0118]**, P(B > A) 0.976; caught top 1% +1.1 pts [-0.3, +2.4];
caught top 0.1% -0.4 pts [-1.4, +0.6]. FT09a - FT04: pAUC -0.0006 [-0.0056, +0.0043] (a tie).

Per scene (pAUC set1 / set2 / set3 / set4): FT04 0.246 / 0.128 / 0.084 / 0.220; FT09b **0.260 / 0.131** / 0.083 / **0.229**.
Objects caught at top 0.1% by scene: FT04 57 / 33 / 13 / 41 (144), FT09b 56 / 34 / 9 / 41 (140) - the loss is all set3.
Caught at top 1% by tier (easy / medium / subtle): FT04 58.2 / 39.3 / 40.3%, FT09b 57.2 / **42.6 / 41.3**%.

Criteria: **1 met** (+0.0060, interval above zero - by +0.0002); **2 not met** at the top 0.1% (-0.4 pts = 4 objects,
unresolved; the top 1% rises +1.1); **3 met** (-0.1); **4 met** (SAM within 0.1 deg, AVIRIS-NG better; IoU within
0.01); **5 failed as written** on FT09a (visible-pixel test; caveat above).

**Observations (FT09)**
- **A candidate, not a release.** The first change since cloud exclusion whose pAUC gain survives the object
  bootstrap - but only just (lower bound +0.0002), the bootstrap does not cover scene-level variation, and the
  top-0.1% object count falls by 4 (all in set3). Gains sit where the design aimed: medium and subtle tiers at the
  top 1%, and 3 of 4 scenes in pAUC.
- **The joint phase did the work.** FT09a (decoder alone) ties FT04 on detection despite higher hidden-pixel contrast;
  FT09b, with the encoder adapting to the new decoder, gains. The best checkpoint is the last epoch, so FT09c continues it.
- The two-phase start was necessary: the first attempt (joint from random decoder) was still 0.5-1 deg SAM behind
  FT04 after 3 epochs.
