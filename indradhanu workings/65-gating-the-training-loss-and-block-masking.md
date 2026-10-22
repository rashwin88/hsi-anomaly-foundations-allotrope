# 65 - Gating the training loss, and block masking: FT07 and FT08

> **The one thing this part teaches:** FT07 left each batch's worst-reconstructed 1% of pixels (plus a
> 2 px halo) out of the training loss, and FT08 also hid tokens in 2 x 2 blocks so a target and its
> halo vanish together - both cost a small but real amount of pAUC, because the worst-reconstructed
> pixels in training are mostly hard *normal* pixels rather than targets, and blocks rebuild ordinary
> texture less exactly; since FT04, every change except cloud exclusion has been inside bench noise or
> a small cost.

## Where we are

**Status: FT07 and FT08 both run 2026-10-21, both failed criterion 1, neither adopted.** Indradhanu
V2 (FT04, part 61) is unchanged. These are the first runs judged by part 64's rule.

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-finetune/FT07-gated/README.md` | the question, design, pass criteria, results and observations for **both** runs |
| `research/experiments/indradhanu-finetune/FT07-gated/finetune_metrics.csv`, `eval_report.md` | FT07's per-epoch validation and full evaluation |
| `research/experiments/indradhanu-finetune/FT08-gated-blocks/` | FT08's `config.json`, `finetune_metrics.csv`, `eval_report.md` |
| `app/foundation_models/trainers/finetune_trainer.py` | `gate_pixels`, its use in `_mixed_loss`, and `_build_prediction_mask` with `mask_block` |
| `app/foundation_models/inferencers/spectral_stem_inferencer.py` | `_checkerboard_keep_mask` - inference cells that match training blocks |
| `app/models/training/training_config.py` | `gate_pct`, `gate_halo`, `mask_block` on `SpectralStemFinetuneConfig` |
| `docs/lld/finetune.md` | the trainer LLD, with both settings |

> **Analogy.** A piano student is told: "the bars you play worst, skip them in practice, so you never
> get used to the strange passage in bar 40". The idea is that the strange passage stays strange. But
> the bars the student plays worst are mostly the merely difficult ones - fast runs, awkward
> fingerings - not the strange passage. Skip those and the student gets a little worse at difficult
> music in general, and the strange passage, which was only one bar among many, is barely affected.

## The question

A masked autoencoder fine-tuned on shards that contain real targets can learn to reconstruct them,
and a target it reconstructs well leaves no residual to score (part 06). GT-HAD's gating keeps likely
anomalies out of what the network learns from. The anomaly course's part 25 tried that per scene, at
inference time, and found a short run never learns the targets anyway, so the gate changed nothing.
Five epochs over the whole training mix is different: it sees every target-like pixel many times. So
two runs, both starting from FT04's `best.pt`, 5 epochs, learning rate 5e-5 on a cosine schedule,
FT04's config otherwise (joint, mixed batches, segmentation loss on its pseudo-labels):

| run | change | config fields |
|---|---|---|
| FT07 | gated loss | `gate_pct: 1.0`, `gate_halo: 2` |
| FT08 | gated loss + 2 x 2-token block masking | as FT07, plus `mask_block: 2` |

**Expected, written before launch (README):** FT07 a small gain at the top 0.1% and slightly worse
reconstruction SAM; FT08 the larger detection change, either way, and worse SAM.

## Mechanism 1 - `gate_pixels`

`_mixed_loss` (part 52's mixed batches) handles each source group - each sensor's tiles in the batch -
separately. With `gate_pct` set, it shrinks that group's `loss_mask` before the L1 and SAM are taken:

```python
loss_mask = loss_mask * ~gate_pixels(x_hat.detach().float(), pixels, band_valid, loss_mask,
                                     cfg.gate_pct, cfg.gate_halo)
```

`gate_pixels` does four things:

```
err   per-pixel L1, mean over valid bands                        # (B, 1, H, W)
cut   torch.quantile(err over loss pixels, 1 - pct / 100)        # the group's top pct % threshold
gate  (err >= cut) & loss pixel
gate  max_pool2d(gate, 2 * halo + 1, stride 1, padding halo)     # dilate: a (2 x halo + 1)-wide square
```

Three details worth noticing:

- `x_hat` is **detached**: the gate is a selection, chosen without gradient. The model cannot learn
  to game which pixels get gated.
- The quantile is taken over **loss pixels only** - hidden, valid pixels - since only they carry loss.
- The quantile reads at most the first 1,000,000 loss pixels (`[:1_000_000]`), which keeps its input
  bounded (`torch.quantile` rejects very large tensors). Eight 128 px tiles are
  `8 x 128 x 128 = 131,072` pixels, so at FT07's batch size of 8 the cap is never reached.

`training_config.py` rejects `gate_pct` without `mixed_batches` ("gate_pct is implemented for
mixed_batches").

### By hand: 100 loss pixels, top 1%

Do this on paper. A group has 100 loss pixels laid out as a 10 x 10 patch, all hidden. Their errors
are distinct.

```
quantile level          1 - 1 / 100 = 0.99
position in the sort    0.99 x (100 - 1) = 98.01        (0-based, torch interpolates linearly)
cut                     between the 99th and 100th smallest errors, 1% of the way up
err >= cut              only the 100th smallest - the single largest error
gated before halo       1 pixel = 1% of 100
```

Now the halo, `gate_halo = 2`. `max_pool2d` with kernel `2 x 2 + 1 = 5` turns one pixel into a
5 x 5 square:

```
gated pixel in the middle of the patch     5 x 5 = 25 pixels    loss pixels left 100 - 25 = 75
gated pixel in a corner                    3 x 3 =  9 pixels    loss pixels left 100 -  9 = 91
```

**A 1% gate removed a quarter of this group's loss.** On a real tile the share is smaller, because
the high-error pixels come in clumps whose squares overlap, and because some of the square lands on
visible pixels that were not in the loss anyway. How large it was in FT07 is not recorded: the run
logged no gated share.

## Mechanism 2 - block masking, `mask_block`

**Why blocks.** Part 63 found the uncertainty head could see a target "through the neighbours": the
encoder hides whole 4 x 4-pixel tokens, and a target's PSF halo (anomaly course, part 20) reaches
into the next tokens, which are often visible. A model asked to fill a hidden token can then copy the
odd spectrum from next door. Hide bigger units and less of the target is left in view.

Picture a 1 px target whose light reaches 2 px either side - a footprint `2 + 1 + 2 = 5` px wide.
Count where it can sit on a fixed grid (this is illustrative arithmetic, not a number from the runs):

```
4 px token     a 5 px footprint never fits inside one token          0 of 4 offsets
8 px block     it fits if it starts 0, 1, 2 or 3 px into the block   4 of 8 offsets = 1/2 per axis
               both axes                                             1/2 x 1/2 = 1/4
```

So a 2 x 2-token block contains the whole footprint about a quarter of the time; otherwise it
straddles a block border and leaks unless the neighbouring block is hidden too.

**How `_build_prediction_mask` does it** (`finetune_trainer.py`, with `mask_block = b > 1`):

```
tokens   valid-token mask                                     # (B, N), N = Ht x Wt
grid     tokens.reshape(B, 1, Ht, Wt)
blocks   max_pool2d(grid, b, ceil_mode=True) > 0              # (B, 1, Ht/b, Wt/b): a block is valid if any token is
draw     generate_prediction_mask(blocks, mask_ratio)         # mask_ratio of the VALID blocks
expand   interpolate(scale_factor=b, mode="nearest")          # each drawn block back to its b x b tokens
pred     expanded[..., :Ht, :Wt] * tokens                     # invalid tokens inside a block drop out
```

### By hand: a 128 px tile at ratio 0.65

```
tokens per side       128 / 4 = 32           tokens   32 x 32 = 1,024
blocks per side       32 / 2  = 16           blocks   16 x 16 =   256
blocks to hide        256 x 0.65 = 166.4     .long() truncates  ->  166
tokens hidden         166 x 4 = 664          share    664 / 1,024 = 0.648
```

The same tile with single tokens: `1,024 x 0.65 = 665.6`, truncated to 665. So FT08 hides almost
exactly as much as FT04 did; it only hides it in bigger pieces.

### Inference cells match training

Part 29's inference fills the scene in two complementary checkerboard passes, each pixel taken from
the pass in which it was hidden. A model trained to fill 8 px holes but asked to fill 4 px ones sees
more context than it ever trained on - a train-test mismatch. So `SpectralStemInferencer` overrides
`_checkerboard_keep_mask`: it reads `mask_block` from the checkpoint's model config and, if it is
larger than the inference config's `checkerboard_cell_size`, builds the checkerboard with cells of
`mask_block` tokens.

```
32 x 32 tokens, cell 2     16 x 16 cells     each pass hides 128 cells = 512 tokens = 50%
```

**Source and description disagree, and the source wins.** `InferenceConfig.checkerboard_cell_size`
(`app/models/training/inference_config.py`) is described as the "Size of each checkerboard cell in
pixels". The code passes it to `checkerboard_token_mask` as `cell_size`, which that function documents
as "checkerboard cell size in tokens", and the new override compares it directly with `mask_block`,
also in tokens. It is a token count.

### What validation measures

Both runs' per-epoch validation is **ungated and uses single-token masks**, so its numbers compare
with FT04's. You can check it: `compute_validation_loss` in `spectral_stem_mae_trainer.py` builds two
complementary random token masks itself, without `_build_prediction_mask` and without `gate_pixels`.
For FT08 that means validation SAM is measured under a different masking from its training and its
inference.

## Pass criteria (fixed before launch)

Against FT04 + RM with cloud excluded on the real PRISMA bench:

1. pAUC <= 1% at least +0.005 **and its paired-bootstrap 95% interval above zero** (part 64);
2. objects caught at the top 1% and top 0.1% not lower;
3. bench-v2.1 caught top 1% not lower by more than 0.5 points;
4. product guards: per-sensor validation SAM within 0.1 deg of FT04 (3.88 / 4.11 / 5.46 for EnMAP /
   PRISMA / AVIRIS-NG) and EnMAP IoU water / cloud within 0.01 (0.839 / 0.810).

## FT07 results

Validation, from `finetune_metrics.csv` (best.pt = epoch 4, validation score 0.05604; FT04 0.05578):

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (the bar) | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 3.93 / 4.25 / 5.60 | 0.841 / 0.810 |
| 3 | 3.94 / 4.14 / 5.48 | 0.835 / 0.810 |
| **4** | **3.90 / 4.16 / 5.47** | **0.840 / 0.813** |
| 5 | 3.91 / 4.17 / 5.59 | 0.838 / 0.815 |

Detection, `configs/eval/ft07.json`, cloud excluded on FT04's masks:

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | v2.1 caught top 1% |
|---|---|---|---|---|
| FT04 | **0.170** | 46.0% / 15.8% | 0.065 | 36.2% |
| FT07 | 0.165 | 46.1% / 15.4% | 0.062 | 36.4% |
| FT07 epoch 1 | 0.169 | 46.3% / 15.5% | 0.063 | - |

Paired bootstrap, FT07 - FT04 (1000 draws):

| measure | B - A [95% interval] | verdict |
|---|---|---|
| pAUC <= 1% | **-0.0047 [-0.0082, -0.0020]** | a real loss, small |
| caught top 0.1% | -0.4 pts [-1.2, +0.2] | tie |
| caught top 1% | +0.1 pts [-0.7, +0.9] | tie |

Criterion 1 failed (a real loss); 2 a tie; 3 met; 4 met at epoch 4. **Not adopted.** The cost grew with
training: epoch 1 was 0.169 (-0.001), epoch 4 0.165 (-0.005).

## FT08 results

Validation, from `FT08-gated-blocks/finetune_metrics.csv` (single-token masks; best.pt = epoch 4,
validation score 0.05559 - below FT04's 0.05578):

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (the bar) | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 3 | 3.95 / 4.28 / 5.78 | 0.840 / 0.798 |
| **4** | **3.92 / 4.17 / 5.33** | **0.841 / 0.807** |
| 5 | 3.92 / 4.19 / 5.60 | 0.837 / 0.808 |

Detection, `configs/eval/ft08.json`, cloud excluded on FT04's masks, FT08 inferring with 2 x 2-token
checkerboard cells:

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | real ROC-AUC | v2.1 caught top 1% |
|---|---|---|---|---|---|
| FT04 | **0.170** | 46.0% / 15.8% | 0.065 | 0.852 | 36.2% |
| FT08 | 0.164 | 46.4% / 16.1% | 0.062 | 0.844 | 36.7% |

| measure | FT08 - FT04 [95% interval] | verdict |
|---|---|---|
| pAUC <= 1% | **-0.0058 [-0.0106, -0.0010]** | a real loss, small |
| caught top 0.1% | +0.2 pts [-1.0, +1.3] | tie |
| caught top 1% | +0.4 pts [-1.1, +2.0] | tie |

Criterion 1 failed; 2 met on point values but unresolved; 3 met (+0.5); 4 met at epoch 4 (AVIRIS-NG SAM
even better than FT04's). **Not adopted.**

## Why both failed

**Gating hit the wrong pixels.** Real targets are under 0.1% of pixels, and many of them are not the
worst-reconstructed pixels in a batch. The top 1% of training error is mostly hard *normal* pixels:
field edges, texture, glint, shoreline (README). Gating those leaves the model slightly worse exactly
where residual Mahalanobis's false alarms come from - more background pixels with large residuals, so
targets are outranked more often - and does little to the targets. That is a pAUC cost with objects
caught flat, which is what the tables show. Reconstruction and masks held (criterion 4), so the cost is
specific to detection, not a damaged model.

**Blocks moved objects one way and pixels the other.** Objects caught went up a little for FT08 (+0.2 /
+0.4 points, unresolved), while pAUC fell 0.006 (resolved) and ROC-AUC 0.008 (not bootstrapped). Hiding 8 px
blocks makes the model reconstruct each hidden pixel from further away, so ordinary texture inside a
block is rebuilt less exactly - more background residual everywhere - while the targets gain little
extra contrast.

**The sentence to remember: making the training ignore what it gets wrong does not make it ignore
targets, because most of what it gets wrong is ordinary hard background.**

One more caution from the README: the epoch-to-epoch swing in validation SAM (FT08's AVIRIS-NG 5.78 at
epoch 3, 5.33 at epoch 4, 5.60 at epoch 5) is as large as the 0.1 deg guard itself. Per-epoch numbers on
these test splits are noisy.

## The cumulative lesson since FT04

| change tried on FT04 | taught in | pAUC change [95% interval] | verdict |
|---|---|---|---|
| cloud exclusion | parts 57, 61, 64 | **+0.0074 [+0.0047, +0.0101]** | real gain, adopted |
| score-side ideas A1-A3 | anomaly course part 24 | not rerunnable (cache keys) | none beat RM on real objects; RM stays |
| FT06 residual / b | part 63 | 0.170 -> 0.125 (not bootstrapped) | large loss |
| FT06 residual / sqrt b | parts 63, 64 | -0.0067 [-0.0174, +0.0037] | tie |
| gated scene adaptation, N = 50 | anomaly course part 25, part 64 | -0.0091 [-0.0176, -0.0028] | real loss |
| FT07 gated loss | this part | -0.0047 [-0.0082, -0.0020] | real loss, small |
| FT08 gated loss + blocks | this part | -0.0058 [-0.0106, -0.0010] | real loss, small |

Read down the last column. **Since FT04, every change except cloud exclusion has been inside the
bench's noise or a small, consistent cost.** Training-time changes, like the score-side and adaptation
ones before them, move FT04's detection by less than the 4-scene, 909-object bench can resolve - except
that small pAUC cost. That is itself a finding: the next gain will have to be bigger than about 0.01
pAUC to be seen at all on this bench, or the bench has to grow.

## Common confusions

**"Top 1% gated means 1% of the loss is gone."**
The halo multiplies it. One isolated pixel with `gate_halo = 2` becomes a 5 x 5 square, 25 pixels.
Clumps overlap and cost less, but the share can be many times 1%.

**"This is the same dilation as the anomaly course's scene adaptation."**
No. Scene adaptation uses `binary_dilation`, a cross that grows one pixel into 13 over two passes.
`gate_pixels` uses `max_pool2d`, a square: 25.

**"FT08 hides more of the tile than FT04."**
It hides almost the same share - 664 against 665 tokens on a 128 px tile - in bigger pieces.

**"FT08's validation score beat FT04's, so FT08 reconstructs better."**
Validation uses single-token masks, not FT08's blocks, and SAM swings by about the size of the guard
from epoch to epoch. Detection is the criterion that counts, and it lost.

**"`checkerboard_cell_size` is in pixels - the config says so."**
The description said pixels; the code treats it as tokens. The source wins - and the description was corrected to
"stage-1 tokens (4 px each)" on 2026-10-21, after this part found it.

**"Objects caught went up for FT08, so it is a partial win."**
The intervals cross zero: under part 64's rule that is a tie, not a gain.

## Check yourself

1. A group has 200 loss pixels and `gate_pct = 1`. Before the halo, how many are gated? With
   `gate_halo = 2`, at most how many loss pixels does that remove, if both gated pixels are far apart
   and away from any edge?
2. With `gate_halo = 1`, how wide is the square, and how many pixels does one gated pixel become?
3. A 256 px tile with `mask_block = 2` and ratio 0.65: how many blocks, how many are hidden, how many
   tokens is that?
4. Why is `x_hat` detached before `gate_pixels`?
5. Why does FT08's inference use 2 x 2-token checkerboard cells, and where is that decided?
6. FT07's pAUC point value was 0.165 against 0.170. Why is that a "real loss" and not a tie?

<details>
<summary>Answers</summary>

1. `200 x 0.01 = 2` pixels (the 0.99 quantile sits between the 2nd- and 3rd-largest errors, so the two
   largest are at or above it). Each becomes a 5 x 5 square, `25 + 25 = 50` pixels, so at most 50 of
   200 loss pixels are removed: 25%.
2. Kernel `2 x 1 + 1 = 3`: a 3 x 3 square, 9 pixels.
3. Tokens per side `256 / 4 = 64`, blocks per side `64 / 2 = 32`, blocks `32 x 32 = 1,024`. Hidden
   `1,024 x 0.65 = 665.6`, truncated to 665 blocks. Tokens `665 x 4 = 2,660` of `64 x 64 = 4,096`.
4. The gate is a choice of which pixels to leave out, not something to learn. Detaching means no
   gradient flows through the selection, so the model cannot shape its errors to change what is gated.
5. A model trained to fill 8 px holes should be asked to fill 8 px holes; 4 px ones give it more
   context than it trained with. `SpectralStemInferencer._checkerboard_keep_mask` reads `mask_block`
   from the checkpoint's model config and uses it as the cell size when it is larger than
   `checkerboard_cell_size`.
6. Its paired-bootstrap interval, [-0.0082, -0.0020], lies wholly below zero (part 64's rule).

</details>

---

Indradhanu V2 (part 61) stands; the gains tried since have all been smaller than this bench can see,
apart from cloud exclusion. Every change so far altered the loss, the masking or the score, never the
shape of the network. Next: part 66 asks whether the decoder itself is the bottleneck.
