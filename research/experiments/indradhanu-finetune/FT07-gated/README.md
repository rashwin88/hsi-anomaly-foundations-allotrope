# FT07 / FT08 - gated training loss, and block masking

**Status:** FT07 done 2026-10-21 17:36 UTC, exit 0, best.pt = epoch 4 - **failed criterion 1 (a small, significant pAUC loss); not adopted.** FT08 done 19:33 UTC, best.pt = epoch 4 - **also failed criterion 1; not adopted.** Both on `i-0a3afdd4efbcafe92`; FT08 queued behind it
(`/opt/ft_chain.sh`). Code bundle 242d950 (faa96a0's working tree + gating and block masks). Launched on the
user's "Go ahead please and don't stop for my approval." FT08's config is in [../FT08-gated-blocks/](../FT08-gated-blocks/config.json).

## Question
A masked autoencoder fine-tuned on shards that contain real targets learns to reconstruct them - and a target it
reconstructs well leaves no residual to score. GT-HAD's gating keeps likely anomalies out of what the network learns
from. Scene adaptation showed a short run never learns the targets (gated = ungated); 5 epochs over the whole mix
sees every target-like pixel many times. Two runs from FT04:

- **FT07, gated loss:** in each source group of each batch, the hidden pixels with the top 1% reconstruction error,
  dilated 2 px, are left out of the reconstruction loss (`gate_pct`, `gate_halo`; `finetune_trainer.gate_pixels`).
- **FT08, gated loss + block masking:** as FT07, and MAE masks drawn in 2 x 2-token (8 px) blocks so a target and its
  halo are hidden together - FT06 showed a target leaks through the neighbouring tokens of a 4 px one (`mask_block`;
  inference uses the same block as its checkerboard cell, `SpectralStemInferencer._checkerboard_keep_mask`).

Both: FT04's config otherwise (joint, mixed batches, segmentation loss with its pseudo-labels), init FT04 `best.pt`,
5 epochs, lr 5e-5 cosine. Validation is ungated and uses single-token masks, so its numbers compare with FT04's.

**Expected, written before launch:** FT07 a small gain at the top 0.1% (targets no longer rebuilt) and slightly worse
reconstruction SAM (hard-normal pixels also gated); FT08 the larger detection change, either way, and worse SAM.

## Pass criteria (fixed before launch)
Against FT04 + RM with cloud excluded on the real PRISMA bench:
1. pAUC <= 1% at least +0.005 **and its paired-bootstrap 95% interval above zero** (research/experiments/bench-resolution/);
2. objects caught at top 1% and top 0.1% not lower;
3. bench-v2.1 caught top 1% not lower by more than 0.5 points;
4. product guards: per-sensor validation SAM within 0.1 deg of FT04 (3.88 / 4.11 / 5.46) and EnMAP IoU water / cloud
   within 0.01 (0.839 / 0.810).

## Results

### FT07 (gated loss)

Validation per epoch ([finetune_metrics.csv](finetune_metrics.csv)); best.pt = epoch 4 (val 0.05604; FT04 0.05578):

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (the bar) | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 1 | 3.93 / 4.25 / 5.60 | 0.841 / 0.810 |
| 3 | 3.94 / 4.14 / 5.48 | 0.835 / 0.810 |
| **4** | **3.90 / 4.16 / 5.47** | **0.840 / 0.813** |
| 5 | 3.91 / 4.17 / 5.59 | 0.838 / 0.815 |

Detection (`configs/eval/ft07.json`, C:/es/eval/ft07, [eval_report.md](eval_report.md)), cloud excluded on FT04's masks:

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | v2.1 caught top 1% |
|---|---|---|---|---|
| FT04 | **0.170** | 46.0% / 15.8% | 0.065 | 36.2% |
| FT07 | 0.165 | 46.1% / 15.4% | 0.062 | 36.4% |
| FT07 epoch 1 | 0.169 | 46.3% / 15.5% | 0.063 | - |

Paired bootstrap, FT07 - FT04 (1000 draws): pAUC **-0.0047 [-0.0082, -0.0020]**; caught top 0.1% -0.4 pts [-1.2, +0.2];
caught top 1% +0.1 pts [-0.7, +0.9].

Criteria: 1 failed (a real loss, small); 2 tie (top 1% +0.1, top 0.1% -0.4, neither resolved); 3 met; 4 met at
epoch 4 (SAM within 0.1 deg on all three sensors, IoU within 0.01).

**Observations (FT07)**
- **Gating the training loss does not help detection; it costs a little pAUC.** Expected a small top-0.1% gain; got a
  tie on objects and a resolved -0.005 on pAUC. The cost grows with training (epoch 1 -0.001, epoch 4 -0.005).
- Likely reason: the top 1% of reconstruction error in a training batch is mostly hard *normal* pixels (field edges,
  texture, glint, shoreline) - real targets are < 0.1% of pixels and many are not the worst-reconstructed. Gating them
  leaves the model a little worse exactly where RM's false alarms come from, and does little to the targets.
- Reconstruction and masks held (criterion 4), so the cost is detection-specific, not a damaged model.

### FT08 (gated loss + 2 x 2-token block masking)

Validation per epoch ([../FT08-gated-blocks/finetune_metrics.csv](../FT08-gated-blocks/finetune_metrics.csv); validation
uses single-token masks); best.pt = epoch 4 (val 0.05559 - below FT04's 0.05578):

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 (the bar) | 3.88 / 4.11 / 5.46 | 0.839 / 0.810 |
| 3 | 3.95 / 4.28 / 5.78 | 0.840 / 0.798 |
| **4** | **3.92 / 4.17 / 5.33** | **0.841 / 0.807** |
| 5 | 3.92 / 4.19 / 5.60 | 0.837 / 0.808 |

Detection (`configs/eval/ft08.json`, C:/es/eval/ft08, [../FT08-gated-blocks/eval_report.md](../FT08-gated-blocks/eval_report.md)),
cloud excluded on FT04's masks; FT08 infers with 2 x 2-token checkerboard cells:

| | real pAUC | real caught top 1% / 0.1% | real TPR @ 0.1% | real ROC-AUC | v2.1 caught top 1% |
|---|---|---|---|---|---|
| FT04 | **0.170** | 46.0% / 15.8% | 0.065 | 0.852 | 36.2% |
| FT08 | 0.164 | 46.4% / 16.1% | 0.062 | 0.844 | 36.7% |

Paired bootstrap, FT08 - FT04: pAUC **-0.0058 [-0.0106, -0.0010]**; caught top 0.1% +0.2 pts [-1.0, +1.3]; caught top
1% +0.4 pts [-1.1, +2.0].

Criteria: 1 failed (a real pAUC loss); 2 met on points (both object counts slightly up) but unresolved; 3 met
(+0.5); 4 met at epoch 4 (SAM within 0.1 deg - AVIRIS-NG better than FT04 - and IoU within 0.01).

**Observations (FT08)**
- **Block masking moved objects the right way and pixels the wrong way, both within or near noise.** Objects caught
  +0.2 / +0.4 points (not resolved); pAUC -0.006 and ROC-AUC -0.008 (resolved). Hiding 8 px blocks makes the model
  reconstruct from further away, so normal texture inside a block is rebuilt less exactly - more background residual
  - while targets gain little extra contrast.
- FT07 and FT08 together: **training-time gating does not help detection on this bench.** Training-time changes,
  like the score-side and adaptation ones before them, move FT04's detection by less than the bench resolves - except
  a small, consistent pAUC cost.
- The epoch-to-epoch swing in validation SAM (AVIRIS-NG 5.33 -> 5.60 between epochs 4 and 5; 5.78 at epoch 3) is as
  large as the guard itself; per-epoch numbers on these test splits are noisy.
