# FT05 - implant-denoising: learn to reconstruct the background under an anomaly

**Status:** done 2026-10-18 14:46 UTC, exit 0; **did not pass** (criterion 1). best.pt = epoch 5 (lowest validation loss). Launched 2026-10-17 on the user's go. Design: docs/proposals/detection-supervised-finetuning.md.

## Question
The detection score is how badly the model reconstructs a pixel from its surroundings. Trained with anomalies in
the input and the clean tile as the target, does the model learn to reconstruct the background beneath them - and
does real detection improve?

**Expected, written before launch:** the real PRISMA bench improves at the top of the ranking (the effect is largest
for objects bigger than one 4 x 4 token, whose own pixels the model can see in neighbouring tokens, and for the PSF
halo); bench-v2 improves more (same generator family - optimistic); reconstruction on clean tiles about unchanged.

## Setup
| | |
|---|---|
| config | [config.json](config.json): FT03's (joint, mixed batches, seg_weight 0.1, SAM 0.5 throughout), init FT03 `best.pt`, 10 epochs; `implant_prob 0.5`, `implant_max_objects 4`, `implant_near_weight 5`, `implant_psf_sigma 1.4`, sources `signature` + `material` (bench-v2's `host` source held out), `implant_seed 1` (bench-v2 used 0) |
| implants | `app/evaluation/implant.implant_tile`, per 128 px training tile, in the loader workers (`finetune_trainer.implant_sample`); same solve as bench-v2 (`solve_realised`: contrast and foreignness paired from one real object, solved on the realised object); placed anywhere valid, water included (vessels in scope) |
| loss | MAE loss against the CLEAN tile on hidden tokens; hidden pixels within 3 px of an implant weighted x5 |
| real objects | `real_v2.json` (909 PRISMA bench objects: size, z, foreignness, halo, signature) staged to /opt/dlami/nvme/ft/ |
| instance | `i-0a3afdd4efbcafe92`, g6e.2xlarge (L40S), kept; training data staged |

## Pass criteria (fixed before launch)
1. **Real PRISMA bench decides:** not below FT03 on pAUC <= 1%, TPR @ 0.1% and caught top 0.1%, and above on at
   least two of the three (FT03, residual Mahalanobis: 0.158 / 0.058 / 14.3%).
2. bench-v2: above FT03 on caught top 1% and pAUC (FT03 32.3%, 0.164) - reported, optimistic, cannot pass the run
   alone; the held-out `host` source must not fall.
3. Per-sensor validation SAM within 0.1 deg of FT03 (3.92 / 4.21 / 5.48) on clean tiles; water / cloud IoU within
   0.02 of FT03 (0.841 / 0.815).

## Results

**Pace:** ~2 h 40 min per epoch (FT03: ~18 min) - the loader workers are CPU-bound on the implant generation
(GPU 0-7%); diagnosed and fixed at the epoch-6 resume - see below. A first launch was restarted after a subspace speed fix (558ff5e).

**Epoch 5** (`configs/eval/ft05_ep5.json`, C:/es/eval/ft05_ep5), residual Mahalanobis:

| | real bench pAUC <= 1% | TPR @ 0.1% | caught top 0.1% | caught top 1% | bench-v2 caught top 1% | v2 pAUC | v2 `host` (held out) |
|---|---|---|---|---|---|---|---|
| FT03 ep10 | 0.158 | 0.058 | 14.3% | 45.4% | 32.3% | 0.164 | 29.4% |
| FT05 ep5 | 0.158 | 0.059 | 14.3% | 44.8% | 32.8% | 0.167 | 29.6% |

With cloud exclusion: real pAUC 0.166 / 0.166, caught top 0.1% 15.8% / 15.6%. Validation SAM 3.94 / 4.20 / 5.33 deg
(FT03 3.92 / 4.21 / 5.48); IoU water / cloud 0.838 / 0.818 (FT03 0.841 / 0.815).

At epoch 5: criterion 1 not met (above FT03 on one of three measures, by 0.001); criterion 2 a small rise, the
held-out source holds; criterion 3 met. Implant-denoising has not moved real detection by mid-run.

**Resumed at epoch 6** (2026-10-18 11:54, f472aa8) with single-thread loader workers, 2 per source: 43 min per epoch
(was 2 h 41 min; the loaders had been ~8x oversubscribed - 18 workers each with its own BLAS thread pool on 8 vCPUs).

| epoch | SAM EnMAP / PRISMA / AVIRIS-NG | IoU water / cloud | validation loss |
|---|---|---|---|
| 5 (best.pt) | 3.94 / 4.20 / 5.33 | 0.838 / 0.818 | 0.0559 |
| 10 (last.pt) | 3.90 / 4.20 / **5.67** | 0.844 / 0.814 | 0.0570 |

Epoch 10 fails criterion 3 on AVIRIS-NG (+0.19 deg over FT03's 5.48).

**Verdict: not passed.** Implant-denoising did not lift real detection by mid-run (criterion 1: above FT03 on one of
three measures, by 0.001) and later epochs cost AVIRIS-NG reconstruction. **Caveat:** half the training implants were
`signature` implants, whose spectra were garbled by a descending-wavelength np.interp bug (fixed in b94dbfd); they
still matched real contrast and foreignness. A rerun with correct signatures would be the fair test; nothing here
suggests it would flip the result.

## Observations
