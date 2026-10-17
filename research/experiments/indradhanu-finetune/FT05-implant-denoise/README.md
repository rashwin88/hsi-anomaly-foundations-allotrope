# FT05 - implant-denoising: learn to reconstruct the background under an anomaly

**Status:** launched 2026-10-17 on the user's go ("Go for it"). Design: docs/proposals/detection-supervised-finetuning.md.

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
## Observations
