# FTH2 / FTB2 - Heavy and the control after 18 joint epochs: Heavy is the better detector

**Status:** done 2026-10-29 (FTH2 05:35-07:26 UTC, FTB2 07:26-09:12) on the H100 capacity block. **Heavy beats the
same-recipe control and shipped FT04 on pAUC and objects caught at the top 1%, at all three neighbouring checkpoints,
every interval above zero.** FTH3 (+6 epochs) running after.

FTH2 = FTH1 + 8 joint epochs (lr 5e-5 cosine; [config](config.json)); FTB2 = FTB1 + the same
([config](../FTB2-base-joint-more/config.json)). Cloud exclusion with each model's **own thick-cloud-only** masks
(`ModelSpec.cloud_rule: thick`; FTH1 mask diagnosis: thin cloud fires on haze).

## Validation (epoch 8 = 18 joint epochs)
| | SAM EnMAP / PRISMA / AVIRIS-NG (deg) | IoU water / cloud |
|---|---|---|
| FT04 | 3.88 / **4.11** / 5.46 | **0.839 / 0.810** |
| FTB2 | **3.85** / 4.36 / **5.33** | 0.806 / 0.729 |
| FTH2 | 3.88 / 4.48 / 5.39 | 0.818 / 0.736 |

## Detection (`configs/eval/fth2_ftb2.json`, real PRISMA bench; [eval_report.md](eval_report.md), [bootstrap.txt](bootstrap.txt))
| | pAUC ep 6 / 7 / 8 | caught top 1% | caught top 0.1% |
|---|---|---|---|
| FT04 shipped (cloud or thin) | 0.170 | 46.0% | **15.8%** |
| FTB2 (own thick) | 0.160 / 0.163 / 0.162 | 46.1 / 46.4 / 46.8% | 15.1 / 15.1 / 15.1% |
| **FTH2 (own thick)** | **0.185 / 0.184 / 0.186** | **48.2 / 49.0 / 49.0%** | 15.3 / 15.2 / 15.7% |

Paired bootstrap (1000 draws):
- **FTH2 - FTB2:** pAUC +0.0245 [+0.0140, +0.0352] / +0.0209 [+0.0102, +0.0321] / +0.0244 [+0.0135, +0.0362];
  caught top 1% +2.1 [+0.2, +4.1] / +2.5 [+0.6, +4.3] / +2.2 [+0.3, +4.0].
- **FTH2 - FT04 shipped:** pAUC +0.0151 [+0.0036, +0.0257] / +0.0143 [+0.0032, +0.0247] / +0.0168 [+0.0053, +0.0275];
  caught top 0.1% -0.6 / -0.7 / -0.1 pts (all unresolved - a tie).
- **FT04 thick rule vs shipped rule:** -0.0004 [-0.0015, +0.0015] - switching the product to thick cloud costs V2 nothing.

## Observations
- **The capacity question is answered on this bench: the bigger pretrained model is the better detector** - ~+0.023
  pAUC over the same-recipe control (about three times what cloud exclusion bought, +0.0074), holding at three neighbouring
  checkpoints; +0.015 over shipped FT04, which had a four-stage curriculum.
- The gap grew with fine-tuning (FTH1 vs FTB1 ~+0.013 -> FTH2 vs FTB2 ~+0.023), while reconstruction did not separate
  them - Heavy's PRISMA SAM is still worse than the control's. Better detection without better reconstruction.
- Heavy's masks with the thick-cloud rule are the best of the three for exclusion (FT04 + FTH2 thick masks: 0.174).
- Open: the top 0.1% is a tie everywhere; 4 scenes, object bootstrap only. Product: Heavy as Indradhanu V2 would need
  the thick-cloud rule in `segment_full_scene` and the manifest / resolver path for its architecture (progressive
  decoder, 34 M parameters - worker CPU cost to measure).
