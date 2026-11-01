# N003 - spectral namer, level 2 generalisation (library augmentation)

**Why:** [N001b](../N001b/README.md) passed level 1 by epoch 8 (man-made recall at ACE's false-alarm rate, fill 0.25:
0.81 / 0.88 / 0.87 against ACE 0.66 / 0.77 / 0.76) but its held-out class accuracy sat at 0.65-0.69 from epoch 5
while the training class loss kept falling (1.03 -> 0.48), and it was the same at fill 1.0 as at 0.25. So the limit
is generalising from ~125 training spectra to unseen ones, not unmixing. User approved the follow-up 2026-10-31.

**Change from N001b** (everything else identical, so the comparison is clean). Config: [config.json](config.json)
(= `configs/namer/N003.json`); code in `app/spectral_match/namer/generator.py` (`warp`, `augment_spectra`,
`mix_within_class`), `model.py` (`dropout`), `train.py` (`lib_aug`).

- **Within-class mixing** of the planted spectrum, p 0.5: w t_i + (1 - w) t_j, j a training spectrum of the same
  class (natural with natural), w ~ U(0.5, 1).
- **Wavelength warp**: read at lambda (1 + s) + delta, s ~ U(+/-1%), delta ~ U(+/-5 nm).
- **Smooth tilt** x (1 + 0.15 curve), the generator's quadratic curve.
- The warp and tilt also on the **library tower's inputs** each training step (`library_side`), never at evaluation.
- **Dropout 0.1** in both towers' MLPs; **weight decay 0.05** (was 0.01).

Held-out spectra and evaluation are unchanged (no augmentation at test), so N003's numbers compare directly with
N001b's and with the ACE baseline in N001b's README.

**Run:** bundle `allotrope-f4f3abd-namer4`, queued on the block-2 H100 to start when the N001b + N002 driver finishes
(`/opt/n3_after.sh`, log `/var/log/namer3.log`); outputs on S3 under
experiments/indradhanu-finetune/spectral-namer/N003/. CPU smoke (laptop, 20 steps) ran; 11 unit tests pass.

## Results

Pending.
