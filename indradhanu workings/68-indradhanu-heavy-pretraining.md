# 68 - Indradhanu Heavy: pretraining a wider model on an H100

> **The one thing this part teaches:** every change since FT04 kept the same small encoder and none
> survived the bench's noise, so the untested question was capacity; Indradhanu Heavy re-pretrained a
> model about six times FT04's size (33.9 M parameters) for 80 epochs on a rented H100, reconstructed
> better than F001 without overfitting, and as a pretrained model ranked targets more broadly but did
> worse at the very top on hazy set4 - and the user made it the base model.

## Where we are

**Status (2026-10-29): F002-heavy done 2026-10-28 22:55 UTC, 80 epochs, `best.pt` = epoch 72. User
decision: "I think heavy is our base model now." The product is unchanged: Indradhanu V2 (FT04, part
61) is opt-in and v0.2.0 is the default. Part 69 teaches the fine-tuning that followed.**

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-se-pretrain/F002-heavy/README.md` | question, setup, GPU log, incidents, results, decision |
| `research/experiments/indradhanu-se-pretrain/F002-heavy/config.json` | the run's configuration |
| `research/experiments/indradhanu-se-pretrain/F002-heavy/heavy_run.sh` | the driver: shard rotation, restarts, S3 sync |
| `research/experiments/indradhanu-se-pretrain/F002-heavy/metrics.csv` | one row per epoch: train and val L1 / SAM, lr, seconds |
| `research/experiments/indradhanu-se-pretrain/F002-heavy/eval_report.md` | pretrained detection against F001 and FT04 |
| `research/experiments/indradhanu-finetune/h100_chain.sh` | what ran after pretraining (part 69) |
| `app/models/training/training_config.py` | `SpectralStemMAEConfig.decoder_type` and `decoder_stage_dims` |

> **Analogy.** A translator who has learned 2,000 words of a language can render most of a novel, but
> every unusual idiom gets flattened into the nearest common phrase. You can coach them on technique
> for months (parts 58 to 67) and the flattening stays, because it is a vocabulary limit. At some point
> the only test left is to hire someone who knows 6,000 words and see whether the idioms survive.

## Why capacity was the untested question

Look back at part 67's cumulative table. Cloud exclusion was a real gain. FT07 and FT08 cost a little.
FT09b gained by a hair and FT09c showed that gain was a lucky checkpoint. Every one of those runs
**kept FT04's encoder at the same size**: widths 32, 64, 160 and 256 channels across its four stages.

Part 66 found the limit those runs could not move: FT04 redraws pixels it can see barely better than
pixels it must guess, and each stage-1 token squeezes 512 numbers into 32. FT09 attacked the decoder
side of that; it did not change how much the encoder can hold. The F002 README states it in one line:
"A wider encoder needs re-pretraining." You cannot widen a trained encoder and keep its weights,
because every weight matrix changes shape.

Define the word first. **Capacity** is how much a model can represent, and for a network of fixed
depth it grows with its **width**: the number of channels each layer carries. More width means more
parameters (the learned numbers, part 08) and more arithmetic per pixel.

## The architecture

`config.json` against F001 (FT04's pretrained parent, part 43), both built by
`SpectralStemSegFormerMAE`:

| setting | F001 / FT04 | Heavy | ratio |
|---|---|---|---|
| `embed_dims` (stage widths) | 32, 64, 160, 256 | 96, 192, 480, 768 | 3x |
| `num_heads` | 2, 2, 5, 8 | 3, 6, 10, 12 | |
| channels per head | 16, 32, 32, 32 | 32, 32, 48, 64 | |
| `compressed_channels` (spectral latent) | 32 | 64 | 2x |
| `token_dim` (wavelength encoding) | 32 | 32 | 1x |
| decoder | SegFormer, `decoder_dim` 256 | progressive (part 67), `decoder_dim` 512, stages 256 / 128 | |
| `reduction_ratios`, `num_blocks`, `mask_ratio` | 8 4 2 1, 2 2 2 2, 0.65 | same | |

**The name and the config disagree, and the config wins.** The README title, the config's
`_description` and the commit all say "4x wide", from the user's launch instruction "Go, but 4x wide".
The encoder widths are 3x FT04's (96 / 32 = 3, 768 / 256 = 3); the spectral latent and the decoder are
2x. Nothing in the model is 4x. Read "4x" as the name of the run, not a measurement.

`token_dim` stays 32 because, in the config's words, "the reduced stem's wavelength encoding is 32
wide". The decoder is FT09's progressive one. Part 67 showed it did not help FT04 on its own; here it
is pretrained from the start, so it never has the warm-start problem of part 67. Since 2026-10-28 the
pretraining config itself carries `decoder_type` and `decoder_stage_dims`
(`app/models/training/training_config.py`, `SpectralStemMAEConfig`); before, only the fine-tuning
config had them.

### By hand: the stage-1 squeeze

Stage 1's patch embedding is a 4 x 4 convolution with stride 4 (part 11; `seg_former_encoder.py`
builds it with kernel 4, not the 7 its docstring example shows). Each token reads a 4 x 4 block of
pixels, each pixel carrying the spectral latent:

```
FT04    16 pixels x 32 channels = 512 numbers in     ->  32 out     512 / 32  = 16 to 1
Heavy   16 pixels x 64 channels = 1,024 numbers in   ->  96 out   1,024 / 96  = 10.7 to 1
```

Per pixel: FT04's token keeps `32 / 16 = 2` numbers for each of its 16 pixels, Heavy's
`96 / 16 = 6`. That is the bottleneck of part 66, three times wider.

The convolution's parameters (`in x out x k x k + out`, part 08), then its LayerNorm (a weight and a
bias per channel):

```
Heavy   64 x 96 x 4 x 4 + 96 = 98,304 + 96 = 98,400     LayerNorm 2 x 96 = 192     total 98,592
FT04    32 x 32 x 4 x 4 + 32 = 16,384 + 32 = 16,416     LayerNorm 2 x 32 =  64     total 16,480
```

Building both models on 2026-10-29 and counting `encoder.patch_embeds[0]` gives 98,592 and 16,480.

### By hand: one encoder block at stage 1

A block (part 15) is LayerNorm, efficient self-attention (part 13), LayerNorm, Mix-FFN (part 14).
At stage 1, Heavy has `C = 96`, reduction ratio 8, expansion 4 (hidden width `4 x 96 = 384`):

```
norm1                       2 x 96                              =       192
q, k, v, proj  4 x (96 x 96 + 96) = 4 x 9,312                    =    37,248
reduction conv  96 x 96 x 8 x 8 + 96 = 589,824 + 96              =   589,920
reduction_norm              2 x 96                              =       192
norm2                       2 x 96                              =       192
fc1             96 x 384 + 384 = 36,864 + 384                    =    37,248
dwconv         384 x 1 x 3 x 3 + 384 = 3,456 + 384               =     3,840
fc2            384 x 96 + 96 = 36,864 + 96                       =    36,960
                                                        one block   705,792
```

Stage 1 has two blocks: `2 x 705,792 = 1,411,584`, which is what the built model reports for
`encoder.blocks[0]`. Notice where the parameters are: the 8 x 8 reduction convolution alone is
`589,920 / 705,792 = 84%` of the block.

**Why 3x width gives about 9x parameters.** Almost every weight here is a `C x C` matrix (or
`C x C x k x k`), so tripling `C` multiplies it by `3 x 3 = 9`. FT04's stage-1 block is 79,616
(you count it in Check yourself), and `705,792 / 79,616 = 8.9`. The biases and norms grow only 3x,
which is why it is a little under 9.

### The whole model

Counted from the built model:

| part | FT04 size (no seg head) | Heavy |
|---|---|---|
| `spectral_in` (stem, part 40) | 14,721 | 17,825 |
| `norm` (BatchNorm) | 64 | 128 |
| encoder | 3,331,072 | 29,580,288 |
| decoder | 2,164,736 (SegFormer) | 4,279,616 (progressive) |
| `spectral_out` | 4,257 | 6,337 |
| **total** | **5,514,850** | **33,884,194** |

That is the README's "33.9 M parameters (encoder 29.6 M, decoder 4.3 M) vs FT04 5.9 M". The 5.9 M
is the shipped FT04 with its segmentation head (5,886,451, part 61); without the head, FT04's size is
5.5 M, and `33,884,194 / 5,514,850 = 6.1`. Heavy is about six times FT04, with nine times its
encoder.

## The capacity block, and what it cost in time

The run did not use the usual L40S. It used an **H100 capacity block**: AWS rents a GPU instance for a
fixed window booked in advance, here a `p5.4xlarge` (one H100 with 80 GB, 16 vCPU) from 2026-10-28
11:30 to 2026-10-29 11:30 UTC. The window starts and ends on the clock whether or not your job is
running, like a hired hall: an hour spent setting up is an hour of the booking gone. A second block
(2026-10-31 to 11-02) is booked.

Block 1, from the READMEs:

| UTC | what happened |
|---|---|
| 10-28 11:30 | block starts |
| 12:38 | launch (68 minutes into the block) |
| 12:48 | first start; crashes at the first validation, three times |
| 13:10 | driver gives up after the third failure; GPU idle |
| 14:05 | restart with the fix; batch 32 x 1 |
| 14:14 | `h100_chain.sh` starts staging fine-tune data alongside |
| 22:55 | 80 epochs done |
| 10-29 01:02 - 10:39 | FTH1, FTB1, FTH2, FTB2, FTH3 (part 69) |
| 11:30 | block ends |

The README's own count: "~2.5 h of block 1 were lost". From 11:30 to 14:05 is 2 h 35 min, about
`2.6 / 24 = 11%` of the booking.

### The crash: 10 test shards, 12 loader workers

Training data arrives in **shards**: tar files of patches, read by `webdataset`. A data **loader
worker** is a separate process that reads and decodes data while the GPU trains; the config asked
for 12 (`num_workers: 12`). Webdataset deals shards out to workers, and a worker with no shard has
nothing to yield.

The validation set was 10 staged test shards. Deal 10 shards to 12 workers and workers 10 and 11
(counting from 0) get none. Webdataset raised "No samples found ... fewer shards than workers", the
trainer died at the first validation, the driver retried twice more and stopped. The fix was
`test_shards_per_size: 16`: 16 shards over 12 workers gives eight workers one shard and four workers
two, and nobody is empty.

Why it went unseen for about 55 minutes, in the README's words: "My epoch watcher only looked for an
epoch-1 row". A watcher that waits for success says nothing when the job fails; it waits. The watcher
was changed to alert "on any trainer exit or the driver stopping".

### The stale log line that stopped the chain

The second trap was in `h100_chain.sh`, the script that waits for pretraining and then runs
everything in part 69. The committed version (commit a0b6930) waited for a sentence in a log file:

```bash
until grep -q "F002-heavy driver finished" /var/log/heavy.log 2>/dev/null; do sleep 120; done
E=$(( $(wc -l < $N/runs/F002-heavy/metrics.csv) - 1 ))
[ $E -ge 80 ] || { log "F002-heavy stopped at $E epochs - chain stops"; exit 1; }
```

`heavy_run.sh` prints "F002-heavy driver finished" whenever its main loop ends, including the
`break` after the three failed starts at 13:10. The working-copy fix explains what went wrong in its comment: "heavy.log is
appended across restarts and keeps old 'finished' lines". So the `grep` found the crashed run's
sentence, the epoch count was below 80, and the chain stopped itself. The fix waits on the thing it
actually needs, the epoch count, and stops only if the driver process is gone:

```bash
# wait on the epoch count, not a log line: heavy.log is appended across restarts and keeps old "finished" lines
epochs() { echo $(( $(wc -l < $N/runs/F002-heavy/metrics.csv 2>/dev/null || echo 1) - 1 )); }
until [ $(epochs) -ge 80 ]; do
  pgrep -f heavy_run.sh >/dev/null || { log "F002-heavy driver not running at $(epochs) epochs - chain stops"; exit 1; }
  sleep 120
done
```

The records do not say when the stop was noticed or how much time it cost. What they do show: the
last epoch ended at 22:55 and FTH1 started at 01:02, a gap of about 2 hours that also holds the FT04
pseudo-labelling step. That fix is in the working tree, not yet committed.

**The lessons, stated plainly.** Watch for failure, not only for success. Wait on the state you need
(80 rows in `metrics.csv`), never on a message that an earlier run may have left behind. And run the
validation path once, briefly, before a booked window starts - that last one is this course's
suggestion; neither README says it.

### Batch size and the GPU

The first start used batch 16 with 2 accumulation steps (part 28); the restart used 32 x 1. The
effective batch is 32 either way, F001's. The GPU log in the README:

| | GPU utilisation | memory | power |
|---|---|---|---|
| 16 x 2 (first start) | 63-85% | 15.9 GB | ~375 W |
| 32 x 1 (restart) | 85-100% | 26.5 GB | ~391-404 W |

With 16 x 2 the README found "86% of the 16 vCPU idle - the GPU, not the loaders, sets the pace; the
headroom is batch size". At 32 x 1 an epoch took 369 seconds, 81.2 patches per second.

## The 80-epoch schedule

F001 ran one 200-epoch cosine schedule (part 43). Heavy runs 80, the config says "cosine fully
decayed", on the user's "we will lose steam after epoch 50-75". A **cosine schedule** lowers the
learning rate (the step size of each update, part 28) along half a cosine wave, from 3e-4 at the start
to `min_lr` 1e-6 at the end:

```
lr(e) = 1e-6 + (3e-4 - 1e-6) x (1 + cos(pi x e / 80)) / 2
```

At `e = 40`, `cos(pi / 2) = 0`, so `lr = 1e-6 + 2.99e-4 x 0.5 = 1.505e-4`; `metrics.csv` epoch 40 logs
`0.00015050`. Setting the schedule to 80 epochs means the rate actually reaches its floor within the
booking, instead of stopping at a point that was planned to be the middle.

The data rotated as in F001: train shards in blocks of 200, but a new block **every 5 epochs** (F001:
every 20), 16 blocks, 3,200 of the 3,419 train shards. Each epoch draws 30,000 patches.

All 80 epochs took 30,690 seconds of training, 8.5 hours, from 14:05 to 22:55 with validation and
start-up. Most blocks' first epoch took about 440 seconds instead of 370 (epochs 6, 11, ... 71 in
`metrics.csv`); fresh shards not yet in the file cache is the likely cause, not something the README
measured.

## The curve: did it overfit?

From `metrics.csv`, validation L1 (normalised reflectance) and SAM (degrees, part 24):

| epoch | train L1 | val L1 | val SAM | lr |
|---|---|---|---|---|
| 1 | 0.0412 | 0.0285 | 8.84 | 3.0e-4 |
| 20 | 0.0224 | 0.0202 | 5.03 | 2.6e-4 |
| 40 | 0.0207 | 0.0176 | 4.43 | 1.5e-4 |
| 50 | 0.0201 | 0.0168 | 3.98 | 9.3e-5 |
| 60 | 0.0196 | 0.0171 | 4.34 | 4.5e-5 |
| **72 (best)** | 0.0192 | **0.0162** | **3.77** | 8.3e-6 |
| 80 | 0.0191 | 0.0164 | 3.89 | 1.0e-6 |

The user's prediction holds in the numbers. From epoch 1 to 20 val L1 fell `0.0285 - 0.0202 = 0.0083`;
from 50 to 72, `0.0168 - 0.0162 = 0.0006`, fourteen times less over a similar span. The last block
(76-80) is flat.

Against F001, from the F002 README: F001's best in 113 epochs was 0.0182 / 4.06 at epoch 100. Heavy
first beats that L1 at epoch 29 (0.0181) and that SAM at epoch 41 (4.02).

**One caveat the README does not make.** F001 validated on test shards 0-9 (its README); Heavy on 16
test shards, because of the crash fix. Both draw 1,500 patches per validation, but not the same 1,500.
The 0.0182 against 0.0164 crosses two validation sets. The gap is large next to epoch-to-epoch wobble,
so the direction is safe; the exact size is not.

**Overfitting** means a model learns its training examples rather than the pattern behind them: train
loss keeps falling while validation loss rises. Two pieces of evidence say Heavy did not.

1. **The train - val gap is steady.** Train L1 sits above val L1 by 0.0023 to 0.0033 from epoch 20 to 80,
   with no trend. (Train above val is expected here: the training loss hides 65% of tokens, reads a
   random band subset in 80% of batches, `band_augment_prob: 0.8`, and runs with dropout 0.4, while
   validation hides 50% twice with complementary masks and no augmentation, `spectral_stem_mae_trainer.py`.
   The README writes the gap as "-0.003 L1", meaning val minus train.)
2. **Nothing is memorised within a block.** Each block's 200 shards are seen for 5 epochs. A model
   memorising them would show train loss falling inside a block and jumping back up on the first epoch
   of the next. Across the 15 block changes the largest jump is `+0.00012` (epoch 20 -> 21), about 0.5%
   of the loss, and train L1 falls inside all 16 blocks. Validation falls inside 13 of the 16.

## Pretrained detection: broader ranking better, the very top worse

`configs/eval/f002_heavy.json`, epoch 72, real PRISMA bench, residual Mahalanobis (part 45), cloud
excluded on FT04's masks (`eval_report.md`):

| | pAUC | caught top 1% / 0.1% | TPR @ 0.1% |
|---|---|---|---|
| F001 (pretrained, FT04 size) | 0.155 | 45.1% / **14.0%** | 0.059 |
| **Heavy (pretrained)** | **0.166** | **47.4%** / 12.4% | 0.058 |
| FT04 (F001 + fine-tuning) | 0.170 | 46.0% / 15.8% | 0.065 |

Paired bootstrap (part 64), Heavy - F001: pAUC **+0.0111 [-0.0003, +0.0223]**, P 0.971 - the interval
touches zero, so by part 64's rule it is not resolved, though by a hair. Caught at the top 0.1%:
**-1.5 points [-3.0, -0.2]**, a resolved loss.

Per scene, pAUC: Heavy 0.259 / 0.150 / 0.071 / 0.185 against F001 0.220 / 0.126 / 0.080 / 0.195.
Check the bench value by hand:

```
Heavy   0.259 + 0.150 + 0.071 + 0.185 = 0.665     / 4 = 0.166
F001    0.220 + 0.126 + 0.080 + 0.195 = 0.621     / 4 = 0.155
```

Objects caught at the top 0.1%, per scene: Heavy 53 / 23 / 12 / 25, F001 52 / 27 / 11 / 37.

```
Heavy   53 + 23 + 12 + 25 = 113      113 / 909 = 12.4%
F001    52 + 27 + 11 + 37 = 127      127 / 909 = 14.0%
loss    14 objects, of which set4: 37 - 25 = 12
```

**Twelve of the fourteen lost objects are on set4**, the hazy scene (the 2026-10-18 correction noted in part 55: its
"cloud" is haze and a cloud / smoke streak). The README's reading: on set4 "Heavy's highest scores land on something
other than the labelled targets". Meanwhile caught at the top 1% rises on every tier (easy / medium /
subtle F001 55.9 / 38.7 / 40.7% -> Heavy 58.2 / 40.7 / 43.3%), above even fine-tuned FT04's 46.0%
overall.

How far does pretraining alone get? FT04's lead over F001 is `0.170 - 0.155 = 0.015`; Heavy recovers
`0.166 - 0.155 = 0.011` of it, `0.011 / 0.015 = 73%`. The README rounds this to "~75%".

## The decision

The README, 2026-10-29: "**Indradhanu Heavy is the base pretrained model** (user: 'I think heavy is our
base model now'). New fine-tunes and heads start from this checkpoint (`checkpoints/best.pt`, epoch 72),
not F001."

Be clear what the evidence was at that moment: better reconstruction, a near-resolved pAUC gain as a
pretrained model, and a resolved loss at the top 0.1%. It was a judgement on the better general
representation, not a detection win - that came later (part 69). The same paragraph says "Its one-run
fine-tune (FTH1) did not beat the same-recipe control yet"; it was written before FTH1's detection was
evaluated, and part 69 shows that sentence was overtaken the same morning.

## Common confusions

**"Heavy is 4x wide."**
That is its launch name. The encoder widths are 3x FT04's, the spectral latent and decoder 2x. The
parameter count is 6.1x overall and 8.9x for the encoder.

**"Six times the parameters should mean six times slower."**
Not on a GPU with headroom. Heavy ran at 81 patches per second on an H100; time per epoch depends on
the hardware and on whether the GPU is kept busy (63-85% at 16 x 2, ~100% at 32 x 1). The worker's CPU
cost in the product has not been measured; part 69 lists it as an open item.

**"Lower validation loss than F001 means Heavy is the better detector."**
Part 66 already showed reconstruction and detection can come apart. Pretrained Heavy reconstructs
better and catches fewer objects at the top 0.1%. Part 69 sharpens this into a rule.

**"Train loss above val loss means something is broken."**
Training is the harder task here: more hidden tokens, missing bands, dropout. Watch the gap's trend,
not its sign.

**"The 2.5 lost hours were the shard bug."**
Partly. From the timeline: 68 minutes before launch, 10 minutes of start-up, 22 minutes in three
failed starts, and 55 minutes idle because the watcher waited for success - 155 minutes in all. The stale-log stop came later and its cost is not recorded.

## Check yourself

1. Count the parameters of FT04's stage-1 encoder block (`C = 32`, reduction ratio 8, expansion 4),
   line by line as above. Then the ratio to Heavy's 705,792.
2. Heavy's stage-2 patch embedding is a 3 x 3 convolution, stride 2, from 96 to 192 channels, plus
   LayerNorm. Count it.
3. Using the cosine formula, what learning rate does epoch 60 log? (`cos(3 pi / 4) = -0.7071`.)
4. With 16 test shards and 12 workers, how many shards does each worker get? What is the largest
   number of workers you could use with 16 shards without the crash?
5. Pretrained Heavy caught 113 of 909 objects at the top 0.1% and F001 127. Why does a +0.011 pAUC
   gain sit beside this loss without contradiction?
6. Give the two pieces of evidence that Heavy did not overfit, and the one caveat on comparing its
   validation loss with F001's.

<details>
<summary>Answers</summary>

1. ```
   norm1                 2 x 32                          =     64
   q, k, v, proj  4 x (32 x 32 + 32) = 4 x 1,056          =  4,224
   reduction      32 x 32 x 8 x 8 + 32 = 65,536 + 32      = 65,568
   reduction_norm        2 x 32                          =     64
   norm2                 2 x 32                          =     64
   fc1            32 x 128 + 128 = 4,096 + 128            =  4,224
   dwconv        128 x 9 + 128 = 1,152 + 128              =  1,280
   fc2           128 x 32 + 32 = 4,096 + 32               =  4,128
                                                 total     79,616
   ```
   The built model reports `2 x 79,616 = 159,232` for `encoder.blocks[0]`. `705,792 / 79,616 = 8.9`.
2. `96 x 192 x 3 x 3 + 192 = 165,888 + 192 = 166,080`, plus LayerNorm `2 x 192 = 384`: 166,464, which
   is what the built model reports for `encoder.patch_embeds[1]`.
3. `(1 + (-0.7071)) / 2 = 0.2929 / 2 = 0.14645`; `2.99e-4 x 0.14645 = 4.379e-5`; plus 1e-6 is
   `4.479e-5`. `metrics.csv` logs `4.4788e-05`.
4. Dealing 16 shards round-robin to 12 workers: workers 0-3 get two (shards i and i + 12), workers 4-11
   get one. Up to 16 workers works; a 17th would have none.
5. pAUC averages over every target pixel across the first 1% of false alarms, so it rewards many targets
   ranked fairly high. The top-0.1% count only asks about the few highest scores. Heavy ranks more
   targets above most background (top 1% up on every tier) while, on set4, a handful of non-target
   pixels outscore the targets at the very top.
6. The train - val gap stays between 0.0023 and 0.0033 with no trend from epoch 20 to 80; and train loss
   never jumps by more than 0.00012 at a block change, while it falls inside every block. The caveat:
   F001 validated on test shards 0-9 and Heavy on 16 shards, so the two validation sets differ.

</details>

---

Next: part 69 fine-tunes Heavy and puts it against a control trained with the same recipe.
