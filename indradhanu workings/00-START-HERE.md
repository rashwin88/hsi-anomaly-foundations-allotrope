# Indradhanu, from zero

A 30-part course for someone who has **never** seen this codebase, has **never**
worked with satellite imagery, and has **never** built a neural network.

Nothing is assumed. Nothing is skipped. If a part introduces a word, that part
also defines it.

## What Indradhanu is, in one sentence

Indradhanu is the nickname of a computer program called
`hyperspectral_segformer_mae`. It looks at a satellite photograph, tries to
**redraw it from memory**, and then points at the spots where its redrawing came
out wrong. Those spots are the interesting ones.

## An analogy to hold on to for the next 30 parts

Imagine an artist who has spent ten years painting nothing but English
countryside. Fields, hedges, rivers, roads. Thousands of paintings.

Now you cover most of a new photograph with sticky notes and ask them to paint
what is underneath. They will do a superb job on the fields, the hedges and the
river — because they have painted those ten thousand times.

But if there is a **crashed aeroplane** under one of the sticky notes, they will
paint grass there. They have never painted a crashed aeroplane.

Peel the notes off, compare their painting to the photograph, and the place
where they were most wrong is the aeroplane.

That is Indradhanu. The rest of this course is the details.

## How each part is laid out

Every part has the same skeleton:

1. **The one thing this part teaches** — a single sentence.
2. The explanation, in small steps, with real numbers.
3. **Common confusions** — mistakes people actually make.
4. **Check yourself** — a few questions, with the answers underneath.

## The 30 parts

**Background — what and why (01–05)**

| # | File | The one thing it teaches |
|---|---|---|
| 01 | [01-the-problem.md](01-the-problem.md) | Why we cannot just train a classifier |
| 02 | [02-hyperspectral-101.md](02-hyperspectral-101.md) | What the input data actually is |
| 03 | [03-the-165-band-grid.md](03-the-165-band-grid.md) | Where the number 165 comes from |
| 04 | [04-the-vendable.md](04-the-vendable.md) | The exact object the model is handed |
| 05 | [05-where-it-lives.md](05-where-it-lives.md) | Where this sits in the product |

**Foundations — the big idea and the tools (06–08)**

| # | File | The one thing it teaches |
|---|---|---|
| 06 | [06-reconstruct-then-subtract.md](06-reconstruct-then-subtract.md) | Redraw, subtract, and why hiding is essential |
| 07 | [07-shapes-cheatsheet.md](07-shapes-cheatsheet.md) | Every tensor shape, in one place |
| 08 | [08-math-warmup.md](08-math-warmup.md) | Five pieces of maths, worked by hand |

**The machine, piece by piece (09–22)**

| # | File | The one thing it teaches |
|---|---|---|
| 09 | [09-normalisation.md](09-normalisation.md) | Putting every band on the same scale |
| 10 | [10-spectral-compressor.md](10-spectral-compressor.md) | Squeezing 165 numbers into 32 |
| 11 | [11-patch-embedding.md](11-patch-embedding.md) | Chopping the picture into "tokens" |
| 12 | [12-attention.md](12-attention.md) | How tokens consult each other |
| 13 | [13-efficient-self-attention.md](13-efficient-self-attention.md) | Making that consultation affordable |
| 14 | [14-mix-ffn.md](14-mix-ffn.md) | Each token thinking on its own |
| 15 | [15-the-block.md](15-the-block.md) | Wiring those two into one repeatable unit |
| 16 | [16-the-encoder.md](16-the-encoder.md) | Four stages, four zoom levels |
| 17 | [17-masking-1-token-validity.md](17-masking-1-token-validity.md) | Which tokens are real data |
| 18 | [18-masking-2-prediction-targets.md](18-masking-2-prediction-targets.md) | Choosing which ones to hide |
| 19 | [19-masking-3-remove-and-restore.md](19-masking-3-remove-and-restore.md) | Deleting them, then putting them back |
| 20 | [20-mask-erosion.md](20-mask-erosion.md) | Why we discard a border strip |
| 21 | [21-the-decoder.md](21-the-decoder.md) | Rebuilding the full-size picture |
| 22 | [22-full-forward-trace.md](22-full-forward-trace.md) | One patch, start to finish |

**Training and running it (23–30)**

| # | File | The one thing it teaches |
|---|---|---|
| 23 | [23-loss-1-l1.md](23-loss-1-l1.md) | Measuring "how wrong", part 1: brightness |
| 24 | [24-loss-2-sam.md](24-loss-2-sam.md) | Measuring "how wrong", part 2: colour shape |
| 25 | [25-loss-3-combined-and-ramp.md](25-loss-3-combined-and-ramp.md) | Adding the two together, gradually |
| 26 | [26-parameter-budget.md](26-parameter-budget.md) | Counting all 5.5 million knobs |
| 27 | [27-training-data.md](27-training-data.md) | Where the training pictures come from |
| 28 | [28-training-loop.md](28-training-loop.md) | The loop that does the learning |
| 29 | [29-inference.md](29-inference.md) | Running it on a real scene |
| 30 | [30-scoring-and-the-product.md](30-scoring-and-the-product.md) | Turning errors into an answer |

**Sequel - the next Indradhanu: a smarter spectral bottleneck, then pretraining, scoring, fine-tuning and shipping (31-63)**

Written 2026-10-14, after the decision to pretrain the next Indradhanu on SpectralEarth with a
cross-attention spectral bottleneck. Parts 31 to 39 were written as a **design studied before
it was built**; the modules have since been built (`spectral_attention.py`,
`spectral_tokens.py`, `spectral_gate.py`, `spectral_reduced.py` in
`app/foundation_models/components/`), and parts 40 to 42 teach what running them changed.
Assumes all of parts 01 to 30; leans on 08, 10, 12, 13 and 17 to 21.

| # | File | The one thing it teaches |
|---|---|---|
| 31 | [31-why-the-compressor-is-a-ceiling.md](31-why-the-compressor-is-a-ceiling.md) | One fixed recipe for every pixel, and what that costs |
| 32 | [32-cross-attention.md](32-cross-attention.md) | Part 12's attention, with someone else asking |
| 33 | [33-a-spectrum-as-tokens.md](33-a-spectrum-as-tokens.md) | Bands as tokens that know their wavelength |
| 34 | [34-the-cross-attention-compressor.md](34-the-cross-attention-compressor.md) | 32 learned questions, assembled and counted |
| 35 | [35-skipping-a-dead-band.md](35-skipping-a-dead-band.md) | Minus infinity, and mixing sensors in one batch |
| 36 | [36-what-it-costs.md](36-what-it-costs.md) | Same parameters, 163x the arithmetic, and the memory wall |
| 37 | [37-the-spectral-decoder.md](37-the-spectral-decoder.md) | Reconstructing any sensor's bands from one set of weights |
| 38 | [38-trace-and-traps.md](38-trace-and-traps.md) | One pixel start to finish, and the three traps |
| 39 | [39-per-pixel-gating.md](39-per-pixel-gating.md) | The cheap rival: each pixel sets its own band volumes |
| 40 | [40-spectral-reduction.md](40-spectral-reduction.md) | Attention over 40 nm wavelength bins, without losing the bands |
| 41 | [41-scoring-validation.md](41-scoring-validation.md) | Ranking epochs needs a fixed ruler: the ramped-SAM validation trap |
| 42 | [42-one-band-list.md](42-one-band-list.md) | A stem that can read any band list must be trained on many: the PRISMA water failure |
| 43 | [43-f001-rotating-pretraining.md](43-f001-rotating-pretraining.md) | F001: 200 epochs on rotating 30,000-patch blocks, and a detection score that stalled |
| 44 | [44-three-native-datasets.md](44-three-native-datasets.md) | EnMAP, PRISMA and AVIRIS-NG at their own bands: quirks, splits, compression, one band list per batch |
| 45 | [45-scoring-the-direction.md](45-scoring-the-direction.md) | Why size-based scores fail, and residual Mahalanobis worked by hand |
| 46 | [46-the-ring-and-the-comparison.md](46-the-ring-and-the-comparison.md) | The 21/61 ring, integral images, and the eleven-way comparison that chose the default |
| 47 | [47-the-segmentation-head.md](47-the-segmentation-head.md) | A second head for water, cloud, thin cloud, shadow and snow, and a loss for untrusted labels |
| 48 | [48-checking-labels.md](48-checking-labels.md) | Auditing provider labels without ground truth: class 0 is cloud, and dry air makes false cirrus |
| 49 | [49-the-fine-tuning-curriculum.md](49-the-fine-tuning-curriculum.md) | Probe, adapt, joint: stages, gates, sensor shares, the trained-wavelength record |
| 50 | [50-what-ft01-and-ft02-showed.md](50-what-ft01-and-ft02-showed.md) | The probe's IoUs, FT02's gate miss by 0.006, and why `combined` caught up |
| 51 | [51-the-see-saw-and-batchnorm.md](51-the-see-saw-and-batchnorm.md) | Sensor-pure batches and the one BatchNorm: why evaluation depends on which sensor ended the epoch |
| 52 | [52-mixed-batches.md](52-mixed-batches.md) | `forward_mixed`: one stem pass per band list, one encoder pass for all - and what else FT02b changed |
| 53 | [53-edges.md](53-edges.md) | Why correcting the score map for edges fails, and why an edge term under the blind spot is different |
| 54 | [54-options-not-taken-yet.md](54-options-not-taken-yet.md) | Teacher head, per-sensor heads, synthetic targets, regularisation: what each costs and where it stands |
| 55 | [55-ft03-joint.md](55-ft03-joint.md) | FT03: segmentation over the whole mixed batch, seg_weight 0.1, FT01's head as a warm start - IoU up, detection unchanged |
| 56 | [56-what-the-implant-bench-says.md](56-what-the-implant-bench-says.md) | 3,336 implanted objects seemed to sort the stages, with pretraining the big step - **corrected 2026-10-18: that step was a band-order bug**; subtle sits at chance, EnMAP does not move |
| 57 | [57-stage-3-and-what-comes-next.md](57-stage-3-and-what-comes-next.md) | Stage 3: water / cloud exclusion helps every model alike, a model-free PCA-RX wins on bench-v1 and loses on real objects - so bench-v2 comes before detection training |
| 58 | [58-ft05-implant-denoising.md](58-ft05-implant-denoising.md) | FT05: implanted input, clean target - why it teaches "draw the background", the x5 near weight, the held-out source, a CPU-bound loader and its fix; did not pass |
| 59 | [59-the-scoreboard.md](59-the-scoreboard.md) | Every model on every benchmark, read against chance and PCA-RX and with v0.2.0 read correctly: no new model beats the shipped one on detection alone (rewritten 2026-10-19) |
| 60 | [60-ft04-the-release-candidate.md](60-ft04-the-release-candidate.md) | FT04: pseudo-labels with confidence thresholds and a water guard, why the cloud guard was dropped, rare-class weights - and what the model did and did not buy |
| 61 | [61-shipping-indradhanu-v2.md](61-shipping-indradhanu-v2.md) | Shipping FT04 as Indradhanu V2: a new codename (routing is last-wins), the embedded model config, the band-grid check, `read_bands`, cloud zeroed after scoring, and the parity run that guards it |
| 62 | [62-the-uncertainty-head-proposal.md](62-the-uncertainty-head-proposal.md) | **The proposal (run 2026-10-20 as FT06, see 63):** score-side ideas failed, so a small head predicts each pixel's expected error (Laplace likelihood, hidden pixels only) and the score divides by it |
| 63 | [63-the-uncertainty-head-run.md](63-the-uncertainty-head-run.md) | **FT06, failed:** the head learned its own error (Spearman ~0.6) but expects large errors on targets themselves (halo in visible neighbouring tokens), so dividing by it hides anomalies - predicting your error is not detection |

Parts 34 to 39's numbers were first produced by a study prototype and then re-checked against
the real modules (5,345 compressor and 6,401 decoder parameters at `d = D = 32`; 13,162 gate
parameters at 202 bands). Part 40's figures - 14,721 and 4,257 parameters, and the L40S
throughput table measured on 2026-10-14 - are the ones most likely to drift: a change to
`bin_nm`, `_hyper`'s hidden width, or the encoder will move them. The agreed design, including
the evaluation plan, is `docs/proposals/spectral-attention-bottleneck.md`; the runs are
recorded in `research/experiments/indradhanu-se-pretrain/`.

Parts 43 to 49 were written 2026-10-16, while F001 was still training and fine-tuning was at
stage 0. They follow the work of 2026-10-15/16: the final pretraining run (43), the three
native-band fine-tuning datasets (44), the switch of Indradhanu's default score to residual
Mahalanobis (45, 46), and the fine-tuning design - segmentation head, label audit and curriculum
(47 to 49). **Their numbers will drift more than any others in the course:** F001's validation
and bench figures by epoch (parts 43, 45, 46, 49 - the run had reached about epoch 60), the 0.861
stage-2 gate (49), the eleven-method comparison on F001's epoch 52 (46), the shard counts (44), the
label-audit figures and the 0.75 g/cm2 dry-air threshold (47, 48), and the starting sensor shares
and FT01-FT03 settings (49). The fine-tuning parts describe a design studied before its runs; the
code they cite (`finetune_trainer.py`, `targets.py`, the `trained_wl` record) exists and is tested,
but no fine-tuning run had finished. Authoritative: `docs/proposals/fine-tuning-and-segmentation.md`,
`docs/lld/residual-mahalanobis.md`, the LLDs for the EnMAP, PRISMA and AVIRIS-NG shards, and
`research/experiments/indradhanu-finetune/README.md`.

Parts 50 to 54 were written later on 2026-10-16, after FT01 and FT02 had run and while FT02b (mixed
batches) was running: the two runs' results and the decision to build on FT02 (50), the BatchNorm
diagnosis (51), mixed batches (52), the edge diagnosis and the planned FT02c edge term (53), and the
options set aside (54). **Values that will drift:** the status of every option in part 54. FT02b's and FT02c's results were
pending when these parts were written; on 2026-10-17 they were filled in (part 52, "Status, and what FT02b
showed"; part 53, "What FT02c showed"), and dated update notes were added to parts 49, 50 and 54 where
later runs contradicted them. The run READMEs under `research/experiments/indradhanu-finetune/`
and its "Log" section are authoritative; `docs/lld/finetune.md` holds the interfaces and decisions.

Parts 55 and 56 were written 2026-10-17, after FT03 (joint) had run and the implanted-target benchmark
had scored every model from v0.2.0 to FT03. **Values that will drift:** FT03's IoUs and SAMs (55), and
every number in part 56's tables - a rebuilt benchmark (new hosts, new seed) or a new model moves them.
The benchmark itself is taught in the anomaly-detection course, parts 16 to 18
(`docs/training/anomaly-detection/`). Authoritative: `research/experiments/indradhanu-finetune/FT03-joint/README.md`,
`research/experiments/implant-bench/README.md`, `docs/lld/implant-bench.md`.

Part 57 was written later on 2026-10-17, after the stage-3 evaluation through the new `app/evaluation/`
subsystem and while implant bench-v2 was being built. **Values that will drift:** every stage-3 number, the
foreignness and halo medians, and all of section 6 (bench-v2 had no results). Dated update notes were added
on 2026-10-18 to parts 49, 54, 56 and 57: bench-v2 accepted, the 39 real objects inside FT03's mask checked
(13 likely vessels; "no real object sits on water" was wrong), scope narrowed to cloud-only exclusion, and
FT05 launched. The evaluation machinery is
taught in the anomaly-detection course, part 19. Since 2026-10-17 the evaluation code that parts 42 to 56
cite under `scripts/evaluate_prisma_bench.py`, `scripts/evaluate_implant_bench.py` and
`app/utils/evaluation/` lives in `app/evaluation/` and `scripts/evaluate.py`; those parts keep the old
names where they describe what was run at the time. Authoritative: `docs/lld/evaluation.md`,
`docs/lld/implant-bench.md`, `research/experiments/implant-bench/README.md`.

Parts 58 and 59 were written 2026-10-18, while FT05 was at epoch 5 of 10. **Values that will drift:**
everything about FT05 - the epoch-5 detection rows (`C:/es/eval/ft05_ep5/report.md`: a tie with FT03 on
the real bench, half a point ahead on bench-v2), the epoch-5 SAMs and IoUs, the 2 h 40 min epoch and the
0-7% GPU and the loader diagnosis (all in the FT05 README's Results, "Epoch 5" and "Pace"); its final
results will supersede them. Part 59's other rows move with any new model or a rebuilt benchmark. Two open disagreements are
recorded in part 59: the saved v0.2.0 reading check, and bench-v2's per-tier foreignness (the first
resolved and the second superseded on 2026-10-18 - see the next paragraph). The benchmarks
themselves are taught in the anomaly-detection course, parts 20 and 21. Authoritative:
`research/experiments/indradhanu-finetune/FT05-implant-denoise/README.md`,
`docs/proposals/detection-supervised-finetuning.md`, `research/experiments/implant-bench/README.md`.

**The 2026-10-18 corrections, and part 60 (written 2026-10-19).** Two findings on 2026-10-18 changed
earlier parts. (1) PRISMA lists its bands descending and `np.interp` was called unsorted in the
evaluator's resample and in signature implants: v0.2.0's implant-bench rows were a reading artefact and
every transplanted signature was garbled (anomaly-detection course, part 22). Read correctly, on
bench-v2.1, v0.2.0 ties every new model; "pretraining is the big step" was the bug. (2) Set4's "11%
cloud" is haze and a cloud / smoke streak, not bright desert. Dated correction notes were added to parts
49, 50, 53, 54, 55, 56 (and its figure), 57 and 58; part 59 was rewritten; FT05's final verdict (not
passed) and the loader fix were added to part 58. Part 60 teaches FT04, the release candidate. **Values
that will drift:** every number in parts 59 and 60 - FT04's IoUs and SAMs, the real-bench and bench-v2.1
rows, the 0.159 -> 0.170 decomposition, the 0.875 / 0.82 pre-launch pseudo-label checks, the 69% cloud-guard
rejection, `PSEUDO_CONF` 0.9, the NIR 0.15 guard and the `[1, 1, 1, 3, 3]` weights (run configuration, but a
next run may change them), and the 43-minute epoch. The open questions now are part 59's: EnMAP hosts flat
at about 21%, bench-v2.1's easy-tier foreignness (1.60x), and training overlap for v0.2.0. Authoritative:
`research/experiments/indradhanu-finetune/README.md` (the log), `FT04-final-joint/README.md`,
`research/experiments/implant-bench/README.md` ("Corrections" and "bench-v2.1"), `docs/lld/finetune.md`.

**Part 61 (written 2026-10-19)** teaches FT04's shipping as **Indradhanu V2** (Phase 1 of
`docs/proposals/ship-ft04.md`: opt-in, v0.2.0 stays the default). Part 60 got a dated note: the
like-for-like v0.2.0 + exclusion run (0.165 against 0.170) has been made. **Values that will drift:** the
manifest figures (`v2.0.0`, epoch 10, 5,886,451 parameters, val 0.05578), the parity numbers (set1 0.2460,
set4 0.2106 / 0.2199, 11.83% excluded, 185 / 186 bands read), and everything Phase 2 would change - the
defaults, `band_filter_apply`'s grid default, and the "one action cannot run both models" limit.
Authoritative: `docs/lld/spectral-stem.md` ("Serving"), `docs/04-models.md` (Inference),
`docs/09-known-issues.md` ("Spectral-stem models - open gaps"), `final design/ROADMAP.md` (OB-14).

**Part 62 (written 2026-10-19)** is short and describes a **proposal**: `docs/proposals/uncertainty-head.md`,
written after the scoring-gains experiment (`research/experiments/scoring-gains/`, taught in the
anomaly-detection course's part 24) found no score-side change that beats residual Mahalanobis on real
objects caught. Nothing in it is trained or measured except the head's parameter count (95,632, checked
by building the decoder). **Values that will drift:** all of it - the design, the ~2.5 h estimate, the
pass criteria (pAUC 0.170 -> 0.180, Spearman >= 0.5) - once the proposal is agreed, changed or run.

**Part 63 (written 2026-10-20)** teaches the run: FT06, the uncertainty head on frozen FT04, 3 epochs,
head only. It **failed detection and is not adopted**; part 62 got a dated note at its top and now ends
pointing here. **Values that will drift:** the per-epoch Spearman and NLL table (0.584-0.640, -1.331 to
-1.341), the detection table (pAUC 0.170 / 0.125 / 0.163, epoch-1 sqrt 0.176), the target / ring `b` ratios
(1.05 / 1.12 / 1.04 at epoch 1, 1.07 / 1.19 / 1.13 at epoch 3), the 87th-90th percentile, and the status of the
two untried rescue ideas - if either is tried, this part's verdict may change. Authoritative:
`research/experiments/indradhanu-finetune/FT06-uncertainty/README.md` (and its `eval_report.md`,
`b_at_targets.py`), `docs/proposals/uncertainty-head.md`, `docs/lld/finetune.md` (mode `scale`).

## Figures

Most parts carry one or more numbered figures (Figure 16.1 is the first figure of part 16).
They are SVG files in `figures/`, drawn in one consistent visual language so a colour means
the same thing everywhere:

| colour | means |
|---|---|
| teal | spectral - bands, spectra, the compressor and decompressor |
| indigo | spatial - tokens, feature maps, the SegFormer encoder and decoder |
| plum | attention - queries, keys, values, softmax weights |
| amber | loss and error - residuals, scores, L1, SAM |
| grey hatching | masked, hidden, invalid or dead |

Tensor shapes are written in monospace, like `(B, 32, H, W)`, matching part 07.

## How to actually study this

- **Read in order.** Part 14 assumes part 13. There are no shortcuts.
- **Do the arithmetic.** Every worked example uses tiny numbers (three bands, a
  2x2 grid) so you can do it on paper in about a minute. Reading the arithmetic
  is not the same as doing it.
- **Keep a terminal open at the repo root.** Every file path in this course is
  real. Open the file. Look at the line. That habit is worth more than anything
  written here.
- **One part per sitting is fine.** This is a fortnight of study, not an
  afternoon.

## If you get stuck

Skip to the **Check yourself** questions at the end of the part. If you can
answer them, move on — you understood more than you felt you did. If you cannot
answer the first one, reread the first section only.

## Ground rules for this folder

This folder is checked into the repository, so it travels with a clone. It is
**study material, not product documentation** — a course to be read once, not a
reference to be consulted.

The authoritative docs are `docs/01-orientation.md` through
`docs/10-code-style.md`, plus
`research/model_break_down/05_hyperspectral_segformer_mae.md`.

It was written against the source as of 2026-08-24. The parts most likely to go
stale are those quoting specific numbers: the training configuration values in
parts 25 to 28, the resolver defaults in part 05, and the checkpoint metrics in
parts 05, 10 and 26. A retrain or a config change will move those. The
architecture parts (09 to 24) will age much more slowly.

If this course and the source code ever disagree, **the source code is right**.
Documentation in this repo has drifted before — which is exactly why every claim
here points at a file you can open and check.
