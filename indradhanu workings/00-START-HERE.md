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
