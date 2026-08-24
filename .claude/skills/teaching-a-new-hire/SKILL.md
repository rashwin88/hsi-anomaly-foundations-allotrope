---
name: teaching-a-new-hire
description: The house style for writing training material, onboarding docs, explainers and walkthroughs for someone who does not yet know this codebase or its domain. Use when asked to explain a subsystem to a new engineer, write a training module or course, produce a "how does X work" walkthrough, document a model or pipeline for someone unfamiliar, or make an existing doc easier for a beginner. Enforces a fixed per-part skeleton, worked arithmetic over symbols, verified-against-source claims, analogies, and self-check questions. Triggers on - teach, train, training module, onboarding, explain to a new hire, walkthrough, tutorial, course, primer, from scratch, assume no knowledge, make this simpler, beginner, slow learner.
---

# Teaching a new hire

Written material for someone who knows nothing yet. Not a reference, not an API doc - a
course. The reader should be able to start at part 1 knowing nothing and finish able to
read the source.

## Before you write a word: verify

Every claim points at a file the reader can open. Read the source first; never write from
memory or from an existing doc. This repo has shipped docs that described classes which did
not exist.

When source and comments disagree, **the source wins** - and say so in the text. Those
disagreements are among the most valuable things you can hand a new hire. Three real
examples found while writing the Indradhanu module: a docstring claiming stage 1 uses
kernel 7 when the encoder builds it with 4; an inference threshold of 0.1 under a comment
claiming it "matches training" at 0.4; and a `warmup_epochs` config field that is
documented, defaulted, set in every config, and never read.

Show the reader how you know: paste the grep, cite the line, name the file.

## The per-part skeleton

Every part gets the same four sections, in this order. The sameness is the point - a
struggling reader always knows where they are.

1. **"The one thing this part teaches"** - one sentence, in a blockquote, before any
   detail. If you cannot write it, the part is doing too much; split it.
2. The explanation.
3. **Common confusions** - the mistakes people actually make, as questions with answers.
   Include the naming collisions and the two-things-called-the-same-word traps.
4. **Check yourself** - about five questions, answers in a `<details>` block. At least one
   must require arithmetic the reader does by hand.

End with a one-line pointer to the next part. Start the set with an index that lists every
part and its one-sentence purpose.

## Sequencing

- **Strictly linear.** Part N may assume parts 1..N-1 and nothing else. No forward
  references except "part 21 covers this".
- **One sitting per part.** If a part cannot be finished in a sitting, split it. The
  masking machinery became three parts for this reason.
- **Motivation before mechanism.** Say what problem exists and why the obvious fix fails,
  then show the code. Never open with the implementation.
- **Context before component.** Domain, product placement and vocabulary come before any
  architecture.

## Explaining

**Define before use.** Every term gets a plain-language definition at first use, before any
formula. That includes words you think are universal: tensor, parameter, batch, loss,
kernel, stride, gather, scatter, epoch.

**Numbers, not symbols.** Replace every abstract claim with a concrete one.

- Not "softmax saturates for large inputs" but `exp(20) = 485,165,195` next to
  `exp(-15) = 0.0000003`.
- Not "arccos is unstable near 1" but a table of its derivative at u = 0.9, 0.99, 0.999,
  0.9999, infinity - beside the observation that a well-trained model sits at 0.999+.

**Show every arithmetic step.** Never `mean = 0.070 / 3 = 0.0233`. Show each difference,
each absolute value, the sum, then the division, on separate lines.

**Keep worked examples tiny.** Three bands, a 2x2 grid, six tokens. Small enough to do on
paper in a minute, because the reader must actually do it. State that expectation.

**Use analogies, and make them do work.** One good analogy per part, tied to something
outside software. The countryside painter who has never painted a crashed aeroplane;
the cloze test; exam marks converted to standard deviations; a 48-track mixdown to stereo;
reading an executive summary instead of the full report. Retire an analogy once it stops
being accurate rather than stretching it.

**Prove the reader understands by predicting a real number.** The strongest device
available: derive a layer's parameter count by hand, then show it matching the real model's
reported figure. Do this repeatedly. Nothing else builds confidence as fast.

**Explain the "why", especially for what looks like a mistake.** Asymmetries, missing
normalisations and odd defaults are usually deliberate. Say so, give the reason, and say
what would break if somebody "tidied" it.

## Formatting

- Tables for anything with parallel structure: shapes, configurations, comparisons,
  before/after.
- ASCII diagrams for data flow. No image dependencies.
- Short paragraphs. Two to four sentences.
- Annotate code snippets with shapes, matching the repo's own convention
  (`# x: (B, N, C)`).
- Quote source comments verbatim when they explain a decision well. Attribute them.
- Bold the sentence a reader should remember; do not bold more than one per section.
- Plain hyphens, not em dashes, matching the rest of `.claude/skills/`.

## Tone

- Second person. "You will trip over this."
- Say when something is hard, surprising, or an actual known trap.
- Say when a design decision is a genuine trade-off rather than a clear win.
- Never write "simply", "just", "obviously", or "as you know".
- Do not apologise for complexity and do not oversell simplicity.

## Scope of a module

For a subsystem the size of one model or one pipeline, roughly:

| Parts | Content |
|---|---|
| 5 | domain, vocabulary, product placement, the data it consumes |
| 3 | the core idea, notation and shapes, prerequisite maths |
| 14 | the machinery, one component per part, ending in one end-to-end trace |
| 8 | how it is trained, run, and how its output reaches a user |

Adjust the proportions, keep the shape. The end-to-end trace part is not optional: it is
where the components stop being separate.

## Housekeeping

- Put a training module in a gitignored working folder unless asked otherwise, and say so
  in the index. Tell the user it will not travel with a clone.
- Note in the index which parts cite values that will drift (config numbers, defaults,
  checkpoint metrics) so a future reader knows what to re-verify.
- Point at the authoritative docs and say that this module is study material, subordinate
  to them and to the source.

## The finishing test

Read part 1 as if you know nothing. Then read the last part. If a term in the last part
was never defined, or a formula was never worked with real numbers, go back.
