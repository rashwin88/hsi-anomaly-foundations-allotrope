---
name: iterative-nano-chunking
description: The default coding workflow for this repo. Use for ANY code change beyond a one-line typo fix - features, bugfixes, refactors, migrations, new action types, new models. Enforces a four-phase loop - restate the problem, propose a design and STOP for agreement, break the work into nano chunks of at most 20 changed lines, then execute exactly ONE chunk per message and STOP for feedback before the next. Finishes by updating design docs and LLDs. Triggers on - implement, build, add, create, fix, debug, refactor, rework, change, migrate, wire up, hook up, extend, or any request to write or modify code.
---

# Iterative nano chunking

A change lands in small, reviewable steps with a human in the loop at every one. The point
is not speed. The point is that the human sees and approves each increment, so a wrong turn
costs one chunk instead of an afternoon.

## The four phases

### Phase 0 - Understand

Restate the problem in your own words in 2-4 sentences. Name the files you expect to touch.

Ask blocking questions **now**, before designing. A blocking question is one where different
answers produce different designs. Do not ask questions you can answer by reading the code.

### Phase 1 - Propose a design, then STOP

Write the design. **No code yet.** Cover:

- **Approach** - what you're going to do, in prose.
- **Files touched** - each with a one-line reason.
- **Interfaces** - new or changed function signatures, schemas, endpoints, config fields.
- **Trade-offs** - what you considered and rejected, and why. If there's a real fork in the
  road, say so and recommend one.
- **Risks** - what could break, what you can't verify in this environment.

Then **stop and ask for agreement.** Do not produce a chunk plan in the same message.

### Phase 2 - Chunk plan, then STOP

Once the design is agreed, break it into numbered chunks.

**Chunk rules:**

- **At most 20 changed lines** per chunk - added plus modified, counting the diff body.
- **One coherent step.** A chunk should be describable in one sentence without "and".
- **Usually one file.** Two only when they must change together to stay consistent.
- **Leave the tree working** where possible. If a chunk must leave it temporarily broken,
  say so explicitly and name the chunk that repairs it.
- **Order by dependency.** Types and schemas before the code that uses them.
- If a step won't fit in 20 lines, split it. If it genuinely can't be split, say so and
  explain why before proceeding.

Present the full numbered list with a one-line goal each, then **stop and ask for
agreement.** Do not start chunk 1 in the same message.

### Phase 3 - Execute, one chunk per message

This is the phase agents get wrong. Read it twice.

> **Execute exactly ONE chunk per response, then stop and wait.**
>
> Not two. Not "these next three are trivial so I'll batch them." Not "the user approved
> chunk 2 so they probably want 3 as well." Not even when the chunk is a two-line import.
> One chunk, then stop. Blanket permission given earlier does not carry forward - the loop
> exists precisely so approval is renewed at each step.

Format each chunk message as:

```
Chunk N of M - <title>

Goal:     one sentence
File:     path/to/file.py
Lines:    <count> changed
```

...then make the edit, then:

```
Verified: <what you actually checked, or "could not verify - <reason>">
Next:     Chunk N+1 - <title>
Remaining: N+1, N+2, ... <one-line titles>
```

...and ask whether to proceed.

Restating the remaining chunks every message is deliberate: it survives context compaction
and lets the human re-steer cheaply.

**Reading the response:**

- "ok" / "yes" / "next" / "go" / "proceed" - do the next chunk.
- A question - answer it. Do **not** also do the next chunk.
- Feedback or a correction - revise **this** chunk and re-present it. The chunk number does
  not advance.
- Silence or an unrelated message is **not** approval.

**When a chunk proves the plan wrong:** stop the loop immediately. Say what you learned, what
it invalidates, and propose a revised plan. Return to Phase 1 or 2 as appropriate. Never
quietly improvise a different design mid-loop.

**Verification per chunk.** Chunks are small, so checking is cheap - run the narrowest thing
that proves it: a single test, an import, `tsc`, a grep for remaining call sites. If tooling
isn't available in this environment, say plainly what you could not verify rather than
implying it passed.

### Phase 4 - Close out

A task is not done when the code works. Do these, each as its own chunk:

1. **Update the affected doc** in `docs/01-orientation.md` .. `docs/09-known-issues.md`.
   Behaviour change, new convention, new Action type, new endpoint, changed default - the
   relevant numbered doc gets updated in the same session.
2. **If you fixed a known issue, delete its entry** from `docs/09-known-issues.md`. Don't
   annotate it as fixed - remove it.
3. **If you introduced a new issue or a deliberate limitation, add an entry** there. This
   repo has a zero `TODO`/`FIXME` marker policy - `docs/09-known-issues.md` is the register.
4. **Write or update an LLD** in `docs/lld/<subsystem>.md` when the change adds a subsystem,
   or logic whose *why* isn't obvious from reading it. Template:

   ```markdown
   # <Subsystem>

   **Purpose** - what problem it solves, in two sentences.
   **Interfaces** - public functions/classes/endpoints, with signatures.
   **Data flow** - what comes in, what goes out, what it touches on disk or in the DB.
   **Invariants** - what must always hold. What breaks if it doesn't.
   **Failure modes** - how it fails and what the caller sees.
   **Decisions** - what was chosen, what was rejected, why.
   ```

   Small changes to existing behaviour do not need a new LLD - update the existing one.
5. **Product or UX change** - update `final design/ROADMAP.md` and the relevant `-spec.md`.

State explicitly which docs you updated, and which you judged unaffected.

## When to use a lighter touch

Skip to a single chunk, with the design stated in one line, when the change is a typo, a
constant, a comment, or a one-line fix the human has already fully specified. Say that you're
doing so.

Everything else gets the full loop. When unsure, use the full loop - the cost of an extra
approval round is one message; the cost of an unreviewed wrong turn is the whole task.
