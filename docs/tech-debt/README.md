# Tech debt

Code that **works** but is structurally awkward, and the cost of straightening it.

## How this differs from `docs/09-known-issues.md`

They are easy to confuse, so the line is:

- **`09-known-issues.md`** — things that are **wrong**. Broken imports, failing builds,
  config knobs that do nothing, train/inference mismatches. A defect register.
- **This folder** — things that are **right but expensive**. Duplication, leaky
  abstractions, missing tests, coupling that blocks a change we want to make. Nothing here
  is a bug; every entry describes working code.

If a reader would be surprised the software runs at all, it belongs in `09`. If they would
wince but agree it works, it belongs here.

## What an entry must contain

Vague debt never gets paid. Each entry states:

1. **The extent, counted.** "8 files define the same constant", not "the bucket name is
   duplicated". A number turns an opinion into a task.
2. **What it blocks.** Debt with no consequence is not debt, it is taste.
3. **Why not now.** Every entry exists because someone decided against fixing it. Record
   the reason, so the next person can tell whether it still holds.
4. **The size of the fix**, in chunks, so it can be scheduled rather than dreaded.

## Lifecycle

Delete an entry when the debt is paid. Do not annotate it as done — same rule as
`09-known-issues.md`. If a decision reverses and the debt becomes permanent by choice,
move the reasoning into the relevant numbered doc and delete the entry.

*(No open entries. `s3-coupling-in-sharding.md` was paid off on 2026-08-26 and deleted,
per the lifecycle above.)*
