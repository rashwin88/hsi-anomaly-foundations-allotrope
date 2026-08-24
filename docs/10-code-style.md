# 10. Code style

House rules for `app/` and `backend/`. Written for the refactor, but they apply to new code
too.

The codebase is ~39,000 lines at roughly 10% comment density. That ratio is about right —
**the problem has never been volume, it's placement.** A file that explains its own reason
for existing is worth ten files of line-by-line narration.

## Every module opens with a header

Four questions, in this order. Two of them are usually one line each.

```python
"""
<One line: what this does.>

<Where it sits: what calls this, what it feeds. Name the actual modules.>

<Why it exists: the problem it solves, and what breaks without it. This is the
part that earns its keep — it is the thing a reader cannot recover from the code.>

<Gotchas: units, shapes, orderings, anything that will bite.>
"""
```

The best example in the repo is [`app/utils/pixel_fill/nearest_valid_fill.py`](../app/utils/pixel_fill/nearest_valid_fill.py).
It spends ten lines explaining that a 7×7 patch-embedding kernel straddling a valid/invalid
boundary sees a cliff from 0 to −2.3, that attention broadcasts that contamination globally,
and that the resulting artefacts *outrank real detections*. No amount of reading
`distance_transform_edt` tells you that. Then it lists what changes and what deliberately
doesn't.

Aim for that. Six lines beats sixty.

## Comment what the code cannot say

Code already states *what*. Comments are for *why*, and for what would otherwise be
invisible.

```python
# Good — the reader cannot deduce any of this:
# float64 for the moments: HotSat DN sits near 5000±400, so var = E[x²] − E[x]²
# loses catastrophically in float32.
# Percentile-spaced, not linear: score distributions are heavy-tailed, so linear
# spacing wastes almost all its resolution in the empty upper range.
# Stage 1 uses NON-overlapping patches so token removal leaks nothing. Stages 2-4
# overlap deliberately.

# Noise — delete on sight:
# increment the counter
i += 1
# loop over the bands
for b in range(n_bands):
```

If a comment would survive a rewrite of the line beneath it, it's probably a good comment.
If it dies with the line, it was probably restating it.

## Docstrings

- **Public functions and classes** — what it does, what it returns, what it raises. Document
  arguments only where the name and type annotation don't already say it. Units, expected
  shapes and orderings always earn a line: `(C, H, W), reflectance, ascending wavelength`.
- **Private helpers** — one line if the purpose isn't obvious from the name. Nothing if it is.
- **Never restate the signature.** The annotations are right there.

## Constants

A magic number gets a name and a note on where it came from:

```python
# Clusters colder than this below the scene median are cloud. 12 °C is
# empirical — see the GMM anchors in b10_adaptive_cloud_masker.
CLOUD_TEMPERATURE_MARGIN_C = 12.0
```

"Where it came from" matters more than the value. A number nobody can re-derive is a number
nobody can safely change.

## Things not to do

- **No line-by-line commentary.** It rots, and it trains readers to skip comments.
- **No comments that reference line numbers**, or say "currently" / "for now" without saying
  what would change it.
- **No `TODO` / `FIXME` / `HACK`.** Zero-marker policy — unfinished work goes in
  [`09-known-issues.md`](09-known-issues.md), which is the register.
- **Don't reformat while commenting.** A docstring commit whose diff also reflows code is
  unreviewable. Comments and structure move in separate commits.
- **Don't document what you haven't verified.** Prose in this repo has drifted ahead of the
  code before — a whole doc tree once described classes that never existed. If you're
  describing behaviour, read the behaviour first.

## Module size

Roughly 300 lines is where a file starts costing more than it gives. Forty files are over
that today; ten are over 500. Not every long file is wrong — but if you're adding to one,
that's the moment to ask whether the thing you're adding is really the same concern.

Splitting is a **structural** change: it needs a test run, and it does not belong in the same
commit as a docstring pass.

---

**See also:** [6. Backend](06-backend.md) for the action-type contract and the lazy-import
rule, [1. Orientation](01-orientation.md) for the `app/` vs `backend/` layering rule.
