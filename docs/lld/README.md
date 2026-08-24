# Low-level designs

One file per subsystem whose *why* is not recoverable by reading it.

These are not API docs. The numbered docs (`docs/01-orientation.md` onward) tell you what
the system does; an LLD here tells you why one piece of it is built the way it is, what
must stay true, and what breaks if somebody tidies it up.

A subsystem earns a file here when it was newly added, or when its rationale lives only in
someone's head. Small changes to existing behaviour update the existing file rather than
adding another.

| file | subsystem |
|---|---|
| `action-api.md` | the five modules `actions.py` was split into |
| `pixel-stats.md` | resolving per-band normalisation statistics |
| `local-background.md` | batched Mahalanobis for the annulus detectors |
| `frequency-destriper.md` | the FFT notch filter and its significance test |
| `test-harness.md` | how anything gets verified in this repo |

## How these are written

Every claim points at a file you can open, and was checked against the source rather than
against an existing doc. Where a source comment and the code disagree, **the code wins and
the disagreement is written down** - those contradictions are the most useful thing here,
because they are what will mislead you at 2am.

Numbers are concrete. "Uses a lot of memory" is not a statement; "24 bytes per padded pixel,
52 MB per padded PRISMA band, 12.4 GB for all 239 at once" is.

Values that drift are marked where they appear. Config defaults, checkpoint metrics and
line counts all rot; if one matters to your decision, re-verify it.

These are subordinate to the source and to `docs/01-09`. If they contradict either, they
are wrong.
