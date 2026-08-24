# Proposals

Ideas that have been thought through but **not decided**. Nothing here describes shipped
behaviour.

This folder is deliberately outside the authoritative set. `docs/01-orientation.md` ..
`docs/09-known-issues.md` document what the system *does*; a file in here documents what it
*could* do and what that would cost. Do not cite a proposal as though it were current
behaviour, and do not treat one as a commitment to build.

Each file states its status at the top. When a proposal is accepted and built, the change
gets written into the numbered docs (and an LLD under `docs/lld/` if it adds a subsystem),
and the proposal is deleted — it is not left behind annotated as "done".

If a proposal is rejected, delete it too. The reasoning that mattered belongs in whichever
numbered doc explains the design that won.

| file | subject |
|---|---|
| `spectral-attention-bottleneck.md` | replacing Indradhanu's fixed linear spectral compressor with band-level attention |
| `inference-harness-adoption.md` | adopting or deleting `InferenceHarness`, which is fully built, tested, and unused |
