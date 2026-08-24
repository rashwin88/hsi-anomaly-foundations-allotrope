# Finding anomalies in satellite imagery

A course for someone who has never worked with hyperspectral data and has never opened this
repository. Start at part 1 knowing nothing; finish able to read the source.

Fourteen parts, each finishable in a sitting. **Read them in order.** Part N assumes parts
1 to N-1 and nothing else, so skipping ahead will cost you more time than it saves.

| part | one-sentence purpose |
|---|---|
| [01](01-the-problem.md) | What problem this system solves, and why it is hard |
| [02](02-light-bands-cubes.md) | What a hyperspectral cube actually is |
| [03](03-scene-project-action.md) | Where you are: the product's four nouns |
| [04](04-the-sensors.md) | Five sensors, and what each one hands you |
| [05](05-the-vendable.md) | The one data structure everything downstream consumes |
| [06](06-anomaly-as-surprise.md) | The core idea, and why there are no labels |
| [07](07-the-band-pipeline.md) | Making five sensors comparable |
| [08](08-rx-detection.md) | Statistical detection, and one banned technique |
| [09](09-local-rx.md) | Scoring a pixel against its neighbours |
| [10](10-reconstruction-models.md) | Teaching a network what normal looks like |
| [11](11-masking.md) | Why a pixel must never help reconstruct itself |
| [12](12-residual-to-score.md) | Turning reconstruction error into a decision |
| [13](13-end-to-end.md) | One scene, one anomaly, start to finish |
| [14](14-material-and-export.md) | Naming the material and getting it out |

## How to use this

**Do the arithmetic.** Every part ends with check-yourself questions, and at least one in
each needs a pen. The worked examples are deliberately tiny - three bands, a 2x2 grid - so
you can do them in a minute. Reading them is not the same as doing them.

**Open the files.** Every claim points at a path you can open. Where a source comment and
the code disagree, this course says so and follows the code - those disagreements are worth
more to you than the agreements.

## What this is not

Study material, subordinate to two things:

- **The source.** If this course contradicts the code, the code is right.
- **`docs/01-orientation.md` through `docs/10-code-style.md`**, which are the reference
  documentation, plus `docs/lld/` for individual subsystems.

This is a path *into* those, not a replacement.

## Values that will drift

Some numbers here are read from configs and checkpoints and will go stale. Re-verify before
relying on them:

| part | what will drift |
|---|---|
| 04 | band counts per sensor |
| 07 | `BandFilterConfig` defaults, the 165-band grid |
| 08, 09 | detector defaults - windows, regularisation, component counts |
| 10 | parameter counts, checkpoint validation losses |
| 12 | scoring defaults and threshold percentiles |

Structural claims - what a vendable is, why masking works the way it does - will outlive
the numbers.

## A note on where this lives

The house style says training modules go in a gitignored working folder. This one is tracked
instead, because it exists to onboard people and a doc that does not survive `git clone`
cannot do that. It is separated from `docs/01-10` so nobody mistakes a course for reference
documentation.
