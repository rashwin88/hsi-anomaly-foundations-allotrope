# Finding anomalies in satellite imagery

A course for someone who has never worked with hyperspectral data and has never opened this
repository. Start at part 1 knowing nothing; finish able to read the source.

Twenty-three parts, each finishable in a sitting. **Read them in order.** Part N assumes parts
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
| [15](15-naming-the-family.md) | Naming the material family of a pixel the object only partly fills |
| [16](16-detection-measures.md) | Measuring detection where the analyst looks: pAUC, TPR at 0.1%, objects caught, and chance |
| [17](17-public-benchmarks.md) | The public benchmarks (HAD100, ABU), and why none of them tests this product |
| [18](18-the-implant-benchmark.md) | The implanted-target benchmark: real held-out scenes, synthetic objects like the real ones |
| [19](19-running-an-evaluation.md) | Running an evaluation: one config, scene-major loop, cached score maps, chance - and a classical baseline that wins on the implants and loses on real objects |
| [20](20-bench-v2.md) | bench-v2: copying contrast, foreignness and halo from the real objects, solving on the realised object, and the ordering test that lets a benchmark be trusted |
| [21](21-scope-and-masks.md) | Scope: cloud excluded, water kept for vessels, and what a mask costs - the 39 real objects inside FT03's mask, and set4's "desert" that was haze |
| [22](22-the-sorting-bug.md) | How a sorting bug rewrote the scoreboard: `np.interp` needs ascending wavelengths, PRISMA lists them descending, and the cross-check against the product path that found it |
| [23](23-using-indradhanu-v2.md) | Using Indradhanu V2 in an action chain: a native-band band filter, the codename, the error you get otherwise, the new cloud mask and class map, and why one action cannot run both Indradhanus |

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
| 15 | family counts in the library, `family_alpha`, the ~2.5 ms per pixel cost, PRISMA results |
| 16 | `compute_roc`'s threshold count formula, the FT02b bench measures |
| 17 | the state of HAD100 / ABU (ABU's page was already 404 on 2026-10-16), the leakage distances |
| 18 | bench-v1's object counts and every results row - a rebuild (new hosts, new seed) or a new model moves them; the tier edges 0.502 / 0.934. Since 2026-10-18 its v0.2.0 row is void and its signature implants are known garbled (part 22) |
| 19 | every stage-3 number (exclusion, PCA-RX), the diagnosis medians (0.006 / 0.74); dated update notes record what changed since (cache key, `dirty` flag, the 39 objects, bench-v2 accepted, the resample bug) |
| 20 | every bench-v2 number (2,913 objects, the model and PCA-RX rows), the `real_v2.json` medians (0.0071 / 0.643), the per-tier ratios recomputed from `objects.json`, sigma 1.4, the 0.25 -> 0.02 s subspace timing. The v0.2.0 reading question was resolved on 2026-10-18 (part 22): its v0.2.0 row is void and bench-v2 is superseded by bench-v2.1 |
| 21 | the cloud-only and water + cloud numbers, the per-scene object counts, the 39-object breakdown and FT03's 11% set4 "cloud" (haze, corrected 2026-10-18); all move with a new segmentation head - FT04 passed on 2026-10-18 and its masks are the release candidate's |
| 22 | every bench-v2.1 number (2,817 objects, the model rows, the 1.05 / 1.15 / 1.60x foreignness), the resample-check rows, the PRISMA band orders quoted from the dev laptop's files, NumPy 2.5.2's `np.interp` behaviour on unsorted input (undefined by contract - another version may differ) |
| 23 | Indradhanu V2's status (opt-in, v0.2.0 the default - Phase 2 of `docs/proposals/ship-ft04.md` would change both defaults and the "one action cannot run both" limit), the 0.165 / 0.170 like-for-like, set4's 11.83% excluded, the missing viewer panel, the unmeasured CPU cost |

Structural claims - what a vendable is, why masking works the way it does - will outlive
the numbers.

## A note on where this lives

The house style says training modules go in a gitignored working folder. This one is tracked
instead, because it exists to onboard people and a doc that does not survive `git clone`
cannot do that. It is separated from `docs/01-10` so nobody mistakes a course for reference
documentation.
