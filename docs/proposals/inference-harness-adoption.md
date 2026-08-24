# InferenceHarness: adopt, or delete

**Status:** proposal — not decided, not scheduled, nothing built.
**Raised:** 2026-08-24, during the `app/` + `backend/` refactor.
**Scope:** how classical detectors are driven at inference time. Does not touch the
foundation-model path, the scoring module, or any Action contract.

---

## The question this answers

*Why are there two ways to run a classical detector, and why does production use the
worse one?*

`InferenceHarness` is fully implemented, documented, and covered by 22 tests — and nothing
in production calls it. It has been dead code since it was written. The decision is to
adopt it or delete it; leaving a tested, documented abstraction unused is the one option
that costs something every time a new reader finds it and assumes it matters.

Deferred deliberately during the refactor: adopting it means changing a working scoring
path, which is not something to fold into a cleanup pass.

## What exists today

Two paths that do the same job.

**The one production uses** — `backend/allotrope/action_types/_anomaly_scoring_run.py`:

```python
detector_cls = get_detector(ADModel(m.detector_key))
detector = detector_cls(vendable)
detector.fit()
detector._spatial_mask &= eroded_keep_mask     # reaches into a private attribute
scores = detector.detect(cube_np, validity_np)
```

**The one nothing uses** — `app/utils/anomaly_detection/inference_harness.py`:

```python
harness = InferenceHarness(config=InferenceHarnessConfig(patch_config=..., batch_size=8))
result = harness.run(vendable, detector)      # -> AnomalyDetectionResult
```

The harness extracts cube and validity polymorphically from any of the three vendable
types, optionally fits, then runs either full-scene or patched with overlap-averaging.

Source: `app/utils/anomaly_detection/inference_harness.py`,
`app/models/anomaly_detection/harness_config.py`,
`backend/allotrope/action_types/_anomaly_scoring_run.py`.

### Why the current path is the weaker one

- **It mutates a private attribute of the detector.** `detector._spatial_mask &= keep_mask`
  is done from the outside, between `fit()` and `detect()`, to keep clouds and water out of
  the covariance estimate. It works, it is commented at the call site, and it is exactly the
  kind of coupling that breaks silently when a detector is refactored.
- **It cannot patch.** The harness supports patched inference with overlap averaging; the
  production path is full-scene only. For LRX on a large scene that is a memory ceiling,
  and it is the reason `stride` exists as a subsampling escape hatch in the detector itself.
- **The vendable-type branching is inline.** `_extract_cube_and_validity` already solves
  this in the harness and is duplicated, less completely, in the run module.

## Three options

### Option A — delete the harness (least work)

Remove `inference_harness.py`, `harness_config.py`, `AnomalyDetectionResult`, and the 22
tests. Roughly 250 lines and one doc section.

Honest, and it removes the trap. But it discards the better abstraction and keeps the
private-attribute reach-through, and if patched classical inference is ever needed the work
comes back.

### Option B — adopt it as-is

Rewrite the classical branch of `_anomaly_scoring_run.py` to construct an
`InferenceHarnessConfig` and call `harness.run()`.

Blocked on one real gap: **the harness has no way to narrow the background mask.** The
production path's `detector._spatial_mask &= keep_mask` has no equivalent, so adopting it
as-is would silently drop cloud and water exclusion from the covariance estimate — a
correctness regression, not a refactor.

### Option C — extend, then adopt (recommended)

Add a `keep_mask` to `InferenceHarnessConfig`, applied inside `run()` between `fit()` and
`detect()`. That moves the reach-through *into* the abstraction that owns the detector
lifecycle, where it can be documented and tested, and makes Option B safe.

Then switch the classical branch over, and delete the inline extraction and mask mutation
from `_anomaly_scoring_run.py`. That module is 1,000 lines and is a Stage 3d split
candidate anyway; this removes a chunk of it.

## Costs, risks, and what would have to be proven

- **No behavioural change is acceptable here.** Classical detection is a shipped feature.
  Before and after must produce identical score maps for the same scene and detector — bit
  identical, not visually similar. There is no test asserting that today, so the first task
  is a characterisation test, not a refactor.
- **`backend/` coverage is thin.** `tests/test_backend/test_imports.py` proves the module
  graph resolves and nothing more. The 22 harness tests exercise the harness, not the
  integration.
- **The keep_mask semantics need pinning down.** The production path ANDs it into the mask
  *after* `fit()`, so the covariance is estimated on the full valid region and only the
  scoring is narrowed. Whether that is intended or incidental should be settled before it
  is enshrined in a config field — the alternative (mask before fit) changes the numbers.
- Detectors vary in whether `fit()` is meaningful; `AnomalyDetector.fit()` is a no-op by
  default, and `fit_on_full_scene` already exists in the config to express that.

## Related

- `docs/05-detectors.md` — the RX family and where the keep_mask comes from
- `docs/09-known-issues.md` — the registry gaps (`LRX` and `MNF_LRX` are reachable in the
  detector registry but absent from the backend capabilities table, so adopting the harness
  does not by itself make them selectable)
- Stage 3d of the refactor — splitting `_anomaly_scoring_run.py`

## Open questions

1. Is patched classical inference actually wanted, or is full-scene sufficient forever? If
   the latter, Option A gets much more attractive.
2. Should the keep_mask apply before or after `fit()`? The current behaviour may be
   incidental.
3. Is `AnomalyDetectionResult` (score map plus provenance) worth keeping as the classical
   return type, or should the classical and foundation paths keep returning bare arrays?
