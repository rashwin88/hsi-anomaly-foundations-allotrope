# The Action API modules

**Files:** `backend/allotrope/api/actions.py`, `action_threshold.py`, `action_files.py`,
`action_export.py`, `_action_common.py`

## Purpose

Everything a user runs in Allotrope is an Action. All of it used to live in one 1,525-line
`actions.py`. This records how it was split, why along those lines, and the one coupling
that must not be broken.

## The shape now

| file | lines | the question its endpoints answer |
|---|---|---|
| `actions.py` | 544 | create, list, fetch, delete - the Action lifecycle |
| `action_threshold.py` | 525 | "is this threshold right?" - the interactive flow |
| `action_export.py` | 326 | "get a result out of the system" |
| `action_files.py` | 247 | "give me bytes this Action produced" |
| `_action_common.py` | 71 | the pieces more than one of them needs |

Line counts drift. The grouping is the point, not the numbers.

## The coupling you must not break

`action_threshold.py` holds three endpoints:

```
POST /actions/{id}/anomaly_detection_preview        try a threshold
GET  /actions/{id}/anomaly_detection_preview_mask   fetch what it looked like
POST /actions/{id}/anomaly_detection_commit         keep it
```

**These three must stay in one module.** `preview` renders a mask PNG and stashes it in a
module-level dictionary; `preview_mask` reads it back out:

```python
_PREVIEW_MASK_CACHE_MAX = 16
_preview_mask_cache: "dict[tuple, bytes]" = {}
_preview_mask_lru: "list[tuple]" = []
```

Separate them into different modules and each gets its own `_preview_mask_cache`. The
preview endpoint writes to one dictionary, the mask endpoint reads an empty one, and every
request 404s with `preview_mask_not_cached_repost_apply`. Nothing in the logs would point at
the cause, because from each endpoint's own perspective it is behaving correctly.

That cache is also why **this api must run as a single process**. A second uvicorn worker
gets its own copy of the dictionary, and roughly half of all mask requests land on the
process that did not render it. The container CMD is single-process today; adding
`--workers` would break preview silently.

## Why the threshold flow is interactive at all

An absolute score cut cannot work across scenes. Typical residuals differ by an order of
magnitude between a calm lake and a fire-affected scene, so a fixed threshold flags
everything in one and nothing in the other. A human picks a percentile by eye instead.

The consequence is a lifecycle state most systems do not have. `anomaly_detection_prep`
finishes its compute and then parks at `needs_threshold` rather than `complete`, waiting.
That is the one documented exception to the invariant in `models/action.py`: normally
`status == 'complete'` implies exactly one `ActionOutput` row, but a prep action has an
output while still sitting at `needs_threshold`.

The mechanism is `TERMINAL_STATUS`, read reflectively by the worker:

```python
terminal = getattr(spec, "TERMINAL_STATUS", "complete")
```

## Interfaces in `_action_common.py`

```python
class ActionOutputPublic(BaseModel)          # wire shape for ActionOutput rows
def action_or_404(action_id_wire, db) -> Action
def output_for_action(action_id, db) -> ActionOutput | None
def output_to_wire(o: ActionOutput) -> ActionOutputPublic
```

**The bar for living here is being needed by more than one Action module - nothing else.**
`ActionOutputPublic` qualifies because `actions.py` embeds it in `ActionDetail` and
`action_files.py` returns it directly. A response model used by exactly one module stays
with that module.

That rule was already revised once. `_action_common.py` was created as helpers-only, no
schemas; `ActionOutputPublic` was moved in one commit later when the second cut needed it.
The looser rule is the honest one, but it only works if it is applied - this module becomes
useless the moment it turns into a bag of everything.

## Routing

All four routers share the `/actions` prefix and mount separately in `main.py`. That is safe
because **no two Action routes are ambiguous** - no path-and-method pair can match more than
one handler - so registration order carries no meaning.

Check that before adding a route. If you add something that could collide with an existing
pattern, mount order suddenly matters and this note stops being true.

## Invariants

- Wire ids are prefixed. `action_<uuid>`, never a bare UUID. Parse with
  `api/wireformat.py`; returning a raw UUID is a bug.
- Artifact paths are stored **relative** in the database and joined to the artifacts volume
  at read time. A stored path can never escape it.
- Every handler is a sync `def`. The DB session is sync; one `async` endpoint would block
  the loop.

## Failure modes

| Condition | Response |
|---|---|
| Unknown action id | 404 `action_not_found` |
| Preview mask evicted or api restarted | 404 `preview_mask_not_cached_repost_apply` |
| Export on a scene whose CRS cannot be resolved | 422 `crs_missing` |
| Delete on a running Action | 409 |

The export case is worth understanding rather than just catching. Vendables carry no spatial
reference - every GeoTIFF an Action writes has an identity transform - so CRS and affine are
recovered at export time by re-reading the raw scene through `app/georef/`. When that fails,
422 is deliberate: a bundle with no projection would be disqualified downstream, so refusing
is better than shipping one.

## Decisions

**Split by question, not by size.** The alternative was cutting at line boundaries into
`actions_1.py`, `actions_2.py`. Grouping by what an endpoint is *for* means a reader looking
for "how does the UI get its PNG" opens one file and finds all three answers.

**Export runs synchronously in the api process.** Unlike project export, which is a queued
job. That is why the supposedly lightweight api image carries geopandas, fiona, shapely and
rasterio. It is a real trade-off, not an oversight: the bundle is small and users expect a
download, not a job to poll. Revisit it if scenes get much larger.

## Verification

The split was checked by capturing the full route table before any change and diffing it
after: 70 routes, zero difference, checked twice. That proves the refactor is
behaviour-preserving at the routing layer, which import checks alone cannot.

Do the same if you move an endpoint. `backend/` has no behavioural tests for these handlers -
only the import smoke test in `tests/test_backend/` - so the route table is the strongest
signal available.
