# The test harness

**Files:** `scripts/run_tests.ps1`, `scripts/run_tests.sh`, `tests/test_backend/test_imports.py`

## Purpose

How anything gets verified in this repo, why the tests run inside a container, and what the
harness still cannot see.

## Running it

```bash
./scripts/run_tests.sh        # macOS / Linux / WSL
scripts\run_tests.ps1         # Windows PowerShell
```

Expect **103 passed, 51 deselected** in about eight seconds. Extra arguments pass through,
so `scripts\run_tests.ps1 -k spectral -v` works.

## Why it runs in the worker container

The suite imports `app.*`, which needs numpy, torch, rasterio, scipy and h5py. The worker
image already has all of them. Installing that stack on a developer machine is the thing
this repo is built to avoid - `rasterio` alone needs GDAL, and on Windows it may not import
at all under an application-control policy.

`tests/` is deliberately **not** COPY'd into the image. The runner bind-mounts it:

```
-v tests:/srv/tests
-v pytest.ini:/srv/pytest.ini
-v app:/srv/app
-v backend/allotrope:/srv/allotrope
-v backend/allotrope_worker:/srv/allotrope_worker
```

The last three matter more than they look. **Without them the suite tests the image, not
your working tree.** The harness shipped that way initially, and for a while every "suite
still green" was measured against whatever was baked at the last build. It only went
unnoticed because the changes at the time were docstrings.

`--no-deps` keeps Postgres and bootstrap out of it. The `app/` tests touch no database.

## What the markers mean

`pytest.ini` defines three: `large_files`, `large_benchmarks`, `network_access`. The runner
excludes all three.

The 51 deselected tests need `tests/test_payloads/` - real multi-gigabyte scene files that
are gitignored. If you have them locally, run that set with `-m large_files`.

**That marker was lying until recently.** Thirteen tests needed payloads without carrying
the marker, so `-m "not large_files"` promised a runnable subset and did not deliver one:
the baseline was *5 failed, 67 passed, 8 errors*. They are now gated at module level where
every test in the file needs a payload, and per-function in `test_enmap_stac.py`, where
`test_filename_parser_enmap` parses strings only and still passes.

A green baseline is worth more than a bigger one. *"67 passed, 5 failed, 8 errors"* forces a
judgement call after every change about whether a failure is yours; *"103 passed"* does not.

## The backend smoke test

`tests/test_backend/test_imports.py` - 28 tests, no database, about three seconds. It imports
every worker entry point, every sensor helper and builder, and each action type's heavy
`_<kind>_run` sibling.

It exists because two total outages reached `main` undetected, both a plain `ImportError`:

- `anomaly_scoring` imported a `PixelStatsOverride` that was never defined, so every scoring
  job failed.
- `envi_helper` and `hotsat_helper` imported `ENVIFileComponents` and
  `HotSatFileComponents`, neither of which existed anywhere in the repo. **The worker could
  not start at all** - no scene onboarding, no action runs, no exports.

The second is the instructive one. **A dead worker looks like a healthy system.** The api's
`/healthz/db` stays green throughout, because the lazy-import rule keeps `scene_onboard` out
of the api's startup path. Nothing imports the worker except the worker, so nothing noticed.

The tests are deliberately shallow. They assert that the module graph resolves - not that
anything behaves correctly - because that is the failure mode that has actually bitten this
repo twice.

**It was proved capable of failing.** Renaming `ENVIFileComponents` reproduced the outage and
failed six tests, including `allotrope_worker.runner`, in 3.4 seconds. A test that cannot
fail is worth nothing, and the only way to know is to break something on purpose.

## What the harness cannot see

Be clear about the gaps, because green here does not mean safe.

| Not covered | Consequence |
|---|---|
| `backend/` behaviour | Endpoints are import-checked only. Use the route table for API changes |
| `frontend/` entirely | No tests exist. `npm run build` is the only gate |
| `app/foundation_models/` | No trainer, inferencer or architecture is exercised |
| Anything needing a real scene | `vend_dataset`, full-scene inference, real scoring |
| **Text integrity** | See below |

That last row is the one that caught me out. Encoding damage - a UTF-8 BOM, or mojibake from
an ANSI read - lives in comments and docstrings. Python parses it, imports succeed, all 103
tests pass, the route table is byte-identical. Every gate here measures *behaviour*, and
this class of damage does not change behaviour. It went unnoticed across three commits and
nine files.

If you edit files in bulk on Windows, check afterwards:

```powershell
Select-String -Path "app\**\*.py" -Pattern 'â€|Â°|Î¼|Ïƒ|â†'   # mojibake
# and check byte 0 for 0xEF (BOM)
```

## Verification techniques used in this repo

Beyond the suite, three that have earned their place:

**Route-table diff.** Capture `app.routes` before a change and after, and compare. Proved the
`actions.py` split was behaviour-preserving at 70 routes with zero difference.

**Side-by-side bit-identical comparison.** Load the pre-change module from git alongside the
new one, run both over the same inputs, assert equality. Used for the destriper
decomposition: `max_abs_diff = 0` over three scenes.

**Independent reference implementation.** Write the same computation a second way and compare.
`batch_mahalanobis` against plain numpy: `3.55e-15`.

All three prove more than "the tests pass", and all three are cheap. Reach for them when
changing code the suite does not cover - which, in `backend/`, is all of it.

## If you add a test

Put `app/` tests under `tests/` mirroring the source layout. Put backend tests under
`tests/test_backend/`. Mark anything needing `tests/test_payloads/` as `large_files` at
module level if the whole file needs it, per function otherwise.

The single highest-value test still missing is one that starts the stack and asserts all
five containers reach `running`. `exited` and `restarting` both look like progress in the
logs, and a crash-looping worker is invisible to every check here.
