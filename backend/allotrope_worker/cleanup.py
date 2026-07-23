"""Post-job memory reclamation.

After an action_run completes, the worker process still holds:
  - the unpickled VendableDataset (PRISMA pickles are ~1.5 GiB)
  - the last torch model + decoder weights
  - per-batch numpy / torch staging tensors
  - whatever malloc arena state glibc decided to retain

Python's gc handles the object graph, but glibc's malloc doesn't return
freed pages to the OS — RSS stays high until the process exits. On
Linux we can force the arena to release via `malloc_trim(0)`. On macOS
the equivalent isn't reliable (jemalloc / libmalloc don't expose it);
the gc + torch sweep still helps, the trim call is just a no-op.

Costs: gc.collect() is ~10–40 ms on a freshly-completed action.
malloc_trim(0) is ~20–80 ms when it has work to do, free when it doesn't.
Combined per-job overhead is well under 100 ms — invisible next to a
minute-plus inference run.

Call sites:
  - runner._process_one_tick after each terminal transition (success and
    failure both — failure leaks tensors just as readily).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import sys

logger = logging.getLogger("allotrope.worker.cleanup")


def _try_malloc_trim() -> bool:
    """Best-effort: ask glibc to release unused arena pages to the OS.

    Returns True iff the call actually executed. macOS / musl / non-glibc
    Linux builds return False without raising — the loader simply can't
    find a libc that exposes malloc_trim.
    """
    if sys.platform == "darwin":
        # libmalloc on macOS doesn't expose an equivalent. Skip silently.
        return False
    libc_path = ctypes.util.find_library("c")
    if libc_path is None:
        return False
    try:
        libc = ctypes.CDLL(libc_path)
        if not hasattr(libc, "malloc_trim"):
            return False
        libc.malloc_trim(0)
        return True
    except OSError:
        return False


def _try_torch_empty_cache() -> None:
    """Drop torch's CUDA / MPS caching allocator. No-op if cpu-only."""
    try:
        import torch  # imported lazily — worker may not have torch loaded
    except ImportError:
        return
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass
    try:
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


def reclaim_memory(reason: str = "post_job") -> None:
    """Sweep gc, drop torch caches, then ask glibc to release pages.

    Safe to call from any thread; cheap enough to call every tick.
    """
    collected = gc.collect()
    _try_torch_empty_cache()
    trimmed = _try_malloc_trim()
    logger.debug(
        "reclaim_memory reason=%s gc_collected=%d malloc_trim=%s",
        reason,
        collected,
        "yes" if trimmed else "skipped",
    )
