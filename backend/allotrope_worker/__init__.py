"""Allotrope worker — long-running queue consumer.

Pulls jobs from the Postgres-backed `jobs` table (abstractions-spec § 5.13)
and dispatches to type-specific handlers. One process per worker container;
horizontal scale by replica count (compose's `deploy.replicas`).

Entry point: `python -m allotrope_worker` → `runner.main()`.
Sequence diagram (added in step 6e): final design/diagrams/worker-loop.drawio
"""

# Force the headless matplotlib backend before anything else can import
# matplotlib transitively. The Dockerfile also sets MPLBACKEND=Agg in
# the env; this is belt-and-braces in case someone runs the package
# outside the container.
import os as _os

_os.environ.setdefault("MPLBACKEND", "Agg")
# Cap BLAS thread fan-out; over-subscription has been seen to SIGSEGV
# mid-handler on the existing PRISMA pipeline (numpy + numexpr + torch
# each defaulting to cpu_count() threads).
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, "2")
