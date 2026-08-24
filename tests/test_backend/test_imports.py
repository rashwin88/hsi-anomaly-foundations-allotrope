"""
Smoke test: every backend entry point must import.

This exists because two total outages reached main undetected, both of them a
plain ImportError:

  - anomaly_scoring imported a PixelStatsOverride that was never defined, so
    every scoring job failed.
  - envi_helper and hotsat_helper imported ENVIFileComponents and
    HotSatFileComponents, which had never existed, so the WORKER COULD NOT
    START - killing scene onboarding, every action run and every export.

Neither was visible to reading, and the second was invisible to monitoring too:
the api's /healthz/db stays green while the worker is dead, because the
lazy-import rule keeps worker-only modules out of the api's startup path.
Nothing imported the worker except the worker.

These tests are deliberately shallow. They do not check behaviour - they check
that the module graph resolves, which is the failure mode that has actually bitten
us. They run in milliseconds and need no database.
"""

import importlib

import pytest

# The worker's real startup chain. runner imports claim, which imports handlers,
# which imports every job handler - so importing these covers the whole tree
# that failed before.
WORKER_MODULES = [
    "allotrope_worker.runner",
    "allotrope_worker.handlers",
    "allotrope_worker.claim",
    "allotrope_worker.heartbeat",
    "allotrope_worker.reaper",
    "allotrope_worker.cleanup",
    "allotrope_worker.action_run",
    "allotrope_worker.scene_onboard",
    "allotrope_worker.annotation_attach",
    "allotrope_worker.project_export",
    "allotrope_worker.visualizations",
]

# Sensor file helpers. The AVIRIS-NG and HotSat ones are where the enums went
# missing; they reach the worker only via scene_onboard's builder imports.
SENSOR_MODULES = [
    "app.utils.files.he5_helper",
    "app.utils.files.tif_helper",
    "app.utils.files.enmap_helper",
    "app.utils.files.envi_helper",
    "app.utils.files.hotsat_helper",
    "app.utils.dataset_builder.prisma_dataset_builder",
    "app.utils.dataset_builder.enmap_dataset_builder",
    "app.utils.dataset_builder.landsat_dataset_builder",
    "app.utils.dataset_builder.aviris_ng_dataset_builder",
    "app.utils.dataset_builder.hotsat_dataset_builder",
]


@pytest.mark.parametrize("module", WORKER_MODULES)
def test_worker_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("module", SENSOR_MODULES)
def test_sensor_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_action_type_registry_loads() -> None:
    """The api imports every action type at startup; a bad one takes it down."""
    from allotrope import action_types

    catalog = action_types.public_catalog()
    assert len(catalog) > 0, "action type registry is empty"


@pytest.mark.parametrize(
    "kind",
    [
        "band_filter_apply",
        "scene_segmentation",
        "cloud_mask",
        "anomaly_scoring",
        "anomaly_detection_prep",
        "spectral_library_match",
    ],
)
def test_action_type_contract(kind: str) -> None:
    """
    Each action type must expose the full contract, and its heavy `_<kind>_run`
    sibling must import.

    Calling spec.run() is what surfaced the anomaly_scoring break in production,
    because the heavy imports live inside run(). We cannot call it without a
    scene, so instead we import the sibling module directly - which executes the
    same import statements at module scope.
    """
    from allotrope import action_types

    spec = action_types.get_spec(kind)
    for attr in ("KIND", "META", "validate_config", "run", "summarize", "preview"):
        assert hasattr(spec, attr), f"{kind} is missing {attr}"

    # Import the heavy sibling if there is one. This is the check that would
    # have caught the PixelStatsOverride break.
    try:
        importlib.import_module(f"allotrope.action_types._{kind}_run")
    except ModuleNotFoundError as exc:
        if f"_{kind}_run" not in str(exc):
            raise  # a real missing dependency, not an absent sibling
