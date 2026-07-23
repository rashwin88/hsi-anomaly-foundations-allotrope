"""Action type registry (abstractions-spec § 5.5, CC-13a equivalent).

Each action type is a module under this package that exposes:

    KIND                     str
    META                     ActionTypeMeta
    validate_config          (raw_cfg: dict, scene: Scene) -> dict
    run                      (ctx: ActionRunContext) -> RunResult        [Step 12d]
    summarize                (ctx, output_dir) -> dict                   [Step 12d]
    preview                  (ctx, output_dir) -> Path                   [Step 12d]

The registry maps KIND strings to the module objects themselves; the api
uses it for submit-time validation, the worker uses it for dispatch.

Adding a new type:
    1. Create `<kind>.py` exposing the seven attributes above.
    2. Import it below and add it to REGISTRY.
    3. Author the META payload (label, description, inputs, outputs,
       sensor compatibility, default config per sensor) — that's what
       drives the Action card and the picker.

Both the api and worker import this module. Don't put torch / rasterio /
scipy / Pillow imports at module top-level — only inside the `run`,
`summarize`, `preview` functions, so the api process (which doesn't
execute action recipes) doesn't pay the import cost.
"""

from __future__ import annotations

from typing import Any, Protocol

from . import (
    anomaly_detection_prep,
    anomaly_scoring,
    band_filter_apply,
    cloud_mask,
    scene_segmentation,
    spectral_library_match,
)
from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


class ActionTypeSpec(Protocol):
    """Per-type contract.

    Each type module is a flat collection of constants + functions;
    we don't construct instances of it. `Protocol` (not abstract base
    class) keeps the modules lightweight.
    """

    KIND: str
    META: ActionTypeMeta

    @staticmethod
    def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
        """API-side validation. Parse `raw_cfg` through this type's Pydantic
        config schema and return the validated, defaulted dict ready to
        serialize into `actions.configuration`.

        Raise `pydantic.ValidationError` (which the api translates to 422)
        for shape errors. Raise `ValueError` for cross-field semantic
        problems (e.g., `input_band_filter_output_id` references an
        ActionOutput from a different Project).
        """
        ...


REGISTRY: dict[str, "ActionTypeSpec"] = {
    band_filter_apply.KIND: band_filter_apply,                # type: ignore[dict-item]
    scene_segmentation.KIND: scene_segmentation,              # type: ignore[dict-item]
    anomaly_scoring.KIND: anomaly_scoring,                    # type: ignore[dict-item]
    cloud_mask.KIND: cloud_mask,                              # type: ignore[dict-item]
    anomaly_detection_prep.KIND: anomaly_detection_prep,      # type: ignore[dict-item]
    spectral_library_match.KIND: spectral_library_match,      # type: ignore[dict-item]
}


def get_spec(kind: str) -> "ActionTypeSpec":
    """Look up a type module by KIND. Raises KeyError on unknown kind."""
    spec = REGISTRY.get(kind)
    if spec is None:
        raise KeyError(f"unknown action type: {kind!r}")
    return spec  # type: ignore[return-value]


def supported_kinds() -> list[str]:
    """Sorted list of registered KIND strings."""
    return sorted(REGISTRY.keys())


def public_catalog() -> list[dict[str, Any]]:
    """Wire shape for the api `/action-types` endpoint.

    Pydantic round-trips the META payload through `model_dump(mode="json")`
    so tuples (e.g., `exclusion_ranges`) come out as lists and nested
    BaseModels collapse to plain dicts.
    """
    return [spec.META.model_dump(mode="json") for spec in REGISTRY.values()]


__all__ = [
    "ActionInputSpec",
    "ActionOutputSpec",
    "ActionTypeMeta",
    "ActionTypeSpec",
    "REGISTRY",
    "get_spec",
    "public_catalog",
    "supported_kinds",
]
