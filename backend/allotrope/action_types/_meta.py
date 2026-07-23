"""ActionTypeMeta — the descriptive payload every action type ships.

Single source of truth for everything the UI needs to display about an
action type: human-friendly labels, a long-form description for the
Action card, the input contract (what kinds of references the Action
accepts), the output manifest (what artifacts it produces), and the
sensor-keyed default configurations that seed system ActionTemplates.

The api exposes this verbatim via `GET /action-types` (Step 12c).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionInputSpec(BaseModel):
    """One reference an Action accepts as input.

    `ref_kind` decides what entity the input points at:
    - "scene"          → scene_<uuid>; the workspace's bound Scene by default
    - "action_output"  → output_<uuid>; produced by a prior Action in the same Project
    - "annotation"     → annotation_<uuid>; attached to the Scene
    """

    key: str = Field(..., description="Stable identifier used inside configuration JSON")
    label: str = Field(..., description="Human-readable label for the input picker")
    description: str = Field("", description="One-line caption shown in the picker")
    ref_kind: Literal["scene", "action_output", "annotation"]
    producing_action_types: list[str] = Field(
        default_factory=list,
        description=(
            "If `ref_kind` is `action_output`, restrict the picker to outputs "
            "produced by these action types. Empty = any action type."
        ),
    )
    required: bool = True
    cardinality_min: int = 1
    cardinality_max: int = 1


class ActionOutputSpec(BaseModel):
    """One artifact the Action produces under its output directory.

    `artifact_type` drives downstream rendering choices in the workspace.
    """

    key: str = Field(..., description="Stable identifier used inside configuration JSON")
    label: str = Field(..., description="Human-readable label for the Output viewer")
    description: str = Field("", description="One-line caption")
    artifact_type: Literal["raster", "vector", "pickle", "json", "image", "directory"]
    filename: str = Field(
        ...,
        description=(
            "Filename relative to the action's output directory: "
            "`projects/<project_id>/actions/<action_id>/output/<filename>`"
        ),
    )
    always_present: bool = True


class ActionTypeMeta(BaseModel):
    """Full descriptive payload for an action type."""

    type: str = Field(..., description="Stable type identifier, e.g. `band_filter_apply`")
    label: str = Field(..., description="Human-readable name, e.g. `Apply spectral band filter`")
    short_description: str = Field(
        ...,
        description="One sentence — used in the picker and Action list rows.",
    )
    description: str = Field(
        ...,
        description=(
            "Two to four paragraphs. Surfaced verbatim on the Action card "
            "in the workspace center pane. Markdown OK."
        ),
    )
    when_to_use: str = Field(
        "",
        description="Plain-language guidance shown beside the picker.",
    )
    inputs: list[ActionInputSpec] = Field(default_factory=list)
    outputs: list[ActionOutputSpec] = Field(default_factory=list)

    accepted_sensor_types: list[str] = Field(
        ...,
        description=(
            "Which Scene sensor types the action applies to. The picker "
            "is gated on the workspace Scene's sensor."
        ),
    )

    default_config_per_sensor: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "One default configuration body per applicable sensor. The "
            "seed-action-templates CLI (Step 12e) materialises these into "
            "ActionTemplate rows with `is_system=True`."
        ),
    )
