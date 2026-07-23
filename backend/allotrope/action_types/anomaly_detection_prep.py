"""Action type: ``anomaly_detection_prep`` — interactive threshold exploration.

Takes the per-pixel score maps from an upstream ``anomaly_scoring``
Output (one map per algorithm) and produces a single **composite score
map** that the user explores live in the viewer:

  1. Worker rescales each algorithm's score map to [0, 1] via min-max
     over its valid pixels, weighted-combines them using the
     user-supplied ``algorithm_weights``, and writes
     ``composite_score.tif`` + ``composite_score.png``.
  2. Action transitions to status ``"needs_threshold"`` (NOT
     ``"complete"`` — this is the only action type today that does so,
     declared via ``TERMINAL_STATUS`` for the action_run dispatcher).
  3. User opens the viewer, drives a threshold slider + dilation knob,
     presses **Apply**. Each Apply call hits the api's
     ``POST /actions/{id}/anomaly_detection_preview`` endpoint which
     binarises the composite, optionally dilates, and recomputes
     precision/recall/F1 against the attached ground truth (if any).
  4. The mask + metrics returned by Apply are **ephemeral** —
     recomputed each call, never written to the action output dir.

Finalisation (locking the threshold + writing the canonical
``anomaly_mask.tif``) is deferred to a separate ``anomaly_detection_commit``
action that can only fire from a prep that's in ``needs_threshold``. See
ROADMAP Step 14.6.

Output layout under
    projects/<project_id>/actions/<action_id>/output/

    composite_score.tif       float32, the rescaled-and-combined map
    composite_score.png       colormapped preview (inferno + alpha mask)
    rgb.png                   copy of the upstream RGB so this action
                              is self-contained
    summary.json              algorithms used, weights, composite
                              distribution stats, has_gt flag
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "anomaly_detection_prep"

# The action_run worker dispatcher reads this module-level attribute
# to decide the terminal status after a successful run. Defaults to
# "complete" for every other action type — this one breaks the pattern
# because the user still has work to do in the viewer.
TERMINAL_STATUS = "needs_threshold"


# --- Per-type config schema ---------------------------------------------


class AnomalyDetectionPrepConfig(BaseModel):
    """Validated configuration body for an ``anomaly_detection_prep`` Action.

    The configuration is fully set at *submit time* in the
    NewActionDialog — there are no defaults that the user discovers
    later. The threshold + dilation knob live in the viewer's local
    state, NOT in this config (they're passed to the Apply endpoint
    fresh on every click).
    """

    model_config = ConfigDict(extra="forbid")

    input_anomaly_scoring_output_id: str = Field(
        ...,
        description=(
            "Wire-format ``output_<uuid>`` of the upstream "
            "``anomaly_scoring`` ActionOutput whose per-algorithm score "
            "maps will be rescaled and combined. The producing Action "
            "must be in the same Project."
        ),
    )
    algorithm_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-algorithm weight, keyed by codename. Non-negative "
            "floats. Normalised by their sum during the combine so the "
            "composite stays in [0, 1]. Empty dict = equal weights "
            "across all algorithms in the upstream output."
        ),
    )
    input_annotation_id: str | None = Field(
        default=None,
        description=(
            "Optional wire-format ``annotation_<uuid>`` of a ground-truth "
            "annotation attached to the bound Scene. When set, the Apply "
            "endpoint computes precision/recall/F1 against this mask."
        ),
    )


# --- Sensor-keyed defaults ------------------------------------------------

# The recipe is sensor-agnostic — the upstream anomaly_scoring already
# baked in the sensor-specific normalisation. Same default body for all
# accepted sensors; the dialog computes equal weights from the upstream
# at submit time.
_DEFAULT_BODY: dict[str, Any] = {
    "algorithm_weights": {},
    "input_annotation_id": None,
}


# --- META ---------------------------------------------------------------

META = ActionTypeMeta(
    type=KIND,
    label="Anomaly detection (prep)",
    short_description=(
        "Combine per-algorithm anomaly scores into one composite, then "
        "explore thresholds live with precision/recall feedback."
    ),
    description=(
        "Takes the per-pixel anomaly scores from a prior ``anomaly_scoring`` "
        "run (potentially one map per algorithm — Indradhanu, MNF-RX, "
        "Chakshu, …) and assembles them into a single **composite score** "
        "the user can threshold interactively.\n\n"
        "The worker phase: load each algorithm's score map, rescale it "
        "into [0, 1] using its own per-scene min/max over valid pixels "
        "(so a Chakshu reconstruction score in [0, 1] and an MNF-RX "
        "Mahalanobis distance in the thousands become comparable), then "
        "compute a weighted sum across algorithms using the user's "
        "``algorithm_weights``. Weights default to equal across all "
        "algorithms; a weight of 0 effectively drops that algorithm "
        "without re-running anything.\n\n"
        "After the worker writes ``composite_score.tif``, the action "
        "lands in ``needs_threshold`` status. The user opens a viewer "
        "with three side-by-side panels — RGB · composite · binary "
        "anomaly preview — and a threshold slider. Moving the slider "
        "is inert; pressing **Apply** ships the chosen threshold "
        "+ dilation knob to a stateless preview endpoint that returns "
        "the binary mask PNG plus precision/recall/F1 against the "
        "attached ground truth (when present).\n\n"
        "Finalising a threshold choice into a canonical output is a "
        "separate ``anomaly_detection_commit`` action that can only be "
        "triggered from this prep — see roadmap step 14.6."
    ),
    when_to_use=(
        "Run after at least one ``anomaly_scoring`` action has produced "
        "score maps for the scene. The more algorithms in the upstream "
        "run, the more interesting this combination step becomes."
    ),
    inputs=[
        ActionInputSpec(
            key="input_anomaly_scoring_output_id",
            label="Upstream anomaly scoring output",
            description=(
                "The ``anomaly_scoring`` ActionOutput whose per-algorithm "
                "score maps will be combined."
            ),
            ref_kind="action_output",
            producing_action_types=["anomaly_scoring"],
            required=True,
        ),
        ActionInputSpec(
            key="input_annotation_id",
            label="Ground-truth annotation (optional)",
            description=(
                "If attached, the Apply endpoint computes precision / "
                "recall / F1 against this mask."
            ),
            ref_kind="annotation",
            required=False,
        ),
    ],
    outputs=[
        ActionOutputSpec(
            key="composite_score_raster",
            label="Composite score (raster)",
            description=(
                "Float32 composite anomaly score per pixel, rescaled to "
                "[0, 1] and weight-combined across algorithms."
            ),
            artifact_type="raster",
            filename="composite_score.tif",
        ),
        ActionOutputSpec(
            key="composite_score_preview",
            label="Composite score preview",
            description=(
                "Inferno-colormapped preview of the composite score with "
                "an alpha channel masking invalid pixels."
            ),
            artifact_type="image",
            filename="composite_score.png",
        ),
        ActionOutputSpec(
            key="rgb_preview",
            label="RGB reference",
            description=(
                "Copy of the upstream RGB render so the prep viewer is "
                "self-contained — survives even if the upstream action "
                "is deleted."
            ),
            artifact_type="image",
            filename="rgb.png",
        ),
        ActionOutputSpec(
            key="summary",
            label="Summary",
            description=(
                "Algorithms used, normalised weights, composite "
                "distribution stats (min / p50 / p99 / max), has_gt flag, "
                "absolute-to-percentile mapping for the slider."
            ),
            artifact_type="json",
            filename="summary.json",
        ),
    ],
    # Sensor-agnostic — the upstream anomaly_scoring already speaks for
    # the sensor. We accept anything ``anomaly_scoring`` accepted.
    accepted_sensor_types=[
        "prisma",
        "enmap",
        "landsat9",
        "aviris_ng",
        "hotsat1",
    ],
    default_config_per_sensor={
        "prisma": _DEFAULT_BODY,
        "enmap": _DEFAULT_BODY,
        "landsat9": _DEFAULT_BODY,
        "aviris_ng": _DEFAULT_BODY,
        "hotsat1": _DEFAULT_BODY,
    },
)


# --- API-side validation ------------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Parse + lightweight semantic checks.

    Cross-Output semantic checks (the upstream output_id belongs to the
    bound Project, the annotation belongs to the bound Scene) live in
    the api submit endpoint — same pattern as the other action types.
    """
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = AnomalyDetectionPrepConfig.model_validate(raw_cfg)
    # Weights must be non-negative.
    for codename, w in (parsed.algorithm_weights or {}).items():
        if w < 0:
            raise ValueError(
                f"algorithm_weights[{codename!r}] = {w} — weights must be >= 0"
            )
    # At least one weight must be non-zero (when weights are supplied).
    if parsed.algorithm_weights and not any(
        w > 0 for w in parsed.algorithm_weights.values()
    ):
        raise ValueError(
            "algorithm_weights are all zero — at least one algorithm "
            "must have a positive weight to produce a meaningful composite."
        )
    return parsed.model_dump(mode="json")


# --- run / summarize / preview lazy-dispatch ---------------------------


def run(ctx: Any) -> None:
    from ._anomaly_detection_prep_run import run as _run

    return _run(ctx)


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    from ._anomaly_detection_prep_run import summarize as _summarize

    return _summarize(ctx, output_dir)


def preview(ctx: Any, output_dir: Any) -> Any:
    from ._anomaly_detection_prep_run import preview as _preview

    return _preview(ctx, output_dir)
