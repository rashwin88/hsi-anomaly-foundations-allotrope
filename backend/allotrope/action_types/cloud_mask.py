"""Action type: `cloud_mask` — thermal-only adaptive cloud masking.

Wraps the existing `app/statistical_models/b10_adaptive_cloud_masker.py`
GMM-based cloud detector into a worker-side Action. The masker fits a
5-component Gaussian mixture to the B10 brightness-temperature
distribution of the scene and labels the cold clusters as cloud.

Output is a binary `cloud_mask.tif` (1 = cloud) plus a canonical
`keep_mask.tif` (1 = keep, defined as `spatial_validity & ¬cloud`) so
downstream `anomaly_scoring` Actions can consume the keep_mask the
same way they consume `scene_segmentation`'s on the HSI side.

This is the thermal twin of `scene_segmentation`. Thermal scenes have
no Red/Green/NIR bands, so the NDVI/NDWI machinery doesn't apply —
adaptive GMM thresholding on B10 is the right tool instead.

Inputs (carried inside `actions.configuration`):

    sampling_ratio   optional, default 0.10 — fraction of valid pixels
                     used to fit the GMM. Lower = faster, higher = more
                     stable on tiny scenes.

Output layout under
    projects/<project_id>/actions/<action_id>/output/

    summary.json
    diagnostics.json
    cloud_mask.tif      uint8 binary, 1 = cloud
    keep_mask.tif       uint8 binary, 1 = keep (clear + valid)
    preview.png         B10 grayscale + cyan cloud overlay
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "cloud_mask"


# --- Per-type config schema ---------------------------------------------


class CloudMaskConfig(BaseModel):
    """Validated configuration body for a `cloud_mask` Action."""

    model_config = ConfigDict(extra="forbid")

    sampling_ratio: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of valid pixels sampled to fit the GMM. The masker "
            "is robust around 0.05–0.20 on PRISMA-sized scenes; raise "
            "for very small thermal scenes, lower if fit time becomes "
            "the bottleneck."
        ),
    )


# --- Sensor-keyed defaults -------------------------------------------------

_DEFAULT_LANDSAT9: dict[str, Any] = {
    "sampling_ratio": 0.10,
}


# --- META ---------------------------------------------------------------

META = ActionTypeMeta(
    type=KIND,
    label="Cloud mask",
    short_description=(
        "Adaptive GMM cloud detection on Landsat 9 B10. Emits a "
        "binary cloud raster + a keep_mask for downstream scoring."
    ),
    description=(
        "Anomaly scoring on thermal scenes is dominated by cold cloud "
        "pixels — they reconstruct as well as the warm land they "
        "occlude, but their residuals end up enormous because the "
        "model rarely sees ice clouds in training. Masking them out "
        "before scoring is the right answer.\n\n"
        "This action runs `B10AdaptiveCloudMasker` against the scene's "
        "onboarding vendable: probe the brightness-temperature "
        "distribution, anchor a 5-component Gaussian mixture against "
        "physically meaningful temperature percentiles, fit on a "
        "sub-sample of valid pixels, then physically verify which GMM "
        "components correspond to cold-cluster (cloud) pixels.\n\n"
        "The action explicitly reads `vendable.pure_validity_mask` "
        "(scene validity *before* onboarding's own cloud step) so the "
        "GMM sees the raw B10 distribution including cold cloud tops. "
        "Without this, the onboarding-stage cloud mask AND-folded into "
        "`validity_cube` would have already removed the very pixels "
        "we're trying to detect.\n\n"
        "Output is a `cloud_mask.tif` (binary, 1 = cloud) and a "
        "canonical `keep_mask.tif` = `pure_validity ∧ ¬cloud` that "
        "downstream `anomaly_scoring` Actions consume the same way "
        "they consume `scene_segmentation`'s keep_mask on the HSI side. "
        "Re-run with a different `sampling_ratio` to tune fit stability."
    ),
    when_to_use=(
        "Run on any Landsat 9 scene before `anomaly_scoring`, "
        "especially over regions with frequent cloud cover. Skip if "
        "the scene is known to be cloud-free."
    ),
    inputs=[
        # Scene is implicit — taken from the bound Project's scene_id at
        # submit time. The masker reads the onboarding vendable directly,
        # so no upstream Action is required.
    ],
    outputs=[
        ActionOutputSpec(
            key="cloud_mask",
            label="Cloud mask",
            description="Binary uint8 raster, 1 = cloud.",
            artifact_type="raster",
            filename="cloud_mask.tif",
        ),
        ActionOutputSpec(
            key="keep_mask",
            label="Keep mask",
            description=(
                "Canonical input for downstream thermal anomaly Actions. "
                "spatial_validity ∧ ¬cloud_mask."
            ),
            artifact_type="raster",
            filename="keep_mask.tif",
        ),
        ActionOutputSpec(
            key="diagnostics",
            label="Diagnostics",
            description=(
                "GMM component count, anchor temperatures, sample "
                "count, dynamic cloud threshold, cloud / kept percentages."
            ),
            artifact_type="json",
            filename="diagnostics.json",
        ),
        ActionOutputSpec(
            key="preview",
            label="Cloud overlay preview",
            description=(
                "B10 grayscale + cyan cloud-mask overlay at scene "
                "resolution."
            ),
            artifact_type="image",
            filename="preview.png",
        ),
    ],
    accepted_sensor_types=["landsat9"],
    default_config_per_sensor={
        "landsat9": _DEFAULT_LANDSAT9,
    },
)


# --- API-side validation ------------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Parse `raw_cfg` through `CloudMaskConfig` + reject non-thermal."""
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = CloudMaskConfig.model_validate(raw_cfg)
    return parsed.model_dump(mode="json")


# --- run / summarize / preview lazy-dispatch ---------------------------


def run(ctx: Any) -> None:
    from ._cloud_mask_run import run as _run

    return _run(ctx)


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    from ._cloud_mask_run import summarize as _summarize

    return _summarize(ctx, output_dir)


def preview(ctx: Any, output_dir: Any) -> Any:
    from ._cloud_mask_run import preview as _preview

    return _preview(ctx, output_dir)
