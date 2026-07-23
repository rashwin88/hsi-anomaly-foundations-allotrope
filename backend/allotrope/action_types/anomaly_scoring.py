"""Action type: `anomaly_scoring` — multi-model HSI / thermal scoring.

A single Action selects one OR MORE foundation models (by codename),
runs each through its inferencer on the (possibly band-filtered) cube,
and emits a per-model anomaly *score* raster + the model's
*reconstruction* cube. When the bound Scene has a ground-truth
annotation attached, the worker also computes per-model ROC + AUC.

This is *scoring only* — no thresholding, no binary detections.
The Output viewer renders score / reconstruction / RGB side-by-side
so the human can read off where each model thinks the scene deviates
from "normal."

Inputs (all carried inside `actions.configuration`):

  - input_band_filter_output_id   required for HSI scenes — filtered
                                  vendable pickle (forbidden on thermal)
  - input_scene_segmentation_output_id   optional, HSI only — restricts
                                  scoring to `keep_mask`
  - input_annotation_id           optional — triggers ROC computation
  - model_codenames               required — list of 1+ codenames
  - model_overrides               optional — per-codename dict of
                                  {scoring_method, patch_size, stride,
                                   batch_size, sam_l1_alpha} fields,
                                  any subset; missing fields fall
                                  back to the capability defaults

Output layout under
    projects/<project_id>/actions/<action_id>/output/

    summary.json
    diagnostics.json                rich payload (lazy-loaded by viewer)
    preview.png                     thumbnail (RGB + heatmaps; small)
    roc.json                        per-model FPR/TPR/AUC (only when GT)
    rgb.png                         the scene rendered RGB at full res
    models/<codename>/
        anomaly_score.tif           float32 raster
        anomaly_score.png           pre-rendered viridis preview
        reconstruction.tif          float32 cube (C, H, W); HSI=many bands
        reconstruction.png          RGB-rendered preview of the recon
        stats.json                  per-model timing + score range

`run` / `summarize` / `preview` are implemented in
`_anomaly_scoring_run.py` to keep this module light on import (api
process doesn't pay for torch / rasterio / matplotlib).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "anomaly_scoring"


# --- Per-type config schema ---------------------------------------------


_VALID_METHODS = {"L1", "MSE", "SAM", "combined", "mahalanobis"}


class ModelOverride(BaseModel):
    """Per-codename optional overrides. Every field is optional —
    leaving a field unset means "use the model's capability default."
    """

    model_config = ConfigDict(extra="forbid")

    scoring_method: str | None = Field(
        default=None,
        description="L1 | MSE | SAM | combined. Must be supported by the model.",
    )
    patch_size: int | None = Field(default=None, ge=32, le=1024)
    stride: int | None = Field(default=None, ge=8, le=1024)
    batch_size: int | None = Field(default=None, ge=1, le=128)
    sam_l1_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weight on L1 in the 'combined' score: "
            "score = alpha * L1_norm + (1 - alpha) * SAM_norm. "
            "Ignored unless scoring_method='combined'."
        ),
    )
    erosion_kernel_size: int | None = Field(
        default=None,
        ge=1,
        le=129,
        description=(
            "Kernel size (odd integer) for eroding the validity mask "
            "before reconstruction accumulation. Pixels within "
            "kernel_size//2 of any invalid pixel are excluded from "
            "scoring. Default per the model's InferenceConfig is 15 "
            "(7-pixel buffer). Larger values = wider edge exclusion. "
            "SegFormer-MAE family only; ignored by autoencoder "
            "architectures and classical detectors."
        ),
    )
    keep_mask_erosion_kernel_size: int | None = Field(
        default=None,
        ge=1,
        le=129,
        description=(
            "Kernel size (odd integer) for eroding the upstream "
            "keep_mask (scene_segmentation / cloud_mask output) before "
            "it's applied to the score. Strips off the boundary-rim "
            "artifact where cloud/water edges otherwise score very "
            "high. Applies to both foundation (eroded mask fed to "
            "compute_score) and classical paths (eroded mask AND'd "
            "into _spatial_mask between fit and detect). Default 1 = "
            "no erosion. Try 7 if you see bright rings tracing the "
            "segmentation boundary."
        ),
    )


class AnomalyScoringConfig(BaseModel):
    """Validated configuration body for an `anomaly_scoring` Action."""

    model_config = ConfigDict(extra="forbid")

    input_band_filter_output_id: str | None = Field(
        default=None,
        description=(
            "ActionOutput id from a `band_filter_apply` Action in the "
            "same Project. REQUIRED on hyperspectral scenes; MUST BE "
            "ABSENT on thermal scenes."
        ),
        pattern=r"^output_[0-9a-fA-F-]{36}$",
    )

    input_scene_segmentation_output_id: str | None = Field(
        default=None,
        description=(
            "Optional ActionOutput id from a `scene_segmentation` "
            "Action in the same Project. HSI only — when set, anomaly "
            "score is computed only on its keep_mask."
        ),
        pattern=r"^output_[0-9a-fA-F-]{36}$",
    )

    input_cloud_mask_output_id: str | None = Field(
        default=None,
        description=(
            "Optional ActionOutput id from a `cloud_mask` Action in "
            "the same Project. Thermal only — when set, anomaly score "
            "is computed only on its keep_mask (clear, valid pixels)."
        ),
        pattern=r"^output_[0-9a-fA-F-]{36}$",
    )

    input_annotation_id: str | None = Field(
        default=None,
        description=(
            "Optional Annotation id (annotation_<uuid>) on the bound "
            "Scene. When set, the worker computes a per-model ROC + AUC."
        ),
        pattern=r"^annotation_[0-9a-fA-F-]{36}$",
    )

    model_codenames: list[str] = Field(
        ...,
        description=(
            "Codenames of the models to run. Order is preserved in the "
            "Output viewer's per-model tabs."
        ),
        min_length=1,
    )

    model_overrides: dict[str, ModelOverride] = Field(
        default_factory=dict,
        description=(
            "Optional per-codename overrides for scoring method, "
            "patch_size, stride, batch_size, and sam_l1_alpha. Codenames "
            "absent from this dict use the capability defaults."
        ),
    )


# --- Sensor-keyed defaults -------------------------------------------------

_DEFAULT_PRISMA: dict[str, Any] = {
    "model_codenames": ["Indradhanu"],
    "model_overrides": {},
}
_DEFAULT_ENMAP: dict[str, Any] = {
    "model_codenames": ["Indradhanu"],
    "model_overrides": {},
}
_DEFAULT_LANDSAT9: dict[str, Any] = {
    "model_codenames": ["Chakshu"],
    "model_overrides": {},
}
_DEFAULT_AVIRIS_NG: dict[str, Any] = {
    "model_codenames": ["Indradhanu"],
    "model_overrides": {},
}
_DEFAULT_HOTSAT1: dict[str, Any] = {
    # HotSat is single-band thermal but ships uncalibrated DN, not
    # Kelvin. The action handler overrides Chakshu's pixel-norm stats
    # with per-scene (mean, std) so the input distribution matches what
    # the model was trained against.
    "model_codenames": ["Chakshu"],
    "model_overrides": {},
}


# --- META ---------------------------------------------------------------

META = ActionTypeMeta(
    type=KIND,
    label="Anomaly scoring",
    short_description=(
        "Run one or more foundation models on the cube. Per-model "
        "anomaly score raster + reconstruction · zoomable side-by-side "
        "viewer · optional ROC vs GT."
    ),
    description=(
        "The headline scoring action. Loads the (filtered, on HSI) "
        "vendable, fans out across every selected model codename: load "
        "the checkpoint, run patch-based two-pass reconstruction, "
        "compute the per-pixel anomaly score (L1 · SAM · combined · "
        "MSE — defaulted per model, overridable per codename).\n\n"
        "When a scene_segmentation Output is attached the score is "
        "masked to its keep_mask. When a ground-truth annotation is "
        "attached on the Scene, the worker computes a per-model "
        "ROC + AUC against it.\n\n"
        "Output is a per-model triplet (score raster + reconstruction "
        "cube + stats), pre-rendered preview PNGs, a side-by-side "
        "zoomable viewer in the UI, and a diagnostics file with score "
        "distributions and (if GT) ROC arrays. No thresholding — this "
        "is the raw scoring pass; downstream actions can threshold."
    ),
    when_to_use=(
        "After band_filter_apply (HSI) and ideally after "
        "scene_segmentation (recommended). Pick one model to scout, or "
        "several to compare scoring behaviour on the same scene."
    ),
    inputs=[
        ActionInputSpec(
            key="input_band_filter_output_id",
            label="Filtered vendable (HSI only)",
            description=(
                "Hyperspectral scenes: required — pick a completed "
                "band_filter_apply Output. Thermal scenes: not used."
            ),
            ref_kind="action_output",
            producing_action_types=["band_filter_apply"],
            required=False,
        ),
        ActionInputSpec(
            key="input_scene_segmentation_output_id",
            label="Keep mask · scene segmentation (HSI)",
            description=(
                "Optional, HSI only. Output of a scene_segmentation "
                "Action in this project. When set, anomaly score is "
                "computed only on its keep_mask."
            ),
            ref_kind="action_output",
            producing_action_types=["scene_segmentation"],
            required=False,
        ),
        ActionInputSpec(
            key="input_cloud_mask_output_id",
            label="Keep mask · cloud mask (thermal)",
            description=(
                "Optional, thermal only. Output of a cloud_mask "
                "Action in this project. When set, anomaly score is "
                "computed only on its keep_mask (clear, valid pixels)."
            ),
            ref_kind="action_output",
            producing_action_types=["cloud_mask"],
            required=False,
        ),
        ActionInputSpec(
            key="input_annotation_id",
            label="GT annotation (optional)",
            description=(
                "Optional. A raster annotation on the bound Scene. When "
                "set, the worker computes per-model ROC + AUC against it."
            ),
            ref_kind="annotation",
            required=False,
        ),
    ],
    outputs=[
        ActionOutputSpec(
            key="preview",
            label="Heatmap montage (thumbnail)",
            description=(
                "Static RGB + heatmap composite. The interactive "
                "zoomable viewer is built from the per-model rasters."
            ),
            artifact_type="image",
            filename="preview.png",
        ),
        ActionOutputSpec(
            key="diagnostics",
            label="Diagnostics",
            description=(
                "Per-model score distribution, runtime, sensor + arch "
                "tags, override resolution, and (when GT attached) ROC "
                "curve data."
            ),
            artifact_type="json",
            filename="diagnostics.json",
        ),
        ActionOutputSpec(
            key="anomaly_score_dir",
            label="Per-model artifacts",
            description=(
                "Directory: models/<codename>/{anomaly_score.tif, "
                "anomaly_score.png, reconstruction.tif, reconstruction.png, "
                "stats.json}."
            ),
            artifact_type="directory",
            filename="models",
        ),
        ActionOutputSpec(
            key="roc",
            label="ROC curves",
            description=(
                "Per-model FPR / TPR / AUC. Present only when a GT "
                "annotation was attached at submit time."
            ),
            artifact_type="json",
            filename="roc.json",
            always_present=False,
        ),
    ],
    accepted_sensor_types=["prisma", "enmap", "landsat9", "aviris_ng", "hotsat1"],
    default_config_per_sensor={
        "prisma": _DEFAULT_PRISMA,
        "enmap": _DEFAULT_ENMAP,
        "landsat9": _DEFAULT_LANDSAT9,
        "aviris_ng": _DEFAULT_AVIRIS_NG,
        "hotsat1": _DEFAULT_HOTSAT1,
    },
)


# --- API-side validation -----------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Parse `raw_cfg` through the schema + cross-check that:

      - each model_codename resolves AND accepts `sensor_type`
      - every model_overrides key is one of the picked codenames
      - per-model overrides are valid for that model (scoring_method
        in its supported set, patch_size in its valid_patch_sizes)
    """
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = AnomalyScoringConfig.model_validate(raw_cfg)

    from pathlib import Path

    from ..config import settings
    from ..foundation_models.resolver import list_catalog, sensor_family

    catalog = list_catalog(Path(settings.models_dir))
    by_codename = {m.codename.lower(): m for m in catalog}
    scene_family = sensor_family(sensor_type)

    picked: dict[str, Any] = {}
    for cname in parsed.model_codenames:
        m = by_codename.get(cname.strip().lower())
        if m is None:
            raise ValueError(
                f"unknown model codename {cname!r}; available: "
                f"{sorted(model.codename for model in catalog)}"
            )
        if m.sensor != scene_family:
            raise ValueError(
                f"model {cname!r} (family={m.sensor}) does not accept "
                f"scene sensor {sensor_type!r} (family={scene_family})"
            )
        picked[cname] = m

    if scene_family == "hyperspectral":
        if not parsed.input_band_filter_output_id:
            raise ValueError(
                "anomaly_scoring on a hyperspectral scene requires "
                "input_band_filter_output_id (from a completed "
                "band_filter_apply Action)."
            )
    else:
        if parsed.input_band_filter_output_id:
            raise ValueError(
                "anomaly_scoring on a thermal scene must not set "
                "input_band_filter_output_id — band filtering is a "
                "hyperspectral-only step."
            )
        if parsed.input_scene_segmentation_output_id:
            raise ValueError(
                "anomaly_scoring on a thermal scene must not set "
                "input_scene_segmentation_output_id — scene_segmentation "
                "is hyperspectral-only. Use input_cloud_mask_output_id "
                "instead."
            )

    if scene_family == "hyperspectral" and parsed.input_cloud_mask_output_id:
        raise ValueError(
            "anomaly_scoring on a hyperspectral scene must not set "
            "input_cloud_mask_output_id — cloud_mask is thermal-only. "
            "Use input_scene_segmentation_output_id instead."
        )

    for override_codename, override in parsed.model_overrides.items():
        m = picked.get(override_codename)
        if m is None:
            raise ValueError(
                f"model_overrides references codename {override_codename!r} "
                f"that is not in model_codenames"
            )
        if override.scoring_method is not None:
            method = override.scoring_method
            if method not in _VALID_METHODS:
                raise ValueError(
                    f"model_overrides[{override_codename!r}].scoring_method: "
                    f"unknown method {method!r}; valid: {sorted(_VALID_METHODS)}"
                )
            if method not in m.scoring_methods:
                raise ValueError(
                    f"model {override_codename!r} does not support scoring "
                    f"method {method!r}; supported: {list(m.scoring_methods)}"
                )
        if override.patch_size is not None:
            if override.patch_size not in m.valid_patch_sizes:
                raise ValueError(
                    f"model_overrides[{override_codename!r}].patch_size="
                    f"{override.patch_size} not in {list(m.valid_patch_sizes)}"
                )
        if (
            override.stride is not None
            and override.patch_size is not None
            and override.stride > override.patch_size
        ):
            raise ValueError(
                f"model_overrides[{override_codename!r}].stride must be "
                f"<= patch_size"
            )
        # Classical detectors are whole-cube ops — patch / stride /
        # batch / sam_l1_alpha / erosion_kernel_size don't apply.
        # Reject explicitly so the user gets a clear error instead of
        # silently-ignored knobs.
        if m.family == "classical":
            for forbidden in (
                "patch_size", "stride", "batch_size",
                "sam_l1_alpha", "erosion_kernel_size",
            ):
                if getattr(override, forbidden, None) is not None:
                    raise ValueError(
                        f"model_overrides[{override_codename!r}].{forbidden} "
                        f"is not applicable to classical detector "
                        f"{m.architecture!r}"
                    )
        # erosion_kernel_size must be odd — the underlying
        # TokenMasking.erode_mask uses a centered kernel.
        if (
            override.erosion_kernel_size is not None
            and override.erosion_kernel_size % 2 == 0
        ):
            raise ValueError(
                f"model_overrides[{override_codename!r}].erosion_kernel_size "
                f"must be an odd integer (got {override.erosion_kernel_size})"
            )
        # keep_mask_erosion_kernel_size: same odd-integer constraint.
        # Unlike erosion_kernel_size, it applies to BOTH model families
        # — protects against the boundary-rim score artifact along
        # cloud/water/segmentation edges that both foundation and
        # classical detectors produce.
        if (
            override.keep_mask_erosion_kernel_size is not None
            and override.keep_mask_erosion_kernel_size % 2 == 0
        ):
            raise ValueError(
                f"model_overrides[{override_codename!r}]."
                f"keep_mask_erosion_kernel_size must be an odd integer "
                f"(got {override.keep_mask_erosion_kernel_size})"
            )

    return parsed.model_dump(mode="json")


# --- run / summarize / preview lazy-dispatch ---------------------------


def run(ctx: Any) -> None:
    from ._anomaly_scoring_run import run as _run

    return _run(ctx)


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    from ._anomaly_scoring_run import summarize as _summarize

    return _summarize(ctx, output_dir)


def preview(ctx: Any, output_dir: Any) -> Any:
    from ._anomaly_scoring_run import preview as _preview

    return _preview(ctx, output_dir)
