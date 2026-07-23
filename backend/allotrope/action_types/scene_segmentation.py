"""Action type: `scene_segmentation` — spectral indices + class masks.

Lifts § 9 of `benchmarking/hyperspectral/segformer_mae_benchmark.ipynb`
into a worker-side Action: compute NDVI, NDWI, and VNIR brightness from
a filtered cube, then derive water / cloud / shadow / vegetation class
masks via configurable thresholds. Output is a `keep_mask` raster that
downstream Actions (anomaly_scoring, spectral_detection) consume to
restrict scoring to the "interesting" pixel set.

Step 12b ships META + config schema + `validate_config`. The
`run` / `summarize` / `preview` implementations land in Step 12d.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "scene_segmentation"


# --- Per-type config schema ---------------------------------------------


class SpectralIndexBands(BaseModel):
    """Wavelength targets the worker uses to pick the nearest band on the
    common 10 nm grid. Defaults are taken straight from the segformer_mae
    benchmark notebook."""

    model_config = ConfigDict(extra="forbid")

    red_nm: float = Field(default=660, ge=380, le=2500)
    green_nm: float = Field(default=560, ge=380, le=2500)
    nir_nm: float = Field(default=860, ge=380, le=2500)
    vnir_brightness_end_nm: float = Field(
        default=910,
        ge=380,
        le=2500,
        description="Upper edge of the VNIR brightness average (lower edge = first band).",
    )


class ClassMaskThresholds(BaseModel):
    """Threshold values for the four exclusion classes."""

    model_config = ConfigDict(extra="forbid")

    ndwi_water: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="NDWI > this → water.",
    )
    brightness_cloud: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="VNIR brightness > this → cloud.",
    )
    brightness_shadow: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="VNIR brightness < this → shadow.",
    )
    ndvi_vegetation: float = Field(
        default=0.4,
        ge=-1.0,
        le=1.0,
        description="NDVI > this → vegetation.",
    )


class SceneSegmentationConfig(BaseModel):
    """Validated configuration body for a `scene_segmentation` Action."""

    model_config = ConfigDict(extra="forbid")

    # --- Inputs ---
    #
    # `input_scene_id` is intentionally NOT a client-supplied field. It
    # must equal the bound Project's scene, so the api injects it
    # server-side from `project.scene_id` after this Pydantic schema
    # validates. The persisted `actions.configuration` JSONB carries
    # `input_scene_id` for downstream consumers — same shape as
    # band_filter_apply's configuration.

    input_band_filter_output_id: str = Field(
        ...,
        description=(
            "ActionOutput id (output_<uuid>) from a `band_filter_apply` "
            "Action in the same Project. The filtered vendable's "
            "normalized_hyperspectral_cube is the source for index math."
        ),
        pattern=r"^output_[0-9a-fA-F-]{36}$",
    )

    # --- Recipe params ---

    bands: SpectralIndexBands = Field(default_factory=SpectralIndexBands)
    thresholds: ClassMaskThresholds = Field(default_factory=ClassMaskThresholds)

    classes_to_mask: list[str] = Field(
        default_factory=lambda: ["water", "cloud", "shadow", "vegetation"],
        description=(
            "Which exclusion classes contribute to keep_mask. Must be a "
            "subset of {water, cloud, shadow, vegetation}. Empty = "
            "keep_mask is just spatial validity."
        ),
    )


_DEFAULT_PRISMA: dict[str, Any] = {
    "bands": SpectralIndexBands().model_dump(),
    "thresholds": ClassMaskThresholds().model_dump(),
    "classes_to_mask": ["water", "cloud", "shadow", "vegetation"],
}

# EnMAP and PRISMA share the same 10 nm common grid post-band-filter, so
# the wavelength targets and thresholds carry over. Sensor split kept
# explicit for future divergence (e.g. EnMAP-specific NDSI bands).
_DEFAULT_ENMAP: dict[str, Any] = dict(_DEFAULT_PRISMA)

# AVIRIS-NG shares the same 10 nm common grid post-band-filter so the
# index/threshold defaults carry over.
_DEFAULT_AVIRIS_NG: dict[str, Any] = dict(_DEFAULT_PRISMA)


# --- META ---------------------------------------------------------------

META = ActionTypeMeta(
    type=KIND,
    label="Scene segmentation",
    short_description=(
        "Compute NDVI / NDWI / brightness; derive water · cloud · shadow · "
        "vegetation masks; emit a keep_mask for downstream scoring."
    ),
    description=(
        "Anomaly scoring without scene segmentation gets dominated by "
        "*spectrally distinct but uninteresting* pixels — water (strong "
        "negative NIR slope), clouds (saturated brightness), shadows "
        "(near-zero brightness), dense vegetation (red-edge spike). These "
        "drown out real detections in the top-K ranking.\n\n"
        "This action computes three per-pixel spectral indices from the "
        "filtered cube — NDVI = (NIR − Red) / (NIR + Red), NDWI = "
        "(Green − NIR) / (Green + NIR), and brightness = mean VNIR "
        "reflectance — and thresholds each into a binary class mask. "
        "The four class masks (water · cloud · shadow · vegetation) are "
        "unioned and inverted against the spatial validity mask to "
        "produce a single canonical `keep_mask` raster.\n\n"
        "Downstream HSI anomaly Actions consume `keep_mask.tif` as their "
        "scoring domain — anomaly score is computed only where keep_mask "
        "is 1. This is the second canonical preprocessing Action (after "
        "band_filter_apply) for every PRISMA / EnMAP investigation."
    ),
    when_to_use=(
        "Run after band_filter_apply on any HSI scene where you expect "
        "non-target spectral classes (water bodies, cloud remnants, deep "
        "shadow, dense vegetation) to confound anomaly scoring."
    ),
    inputs=[
        # Scene is implicit — taken from the bound Project's scene_id
        # at submit time. The picker only asks for the upstream Action.
        ActionInputSpec(
            key="input_band_filter_output_id",
            label="Filtered vendable",
            description=(
                "Output of a prior `band_filter_apply` Action in this "
                "Project. Its filtered cube is the source for index math."
            ),
            ref_kind="action_output",
            producing_action_types=["band_filter_apply"],
            required=True,
        ),
    ],
    outputs=[
        ActionOutputSpec(
            key="ndvi",
            label="NDVI",
            description="(NIR − Red) / (NIR + Red), -1..1 float32 raster.",
            artifact_type="raster",
            filename="ndvi.tif",
        ),
        ActionOutputSpec(
            key="ndwi",
            label="NDWI",
            description="(Green − NIR) / (Green + NIR), -1..1 float32 raster.",
            artifact_type="raster",
            filename="ndwi.tif",
        ),
        ActionOutputSpec(
            key="brightness",
            label="VNIR brightness",
            description="Mean VNIR reflectance, 0..1 float32 raster.",
            artifact_type="raster",
            filename="brightness.tif",
        ),
        ActionOutputSpec(
            key="mask_water",
            label="Water mask",
            description="Binary uint8 raster, 1 = water.",
            artifact_type="raster",
            filename="mask_water.tif",
        ),
        ActionOutputSpec(
            key="mask_cloud",
            label="Cloud mask",
            description="Binary uint8 raster, 1 = cloud.",
            artifact_type="raster",
            filename="mask_cloud.tif",
        ),
        ActionOutputSpec(
            key="mask_shadow",
            label="Shadow mask",
            description="Binary uint8 raster, 1 = shadow.",
            artifact_type="raster",
            filename="mask_shadow.tif",
        ),
        ActionOutputSpec(
            key="mask_vegetation",
            label="Vegetation mask",
            description="Binary uint8 raster, 1 = dense vegetation.",
            artifact_type="raster",
            filename="mask_vegetation.tif",
        ),
        ActionOutputSpec(
            key="keep_mask",
            label="Keep mask",
            description=(
                "Canonical input for downstream anomaly Actions. "
                "spatial_validity & ¬(union of selected class masks)."
            ),
            artifact_type="raster",
            filename="keep_mask.tif",
        ),
        ActionOutputSpec(
            key="diagnostics",
            label="Diagnostics",
            description=(
                "Per-class pixel counts and percentages, kept_pct, "
                "threshold set, band indices used."
            ),
            artifact_type="json",
            filename="diagnostics.json",
        ),
        ActionOutputSpec(
            key="preview",
            label="Mask overlay preview",
            description=(
                "RGB composite with each class mask coloured and the "
                "keep_mask outlined."
            ),
            artifact_type="image",
            filename="preview.png",
        ),
    ],
    # Hyperspectral only — NDVI/NDWI/brightness math needs Red/Green/NIR
    # bands. Single-band thermal scenes don't have them.
    accepted_sensor_types=["prisma", "enmap", "aviris_ng"],
    default_config_per_sensor={
        "prisma": _DEFAULT_PRISMA,
        "enmap": _DEFAULT_ENMAP,
        "aviris_ng": _DEFAULT_AVIRIS_NG,
    },
)


# --- API-side validation ------------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Parse `raw_cfg` through `SceneSegmentationConfig`.

    Cross-field semantic checks — that `input_band_filter_output_id`
    references a *complete* `band_filter_apply` ActionOutput in the same
    Project — live in the api submit endpoint (Step 12c). The type
    module only owns shape validation.
    """
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = SceneSegmentationConfig.model_validate(raw_cfg)
    # Validate classes_to_mask membership.
    allowed = {"water", "cloud", "shadow", "vegetation"}
    invalid = set(parsed.classes_to_mask) - allowed
    if invalid:
        raise ValueError(
            f"unknown classes in classes_to_mask: {sorted(invalid)}; "
            f"allowed: {sorted(allowed)}"
        )
    return parsed.model_dump(mode="json")


# --- Worker-side run (Step 12d) -----------------------------------------


_DIAGNOSTICS_FILENAME = "diagnostics.json"
_SUMMARY_FILENAME = "summary.json"
_PREVIEW_FILENAME = "preview.png"


def _save_raster(arr: Any, path: Any, dtype: str) -> None:
    """Tiny GeoTIFF writer — no CRS, no transform, just a typed
    pixel-grid disk artifact downstream Actions can read with rasterio
    or numpy. Tightens to the dtype expected by the consumer."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got {a.shape}")
    a = a.astype(dtype, copy=False)
    h, w = a.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": a.dtype.name,
        "compress": "lzw",
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a, 1)


def run(ctx: Any) -> None:
    """Compute indices + class masks via `app.utils.spectral_indices`,
    write them as GeoTIFF rasters under `ctx.output_dir`."""
    import json
    import pickle

    import numpy as np

    from app.utils.spectral_indices.scene_indices import (
        ClassThresholds,
        IndexBands,
        compute_scene_segmentation,
    )

    cfg = ctx.configuration
    upstream_output = cfg["input_band_filter_output_id"]

    ctx.on_step("load_filtered_vendable")
    upstream_dir = ctx.resolve_action_output(upstream_output)
    pickle_path = upstream_dir / "filtered_vendable.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"upstream band_filter_apply output is missing filtered_vendable.pkl: "
            f"{pickle_path}"
        )
    with pickle_path.open("rb") as f:
        vendable = pickle.load(f)

    cube = vendable.normalized_hyperspectral_cube
    validity = vendable.validity_cube
    wavelengths = np.asarray(vendable.band_cw_order)

    ctx.on_step("compute_indices_and_masks")
    bands_cfg = cfg.get("bands", {})
    thresh_cfg = cfg.get("thresholds", {})
    seg = compute_scene_segmentation(
        cube=cube,
        validity=validity,
        wavelengths=wavelengths,
        bands=IndexBands(
            red_nm=float(bands_cfg.get("red_nm", 660)),
            green_nm=float(bands_cfg.get("green_nm", 560)),
            nir_nm=float(bands_cfg.get("nir_nm", 860)),
            vnir_brightness_end_nm=float(
                bands_cfg.get("vnir_brightness_end_nm", 910)
            ),
        ),
        thresholds=ClassThresholds(
            ndwi_water=float(thresh_cfg.get("ndwi_water", 0.3)),
            brightness_cloud=float(thresh_cfg.get("brightness_cloud", 0.4)),
            brightness_shadow=float(thresh_cfg.get("brightness_shadow", 0.02)),
            ndvi_vegetation=float(thresh_cfg.get("ndvi_vegetation", 0.4)),
        ),
        classes_to_mask=cfg.get(
            "classes_to_mask", ["water", "cloud", "shadow", "vegetation"]
        ),
    )

    ctx.on_step("write_rasters")
    out = ctx.output_dir
    _save_raster(seg.ndvi, out / "ndvi.tif", "float32")
    _save_raster(seg.ndwi, out / "ndwi.tif", "float32")
    _save_raster(seg.brightness, out / "brightness.tif", "float32")
    _save_raster(seg.mask_water, out / "mask_water.tif", "uint8")
    _save_raster(seg.mask_cloud, out / "mask_cloud.tif", "uint8")
    _save_raster(seg.mask_shadow, out / "mask_shadow.tif", "uint8")
    _save_raster(seg.mask_vegetation, out / "mask_vegetation.tif", "uint8")
    _save_raster(seg.keep_mask, out / "keep_mask.tif", "uint8")

    # --- Per-class mean spectra + index histograms (rich diagnostics) -
    ctx.on_step("compute_per_class_stats")
    from app.utils.spectral_indices.scene_indices import (
        index_histogram,
        per_class_mean_spectra,
    )
    import numpy as np

    spatial_valid = (validity.sum(axis=0) > 0).astype(np.uint8)
    class_masks = {
        "water": seg.mask_water,
        "cloud": seg.mask_cloud,
        "shadow": seg.mask_shadow,
        "vegetation": seg.mask_vegetation,
        "kept": seg.keep_mask,
    }
    mean_spectra = per_class_mean_spectra(cube, class_masks)
    histograms = {
        "ndvi": index_histogram(seg.ndvi, spatial_valid, bins=50, value_min=-1.0, value_max=1.0),
        "ndwi": index_histogram(seg.ndwi, spatial_valid, bins=50, value_min=-1.0, value_max=1.0),
        "brightness": index_histogram(
            seg.brightness, spatial_valid, bins=50, value_min=0.0,
            # brightness can exceed 1 on cloud-saturated scenes — use the
            # observed max if it does.
            value_max=max(0.6, float(seg.brightness.max() if seg.brightness.size else 0.6)),
        ),
    }

    # Lean summary for action_outputs.summary JSONB.
    summary = {
        "upstream_band_filter_output_id": upstream_output,
        "band_indices": seg.diagnostics.get("band_indices", {}),
        "band_wavelengths_nm": seg.diagnostics.get("band_wavelengths_nm", {}),
        "thresholds": seg.diagnostics.get("thresholds", {}),
        "classes_to_mask": seg.diagnostics.get("classes_to_mask", []),
        "pixel_counts": seg.diagnostics.get("pixel_counts", {}),
        "kept_pct_of_valid": seg.diagnostics.get("kept_pct_of_valid", 0.0),
    }

    # Rich diagnostics on disk — heavy arrays and histograms live here.
    diagnostics = {
        **summary,
        "wavelengths_nm": [float(w) for w in wavelengths],
        "mean_spectrum_per_class": mean_spectra,
        "index_histograms": histograms,
    }
    (out / _DIAGNOSTICS_FILENAME).write_text(json.dumps(diagnostics, indent=2))
    (out / _SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))

    ctx.on_step("done")


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    """Lean payload for `action_outputs.summary` JSONB.

    The rich payload (per-class mean spectra, index histograms) lives in
    `diagnostics.json` and is fetched lazily by the Output viewer.
    """
    import json

    summary_path = output_dir / _SUMMARY_FILENAME
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    diag_path = output_dir / _DIAGNOSTICS_FILENAME
    if diag_path.exists():
        # Backwards-compatible: older runs only wrote diagnostics.json.
        return json.loads(diag_path.read_text())
    return {"error": "diagnostics_missing"}


def preview(ctx: Any, output_dir: Any) -> Any:
    """Render a 2x2 montage: NDVI · NDWI · brightness · keep_mask overlay
    on a base RGB composite. Decorative — failure is non-fatal."""
    import pickle

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = ctx.configuration

    try:
        upstream_dir = ctx.resolve_action_output(cfg["input_band_filter_output_id"])
        with (upstream_dir / "filtered_vendable.pkl").open("rb") as f:
            vendable = pickle.load(f)
        cube = vendable.normalized_hyperspectral_cube
        wavelengths = np.asarray(vendable.band_cw_order)

        # Lazy reload of the masks we just wrote (smaller in memory than
        # passing them across function boundaries via globals).
        import rasterio
        with rasterio.open(output_dir / "ndvi.tif") as src:
            ndvi = src.read(1)
        with rasterio.open(output_dir / "ndwi.tif") as src:
            ndwi = src.read(1)
        with rasterio.open(output_dir / "brightness.tif") as src:
            brightness = src.read(1)
        with rasterio.open(output_dir / "keep_mask.tif") as src:
            keep_mask = src.read(1)

        # Downsample for cheap rendering (per visualization-rendering memory).
        max_side = 768
        h, w = keep_mask.shape
        stride = max(1, int(np.ceil(max(h, w) / max_side)))
        ndvi_ds = ndvi[::stride, ::stride]
        ndwi_ds = ndwi[::stride, ::stride]
        bri_ds = brightness[::stride, ::stride]
        keep_ds = keep_mask[::stride, ::stride]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(ndvi_ds, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[0, 0].set_title("NDVI")
        axes[0, 1].imshow(ndwi_ds, cmap="BrBG", vmin=-1, vmax=1)
        axes[0, 1].set_title("NDWI")
        axes[1, 0].imshow(bri_ds, cmap="gray", vmin=0, vmax=0.6)
        axes[1, 0].set_title("Brightness (VNIR mean)")
        axes[1, 1].imshow(keep_ds, cmap="Greens", vmin=0, vmax=1)
        axes[1, 1].set_title(
            f"Keep mask · kept {int(keep_mask.sum()):,} px"
        )
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        out_path = output_dir / _PREVIEW_FILENAME
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path
    except Exception as e:
        import logging
        logging.getLogger("allotrope.worker.action_run").warning(
            "scene_segmentation.preview failed: %s", e
        )
        return None
