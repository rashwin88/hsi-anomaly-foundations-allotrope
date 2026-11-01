"""Action type: `scene_segmentation` — class masks and the keep_mask.

Two methods (config `method`): "indices", below, and "indradhanu_v2" - Indradhanu
V2's segmentation head, implemented in `_scene_segmentation_run.py`
(docs/lld/segmentation-v2.md). A config without `method` is "indices".

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

from typing import Any, Literal

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

    # --- Method (2026-11-01) ---
    #
    # "indices" is the original NDVI / NDWI / brightness thresholding above; a config with no
    # `method` (every action stored before 2026-11-01) is that. "indradhanu_v2" runs Indradhanu
    # V2's segmentation head instead (docs/lld/segmentation-v2.md) and ignores bands /
    # thresholds / classes_to_mask; the three fields below are its own.

    method: Literal["indices", "indradhanu_v2"] = Field(
        default="indices",
        description="indices = NDVI/NDWI/brightness thresholds; indradhanu_v2 = the model's segmentation head.",
    )
    exclude_classes: list[str] = Field(
        default_factory=lambda: ["cloud"],
        description=(
            "indradhanu_v2 only: head classes left out of keep_mask, a subset of "
            "{water, cloud, shadow, snow}. Default thick cloud only - water is kept because "
            "vessels on water are in scope."
        ),
    )
    thin_cloud_is_cloud: bool = Field(
        default=False,
        description="indradhanu_v2 only: count the head's thin-cloud output as cloud (it fires on haze).",
    )
    model_codename: str = Field(
        default="Indradhanu V2",
        description="indradhanu_v2 only: the model whose segmentation head runs.",
    )
    method_if_not_reflectance: Literal["indices"] | None = Field(
        default=None,
        description=(
            "indradhanu_v2 only: run this method instead when the band-filtered scene is not "
            "reflectance (a generic_hsi cube in DN / radiance). None = refuse, as V2 needs reflectance."
        ),
    )


# The classes indradhanu_v2 may exclude: the head's CLASS_NAMES minus land.
V2_EXCLUDABLE_CLASSES = ("water", "cloud", "shadow", "snow")


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

# Tanager: index bands are picked by wavelength, so the defaults carry over too.
_DEFAULT_TANAGER: dict[str, Any] = dict(_DEFAULT_PRISMA)

# Since 2026-11-01 every hyperspectral template runs Indradhanu V2's head (it needs the native-band
# filter, band_filter_apply's default). The index defaults above stay for method="indices".
_DEFAULT_V2: dict[str, Any] = {
    "method": "indradhanu_v2",
    "exclude_classes": ["cloud"],
    "thin_cloud_is_cloud": False,
    "model_codename": "Indradhanu V2",
}

# Any other hyperspectral file: V2 on reflectance; on a scene its reader found not to be
# reflectance (docs/lld/generic-hsi.md) the run falls back to the unit-light index method.
_DEFAULT_GENERIC_HSI: dict[str, Any] = {**_DEFAULT_V2, "method_if_not_reflectance": "indices"}


# --- META ---------------------------------------------------------------

META = ActionTypeMeta(
    type=KIND,
    label="Scene segmentation",
    short_description="Find cloud, water, shadow and snow, and mask out the ones you choose before scoring.",
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
        "band_filter_apply) for every PRISMA / EnMAP investigation.\n\n"
        "method = indradhanu_v2 (the hyperspectral default since 2026-11-01) "
        "replaces the index thresholds with Indradhanu V2's segmentation head: "
        "land · water · cloud · shadow · snow per pixel, written as class_map.tif, "
        "and keep_mask = valid and not in exclude_classes (thick cloud by "
        "default). An anomaly_scoring that runs Indradhanu V2 on this keep_mask "
        "uses these classes instead of running the head again."
    ),
    when_to_use="",
    inputs=[
        # Scene is implicit — taken from the bound Project's scene_id
        # at submit time. The picker only asks for the upstream Action.
        ActionInputSpec(
            key="input_band_filter_output_id",
            label="Band filter result",
            description="",
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
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="ndwi",
            label="NDWI",
            description="(Green − NIR) / (Green + NIR), -1..1 float32 raster.",
            artifact_type="raster",
            filename="ndwi.tif",
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="brightness",
            label="VNIR brightness",
            description="Mean VNIR reflectance, 0..1 float32 raster.",
            artifact_type="raster",
            filename="brightness.tif",
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="mask_water",
            label="Water mask",
            description="Binary uint8 raster, 1 = water.",
            artifact_type="raster",
            filename="mask_water.tif",
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="mask_cloud",
            label="Cloud mask",
            description="Binary uint8 raster, 1 = cloud.",
            artifact_type="raster",
            filename="mask_cloud.tif",
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="mask_shadow",
            label="Shadow mask",
            description="Binary uint8 raster, 1 = shadow.",
            artifact_type="raster",
            filename="mask_shadow.tif",
            always_present=False,   # method="indices" only
        ),
        ActionOutputSpec(
            key="mask_vegetation",
            label="Vegetation mask",
            description="Binary uint8 raster, 1 = dense vegetation.",
            artifact_type="raster",
            filename="mask_vegetation.tif",
            always_present=False,   # method="indices" only
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
        # method="indradhanu_v2" only (_scene_segmentation_run.py).
        ActionOutputSpec(
            key="class_map",
            label="Class map",
            description="uint8 head classes: 0 land, 1 water, 2 cloud, 3 shadow, 4 snow; 255 outside valid.",
            artifact_type="raster",
            filename="class_map.tif",
            always_present=False,
        ),
        ActionOutputSpec(
            key="class_map_png",
            label="Class map (colours)",
            description="The class map in fixed per-class colours; transparent outside valid.",
            artifact_type="image",
            filename="class_map.png",
            always_present=False,
        ),
        ActionOutputSpec(
            key="class_map_overlay",
            label="Class overlay",
            description="Scene RGB with the class colours blended on top.",
            artifact_type="image",
            filename="class_map_overlay.png",
            always_present=False,
        ),
        ActionOutputSpec(
            key="rgb",
            label="Scene RGB",
            description="Full-resolution RGB the viewer draws the classes on.",
            artifact_type="image",
            filename="rgb.png",
            always_present=False,
        ),
    ],
    # Hyperspectral only — NDVI/NDWI/brightness math needs Red/Green/NIR
    # bands. Single-band thermal scenes don't have them.
    accepted_sensor_types=["prisma", "enmap", "aviris_ng", "tanager", "generic_hsi"],
    default_config_per_sensor={
        "prisma": _DEFAULT_V2,
        "enmap": _DEFAULT_V2,
        "aviris_ng": _DEFAULT_V2,
        "tanager": _DEFAULT_V2,
        "generic_hsi": _DEFAULT_GENERIC_HSI,
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
    if parsed.method == "indradhanu_v2":
        _validate_v2(parsed)
    # Validate classes_to_mask membership.
    allowed = {"water", "cloud", "shadow", "vegetation"}
    invalid = set(parsed.classes_to_mask) - allowed
    if invalid:
        raise ValueError(
            f"unknown classes in classes_to_mask: {sorted(invalid)}; "
            f"allowed: {sorted(allowed)}"
        )
    return parsed.model_dump(mode="json")


def _validate_v2(parsed: SceneSegmentationConfig) -> None:
    """exclude_classes are head classes, and model_codename is a catalog model with a segmentation
    head (the spectral_stem_finetune architecture - Indradhanu V2)."""
    from pathlib import Path

    from ..config import settings
    from ..foundation_models.resolver import list_catalog

    invalid = set(parsed.exclude_classes) - set(V2_EXCLUDABLE_CLASSES)
    if invalid:
        raise ValueError(f"unknown classes in exclude_classes: {sorted(invalid)}; "
                         f"allowed: {list(V2_EXCLUDABLE_CLASSES)}")
    heads = {m.codename.lower(): m.codename for m in list_catalog(Path(settings.models_dir))
             if m.architecture == "spectral_stem_finetune"}
    if parsed.model_codename.strip().lower() not in heads:
        raise ValueError(f"model {parsed.model_codename!r} has no segmentation head; "
                         f"available: {sorted(heads.values())}")


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
    write them as GeoTIFF rasters under `ctx.output_dir`. method="indradhanu_v2"
    runs the model's segmentation head instead (`_scene_segmentation_run.py`)."""
    if ctx.configuration.get("method", "indices") == "indradhanu_v2":
        from ._scene_segmentation_run import run as _run_v2, upstream_not_reflectance

        if not (upstream_not_reflectance(ctx) and ctx.configuration.get("method_if_not_reflectance") == "indices"):
            return _run_v2(ctx)
        ctx.on_step("not_reflectance: method indices instead of indradhanu_v2")
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
    on a base RGB composite. Decorative — failure is non-fatal. The V2 method
    renders its own preview inside run."""
    if ctx.configuration.get("method", "indices") == "indradhanu_v2":
        p = output_dir / _PREVIEW_FILENAME
        return p if p.exists() else None
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
