"""Action type: `band_filter_apply` — re-vend a Scene with BandFilterConfig.

Lifts the call your benchmark notebooks make:

    builder = PrismaDatasetBuilder(FileSourceConfig(source_path=he5_path))
    vendable = builder.vend_dataset(band_filter_config=BandFilterConfig(...))

into a worker-side Action. The Output is a fresh
`VendableHyperspectralDataset` pickle attuned to inference — atmospheric
exclusion windows applied, edge bands trimmed, partial spectra PCHIP-filled,
spatial mask sharpened by `max_invalid_voxel_fraction`, EnMAP quality
masks honoured (when applicable), spectrally resampled to the 10 nm
common grid, optional nearest-valid-fill for transformer boundaries.

Step 12b ships the META + config schema + `validate_config`. The
`run` / `summarize` / `preview` implementations land in Step 12d.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "band_filter_apply"


# --- Per-type config schema ---------------------------------------------


class BandFilterApplyConfig(BaseModel):
    """Validated configuration body for a `band_filter_apply` Action.

    Field names + defaults mirror `app/models/dataset/vendables.py`'s
    `BandFilterConfig`. The worker-side `run` (12d) instantiates a
    `BandFilterConfig` from this dict and calls `vend_dataset` with it.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Inputs (config-time references; api validates at submit) ---

    input_scene_id: str = Field(
        ...,
        description="Scene to re-vend, in wire format scene_<uuid>.",
        pattern=r"^scene_[0-9a-fA-F-]{36}$",
    )

    # --- Filter recipe ----------------------------------------------------

    exclusion_ranges: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0, 450),       # low SNR / detector noise
            (912, 978),     # water vapor
            (1131, 1152),   # water vapor
            (1350, 1450),   # water vapor absorption
            (1800, 1950),   # water vapor + CO2 absorption
        ],
        description="Wavelength ranges (nm) to exclude. (low, high) inclusive.",
    )

    edge_bands_to_trim: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Bands trimmed from each end of each detector (VNIR, SWIR).",
    )

    min_valid_pixel_pct: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Minimum valid pixel percentage to keep a band.",
    )

    max_invalid_voxel_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Pixels with more than this fraction of invalid bands are fully "
            "invalidated."
        ),
    )

    quality_masks_to_apply: list[str] = Field(
        default_factory=list,
        description=(
            "Quality mask layers (EnMAP only). Pixels flagged by any of "
            "these are zeroed out before voxel-fraction masking. "
            "Available: cloud, cirrus, haze, cloud_shadow, snow."
        ),
    )

    use_common_wavelength_grid: bool = Field(
        default=True,
        description=(
            "If true, resample to the 10 nm common grid "
            "(DEFAULT_COMMON_WAVELENGTH_GRID). Required for cross-sensor "
            "models like Indradhanu."
        ),
    )

    apply_nearest_valid_fill: bool = Field(
        default=True,
        description=(
            "Run nearest-valid-pixel fill after band-filter to eliminate "
            "boundary artefacts in transformer-based inference (SegFormer "
            "OPE thermal-cliff issue)."
        ),
    )


# --- Sensor-keyed defaults that seed system ActionTemplates ---------------

_DEFAULT_PRISMA: dict[str, Any] = {
    # PRISMA has no quality-mask product, so omit. Other defaults are the
    # same as BandFilterApplyConfig defaults.
    "exclusion_ranges": [
        [0, 450], [912, 978], [1131, 1152], [1350, 1450], [1800, 1950],
    ],
    "edge_bands_to_trim": 3,
    "min_valid_pixel_pct": 20.0,
    "max_invalid_voxel_fraction": 0.4,
    "quality_masks_to_apply": [],
    "use_common_wavelength_grid": True,
    "apply_nearest_valid_fill": True,
}

_DEFAULT_AVIRIS_NG: dict[str, Any] = {
    # AVIRIS-NG carries a bbl (bad-band list) which the builder already
    # honours via band_validity_by_position; no quality-mask product to
    # apply on top.
    "exclusion_ranges": [
        [0, 450], [912, 978], [1131, 1152], [1350, 1450], [1800, 1950],
    ],
    "edge_bands_to_trim": 3,
    "min_valid_pixel_pct": 20.0,
    "max_invalid_voxel_fraction": 0.4,
    "quality_masks_to_apply": [],
    "use_common_wavelength_grid": True,
    "apply_nearest_valid_fill": True,
}

_DEFAULT_ENMAP: dict[str, Any] = {
    # EnMAP ships cloud / cloud_shadow / haze quality masks — apply them.
    "exclusion_ranges": [
        [0, 450], [912, 978], [1131, 1152], [1350, 1450], [1800, 1950],
    ],
    "edge_bands_to_trim": 3,
    "min_valid_pixel_pct": 20.0,
    "max_invalid_voxel_fraction": 0.4,
    "quality_masks_to_apply": ["cloud", "cloud_shadow", "haze"],
    "use_common_wavelength_grid": True,
    "apply_nearest_valid_fill": True,
}


# --- META payload (single source of truth for UI surfacing) -------------

META = ActionTypeMeta(
    type=KIND,
    label="Apply spectral band filter",
    short_description=(
        "Re-vend the scene with atmospheric-window exclusion, edge trim, "
        "spectral fill, and 10 nm common-grid resample."
    ),
    description=(
        "Onboarding pickles a *generic* vendable for the scene: "
        "DN→reflectance, sensor-flagged bad bands, validity masks. That's "
        "the right shape for browsing the cube but not for inference — "
        "the foundation models in this repo were trained on cubes with "
        "atmospheric absorption windows excluded, partial spectra filled, "
        "and bands resampled to a common 10 nm grid. Running them on the "
        "raw vendable is statistically out-of-distribution.\n\n"
        "This action produces an inference-ready vendable. The recipe:\n"
        "• drop bands inside atmospheric absorption windows (912–978 / "
        "1131–1152 / 1350–1450 / 1800–1950 nm by default);\n"
        "• trim noisy edge bands at each detector boundary;\n"
        "• prune bands with too few valid pixels and pixels with too many "
        "invalid bands;\n"
        "• apply EnMAP quality masks (cloud / cloud_shadow / haze) when "
        "available;\n"
        "• fill partial spectra with shape-preserving PCHIP interpolation;\n"
        "• resample to the 10 nm common wavelength grid for cross-sensor "
        "consistency;\n"
        "• optionally apply nearest-valid-pixel fill to eliminate boundary "
        "artefacts in transformer-based inference.\n\n"
        "Output is a drop-in `VendableHyperspectralDataset` pickle that "
        "every downstream HSI Action consumes via `input_action_output_ids` — "
        "scene_segmentation, anomaly_scoring, spectral_detection."
    ),
    when_to_use=(
        "First Action on every PRISMA or EnMAP scene. Re-run with adjusted "
        "exclusion ranges or thresholds when investigating sensor-specific "
        "artefacts."
    ),
    inputs=[
        ActionInputSpec(
            key="input_scene_id",
            label="Scene",
            description="The hyperspectral scene to re-vend.",
            ref_kind="scene",
            required=True,
        ),
    ],
    outputs=[
        ActionOutputSpec(
            key="filtered_vendable",
            label="Filtered vendable (pickle)",
            description=(
                "VendableHyperspectralDataset with band-filter, spectral fill, "
                "common-grid resample applied. Drop-in replacement for the "
                "onboarding vendable — same Python type."
            ),
            artifact_type="pickle",
            filename="filtered_vendable.pkl",
        ),
        ActionOutputSpec(
            key="diagnostics",
            label="Filter diagnostics",
            description=(
                "Per-stage band counts, dropped-band wavelengths, kept-pixel "
                "percentages, voxel-fraction stats."
            ),
            artifact_type="json",
            filename="diagnostics.json",
        ),
        ActionOutputSpec(
            key="preview",
            label="Before/after preview",
            description=(
                "Composite RGB rendered from the filtered cube alongside a "
                "spectrum sample at a representative pixel."
            ),
            artifact_type="image",
            filename="preview.png",
        ),
    ],
    # Hyperspectral only — atmospheric absorption windows, edge trim,
    # common-grid resample, EnMAP quality masks are all HSI concerns.
    # Single-band thermal scenes (Landsat 9 B10) don't need this step;
    # their onboarding vendable is already inference-ready.
    accepted_sensor_types=["prisma", "enmap", "aviris_ng"],
    default_config_per_sensor={
        "prisma": _DEFAULT_PRISMA,
        "enmap": _DEFAULT_ENMAP,
        "aviris_ng": _DEFAULT_AVIRIS_NG,
    },
)


# --- API-side validation ------------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Parse `raw_cfg` through the per-type Pydantic schema.

    Returns the validated + defaulted dict ready to persist into
    `actions.configuration`. Raises `pydantic.ValidationError` on shape
    issues; FastAPI translates that to 422.

    Cross-field semantic checks (e.g., `input_scene_id` belongs to the
    bound Project's Scene) live in the api submit endpoint (Step 12c) —
    the type module only owns shape validation.
    """
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = BandFilterApplyConfig.model_validate(raw_cfg)
    return parsed.model_dump(mode="json")


# --- Worker-side run (Step 12d) -----------------------------------------


_DIAGNOSTICS_FILENAME = "diagnostics.json"
_SUMMARY_FILENAME = "summary.json"
_VENDABLE_FILENAME = "filtered_vendable.pkl"
_PREVIEW_FILENAME = "preview.png"


def run(ctx: Any) -> None:
    """Re-vend the Scene with `BandFilterConfig` and pickle the result.

    Heavy imports happen at function scope so the api process doesn't
    pay for `app/`, numpy, or scipy at import time.
    """
    import json
    import pickle

    import numpy as np

    from app.models.dataset.vendables import (
        BandFilterConfig,
        DEFAULT_COMMON_WAVELENGTH_GRID,
    )
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.aviris_ng_dataset_builder import (
        AvirisNGDatasetBuilder,
    )
    from app.utils.dataset_builder.enmap_dataset_builder import (
        EnmapDatasetBuilder,
    )
    from app.utils.dataset_builder.prisma_dataset_builder import (
        PrismaDatasetBuilder,
    )
    from app.utils.pixel_fill.nearest_valid_fill import nearest_valid_fill

    cfg = ctx.configuration
    sensor = ctx.sensor_type

    ctx.on_step("build_vendable_filter_config")
    bf_config = BandFilterConfig(
        exclusion_ranges=[tuple(r) for r in cfg["exclusion_ranges"]],
        edge_bands_to_trim=int(cfg["edge_bands_to_trim"]),
        min_valid_pixel_pct=float(cfg["min_valid_pixel_pct"]),
        max_invalid_voxel_fraction=float(cfg["max_invalid_voxel_fraction"]),
        quality_masks_to_apply=list(cfg.get("quality_masks_to_apply", [])),
        common_wavelength_grid=(
            DEFAULT_COMMON_WAVELENGTH_GRID
            if cfg.get("use_common_wavelength_grid", True)
            else None
        ),
    )

    ctx.on_step("instantiate_dataset_builder")
    # ctx.scene_raw_path points at the raw *directory* (scenes/<id>/raw/).
    # Each builder expects a different shape inside that dir — PRISMA
    # wants the .he5 file, Landsat the .tif, EnMAP the folder containing
    # *-METADATA.XML. Resolver is shared with onboarding so both code
    # paths agree on what they hand to FileSourceConfig.
    from allotrope.sensors.source_path import resolve_source_path

    source_path = resolve_source_path(sensor, ctx.scene_raw_path)
    file_source = FileSourceConfig(source_path=str(source_path))
    if sensor == "prisma":
        builder = PrismaDatasetBuilder(file_source_configuration=file_source)
    elif sensor == "enmap":
        builder = EnmapDatasetBuilder(file_source_configuration=file_source)
    elif sensor == "aviris_ng":
        builder = AvirisNGDatasetBuilder(file_source_configuration=file_source)
    else:
        raise ValueError(
            f"band_filter_apply: unsupported sensor {sensor!r} "
            f"(accepted: {META.accepted_sensor_types})"
        )

    ctx.on_step("vend_dataset_with_band_filter")
    # AVIRIS-NG materialises BSQ + validity to disk; park them next to
    # the filtered_vendable.pkl so all per-action artefacts colocate.
    if sensor == "aviris_ng":
        vendable = builder.vend_dataset(
            band_filter_config=bf_config,
            validity_cube_dir=ctx.output_dir,
        )
    else:
        vendable = builder.vend_dataset(band_filter_config=bf_config)

    band_count = vendable.normalized_hyperspectral_cube.shape[0]
    height = vendable.normalized_hyperspectral_cube.shape[1]
    width = vendable.normalized_hyperspectral_cube.shape[2]
    spatial_validity_pct = float(
        (vendable.validity_cube.sum(axis=0) > 0).mean() * 100
    )

    if cfg.get("apply_nearest_valid_fill", True):
        ctx.on_step("nearest_valid_fill")
        # Collapse per-band validity to a single per-pixel mask (1, H, W).
        spatial_mask = (vendable.validity_cube.sum(axis=0) > 0).astype(np.int8)[
            None, :, :
        ]
        filled = nearest_valid_fill(
            vendable.normalized_hyperspectral_cube, spatial_mask
        )
        vendable.normalized_hyperspectral_cube = filled

    ctx.on_step("pickle_filtered_vendable")
    output_pickle = ctx.output_dir / _VENDABLE_FILENAME
    with output_pickle.open("wb") as f:
        pickle.dump(vendable, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_size_mb = output_pickle.stat().st_size / (1024 * 1024)

    # --- Per-band statistics over valid pixels (rich diagnostics) ----
    ctx.on_step("compute_spectral_stats")
    cube = vendable.normalized_hyperspectral_cube     # (C, H, W) — already filled
    validity = vendable.validity_cube                 # (C, H, W)
    wavelengths = np.asarray(vendable.band_cw_order, dtype=np.float64)

    # Per-pixel spatial validity collapses the per-band stack.
    spatial_valid = (validity.sum(axis=0) > 0)        # (H, W) bool
    n_total = int(spatial_valid.size)
    n_valid = int(spatial_valid.sum())

    # Reshape to (C, N) and select valid columns. Memory-friendly: this
    # is float32 × 165 × n_valid; for a 1.5M-pixel scene that's ~990 MB —
    # fine on the worker box, but compute stats in-place.
    flat = cube.reshape(cube.shape[0], -1)             # (C, H*W)
    flat_valid = flat[:, spatial_valid.ravel()]        # (C, n_valid)

    if n_valid > 0:
        mean_spectrum = flat_valid.mean(axis=1)
        std_spectrum = flat_valid.std(axis=1)
        # P10/P50/P90 across valid pixels for each band — via percentile.
        p_spectra = np.percentile(flat_valid, [10, 50, 90], axis=1)
    else:
        mean_spectrum = np.zeros(cube.shape[0], dtype=np.float32)
        std_spectrum = np.zeros(cube.shape[0], dtype=np.float32)
        p_spectra = np.zeros((3, cube.shape[0]), dtype=np.float32)
    p10_spectrum, p50_spectrum, p90_spectrum = p_spectra

    # --- Lean summary (action_outputs.summary JSONB) -----------------
    summary = {
        "sensor": sensor,
        "band_count": int(band_count),
        "height": int(height),
        "width": int(width),
        "total_pixels": n_total,
        "valid_pixels": n_valid,
        "spatial_validity_pct": round(spatial_validity_pct, 3),
        "wavelengths_min_nm": float(wavelengths.min()),
        "wavelengths_max_nm": float(wavelengths.max()),
        "common_wavelength_grid_applied": bool(
            cfg.get("use_common_wavelength_grid", True)
        ),
        "nearest_valid_fill_applied": bool(
            cfg.get("apply_nearest_valid_fill", True)
        ),
        "pickle_filename": _VENDABLE_FILENAME,
        "pickle_size_mb": round(pickle_size_mb, 2),
        "quality_masks_applied": cfg.get("quality_masks_to_apply", []),
        "exclusion_ranges_nm": cfg["exclusion_ranges"],
        "edge_bands_to_trim": cfg["edge_bands_to_trim"],
    }

    # --- Rich diagnostics on disk (heavy arrays go here, NOT into DB) -
    diagnostics = {
        **summary,
        "wavelengths_nm": [float(w) for w in wavelengths],
        "mean_spectrum": [float(v) for v in mean_spectrum],
        "std_spectrum": [float(v) for v in std_spectrum],
        "p10_spectrum": [float(v) for v in p10_spectrum],
        "p50_spectrum": [float(v) for v in p50_spectrum],
        "p90_spectrum": [float(v) for v in p90_spectrum],
        "spectrum_units": "reflectance (0..~1) over valid pixels only",
    }
    diagnostics_path = ctx.output_dir / _DIAGNOSTICS_FILENAME
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    # Stash the lean summary so summarize() can return it without
    # re-reading + re-parsing the (heavier) diagnostics file.
    summary_path = ctx.output_dir / _SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))

    ctx.on_step("done")


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    """Lean payload for `action_outputs.summary` JSONB.

    The rich payload (mean / p10 / p50 / p90 spectra, per-wavelength
    array) lives in `diagnostics.json` and is fetched lazily by the
    Output viewer via /actions/{id}/files/diagnostics.json.
    """
    import json

    summary_path = output_dir / _SUMMARY_FILENAME
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    diagnostics_path = output_dir / _DIAGNOSTICS_FILENAME
    if diagnostics_path.exists():
        # Backwards-compatible: older runs only wrote diagnostics.json.
        return json.loads(diagnostics_path.read_text())
    return {"error": "diagnostics_missing"}


def preview(ctx: Any, output_dir: Any) -> Any:
    """Render a downsampled RGB-composite preview from the filtered cube.

    Picks three bands closest to 650 / 550 / 450 nm. Applies a per-band
    1st/99th-percentile stretch over valid pixels. Saves to preview.png
    next to the pickle. Best-effort: a render failure does not fail the
    Action — preview is decorative.
    """
    import pickle

    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    pickle_path = output_dir / _VENDABLE_FILENAME
    if not pickle_path.exists():
        return None

    try:
        with pickle_path.open("rb") as f:
            vendable = pickle.load(f)
        cube = vendable.normalized_hyperspectral_cube  # (C, H, W)
        validity = vendable.validity_cube              # (C, H, W)
        wavelengths = np.asarray(vendable.band_cw_order)

        spatial_valid = (validity.sum(axis=0) > 0)

        def nearest(target: float) -> int:
            return int(np.argmin(np.abs(wavelengths - target)))

        idx_r = nearest(650)
        idx_g = nearest(550)
        idx_b = nearest(450)

        # Downsample-before-colormap rule: cap to 1024 px on the long side.
        h, w = cube.shape[1], cube.shape[2]
        max_side = 1024
        if max(h, w) > max_side:
            stride = int(np.ceil(max(h, w) / max_side))
        else:
            stride = 1
        r = cube[idx_r, ::stride, ::stride].astype(np.float32)
        g = cube[idx_g, ::stride, ::stride].astype(np.float32)
        b = cube[idx_b, ::stride, ::stride].astype(np.float32)
        valid_ds = spatial_valid[::stride, ::stride]

        def stretch(channel: np.ndarray) -> np.ndarray:
            # Validity mask BEFORE percentile stretch — invalid pixels
            # would otherwise pull the dynamic range.
            vals = channel[valid_ds]
            if vals.size == 0:
                return np.zeros_like(channel)
            lo, hi = np.percentile(vals, [1.0, 99.0])
            if hi - lo < 1e-6:
                return np.zeros_like(channel)
            stretched = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
            stretched[~valid_ds] = 0.0
            return stretched

        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        # uint8 LUT (per visualization-rendering memory).
        rgb_u8 = (rgb * 255.0).astype(np.uint8)

        out_path = output_dir / _PREVIEW_FILENAME
        # Pillow direct write — avoids matplotlib axis padding for a
        # crisp rectangle that rerenders cleanly in the workspace.
        from PIL import Image
        Image.fromarray(rgb_u8).save(out_path)
        return out_path
    except Exception as e:
        # Preview is decorative — log and move on.
        import logging
        logging.getLogger("allotrope.worker.action_run").warning(
            "band_filter_apply.preview failed: %s", e
        )
        return None
