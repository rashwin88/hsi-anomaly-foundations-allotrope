"""Action type: ``spectral_library_match`` — material family + USGS splib07 SAM.

Names the material of each anomaly pixel two ways:

* **Family matching (the answer).** Calibrated Simplex ACE against the family library
  (``data/family_library``, baked into the worker image at ``/srv/family_library``) names a
  family — roofing, metal, plastic, … — with a calibrated confidence, and only when that
  confidence clears p* = 1 - family_alpha / F. See ``docs/lld/spectral-family-match.md``.
  Skipped in ``mode='all_kept'``: it costs ~2.5 ms per pixel, ~40 min on a full PRISMA scene.
* **SAM (kept alongside).** Spectral Angle Mapper against a sensor-specific USGS splib07
  cache, top-K single spectra per pixel. Kept until the family result has been checked on
  more product scenes; its outputs are unchanged.

SAM pipeline (worker):

  1. Resolve the upstream ``anomaly_detection_prep`` output. Read its
     committed binary anomaly mask.
  2. Load the **native onboarding vendable** (NOT band_filter_apply's
     resampled cube — splib matching needs the un-resampled bands so
     narrow absorption features survive).
  3. Load the per-sensor splib07 cache built earlier by
     ``scripts/build_splib_sensor_cache.py``. The cache filename is
     content-addressed so the same sensor+settings always resolves to
     the same cache.
  4. Pull pixel spectra at the masked locations. SG-smooth.
  5. ``app.spectral_match.match_pixels`` returns top-K SAM angles +
     library indices + n_bands_used per pixel.
  6. Write outputs: matches.parquet (long table), match_map.tif
     (top-1 library index raster), histogram.json, summary.json.

Family matching then writes its own ``family_*`` files beside them, on the raw (unsmoothed)
cube, and a ``family`` block in summary.json. A family failure is recorded there, not raised:
the SAM result still lands.

This module is api-side only — the heavy lifting lives in
``_spectral_library_match_run.py`` (worker process).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._meta import ActionInputSpec, ActionOutputSpec, ActionTypeMeta


KIND = "spectral_library_match"


# --- Per-type config schema ---------------------------------------------


_VALID_MODES = {"anomaly_pixels", "all_kept"}

_DEFAULT_CHAPTERS = ["minerals", "artificial", "soils", "vegetation", "organics"]


class SpectralLibraryMatchConfig(BaseModel):
    """Validated configuration body for a ``spectral_library_match`` Action."""

    model_config = ConfigDict(extra="forbid")

    input_anomaly_detection_output_id: str = Field(
        ...,
        description=(
            "Wire-format ``output_<uuid>`` of a committed "
            "``anomaly_detection_prep`` Action in the same Project. The "
            "binary anomaly mask from this output drives which pixels "
            "get matched (when ``mode='anomaly_pixels'``)."
        ),
        pattern=r"^output_[0-9a-fA-F-]{36}$",
    )

    chapters: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CHAPTERS),
        description=(
            "splib07 chapter slugs to include in the match pool. "
            "Reducing this set speeds up matching and tightens the "
            "search to materials the analyst expects in the scene."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "How many top matches to record per pixel. Capped at 20 to "
            "keep matches.parquet bounded."
        ),
    )

    min_coverage: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Drop library entries whose validity inside the pixel's "
            "valid-band pattern is below this fraction. 0.7 keeps "
            "matching honest without throwing away mostly-good entries."
        ),
    )

    min_band_count: int = Field(
        default=20,
        ge=1,
        le=10_000,
        description=(
            "Pixels with fewer than this many valid bands get no match "
            "(SAM on too few bands is noise). 20 is the empirical floor "
            "for hyperspectral sensors after atmospheric+bbl masking."
        ),
    )

    sg_window_length: int = Field(
        default=7,
        ge=3,
        le=51,
        description=(
            "Savitzky-Golay window (odd integer). 7 bands at ~5-10 nm "
            "sampling suppresses sensor noise without flattening "
            "narrow absorption features."
        ),
    )

    sg_polyorder: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Savitzky-Golay polynomial order. Must be < window length.",
    )

    mode: str = Field(
        default="anomaly_pixels",
        description=(
            "``anomaly_pixels``: match only pixels with anomaly_mask==1 "
            "(fast, default). ``all_kept``: match every valid pixel in "
            "the keep_mask (full-scene material map; minutes of compute). "
            "Family matching runs in ``anomaly_pixels`` mode only."
        ),
    )

    family_alpha: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description=(
            "Family matching: the share of ordinary background pixels you accept being "
            "given a man-made name. Sets the confidence bar p* = 1 - alpha / F "
            "(F = man-made families). 0.05 gives p* = 0.995 with the shipped library."
        ),
    )


# --- Sensor-keyed defaults -------------------------------------------------


_DEFAULT_HSI: dict[str, Any] = {
    "chapters": list(_DEFAULT_CHAPTERS),
    "top_k": 5,
    "min_coverage": 0.7,
    "min_band_count": 20,
    "sg_window_length": 7,
    "sg_polyorder": 2,
    "mode": "anomaly_pixels",
    "family_alpha": 0.05,
}


# --- META ---------------------------------------------------------------


META = ActionTypeMeta(
    type=KIND,
    label="Spectral match (material family + USGS splib07)",
    short_description=(
        "Name the likely material family of each anomaly pixel (calibrated "
        "Simplex ACE), with SAM against USGS splib07 alongside."
    ),
    description=(
        "**Material family.** Each anomaly pixel is compared with the ring of "
        "ordinary pixels around it, in whitened space, against a family library "
        "(metal, plastic, roofing, paint, …). A family is named only when its "
        "score beats the scene's own ordinary pixels by the bar p* = 1 - "
        "``family_alpha`` / F; otherwise the pixel reads \"man-made, type "
        "unconfirmed\" or its natural family. This is a naming step, not a "
        "false-alarm filter: false alarms are unusual too, so they get "
        "man-made names more often than ordinary pixels do.\n\n"
        "**SAM (kept alongside).** Takes a committed ``anomaly_detection_prep`` mask and asks: "
        "*what material is each anomaly pixel?* For each pixel we pull "
        "its spectrum from the **native onboarding cube** (un-resampled, "
        "so narrow absorption features survive), Savitzky-Golay smooth "
        "it, and compute Spectral Angle Mapper against a pre-resampled "
        "USGS splib07 cache built for this sensor. The top-K candidates "
        "per pixel land in a parquet table; the top-1 candidate per "
        "pixel paints a raster ``match_map.tif`` for the viewer.\n\n"
        "Library coverage and per-pixel masking are handled jointly: "
        "library entries are dropped if they don't cover ≥ ``min_coverage`` "
        "of the pixel's valid bands, and the SAM angle is always computed "
        "on the intersection of pixel-valid and library-valid bands. "
        "Pixels with fewer than ``min_band_count`` valid bands get no "
        "match rather than a noisy one.\n\n"
        "Mode ``anomaly_pixels`` (default) only matches pixels with "
        "anomaly_mask==1 — usually a few thousand pixels, runs in "
        "seconds. Mode ``all_kept`` matches every pixel in the keep "
        "mask, producing a full-scene material map — minutes of compute "
        "but useful for context."
    ),
    when_to_use=(
        "After ``anomaly_detection_prep`` has been committed (a binary "
        "anomaly mask exists). Hyperspectral only — thermal sensors "
        "don't have the spectral resolution for splib07 matching."
    ),
    inputs=[
        ActionInputSpec(
            key="input_anomaly_detection_output_id",
            label="Anomaly detection (committed)",
            description=(
                "A committed ``anomaly_detection_prep`` Output. The "
                "binary anomaly mask from this output picks which "
                "pixels get matched in default mode."
            ),
            ref_kind="action_output",
            producing_action_types=["anomaly_detection_prep"],
            required=True,
        ),
    ],
    outputs=[
        ActionOutputSpec(
            key="matches_table",
            label="Top-K matches (parquet)",
            description=(
                "Long table: one row per (pixel, rank). Columns: row, "
                "col, rank, library_ix, material_id, name, chapter, "
                "asd_subtype, angle_deg, n_bands_used."
            ),
            artifact_type="pickle",
            filename="matches.parquet",
        ),
        ActionOutputSpec(
            key="match_map",
            label="Top-1 material map (raster)",
            description=(
                "Int32 raster painting each matched pixel with its top-1 "
                "library index. Sentinel ``-1`` = no match; ``-2`` = "
                "not in the input mask. Companion ``match_map_legend.json`` "
                "maps library_ix → name."
            ),
            artifact_type="raster",
            filename="match_map.tif",
        ),
        ActionOutputSpec(
            key="match_map_preview",
            label="Match map preview",
            description=(
                "Categorical RGBA rendering of the top-1 material map. "
                "Stable per-material colours via a deterministic hash."
            ),
            artifact_type="image",
            filename="match_map.png",
        ),
        ActionOutputSpec(
            key="match_map_legend",
            label="Match map legend",
            description=(
                "JSON mapping library_ix → {name, material_id, chapter}. "
                "Drives the legend on the viewer's match_map panel."
            ),
            artifact_type="json",
            filename="match_map_legend.json",
        ),
        ActionOutputSpec(
            key="rgb_preview",
            label="RGB reference (forwarded)",
            description=(
                "Copy of the upstream RGB so the viewer composes its "
                "layers from one location."
            ),
            artifact_type="image",
            filename="rgb.png",
        ),
        ActionOutputSpec(
            key="match_map_overlay",
            label="Match map overlay (transparent)",
            description=(
                "Same viridis angle heatmap as match_map.png, but every "
                "non-matched pixel is fully transparent — CSS-composited "
                "on top of rgb.png in the viewer."
            ),
            artifact_type="image",
            filename="match_map_overlay.png",
        ),
        ActionOutputSpec(
            key="pixel_spectra",
            label="Anomaly-pixel spectra snapshot",
            description=(
                "NPZ snapshot of the smoothed pixel spectra at every "
                "matched (row, col). Browser preloads this and serves "
                "the spectrum chart locally on every hover/click."
            ),
            artifact_type="pickle",
            filename="anomaly_pixel_spectra.npz",
        ),
        ActionOutputSpec(
            key="library_refl",
            label="Resampled library snapshot",
            description=(
                "NPZ snapshot of the per-sensor library entries used by "
                "this action (refl, valid mask, wavelengths). Powers the "
                "viewer's at-pixel library-spectrum overlay."
            ),
            artifact_type="pickle",
            filename="library_refl.npz",
        ),
        ActionOutputSpec(
            key="histogram",
            label="Top-1 material histogram",
            description=(
                "JSON with the count of pixels per top-1 material. "
                "Drives the viewer's bar-chart side panel."
            ),
            artifact_type="json",
            filename="histogram.json",
        ),
        ActionOutputSpec(
            key="family_matches",
            label="Family matches (parquet)",
            description=(
                "One row per anomaly pixel: row, col, scored, family, family_group, "
                "family_code, confidence, answered, best_entry. best_entry is the "
                "closest library spectrum inside the family — a hint, not the answer."
            ),
            artifact_type="pickle",
            filename="family_matches.parquet",
        ),
        ActionOutputSpec(
            key="family_map",
            label="Material family map (raster)",
            description=(
                "Int16 raster of family codes: ``-2`` not in the mask, ``-1`` not "
                "scored, ``0`` man-made type unconfirmed, ``1..F`` a family. "
                "``family_legend.json`` maps code → family, group, colour."
            ),
            artifact_type="raster",
            filename="family_map.tif",
        ),
        ActionOutputSpec(
            key="family_map_preview",
            label="Family map preview",
            description="Family colours on a dark background, one colour per family.",
            artifact_type="image",
            filename="family_map.png",
        ),
        ActionOutputSpec(
            key="family_map_overlay",
            label="Family map overlay (transparent)",
            description="Same colours, transparent off the mask, for compositing on rgb.png.",
            artifact_type="image",
            filename="family_map_overlay.png",
        ),
        ActionOutputSpec(
            key="family_legend",
            label="Family legend",
            description="JSON: family code → {family, group, colour}, plus p* and alpha.",
            artifact_type="json",
            filename="family_legend.json",
        ),
        ActionOutputSpec(
            key="family_library_refl",
            label="Family library snapshot",
            description=(
                "NPZ of the family library resampled to this scene's bands (names, refl, "
                "valid, wavelengths). Powers the viewer's best-entry hint curve."
            ),
            artifact_type="pickle",
            filename="family_library_refl.npz",
        ),
        ActionOutputSpec(
            key="summary",
            label="Summary",
            description=(
                "Cache used, sensor settings, pixel counts, match "
                "counts, timing breakdown, settings echo. A ``family`` "
                "block carries alpha, p*, family counts, or the error."
            ),
            artifact_type="json",
            filename="summary.json",
        ),
    ],
    # Hyperspectral only. Thermal sensors don't have the spectral
    # resolution for SAM against splib07.
    accepted_sensor_types=["prisma", "enmap", "aviris_ng"],
    default_config_per_sensor={
        "prisma": _DEFAULT_HSI,
        "enmap": _DEFAULT_HSI,
        "aviris_ng": _DEFAULT_HSI,
    },
)


# --- API-side validation -----------------------------------------------


def validate_config(raw_cfg: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    if sensor_type not in META.accepted_sensor_types:
        raise ValueError(
            f"action type {KIND!r} does not accept sensor {sensor_type!r}; "
            f"accepted: {META.accepted_sensor_types}"
        )
    parsed = SpectralLibraryMatchConfig.model_validate(raw_cfg)

    if parsed.mode not in _VALID_MODES:
        raise ValueError(
            f"mode={parsed.mode!r} not in {sorted(_VALID_MODES)}"
        )
    if parsed.sg_polyorder >= parsed.sg_window_length:
        raise ValueError(
            f"sg_polyorder ({parsed.sg_polyorder}) must be < "
            f"sg_window_length ({parsed.sg_window_length})"
        )
    if parsed.sg_window_length % 2 == 0:
        raise ValueError(
            f"sg_window_length must be odd, got {parsed.sg_window_length}"
        )

    # Empty chapters list = "match against everything"; allowed but
    # discouraged because the cache builder skips chapters explicitly.
    if not parsed.chapters:
        raise ValueError(
            "chapters cannot be empty — pick at least one of "
            "minerals, artificial, soils, vegetation, organics, liquids, coatings"
        )

    return parsed.model_dump(mode="json")


# --- run / summarize / preview lazy-dispatch ---------------------------


def run(ctx: Any) -> None:
    from ._spectral_library_match_run import run as _run

    return _run(ctx)


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    from ._spectral_library_match_run import summarize as _summarize

    return _summarize(ctx, output_dir)


def preview(ctx: Any, output_dir: Any) -> Any:
    from ._spectral_library_match_run import preview as _preview

    return _preview(ctx, output_dir)
