"""Worker-side implementation of ``anomaly_detection_prep``.

Kept in its own module so the api-side import of
``anomaly_detection_prep.py`` doesn't pay for numpy / rasterio /
matplotlib — this file is only loaded inside the worker process.

Pipeline:

  1. Resolve the upstream ``anomaly_scoring`` output dir.
  2. Read its ``summary.json`` to discover ``model_codenames``.
  3. Normalise the user's ``algorithm_weights`` (default = equal),
     validate every weighted codename exists upstream.
  4. For each algorithm in the weight set:
       a. Read ``models/<codename>/anomaly_score.tif``.
       b. Rescale to [0, 1] using min/max over its finite (non-NaN)
          pixels. NaNs propagate.
  5. Weighted-combine into a single ``(H, W)`` float32 composite. A
     pixel is finite in the composite iff it's finite in every
     contributing algorithm map (conservative — anomaly_scoring marks
     out-of-keep-mask pixels as NaN per algorithm).
  6. Save outputs:
       - ``composite_score.tif`` (float32 raster)
       - ``composite_score.png`` (inferno + alpha mask)
       - ``rgb.png`` copied from upstream so the action is self-contained
       - ``summary.json``

No commit, no binary mask, no metrics here — those live on the api's
preview endpoint which runs interactively per Apply press.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("allotrope.action_types.anomaly_detection_prep")


_COMPOSITE_TIF = "composite_score.tif"
_COMPOSITE_PNG = "composite_score.png"
_RGB_PNG = "rgb.png"
_SUMMARY_FILENAME = "summary.json"

# Long-edge cap for the colormapped composite preview. Matches the
# convention used by anomaly_scoring's _render_score_png.
_PNG_MAX_EDGE = 4096


def run(ctx: Any) -> None:
    """Worker entry — see module docstring for the recipe."""
    # Local heavy imports so the api process doesn't pay.
    import numpy as np
    import rasterio
    from matplotlib import cm
    from PIL import Image

    cfg = ctx.configuration

    # ---- 1. Resolve the upstream anomaly_scoring output dir --------
    upstream_id = cfg["input_anomaly_scoring_output_id"]
    ctx.on_step("resolve_upstream_anomaly_scoring")
    upstream_dir: Path = ctx.resolve_action_output(upstream_id)
    upstream_summary_path = upstream_dir / "summary.json"
    if not upstream_summary_path.is_file():
        raise FileNotFoundError(
            f"upstream anomaly_scoring output is missing summary.json: "
            f"{upstream_summary_path}"
        )
    with upstream_summary_path.open() as f:
        upstream_summary = json.load(f)

    upstream_codenames: list[str] = list(
        upstream_summary.get("model_codenames") or []
    )
    if not upstream_codenames:
        raise ValueError(
            "upstream anomaly_scoring output declares no model_codenames "
            "in summary.json — nothing to combine."
        )
    logger.info(
        "upstream anomaly_scoring output %s — algorithms: %s",
        upstream_id, upstream_codenames,
    )

    # ---- 2. Resolve weights ----------------------------------------
    raw_weights: dict[str, float] = dict(cfg.get("algorithm_weights") or {})
    if raw_weights:
        # User-supplied. Filter to upstream codenames only — if a user
        # supplies a weight for a codename that isn't in the upstream
        # output we fail loudly rather than silently ignore.
        unknown = [c for c in raw_weights if c not in upstream_codenames]
        if unknown:
            raise ValueError(
                f"algorithm_weights references codenames not in upstream "
                f"output {upstream_id}: {unknown!r}. Upstream codenames: "
                f"{upstream_codenames!r}."
            )
        weights = {c: float(raw_weights.get(c, 0.0)) for c in upstream_codenames}
    else:
        # Default — equal weights across every upstream algorithm.
        weights = {c: 1.0 for c in upstream_codenames}

    # Drop zero-weight algorithms — no point reading their TIFs.
    active = {c: w for c, w in weights.items() if w > 0}
    if not active:
        raise ValueError(
            "All algorithm weights resolved to zero — there's nothing to "
            "combine. Supply at least one positive weight."
        )

    weight_sum = sum(active.values())
    normalised = {c: w / weight_sum for c, w in active.items()}
    logger.info(
        "active algorithm weights (normalised): %s",
        {c: round(w, 4) for c, w in normalised.items()},
    )

    # ---- 3. Load + rescale each algorithm's score ------------------
    rescaled_layers: dict[str, "np.ndarray"] = {}
    per_algo_stats: dict[str, dict[str, float]] = {}
    composite_shape: tuple[int, int] | None = None

    for codename in active:
        ctx.on_step(f"load_score:{codename}")
        # The upstream output stores models under
        #     models/<codename_slug>/anomaly_score.tif
        # where the slug is the codename lowercased with spaces → "_".
        # Matches the convention in anomaly_scoring's renderer +
        # frontend's outputUrl helper.
        slug = codename.lower().replace(" ", "_")
        tif_path = upstream_dir / "models" / slug / "anomaly_score.tif"
        if not tif_path.is_file():
            raise FileNotFoundError(
                f"upstream anomaly_scoring output is missing "
                f"{tif_path.relative_to(upstream_dir)} for codename "
                f"{codename!r}."
            )
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32, copy=False)
        if composite_shape is None:
            composite_shape = arr.shape
        elif arr.shape != composite_shape:
            raise ValueError(
                f"score map for {codename!r} has shape {arr.shape} but "
                f"expected {composite_shape} from a sibling algorithm. "
                f"All algorithms in the upstream output should share the "
                f"scene grid."
            )

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            raise ValueError(
                f"score map for {codename!r} contains no finite pixels — "
                f"cannot rescale."
            )
        finite_vals = arr[finite_mask]
        vmin = float(finite_vals.min())
        vmax = float(finite_vals.max())
        if vmax <= vmin:
            # Degenerate scene — all valid scores are the same value.
            # Rescaled layer is uniformly 0 wherever finite.
            rescaled = np.where(finite_mask, np.float32(0.0), np.float32(np.nan))
        else:
            rescaled = (arr - np.float32(vmin)) / np.float32(vmax - vmin)
            rescaled = np.where(finite_mask, rescaled, np.float32(np.nan))
        rescaled_layers[codename] = rescaled
        per_algo_stats[codename] = {
            "raw_min": vmin,
            "raw_max": vmax,
            "weight_raw": float(weights[codename]),
            "weight_normalised": float(normalised[codename]),
        }

    # ---- 4. Weighted combine ---------------------------------------
    ctx.on_step("combine_composite")
    assert composite_shape is not None
    # Track per-pixel validity — a pixel is composite-valid only if
    # finite in every contributing layer.
    valid_mask = np.ones(composite_shape, dtype=bool)
    composite = np.zeros(composite_shape, dtype=np.float32)
    for codename, layer in rescaled_layers.items():
        valid_mask &= np.isfinite(layer)
    for codename, layer in rescaled_layers.items():
        # Replace NaNs with 0 before weighting — the valid_mask will
        # NaN-them-out at the end.
        safe = np.where(np.isfinite(layer), layer, np.float32(0.0))
        composite += np.float32(normalised[codename]) * safe
    composite = np.where(valid_mask, composite, np.float32(np.nan))

    finite_composite = composite[valid_mask]
    if finite_composite.size == 0:
        raise ValueError(
            "Composite score map has no finite pixels — every algorithm "
            "map must have had completely disjoint validity regions."
        )

    distribution_stats = {
        "min": float(finite_composite.min()),
        "p2": float(np.percentile(finite_composite, 2)),
        "p50": float(np.percentile(finite_composite, 50)),
        "p90": float(np.percentile(finite_composite, 90)),
        "p95": float(np.percentile(finite_composite, 95)),
        "p99": float(np.percentile(finite_composite, 99)),
        "p99_5": float(np.percentile(finite_composite, 99.5)),
        "max": float(finite_composite.max()),
        "mean": float(finite_composite.mean()),
        "std": float(finite_composite.std()),
        "valid_pixels": int(valid_mask.sum()),
        "total_pixels": int(valid_mask.size),
    }

    # ---- 5. Write outputs ------------------------------------------
    output_dir: Path = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx.on_step("write_composite_raster")
    _write_float_raster(
        composite,
        output_dir / _COMPOSITE_TIF,
        upstream_dir / "models" / next(iter(rescaled_layers.keys())).lower().replace(" ", "_") / "anomaly_score.tif",
    )

    ctx.on_step("render_composite_preview")
    _render_composite_png(
        composite=composite,
        valid_mask=valid_mask,
        out_path=output_dir / _COMPOSITE_PNG,
    )

    ctx.on_step("copy_rgb")
    upstream_rgb = upstream_dir / "rgb.png"
    if upstream_rgb.is_file():
        shutil.copy2(upstream_rgb, output_dir / _RGB_PNG)
    else:
        logger.warning(
            "upstream output %s has no rgb.png — viewer will fall back "
            "to the scene-level RGB.", upstream_id,
        )

    # ---- 6. Has-GT check (just for the summary flag) ---------------
    has_gt = bool(cfg.get("input_annotation_id"))

    summary = {
        "upstream_anomaly_scoring_output_id": upstream_id,
        "upstream_codenames": upstream_codenames,
        "active_codenames": list(active.keys()),
        "weights_raw": {c: float(weights[c]) for c in upstream_codenames},
        "weights_normalised": {c: float(normalised[c]) for c in active},
        "per_algorithm": per_algo_stats,
        "composite_shape": [int(composite_shape[0]), int(composite_shape[1])],
        "composite_distribution": distribution_stats,
        "has_gt": has_gt,
        "input_annotation_id": cfg.get("input_annotation_id"),
    }
    (output_dir / _SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))

    ctx.on_step("done")
    logger.info(
        "anomaly_detection_prep done — composite shape=%s valid=%d/%d, "
        "algos=%d (status will be needs_threshold).",
        composite_shape,
        int(valid_mask.sum()),
        int(valid_mask.size),
        len(active),
    )


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    """Read summary.json back — same pattern as cloud_mask."""
    p = Path(output_dir) / _SUMMARY_FILENAME
    if not p.is_file():
        return {}
    with p.open() as f:
        return json.load(f)


def preview(ctx: Any, output_dir: Any) -> Any:
    """The action's "preview" filename per the meta — points the
    workspace at the composite colormap PNG. Same shape the other
    action types use."""
    return Path(output_dir) / _COMPOSITE_PNG


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _write_float_raster(arr: "Any", out_path: Path, profile_src: Path) -> None:
    """Write a (H, W) float32 raster with the same georeferencing as
    ``profile_src``. NaN is the implicit nodata."""
    import numpy as np
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(profile_src) as src:
        profile = src.profile
    profile.update(
        dtype="float32",
        count=1,
        nodata=float("nan"),
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.asarray(arr, dtype=np.float32), 1)


def _render_composite_png(
    *,
    composite: "Any",
    valid_mask: "Any",
    out_path: Path,
) -> None:
    """Inferno-colormapped preview of the composite. Invalid pixels
    rendered with alpha=0 so the panel shows transparent gaps where the
    scene is masked. Downsample-before-colormap.

    Mirrors anomaly_scoring's _render_score_png so the visual style is
    consistent across the action catalog — same LUT, same percentile
    stretch (p2..p98 over valid pixels).
    """
    import numpy as np
    from matplotlib import cm
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = composite.shape
    scale = min(1.0, _PNG_MAX_EDGE / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        composite_ds = np.asarray(
            Image.fromarray(composite.astype(np.float32)).resize(
                (new_w, new_h), Image.BILINEAR
            )
        )
        valid_ds = (
            np.asarray(
                Image.fromarray(valid_mask.astype(np.uint8)).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 0
        )
    else:
        composite_ds = composite
        valid_ds = valid_mask

    sample_mask = valid_ds & np.isfinite(composite_ds)
    if sample_mask.any():
        lo, hi = np.percentile(composite_ds[sample_mask], [2, 98])
        if hi <= lo:
            hi = lo + np.float32(1e-6)
    else:
        lo, hi = 0.0, 1.0

    safe = np.where(np.isfinite(composite_ds), composite_ds, 0.0)
    norm = np.clip((safe - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype(np.float32)
    norm[~valid_ds] = 0.0

    lut = (cm.get_cmap("inferno", 256)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    idx = (norm * 255).astype(np.uint8)
    rgb = lut[idx]
    alpha = (valid_ds.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG", optimize=True)
