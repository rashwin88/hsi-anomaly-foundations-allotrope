"""
Writing anomaly-scoring artifacts to disk, and rendering them for the browser.

Split out of _anomaly_scoring_run.py, which was 1,000 lines. Everything here
answers "turn this array into a file the UI can show":

    _save_raster_2d / _save_raster_3d   GeoTIFF writers
    _save_png_array                     PNG writer with a size cap
    _load_annotation_gt                 read a ground-truth annotation raster
    _render_score_png                   the anomaly heatmap
    _render_recon_png                   the reconstruction panel
    _render_thumbnail                   the multi-model montage
    _make_rgb                           a true-colour view of the cube

Two conventions worth knowing:

  - Every GeoTIFF is written with an IDENTITY transform. Vendables carry no
    spatial reference, so georeferencing is recovered later at export time by
    re-reading the raw scene (app/georef/). Do not add a real transform here;
    the export path expects to supply it.

  - Every heavy import (numpy, rasterio, matplotlib) is done INSIDE its
    function, not at module scope. This module is only reached from
    anomaly_scoring.run(), which is itself lazily imported so the api process
    never loads torch or GDAL - see docs/06-backend.md on the lazy-import rule.
    Keep it that way: a top-level import here would be loaded by the worker on
    every job, not just scoring jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Cap on the longest edge of any written PNG. A full AVIRIS-NG scene is far
# larger than any browser needs, and the UI scales to fit anyway.
_PNG_MAX_EDGE = 4096
_THUMBNAIL_DPI = 110

def _safe_dirname(codename: str) -> str:
    return codename.strip().lower().replace(" ", "_")


def _save_raster_2d(arr: Any, path: Any, dtype: str) -> None:
    """Single-band GeoTIFF writer."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    a = np.asarray(arr).astype(dtype, copy=False)
    if a.ndim != 2:
        raise ValueError(f"expected (H, W); got {a.shape}")
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


def _save_raster_3d(arr: Any, path: Any, dtype: str) -> None:
    """Multi-band (C, H, W) GeoTIFF writer."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    a = np.asarray(arr).astype(dtype, copy=False)
    if a.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {a.shape}")
    c, h, w = a.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": a.dtype.name,
        "compress": "lzw",
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a)


def _save_png_array(rgb01: Any, path: Any) -> None:
    """Write an (H, W, 3) float [0, 1] array as a PNG with PIL."""
    import numpy as np
    from PIL import Image

    arr = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = arr.shape[:2]
    if max(h, w) > _PNG_MAX_EDGE:
        scale = _PNG_MAX_EDGE / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        arr = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


def _load_annotation_gt(ctx: Any, annotation_wire_id: str) -> tuple[Any, dict[str, Any]]:
    """Load a binary GT raster from the Scene's attached annotation."""
    import re
    import numpy as np
    import rasterio

    if not annotation_wire_id.startswith("annotation_"):
        raise ValueError(f"not a wire-format annotation id: {annotation_wire_id}")
    annotation_uuid = annotation_wire_id[len("annotation_"):]
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", annotation_uuid):
        raise ValueError(f"bad annotation uuid: {annotation_uuid}")

    annotation_dir = (
        ctx.data_dir
        / "scenes"
        / str(ctx.scene_id)
        / "annotations"
        / annotation_uuid
    )
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation dir not on disk: {annotation_dir}")
    tifs = sorted(annotation_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(
            f"no .tif under {annotation_dir} â€” supported annotation types "
            "must materialise a raster file."
        )
    with rasterio.open(tifs[0]) as src:
        raw = src.read(1)
    gt = (raw > 0).astype(np.uint8)
    return gt, {
        "annotation_id": annotation_wire_id,
        "raster_filename": tifs[0].name,
        "n_positive": int(gt.sum()),
    }


def _render_score_png(
    *,
    score: Any,
    kept: Any,
    out_path: Path,
    stretch: str = "linear",
) -> None:
    """Render the score raster as an inferno-coloured PNG with invalid
    pixels transparent. Capped at 99.5th percentile for legibility.
    Downsample-before-colormap (per the project's viz feedback).

    `stretch`:
      - "linear": score / p99.5, the original behavior. Works well for
        foundation-model reconstruction errors which are roughly
        unimodal and not heavy-tailed.
      - "sqrt":   sqrt(score) / sqrt(p99.5). Use for RX-family scores.
        Mahalanobis squared-distance is Ï‡Â²-distributed (very long
        right tail; p50 is ~10% of the way to the mean). Linear
        stretching crushes the bulk of pixels into the first few
        inferno indices and the image reads as nearly black. sqrt
        moves the bulk into the mid-LUT range where the colormap has
        the perceptual spread to show structure.
    """
    import numpy as np
    from matplotlib import cm
    from PIL import Image

    h, w = score.shape
    scale = min(1.0, _PNG_MAX_EDGE / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        score_ds = np.asarray(
            Image.fromarray(score).resize((new_w, new_h), Image.BILINEAR)
        )
        kept_ds = (
            np.asarray(
                Image.fromarray(kept.astype(np.uint8)).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 0
        )
    else:
        score_ds = score
        kept_ds = kept

    # Apply the chosen stretch on a copy so the underlying score raster
    # is untouched (the .tif on disk + the percentile stats stay in
    # native units).
    if stretch == "sqrt":
        stretched = np.sqrt(np.clip(score_ds.astype(np.float32), 0.0, None))
    else:
        stretched = score_ds.astype(np.float32)

    # Restrict the percentile to pixels that are BOTH kept AND finite.
    # `kept_ds` (NEAREST-downsampled keep mask) can mark a pixel as kept
    # even when `score_ds` (BILINEAR-downsampled score) is NaN at the
    # same location â€” np.percentile then returns NaN and the entire PNG
    # renders as solid black with only the alpha channel carrying signal.
    sample_mask = kept_ds & np.isfinite(stretched)
    if sample_mask.any():
        vmax = float(np.percentile(stretched[sample_mask], 99.5))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0
    # NaN in `stretched` would propagate through the division. Replace
    # NaN with 0 before normalising; those pixels are masked out below
    # anyway via the alpha channel.
    safe = np.where(np.isfinite(stretched), stretched, 0.0)
    norm = np.clip(safe / vmax, 0.0, 1.0).astype(np.float32)
    norm[~kept_ds] = 0.0

    lut = (cm.get_cmap("inferno", 256)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    idx = (norm * 255).astype(np.uint8)
    rgb = lut[idx]                                   # (H, W, 3)
    alpha = (kept_ds.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG", optimize=True)


def _render_recon_png(
    *,
    recon: Any,
    wavelengths: Any,
    spatial_valid: Any,
    out_path: Path,
) -> None:
    """Render the reconstruction cube the same way as the input RGB:
    pick R/G/B nearest-band slices for HSI, single-channel grey for
    thermal, validity-masked, 2/98 percentile stretch."""
    rgb = _make_rgb(recon, wavelengths, spatial_valid)
    _save_png_array(rgb, out_path)


def _render_thumbnail(
    *,
    output_path: Path,
    rgb: Any,
    keep_mask: Any,
    models_root: Path,
    per_model_records: list[dict[str, Any]],
) -> None:
    """Small static RGB + per-model heatmap montage. Used for the
    Action card thumbnail; the live viewer is built from the per-model
    PNGs."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    n_models = len(per_model_records)
    n_cols = n_models + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.8))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB scene", fontsize=10, pad=6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    kept = keep_mask.astype(bool)
    for i, rec in enumerate(per_model_records):
        ax = axes[i + 1]
        codename = rec["codename"]
        score_path = models_root / _safe_dirname(codename) / "anomaly_score.tif"
        with rasterio.open(score_path) as src:
            score = src.read(1).astype(np.float32, copy=False)
        # Strip NaNs before the percentile cap â€” classical detectors
        # write NaN outside their internal spatial_mask, which can
        # sit inside the raw keep_mask when keep_mask_erosion is on.
        # np.percentile over NaN returns NaN and the whole panel
        # renders black.
        sample = score[kept]
        sample = sample[np.isfinite(sample)]
        if sample.size > 0:
            vmax = float(np.percentile(sample, 99.5))
            vmax = max(vmax, 1e-6)
        else:
            vmax = 1.0
        # Mask both raw kept and NaN positions so the colormap
        # ignores them (renders as the masked color, not bright).
        masked = np.ma.masked_where(~kept | ~np.isfinite(score), score)
        ax.imshow(masked, cmap="inferno", vmin=0.0, vmax=vmax)
        title = f"{codename}\n{rec['method']}"
        if rec.get("auc") is not None:
            title += f" Â· AUC={rec['auc']:.3f}"
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=_THUMBNAIL_DPI)
    plt.close(fig)


def _make_rgb(cube: Any, wl: Any, validity: Any) -> Any:
    """Validity-masked RGB composite. R/G/B nearest bands (660/550/450 nm)
    for HSI; for thermal (single band) we replicate the channel into all
    three after a 2/98 stretch. Returns (H, W, 3) float32 0..1."""
    import numpy as np

    valid = validity.astype(bool)
    if cube.shape[0] == 1:
        v = cube[0]
        vals = v[valid] if valid.any() else v.ravel()
        if vals.size:
            p2, p98 = np.percentile(vals, [2, 98])
            if p98 - p2 > 1e-6:
                stretched = np.clip((v - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched = np.zeros_like(v)
        else:
            stretched = np.zeros_like(v)
        rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)
        rgb[~valid] = 0.0
        return rgb

    def nearest(target: float) -> int:
        return int(np.argmin(np.abs(wl - target)))

    r_idx = nearest(660)
    g_idx = nearest(550)
    b_idx = nearest(450)
    rgb = np.stack(
        [cube[r_idx], cube[g_idx], cube[b_idx]], axis=-1
    ).astype(np.float32)
    for ch in range(3):
        v = rgb[:, :, ch]
        vals = v[valid] if valid.any() else v.ravel()
        if vals.size == 0:
            continue
        p2, p98 = np.percentile(vals, [2, 98])
        if p98 - p2 < 1e-6:
            continue
        rgb[:, :, ch] = np.clip((v - p2) / (p98 - p2), 0.0, 1.0)
    rgb[~valid] = 0.0
    return rgb
