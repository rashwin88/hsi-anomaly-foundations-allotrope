"""Worker-side implementation of ``spectral_library_match``.

Kept separate from the api-side module so that importing
``spectral_library_match`` for validation / catalog purposes doesn't
pull in numpy / rasterio / scipy / pyarrow.

See the spec doc ``spectal_match_sample/WALKTHROUGH.md`` for the
algorithmic design (Days 1-4 explain the curation, missing-bands
strategy, cache build, and pattern-bucketed SAM core).
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("allotrope.action_types.spectral_library_match")


_MATCHES_PARQUET = "matches.parquet"
_MATCH_MAP_TIF = "match_map.tif"
_MATCH_MAP_PNG = "match_map.png"
_MATCH_MAP_LEGEND_JSON = "match_map_legend.json"
_HISTOGRAM_JSON = "histogram.json"
_SUMMARY_JSON = "summary.json"
# Per-action snapshot of the library entries actually loaded for this match
# (refl + valid + wavelengths). The api's /probe endpoint reads this so the
# viewer can overlay the candidate library spectrum on the pixel spectrum
# without re-opening the per-sensor cache (which lives outside the action's
# artifact tree and isn't reachable via /files/).
_LIBRARY_REFL_NPZ = "library_refl.npz"
# Per-action snapshot of the anomaly pixel spectra (and their per-band
# validity). Lets the viewer preload everything the chart needs into the
# browser at modal-open time and serve every hover/click locally — no api
# round-trip per pixel.
_PIXEL_SPECTRA_NPZ = "anomaly_pixel_spectra.npz"
# Same heatmap as match_map.png but transparent everywhere a pixel isn't
# matched, sized identically to rgb.png so it can be CSS-overlaid.
_MATCH_MAP_OVERLAY_PNG = "match_map_overlay.png"

_PNG_MAX_EDGE = 4096

# Sentinel values painted into match_map.tif.
_SENTINEL_NOT_IN_MASK = -2
_SENTINEL_NO_MATCH = -1


def _resolve_cache_path(ctx: Any, cfg: dict, vendable: Any) -> Path:
    """Find the per-sensor splib07 cache file for this run.

    The cache filename embeds a 16-hex-char SHA-256 of every input that
    affects the cached library; ``sensor_cache_key`` recomputes the same
    hash here so we open the right file.
    """
    from app.spectral_match.library import sensor_cache_key
    import numpy as np

    from ..config import settings

    target_wl = np.asarray(vendable.band_cw_order, dtype=np.float64)
    # PRISMA reads FWHM straight from the HE5 attrs and some bands carry a
    # null FWHM; np.asarray(..., dtype=float64) raises TypeError on None.
    # Coerce None → NaN so the array is always all-float (matches the
    # onboarding vendable's own float semantics).
    target_fwhm = np.asarray(
        [np.nan if v is None else v for v in vendable.band_fwhm_order],
        dtype=np.float64,
    )
    band_validity = np.asarray(vendable.band_validity_by_position, dtype=np.uint8)

    # The slim bundle's version tag is constant across cache rebuilds
    # unless the operator bumped --version in curate_splib07.py and
    # re-ran the per-sensor cache CLI. We pull it from the cache dir's
    # ``splib_version.txt`` if present, else fall back to "splib07a".
    cache_dir = Path(getattr(settings, "splib07_cache_dir", "/splib07_cache"))
    version_file = cache_dir / "splib_version.txt"
    splib_version = (
        version_file.read_text().strip() if version_file.exists() else "splib07a"
    )

    key = sensor_cache_key(
        sensor_id=ctx.sensor_type,
        target_wl_nm=target_wl,
        target_fwhm_nm=target_fwhm,
        bad_band_mask=band_validity,
        chapters=cfg["chapters"],
        min_coverage=cfg["min_coverage"],
        splib07_version=splib_version,
    )
    npz_path = cache_dir / f"splib07_{key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"splib07 cache not found at {npz_path}. "
            f"Build it with `scripts/build_splib_sensor_cache.py "
            f"--sensor-spec <{ctx.sensor_type}.json> --chapters "
            f"{' '.join(cfg['chapters'])} --min-coverage {cfg['min_coverage']}`."
        )
    return npz_path


def run(ctx: Any) -> None:
    import numpy as np
    import rasterio

    from app.spectral_match import (
        load_sensor_cache,
        match_pixels,
        savgol_smooth,
    )

    cfg = ctx.configuration
    t0 = time.monotonic()

    # ---- 1. Resolve upstream anomaly_detection_prep output ----------
    ctx.on_step("resolve_upstream")
    upstream_id = cfg["input_anomaly_detection_output_id"]
    upstream_dir: Path = ctx.resolve_action_output(upstream_id)
    mask_tif = upstream_dir / "anomaly_mask.tif"
    if not mask_tif.exists():
        raise FileNotFoundError(
            f"upstream anomaly_detection_prep output is missing "
            f"anomaly_mask.tif (was the action committed?): {mask_tif}"
        )
    with rasterio.open(mask_tif) as src:
        anomaly_mask = src.read(1).astype(np.uint8)
        match_map_profile = src.profile

    # ---- 2. Load the native onboarding vendable --------------------
    # NOT the band_filter_apply output — splib07 matching needs native
    # bands so narrow absorption features survive (see Day 1 of
    # WALKTHROUGH for the why).
    ctx.on_step("load_onboarding_vendable")
    vendable_path = (
        ctx.data_dir / "scenes" / str(ctx.scene_id) / "vendable" / "vendable.pkl"
    )
    if not vendable_path.exists():
        raise FileNotFoundError(f"onboarding vendable missing: {vendable_path}")
    with vendable_path.open("rb") as f:
        vendable = pickle.load(f)

    cube = vendable.normalized_hyperspectral_cube                 # (B, H, W)
    validity = vendable.validity_cube                             # (B, H, W) int8 1/0
    B, H, W = cube.shape

    if anomaly_mask.shape != (H, W):
        raise ValueError(
            f"anomaly_mask shape {anomaly_mask.shape} != cube spatial "
            f"shape {(H, W)} — upstream is on a different grid."
        )

    # ---- 3. Load the per-sensor splib07 cache ----------------------
    ctx.on_step("load_splib_cache")
    cache_path = _resolve_cache_path(ctx, cfg, vendable)
    logger.info("splib07 cache: %s", cache_path)
    library_entries = load_sensor_cache(cache_path)
    if not library_entries:
        raise RuntimeError(f"splib07 cache {cache_path} loaded zero entries")

    library_refl = np.stack([e.refl for e in library_entries], axis=0).astype(np.float32)
    library_valid = np.stack([e.valid for e in library_entries], axis=0).astype(np.uint8)

    if library_refl.shape[1] != B:
        raise ValueError(
            f"cache band count {library_refl.shape[1]} != vendable band "
            f"count {B} — wrong cache file or vendable mismatch."
        )

    # ---- 4. Pick pixels --------------------------------------------
    ctx.on_step("collect_pixels")
    spatial_valid = (validity.sum(axis=0) > 0).astype(np.uint8)
    if cfg["mode"] == "all_kept":
        pixel_mask = spatial_valid.astype(bool)
    else:
        pixel_mask = (anomaly_mask.astype(bool) & spatial_valid.astype(bool))

    rows, cols = np.where(pixel_mask)
    n_pixels = int(rows.size)
    if n_pixels == 0:
        # Nothing to match — write empty outputs but still succeed so
        # the action lands in `complete` instead of `failed`.
        logger.warning("no pixels to match (mask is empty)")
        return _write_empty_outputs(ctx, anomaly_mask, library_entries, cfg, upstream_id,
                                    cache_path, match_map_profile, n_pixels, t0)

    # Build the (P, B) spectra + (P, B) validity arrays.
    # Cube is (B, H, W); transpose for fancy indexing.
    pixel_spectra = cube[:, rows, cols].T.astype(np.float32)        # (P, B)
    pixel_valid = validity[:, rows, cols].T.astype(np.uint8)        # (P, B) 1/0

    # ---- 5. Savitzky-Golay smooth ----------------------------------
    ctx.on_step("smooth_spectra")
    pixel_spectra_smooth = savgol_smooth(
        pixel_spectra,
        window_length=cfg["sg_window_length"],
        polyorder=cfg["sg_polyorder"],
    )

    # ---- 6. Run the pattern-bucketed SAM matcher -------------------
    ctx.on_step("match_pixels")
    t_match = time.monotonic()
    result = match_pixels(
        pixels=pixel_spectra_smooth,
        pixel_valid=pixel_valid,
        library_refl=library_refl,
        library_valid=library_valid,
        top_k=cfg["top_k"],
        min_coverage=cfg["min_coverage"],
        min_band_count=cfg["min_band_count"],
    )
    match_seconds = time.monotonic() - t_match
    logger.info(
        "matched %d pixels in %.2fs (no_match=%d)",
        n_pixels, match_seconds, int(result.no_match.sum()),
    )

    # ---- 7. Write outputs ------------------------------------------
    ctx.on_step("write_outputs")
    output_dir: Path = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_matches_parquet(
        output_dir / _MATCHES_PARQUET,
        rows=rows, cols=cols, result=result, library_entries=library_entries,
    )
    _write_match_map(
        output_dir / _MATCH_MAP_TIF,
        profile=match_map_profile,
        pixel_mask=pixel_mask,
        rows=rows, cols=cols, result=result,
    )
    _render_match_map_png(
        output_dir / _MATCH_MAP_PNG,
        pixel_mask=pixel_mask,
        rows=rows, cols=cols, result=result,
    )
    _render_match_map_png(
        output_dir / _MATCH_MAP_OVERLAY_PNG,
        pixel_mask=pixel_mask,
        rows=rows, cols=cols, result=result,
        transparent_unmatched=True,
    )
    # Forward the upstream rgb.png so the viewer can ride one CSS layer
    # tree (rgb underlay + angle overlay + snap ring) without reaching
    # back into a different action's artifact dir.
    upstream_rgb = upstream_dir / "rgb.png"
    if upstream_rgb.is_file():
        shutil.copy2(upstream_rgb, output_dir / "rgb.png")
    else:
        logger.warning(
            "upstream output %s has no rgb.png — viewer will fall back to "
            "no-underlay mode", upstream_id,
        )
    _write_legend(
        output_dir / _MATCH_MAP_LEGEND_JSON,
        library_entries=library_entries,
        used_ix=set(int(i) for i in result.library_ix[:, 0] if i >= 0),
    )
    # Snapshot the resampled library spectra used by THIS action so the
    # viewer can serve library curves without re-reading the per-sensor
    # cache (which lives at /splib07_cache, outside the action's artifact
    # tree).
    np.savez(
        output_dir / _LIBRARY_REFL_NPZ,
        refl=library_refl,
        valid=library_valid,
        wavelengths=np.asarray(vendable.band_cw_order, dtype=np.float64),
    )
    # Snapshot the smoothed pixel spectra at the matched rows/cols.
    # ~9 MB for a typical PRISMA action — small enough to ship as a single
    # asset to the browser at modal open and serve all hover/click probes
    # synchronously client-side. Parallel arrays keyed by parquet row order.
    np.savez(
        output_dir / _PIXEL_SPECTRA_NPZ,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        spectra=pixel_spectra_smooth.astype(np.float32),
        valid=pixel_valid.astype(np.uint8),
        wavelengths=np.asarray(vendable.band_cw_order, dtype=np.float64),
    )
    histogram = _build_histogram(result, library_entries)
    (output_dir / _HISTOGRAM_JSON).write_text(json.dumps(histogram, indent=2))

    summary = {
        "upstream_anomaly_detection_output_id": upstream_id,
        "sensor_type": ctx.sensor_type,
        "splib_cache_path": str(cache_path),
        "n_library_entries": len(library_entries),
        "mode": cfg["mode"],
        "top_k": cfg["top_k"],
        "min_coverage": cfg["min_coverage"],
        "min_band_count": cfg["min_band_count"],
        "sg_window_length": cfg["sg_window_length"],
        "sg_polyorder": cfg["sg_polyorder"],
        "chapters": cfg["chapters"],
        "n_pixels": n_pixels,
        "n_pixels_matched": int(n_pixels - int(result.no_match.sum())),
        "n_pixels_no_match": int(result.no_match.sum()),
        "timing_seconds": {
            "match": round(match_seconds, 3),
            "total": round(time.monotonic() - t0, 3),
        },
    }
    (output_dir / _SUMMARY_JSON).write_text(json.dumps(summary, indent=2))

    ctx.on_step("done")
    logger.info("spectral_library_match done — %s", summary)


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    p = Path(output_dir) / _SUMMARY_JSON
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def preview(ctx: Any, output_dir: Any) -> Any:
    """No PNG preview — the viewer renders match_map.tif client-side."""
    return Path(output_dir) / _SUMMARY_JSON


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_matches_parquet(
    out_path: Path, *, rows, cols, result, library_entries
) -> None:
    """One row per (pixel, rank). Filtered to ranks that actually matched."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    K = result.angles_deg.shape[1]
    rec = {
        "row": [], "col": [], "rank": [],
        "library_ix": [], "material_id": [], "name": [],
        "chapter": [], "asd_subtype": [],
        "angle_deg": [], "n_bands_used": [],
    }
    for p in range(rows.size):
        if result.no_match[p]:
            continue
        for k in range(K):
            lix = int(result.library_ix[p, k])
            if lix < 0:
                break
            entry = library_entries[lix]
            rec["row"].append(int(rows[p]))
            rec["col"].append(int(cols[p]))
            rec["rank"].append(k)
            rec["library_ix"].append(lix)
            rec["material_id"].append(entry.material_id)
            rec["name"].append(entry.name)
            rec["chapter"].append(entry.chapter)
            rec["asd_subtype"].append(entry.asd_subtype)
            rec["angle_deg"].append(float(result.angles_deg[p, k]))
            rec["n_bands_used"].append(int(result.n_bands_used[p, k]))

    # Explicit int32 / float32 schema so browser-side parquet readers
    # don't get BigInt back for the row/col/library_ix columns (JS can't
    # mix BigInt with Number in arithmetic).
    # snappy (not zstd) so hyparquet in the browser can decode without
    # an extra wasm codec dep. snappy compresses this categorical /
    # integer-heavy table almost as well as zstd in practice.
    schema = pa.schema([
        pa.field("row", pa.int32()),
        pa.field("col", pa.int32()),
        pa.field("rank", pa.int32()),
        pa.field("library_ix", pa.int32()),
        pa.field("material_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("chapter", pa.string()),
        pa.field("asd_subtype", pa.string()),
        pa.field("angle_deg", pa.float32()),
        pa.field("n_bands_used", pa.int32()),
    ])
    table = pa.table(rec, schema=schema)
    pq.write_table(table, out_path, compression="snappy")


def _write_match_map(
    out_path: Path, *, profile, pixel_mask, rows, cols, result
) -> None:
    """Paint the top-1 library index per matched pixel."""
    import numpy as np
    import rasterio

    H, W = pixel_mask.shape
    out = np.full((H, W), _SENTINEL_NOT_IN_MASK, dtype=np.int32)
    out[pixel_mask] = _SENTINEL_NO_MATCH

    if rows.size:
        top1 = result.library_ix[:, 0].astype(np.int32)
        finite = ~result.no_match
        out[rows[finite], cols[finite]] = top1[finite]

    profile = dict(profile)
    profile.update(
        dtype="int32",
        count=1,
        nodata=_SENTINEL_NOT_IN_MASK,
        compress="deflate",
        predictor=2,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def _render_match_map_png(
    out_path: Path, *, pixel_mask, rows, cols, result,
    transparent_unmatched: bool = False,
) -> None:
    """Angle heatmap of the top-1 SAM angle per matched pixel.

    Colour channel is the top-1 angle in degrees, viridis-style colormap
    (smaller angle = brighter = better match).

    Two flavours via ``transparent_unmatched``:
      * False (default) — standalone thumbnail: unmatched-in-mask pixels
        get dim grey so they're visible against the modal's dark background.
      * True — overlay: every non-matched pixel is fully transparent, so
        the PNG can be CSS-composited on top of rgb.png.

    Material identity intentionally lives only in the side rail and at-pixel
    probe — chasing categorical colour identity on a sparse anomaly map
    looked chaotic in practice; an angle heatmap reads cleanly at a glance.
    """
    import numpy as np
    from matplotlib import cm
    from PIL import Image

    H, W = pixel_mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    if not transparent_unmatched:
        # Unmatched-in-mask pixels: dim grey, fully opaque (so they stand out
        # against the transparent background but are visibly distinct from
        # successful matches).
        rgba[pixel_mask] = (32, 32, 36, 255)

    if rows.size:
        finite = ~result.no_match
        good_rows = rows[finite]
        good_cols = cols[finite]
        good_angles = result.angles_deg[finite, 0].astype(np.float64)

        # Stretch to the [p2, p98] window of the angle distribution.
        # Smaller angle = better match, so we invert before colourising
        # so "brightest = best".
        if good_angles.size:
            lo = float(np.percentile(good_angles, 2))
            hi = float(np.percentile(good_angles, 98))
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((good_angles - lo) / (hi - lo), 0.0, 1.0)
            score = 1.0 - norm  # invert: 0° → 1.0 (brightest)

            lut = (cm.get_cmap("viridis", 256)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            idx = (score * 255).astype(np.uint8)
            rgb = lut[idx]

            rgba[good_rows, good_cols, 0] = rgb[:, 0]
            rgba[good_rows, good_cols, 1] = rgb[:, 1]
            rgba[good_rows, good_cols, 2] = rgb[:, 2]
            rgba[good_rows, good_cols, 3] = 255

            # Dilate by 1 pixel in 8-neighbour for the overlay variant so
            # single matched pixels read against any RGB background.
            # The thumbnail (transparent_unmatched=False) keeps the exact
            # 1-pixel rendering — there's a grey backing there, no need.
            if transparent_unmatched:
                halo_rgb = (rgb.astype(np.uint16) * 3 // 4).astype(np.uint8)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = good_rows + dr
                        cc = good_cols + dc
                        m = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                        # Only fill cells that aren't already a centre
                        # pixel (so centre colour stays brightest).
                        sel = m & (rgba[rr.clip(0, H - 1), cc.clip(0, W - 1), 3] == 0)
                        rr_s = rr[sel]
                        cc_s = cc[sel]
                        rgba[rr_s, cc_s, 0] = halo_rgb[sel, 0]
                        rgba[rr_s, cc_s, 1] = halo_rgb[sel, 1]
                        rgba[rr_s, cc_s, 2] = halo_rgb[sel, 2]
                        rgba[rr_s, cc_s, 3] = 220

    # Downsample if the scene is huge (matches other action types).
    if max(H, W) > _PNG_MAX_EDGE:
        scale = _PNG_MAX_EDGE / max(H, W)
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        img = Image.fromarray(rgba, mode="RGBA").resize(
            (new_w, new_h), Image.NEAREST
        )
    else:
        img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_path, format="PNG", optimize=True)


def _write_legend(out_path: Path, *, library_entries, used_ix) -> None:
    """library_ix → {name, material_id, chapter} for the viewer."""
    legend = {
        str(ix): {
            "name": library_entries[ix].name,
            "material_id": library_entries[ix].material_id,
            "chapter": library_entries[ix].chapter,
            "asd_subtype": library_entries[ix].asd_subtype,
        }
        for ix in sorted(used_ix)
    }
    out_path.write_text(json.dumps(legend, indent=2))


def _build_histogram(result, library_entries) -> dict[str, Any]:
    """Count top-1 occurrences per material."""
    counts: dict[int, int] = {}
    for p in range(result.library_ix.shape[0]):
        if result.no_match[p]:
            continue
        lix = int(result.library_ix[p, 0])
        if lix < 0:
            continue
        counts[lix] = counts.get(lix, 0) + 1

    rows = [
        {
            "library_ix": lix,
            "name": library_entries[lix].name,
            "chapter": library_entries[lix].chapter,
            "count": n,
        }
        for lix, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return {
        "n_distinct_top1": len(rows),
        "top1_counts": rows,
    }


def _write_empty_outputs(
    ctx, anomaly_mask, library_entries, cfg, upstream_id, cache_path,
    match_map_profile, n_pixels, t0,
) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio

    output_dir: Path = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = anomaly_mask.shape
    out = np.full((H, W), _SENTINEL_NOT_IN_MASK, dtype=np.int32)
    profile = dict(match_map_profile)
    profile.update(
        dtype="int32", count=1, nodata=_SENTINEL_NOT_IN_MASK,
        compress="deflate", predictor=2,
    )
    with rasterio.open(output_dir / _MATCH_MAP_TIF, "w", **profile) as dst:
        dst.write(out, 1)

    empty_table = pa.table({
        "row": [], "col": [], "rank": [],
        "library_ix": [], "material_id": [], "name": [],
        "chapter": [], "asd_subtype": [],
        "angle_deg": [], "n_bands_used": [],
    })
    pq.write_table(empty_table, output_dir / _MATCHES_PARQUET, compression="snappy")
    (output_dir / _MATCH_MAP_LEGEND_JSON).write_text("{}")
    (output_dir / _HISTOGRAM_JSON).write_text(
        json.dumps({"n_distinct_top1": 0, "top1_counts": []}, indent=2)
    )
    (output_dir / _SUMMARY_JSON).write_text(json.dumps({
        "upstream_anomaly_detection_output_id": upstream_id,
        "sensor_type": ctx.sensor_type,
        "splib_cache_path": str(cache_path),
        "n_library_entries": len(library_entries),
        "mode": cfg["mode"],
        "n_pixels": n_pixels,
        "n_pixels_matched": 0,
        "n_pixels_no_match": 0,
        "timing_seconds": {"total": round(time.monotonic() - t0, 3)},
        "note": "no pixels in the input mask — empty outputs written.",
    }, indent=2))
