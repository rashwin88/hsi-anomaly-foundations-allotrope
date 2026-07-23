"""Worker-side implementation of the `anomaly_scoring` action.

Kept in its own module so the api-side import of `anomaly_scoring.py`
doesn't pay for torch / rasterio / matplotlib / app.foundation_models —
this file is only loaded when the worker calls `run`.

Recipe per the action's META:

  1. Load (filtered, on HSI) vendable.
  2. Optional: load keep_mask from the upstream scene_segmentation Output.
  3. Optional: load GT raster from the bound Scene's annotation.
  4. For each picked model codename:
       a. Resolve codename → checkpoint + inferencer config.
       b. Apply per-codename overrides (scoring_method, patch/stride/
          batch_size, sam_l1_alpha) on top of capability defaults.
       c. predict_full_scene → reconstruction.
       d. Score with the resolved method.
       e. Apply keep_mask if provided.
       f. Write per-model rasters (score + reconstruction) and
          rendered PNG previews under <output>/models/<codename>/.
       g. If GT: compute ROC + AUC.
  5. Render an RGB scene PNG + thumbnail montage.
  6. Write summary.json (lean) + diagnostics.json (rich).

No thresholding, no detection raster — this is the raw scoring pass.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("allotrope.action_types.anomaly_scoring")


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (Inf / -Inf / NaN) with None.

    Python's `json.dumps` emits the literal `Infinity` token for `float('inf')`
    by default; that's valid Python but not valid JSON, so JS `JSON.parse`
    rejects it. The ROC threshold bookends and any other score-range fields
    can carry infinities, so sanitize at the write boundary instead of
    chasing them at every producer.
    """
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False))


_SUMMARY_FILENAME = "summary.json"
_DIAGNOSTICS_FILENAME = "diagnostics.json"
# Cap on dots returned in gt_dots.json. Larger GTs are uniformly
# downsampled. Trade-off: SVG with >5k circles starts to feel laggy on
# panzoom; subsampling preserves spatial distribution of positives so
# the user still gets a visual sanity check.
_GT_MAX_DOTS = 5000
_PREVIEW_FILENAME = "preview.png"
_RGB_FILENAME = "rgb.png"
_MODELS_DIR_NAME = "models"

# Cap the max edge of any rendered PNG. The frontend zooms into these
# directly via a <canvas>; bigger than 4k blows up memory on the host
# without buying more usable detail than panzoom can address.
_PNG_MAX_EDGE = 4096
_THUMBNAIL_DPI = 110


def run(ctx: Any) -> None:
    """Top-level entrypoint. See module docstring."""
    import numpy as np
    import rasterio
    import torch

    from app.foundation_models.inferencers.inferencer_factory import get_inferencer
    from app.models.training.inference_config import (
        InferenceConfig,
        PixelStatsOverride,
    )
    from app.utils.anomaly_detection.scoring import compute_roc, compute_score
    from app.utils.torch_helpers.device_selection import get_device

    from ..config import settings
    from ..foundation_models.resolver import list_catalog, sensor_family

    cfg = ctx.configuration

    # --- 1. Load the vendable -----------------------------------------
    scene_family = sensor_family(ctx.sensor_type)
    is_thermal = scene_family == "thermal"

    bf_output_id = cfg.get("input_band_filter_output_id")
    if is_thermal:
        ctx.on_step("load_onboarding_vendable")
        pickle_path = (
            ctx.data_dir
            / "scenes"
            / str(ctx.scene_id)
            / "vendable"
            / "vendable.pkl"
        )
    else:
        if not bf_output_id:
            raise ValueError(
                "anomaly_scoring on a hyperspectral scene needs "
                "input_band_filter_output_id."
            )
        ctx.on_step("load_filtered_vendable")
        bf_dir = ctx.resolve_action_output(bf_output_id)
        pickle_path = bf_dir / "filtered_vendable.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"vendable pickle missing: {pickle_path}")
    with pickle_path.open("rb") as f:
        vendable = pickle.load(f)

    if is_thermal:
        thermal = vendable.normalized_thermal_cube
        if thermal.ndim == 2:
            cube_np = thermal[None, :, :].astype(np.float32, copy=False)
        else:
            cube_np = thermal.astype(np.float32, copy=False)
        v = vendable.validity_cube
        if v.ndim == 2:
            validity_np = v[None, :, :].astype(np.int8, copy=False)
        else:
            validity_np = v.astype(np.int8, copy=False)
        wavelengths = np.array([10900.0], dtype=np.float64)
    else:
        cube_np = vendable.normalized_hyperspectral_cube
        validity_np = vendable.validity_cube
        wavelengths = np.asarray(vendable.band_cw_order, dtype=np.float64)

    spatial_valid = (validity_np.sum(axis=0) > 0).astype(np.uint8)

    # --- 2. Optional keep_mask --------------------------------------
    # HSI scenes can attach a scene_segmentation output;
    # thermal scenes can attach a cloud_mask output. Cross-namespace
    # combinations are rejected at validate_config time.
    keep_mask: np.ndarray = spatial_valid.copy()
    keep_source = "spatial_validity"
    if cfg.get("input_scene_segmentation_output_id"):
        ctx.on_step("load_keep_mask")
        ss_dir = ctx.resolve_action_output(cfg["input_scene_segmentation_output_id"])
        keep_tif = ss_dir / "keep_mask.tif"
        if keep_tif.exists():
            with rasterio.open(keep_tif) as src:
                km = src.read(1).astype(np.uint8)
            keep_mask = (km & spatial_valid).astype(np.uint8)
            keep_source = "scene_segmentation.keep_mask ∧ spatial_validity"
        else:
            logger.warning("scene_segmentation output missing keep_mask.tif")
    elif cfg.get("input_cloud_mask_output_id"):
        ctx.on_step("load_keep_mask")
        cm_dir = ctx.resolve_action_output(cfg["input_cloud_mask_output_id"])
        keep_tif = cm_dir / "keep_mask.tif"
        if keep_tif.exists():
            with rasterio.open(keep_tif) as src:
                km = src.read(1).astype(np.uint8)
            keep_mask = (km & spatial_valid).astype(np.uint8)
            keep_source = "cloud_mask.keep_mask ∧ spatial_validity"
        else:
            logger.warning("cloud_mask output missing keep_mask.tif")

    # --- 3. Optional GT raster --------------------------------------
    gt_np: np.ndarray | None = None
    gt_meta: dict[str, Any] | None = None
    if cfg.get("input_annotation_id"):
        ctx.on_step("load_ground_truth")
        gt_np, gt_meta = _load_annotation_gt(ctx, cfg["input_annotation_id"])

    # --- 3b. Per-scene normalisation override for uncalibrated sensors --
    # The foundation models (Chakshu thermal, Indradhanu hyperspectral)
    # carry baked per-band normalisation stats that assume the input
    # cube is in the same units as their training data (Kelvin /
    # reflectance respectively). Sensors that ship vendables in a
    # different unit space — currently HotSat-1 L2 Visual, which is
    # uncalibrated 14-bit DN — would be many standard deviations out of
    # the training distribution if fed directly. We override the
    # normalisation stats with per-scene (mean, std) computed from the
    # valid pixels of THIS scene so the model sees an input that is
    # roughly N(0,1) per band, matching what it learned to reconstruct.
    #
    # Trade-off: scores become scene-relative — comparable WITHIN this
    # scene but not across scenes. The action's diagnostics carry the
    # ``normalization_mode`` field so the UI can show a banner saying
    # so. Skip this for hyperspectral scenes (their vendables already
    # come out of band_filter_apply in reflectance units that match the
    # training distribution) and for Landsat (already in Celsius).
    pixel_stats_override: PixelStatsOverride | None = None
    normalization_mode = "baked"
    units = getattr(vendable, "units", None)
    if isinstance(units, str) and units.startswith("DN_"):
        # Compute per-band (mean, std) over keep_mask ∧ spatial_valid.
        # Falls back to spatial_valid if no keep_mask was attached.
        # We use float64 for the moments to avoid catastrophic cancel-
        # lation on the squared term — HotSat DN sits around 5000±400,
        # so var = E[x²] − E[x]² is ~160000 − ~25e6 in raw float32.
        cube_f64 = cube_np.astype(np.float64, copy=False)
        mask_2d = keep_mask.astype(bool)
        if not mask_2d.any():
            mask_2d = spatial_valid.astype(bool)
        means: list[float] = []
        stds: list[float] = []
        for b in range(cube_f64.shape[0]):
            vals = cube_f64[b][mask_2d]
            if vals.size == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            m = float(vals.mean())
            s = float(vals.std())
            # Guard against degenerate scenes where every kept pixel
            # has the same value — a zero std would divide by zero in
            # the model's normalisation layer.
            if not math.isfinite(s) or s < 1e-6:
                s = 1.0
            means.append(m)
            stds.append(s)
        pixel_stats_override = PixelStatsOverride(
            mean=means,
            std=stds,
            source="per_scene_dn_zscore",
        )
        normalization_mode = "per_scene_dn_zscore"
        logger.info(
            "Per-scene DN z-score override (units=%s): mean=%s std=%s "
            "(from %d / %d kept pixels)",
            units, means, stds, int(mask_2d.sum()), int(mask_2d.size),
        )

    # --- 4. Per-model loop ------------------------------------------
    catalog = list_catalog(Path(settings.models_dir))
    by_codename = {m.codename.lower(): m for m in catalog}

    output_dir: Path = ctx.output_dir
    models_root = output_dir / _MODELS_DIR_NAME
    models_root.mkdir(parents=True, exist_ok=True)

    model_overrides: dict[str, dict[str, Any]] = cfg.get("model_overrides", {})

    device = get_device()
    device_label = str(device)

    scene_tensor = torch.from_numpy(cube_np).float()
    mask_tensor = torch.from_numpy(spatial_valid[None]).float()

    per_model_records: list[dict[str, Any]] = []
    roc_records: dict[str, dict[str, Any]] = {}

    for codename in cfg["model_codenames"]:
        ctx.on_step(f"model={codename} · resolve")
        m = by_codename.get(codename.strip().lower())
        if m is None:
            raise ValueError(f"unknown codename at run time: {codename!r}")

        ovr = model_overrides.get(codename, {}) or {}
        method = ovr.get("scoring_method") or m.default_scoring_method
        patch_size = int(ovr.get("patch_size") or m.default_patch_size)
        stride = int(ovr.get("stride") or m.default_stride)
        batch_size = int(ovr.get("batch_size") or m.default_batch_size)
        sam_l1_alpha = float(ovr.get("sam_l1_alpha") or 0.5)
        # Optional erosion kernel override for the SegFormer-MAE
        # family. None → keep the InferenceConfig default (15) baked
        # into the inferencer; SegFormer reads `self.config.erosion_kernel_size`
        # inside predict_full_scene. Autoencoder family ignores it.
        erosion_ks_override = ovr.get("erosion_kernel_size")

        # Optional keep_mask erosion. Applies to BOTH foundation and
        # classical paths — strips off the boundary-rim score artifact
        # where cloud/water/segmentation edges otherwise score high.
        # Default (None) means kernel=1 → no erosion → keep_mask used
        # as-is, preserving existing behavior. Odd-int validation
        # happened at submit time.
        keep_mask_erosion_ks = int(
            ovr.get("keep_mask_erosion_kernel_size") or 1
        )
        if keep_mask_erosion_ks > 1:
            from scipy.ndimage import binary_erosion
            half = keep_mask_erosion_ks // 2
            structure = np.ones(
                (keep_mask_erosion_ks, keep_mask_erosion_ks), dtype=bool
            )
            eroded_keep_mask = binary_erosion(
                keep_mask.astype(bool), structure=structure
            ).astype(np.uint8)
            logger.info(
                "model=%s · keep_mask eroded by %d (kernel=%d): "
                "%d → %d pixels kept",
                codename, half, keep_mask_erosion_ks,
                int(keep_mask.sum()), int(eroded_keep_mask.sum()),
            )
        else:
            eroded_keep_mask = keep_mask

        # Two dispatch paths. The "foundation" path loads a torch
        # checkpoint and produces a reconstruction; the "classical"
        # path instantiates a stateless detector from app.detectors,
        # which returns a score map directly (no reconstruction).
        # Outputs are unified: both write anomaly_score.{tif,png} and a
        # reconstruction.{tif,png} (classical's "reconstruction" is a
        # copy of the input cube — visually meaningful, lets the
        # viewer's three-panel layout stay the same).
        is_classical = m.family == "classical"

        if is_classical:
            from app.models.ad_models.ad_model import ADModel
            from app.utils.anomaly_detection.detector_factory import get_detector

            ctx.on_step(f"model={codename} · instantiate_detector")
            t0 = time.time()
            detector_cls = get_detector(ADModel(m.detector_key))
            detector = detector_cls(vendable)
            load_s = time.time() - t0

            ctx.on_step(f"model={codename} · fit")
            t0 = time.time()
            detector.fit()
            fit_s = time.time() - t0

            # Restrict the classical detector's background ROI to
            # keep_mask. RX/MNF-RX/Thermal-GRX have NO trained prior —
            # their covariance is estimated fresh from whichever pixels
            # the internal `_spatial_mask` selects. The `fit()` step
            # builds that mask from the vendable's validity cube alone
            # and never sees the upstream Action's keep_mask
            # (segmentation or cloud_mask). We AND keep_mask into the
            # spatial_mask here, after fit() and before detect(), so
            # `detect()`'s `pixels = working[:, _spatial_mask]` selects
            # only kept-ROI pixels for the covariance.
            #
            # Foundation models keep the original "score everywhere,
            # mask at render" semantics — pre-masking their input would
            # push it out-of-distribution at mask boundaries.
            #
            # Without keep_mask attached, keep_mask == spatial_valid
            # and this is a no-op.
            if hasattr(detector, "_spatial_mask") and detector._spatial_mask is not None:
                detector._spatial_mask = (
                    detector._spatial_mask & eroded_keep_mask.astype(bool)
                )
                logger.info(
                    "classical detector spatial_mask narrowed to keep_mask: "
                    "%d kept pixels",
                    int(detector._spatial_mask.sum()),
                )

            ctx.on_step(f"model={codename} · detect")
            score = detector.detect(cube_np, validity_np)
            infer_s = time.time() - t0
            # The "reconstruction" panel for a classical model is the
            # input itself — visually clean, no special case in the
            # viewer. For HSI we write the full cube so the panel can
            # composite an RGB; for thermal we write the single band.
            recon_np = cube_np.astype(np.float32, copy=False)
            del detector
        else:
            ic_kwargs: dict[str, Any] = dict(
                foundation_model_name=m.foundation_model_name,
                model_config=m.model_config,
                checkpoint_path=m.checkpoint_abs_path,
                patch_size=patch_size,
                stride=stride,
                inference_batch_size=batch_size,
                pixel_stats_path=m.pixel_stats_abs_path,
            )
            # Per-Action erosion override (SegFormer-MAE family only).
            # InferenceConfig's pydantic field validates odd-ness + range;
            # we already pre-validated at the action submit boundary.
            if erosion_ks_override is not None:
                ic_kwargs["erosion_kernel_size"] = int(erosion_ks_override)
            # Per-scene normalisation override (uncalibrated sensors
            # such as HotSat). When ``pixel_stats_override`` is set we
            # also drop the baked ``pixel_stats_path`` so the
            # inferencer doesn't pointlessly read it from disk; the
            # in-memory override would win anyway.
            if pixel_stats_override is not None:
                # Sanity check: override length must match in_channels
                # — for the thermal SegFormerMAE this is always 1, but
                # we validate to catch any future single-band-by-mistake
                # bugs early.
                in_ch = getattr(m.model_config, "in_channels", None)
                if in_ch is not None and len(pixel_stats_override.mean) != int(in_ch):
                    raise ValueError(
                        "pixel_stats_override length "
                        f"{len(pixel_stats_override.mean)} does not match "
                        f"model in_channels={int(in_ch)} for codename={codename!r}."
                    )
                ic_kwargs["pixel_stats_override"] = pixel_stats_override
                ic_kwargs["pixel_stats_path"] = None
            inference_cfg = InferenceConfig(**ic_kwargs)

            ctx.on_step(f"model={codename} · load_checkpoint")
            t0 = time.time()
            inferencer = get_inferencer(inference_cfg)
            load_s = time.time() - t0

            ctx.on_step(f"model={codename} · predict_full_scene")
            t0 = time.time()
            with torch.no_grad():
                reconstruction = inferencer.predict_full_scene(
                    scene_tensor, mask_tensor
                )
            infer_s = time.time() - t0
            recon_np = reconstruction.detach().cpu().numpy().astype(np.float32, copy=False)

            ctx.on_step(f"model={codename} · score={method}")
            # Use the eroded keep_mask here so cloud/water/segmentation
            # boundary rings don't bleed through into the score. The
            # raw keep_mask is still used for ROC computation below
            # (we want ROC computed over the user's stated ROI, not
            # the eroded one) and for percentile stats.
            score = compute_score(
                cube_np.astype(np.float32, copy=False),
                recon_np,
                eroded_keep_mask,
                method=method,
                combined_weight=sam_l1_alpha,
            )

        # Write per-model rasters + rendered previews.
        model_dir = models_root / _safe_dirname(codename)
        model_dir.mkdir(parents=True, exist_ok=True)

        _save_raster_2d(score, model_dir / "anomaly_score.tif", "float32")
        _save_raster_3d(recon_np, model_dir / "reconstruction.tif", "float32")

        ctx.on_step(f"model={codename} · render")
        # Classical (RX-family) scores are squared Mahalanobis
        # distances — χ²-distributed with a heavy right tail. Render
        # with sqrt-stretch so the bulk of pixels lands in the mid-LUT
        # range where inferno actually has perceptual contrast.
        # Foundation reconstruction errors are roughly unimodal so the
        # linear stretch still works there.
        _render_score_png(
            score=score,
            # Use eroded keep_mask here too — the score raster is only
            # well-defined inside the detector's spatial_mask (which
            # we narrowed to keep_mask AND eroded_keep_mask for
            # classical) and inside compute_score's eroded keep for
            # foundation. Passing raw keep_mask would feed NaN pixels
            # to the percentile cap and the PNG goes black.
            kept=eroded_keep_mask.astype(bool),
            out_path=model_dir / "anomaly_score.png",
            stretch="sqrt" if is_classical else "linear",
        )
        _render_recon_png(
            recon=recon_np,
            wavelengths=wavelengths,
            spatial_valid=spatial_valid,
            out_path=model_dir / "reconstruction.png",
        )

        # ROC if GT.
        roc: dict[str, Any] | None = None
        if gt_np is not None:
            ctx.on_step(f"model={codename} · roc")
            # ROC evaluated over the eroded keep_mask: that's where
            # the score is actually defined. Using raw keep_mask would
            # feed NaN pixels from the eroded ring into the threshold
            # sweep and degenerate the curve. The user's "stated ROI"
            # nuance is preserved because eroded_keep_mask defaults to
            # raw keep_mask when keep_mask_erosion_kernel_size=1.
            roc = compute_roc(score, gt_np, eroded_keep_mask)
            roc_records[codename] = roc

        # Per-model stats (timings + score range).
        #
        # Sample over the ERODED keep_mask so we don't pick up NaN /
        # zero pixels in the ring between raw_keep_mask and the
        # detector's actual spatial_mask. Classical detectors emit NaN
        # outside their internal mask; np.percentile over an array
        # containing NaN returns NaN (→ JSON null), which breaks the
        # frontend diagnostics renderer. Belt-and-braces: also strip
        # any non-finite values that snuck in from foundation-side
        # rounding.
        kept = eroded_keep_mask.astype(bool)
        candidate = score[kept]
        if candidate.size > 0:
            finite_mask = np.isfinite(candidate)
            valid_scores = candidate[finite_mask]
        else:
            valid_scores = candidate
        score_percentiles = (
            np.percentile(valid_scores, [50, 90, 95, 99, 99.5, 99.9]).tolist()
            if valid_scores.size > 0
            else [0.0] * 6
        )
        stats = {
            "codename": codename,
            "architecture": m.architecture,
            "sensor": m.sensor,
            "family": m.family,
            "method": method,
            # patch/stride/batch are meaningless for classical detectors
            # (whole-cube ops). Serialize as None so the viewer's stats
            # table can render "—" instead of "0".
            "patch_size": None if is_classical else patch_size,
            "stride": None if is_classical else stride,
            "batch_size": None if is_classical else batch_size,
            "sam_l1_alpha": sam_l1_alpha if (not is_classical and method == "combined") else None,
            "erosion_kernel_size": (
                None if is_classical
                else (int(erosion_ks_override) if erosion_ks_override is not None else 15)
            ),
            "keep_mask_erosion_kernel_size": keep_mask_erosion_ks,
            # Normalisation provenance — "baked" means the model used its
            # training-time pixel stats, "per_scene_dn_zscore" means the
            # action handler overrode them with per-scene stats (e.g. for
            # HotSat L2 Visual). "n/a" for classical detectors that
            # don't have a normalisation step.
            "normalization_mode": "n/a" if is_classical else normalization_mode,
            "scene_units": units if isinstance(units, str) else None,
            "device": device_label,
            "load_seconds": round(load_s, 3),
            "infer_seconds": round(infer_s, 3),
            "score_min": float(valid_scores.min()) if valid_scores.size else 0.0,
            "score_max": float(valid_scores.max()) if valid_scores.size else 0.0,
            "score_mean": float(valid_scores.mean()) if valid_scores.size else 0.0,
            "score_percentiles": {
                "p50": score_percentiles[0],
                "p90": score_percentiles[1],
                "p95": score_percentiles[2],
                "p99": score_percentiles[3],
                "p99_5": score_percentiles[4],
                "p99_9": score_percentiles[5],
            },
            "auc": (roc or {}).get("auc"),
        }
        _write_json(model_dir / "stats.json", stats)
        per_model_records.append(stats)

        # Drop per-model references so torch/numpy memory doesn't pile
        # up across a multi-codename run.
        #
        # `inferencer` and `reconstruction` are only bound in the
        # foundation branch above (see ~line 290). On the classical
        # branch they never exist, so `del` would raise NameError —
        # guard with `is_classical`. The `# noqa: F821` tells the
        # linter that the conditional `del` IS intentional even
        # though pyflakes can't prove the names are defined in this
        # scope. (`runner._process_one_tick` runs `reclaim_memory()`
        # in its finally block, which adds malloc_trim on top of
        # whatever GC catches here.)
        if not is_classical:
            del inferencer  # noqa: F821
            del reconstruction  # noqa: F821
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 5. Render the RGB scene + thumbnail montage ---------------
    ctx.on_step("render_rgb")
    rgb = _make_rgb(cube_np, wavelengths, spatial_valid)
    _save_png_array(rgb, output_dir / _RGB_FILENAME)

    ctx.on_step("render_thumbnail")
    try:
        _render_thumbnail(
            output_path=output_dir / _PREVIEW_FILENAME,
            rgb=rgb,
            keep_mask=keep_mask,
            models_root=models_root,
            per_model_records=per_model_records,
        )
    except Exception as e:
        logger.warning("thumbnail render failed (non-fatal): %s", e)

    # --- 6. Write summary + rich diagnostics -----------------------
    summary = {
        "n_models": len(per_model_records),
        "model_codenames": cfg["model_codenames"],
        # Scene id (wire format) lets the frontend call the existing
        # /scenes/{id}/spectrum endpoint for click-to-inspect inside
        # the action viewer modal without an extra round-trip to fetch
        # the Action row.
        "scene_id": f"scene_{ctx.scene_id}",
        "sensor_type": ctx.sensor_type,
        "scene_shape": [int(cube_np.shape[1]), int(cube_np.shape[2])],
        "band_count": int(cube_np.shape[0]),
        "kept_pct": (
            round(100.0 * float(keep_mask.sum()) / float(spatial_valid.sum() or 1), 3)
            if spatial_valid.any()
            else 0.0
        ),
        "keep_source": keep_source,
        "device": device_label,
        "has_gt": gt_np is not None,
        # Normalisation provenance for the score panel. "baked" =
        # foundation model used its training-time pixel stats.
        # "per_scene_dn_zscore" = action overrode the stats with
        # per-scene (mean, std) computed from this scene's valid
        # pixels — only set for uncalibrated sensors (HotSat-1 L2
        # Visual). The UI shows a banner explaining that scores are
        # scene-relative when this mode is active.
        "normalization_mode": normalization_mode,
        "scene_units": units if isinstance(units, str) else None,
        "per_model": [
            {
                "codename": r["codename"],
                "architecture": r["architecture"],
                "method": r["method"],
                "auc": r.get("auc"),
                "infer_seconds": r["infer_seconds"],
                "score_p99_5": r["score_percentiles"]["p99_5"],
            }
            for r in per_model_records
        ],
    }
    _write_json(output_dir / _SUMMARY_FILENAME, summary)

    diagnostics = {
        **summary,
        "per_model_full": per_model_records,
        "roc": roc_records if roc_records else None,
        "gt_meta": gt_meta,
        "wavelengths_nm": [float(w) for w in wavelengths],
    }
    _write_json(output_dir / _DIAGNOSTICS_FILENAME, diagnostics)

    if roc_records:
        _write_json(output_dir / "roc.json", roc_records)

    # --- 7. GT-dot sidecar (frontend overlay + click-to-spectrum) --
    # When a GT annotation was attached, materialise the positive-pixel
    # coordinates as a tiny JSON sidecar. The anomaly viewer modal uses
    # this to render an optional cyan-dot overlay on the score / RGB
    # panels and to drive a click-to-inspect spectrum probe.
    #
    # Capped at _GT_MAX_DOTS so very dense GT masks don't blow up the
    # SVG; the frontend just gets a uniform random subsample. The .tif
    # GT raster stays untouched on disk for the ROC + diagnostics path.
    if gt_np is not None:
        import numpy as np

        ctx.on_step("gt_dots_sidecar")
        rows, cols = np.where(gt_np > 0)
        n_positive = int(rows.size)
        if n_positive > _GT_MAX_DOTS:
            rng = np.random.default_rng(seed=42)
            sel = rng.choice(n_positive, size=_GT_MAX_DOTS, replace=False)
            rows = rows[sel]
            cols = cols[sel]
            sampled = True
        else:
            sampled = False
        _write_json(
            output_dir / "gt_dots.json",
            {
                "scene_shape": [int(cube_np.shape[1]), int(cube_np.shape[2])],
                "n_positive": n_positive,
                "n_dots": int(rows.size),
                "sampled": sampled,
                # Pairs of (row, col) — frontend scales to display
                # coords using scene_shape vs the panel's rendered px.
                "pixels": list(zip(rows.tolist(), cols.tolist())),
            },
        )

    ctx.on_step("done")


def summarize(ctx: Any, output_dir: Any) -> dict[str, Any]:
    """Lean payload for `action_outputs.summary` JSONB."""
    p = output_dir / _SUMMARY_FILENAME
    if p.exists():
        return json.loads(p.read_text())
    d = output_dir / _DIAGNOSTICS_FILENAME
    if d.exists():
        return json.loads(d.read_text())
    return {"error": "diagnostics_missing"}


def preview(ctx: Any, output_dir: Any) -> Any:
    """Thumbnail is rendered inside `run`. Return its path if present."""
    p = output_dir / _PREVIEW_FILENAME
    return p if p.exists() else None


# --- Internals ---------------------------------------------------------


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
            f"no .tif under {annotation_dir} — supported annotation types "
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
        Mahalanobis squared-distance is χ²-distributed (very long
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
    # same location — np.percentile then returns NaN and the entire PNG
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
        # Strip NaNs before the percentile cap — classical detectors
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
            title += f" · AUC={rec['auc']:.3f}"
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
