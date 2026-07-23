"""Resolve a codename (Indradhanu / Chakshu / …) to the inference-time
artifacts and capabilities the anomaly_scoring action needs.

The catalog is rooted at the on-disk `<MODELS_DIR>/<architecture>/current.json`
manifests already mounted into the api + worker containers (Step 12g).
Each manifest already carries:

  - architecture slug
  - codename (name · script · meaning · why)
  - sensor (thermal | hyperspectral)
  - current.{file, version, epoch, params, val_loss}
  - normalization (baked_in stats)

This module layers two pieces of information on top of that:

  - `FoundationModelName` enum mapping → the inferencer factory key
  - scoring capabilities (which methods make sense for this architecture +
    sensor combination — thermal models can't use SAM, etc.)

Both are small static tables that mirror the canonical model lineup. New
models added to `checkpoints/` need an entry here AND in their per-arch
`current.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ScoringMethod = Literal["L1", "MSE", "SAM", "combined", "mahalanobis"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Static per-architecture knowledge the resolver layers on top of
    the on-disk `current.json` manifest."""

    architecture: str
    foundation_model_name: str          # value for app's FoundationModelName enum
    scoring_methods: tuple[ScoringMethod, ...]
    default_scoring_method: ScoringMethod
    default_patch_size: int
    default_stride: int
    # "foundation" → torch checkpoint via FoundationInferencer.
    # "classical"  → stateless detector from app.detectors via detector_factory.
    # The worker branches on this; everything else (output layout, viewer,
    # ROC) is identical so the two paths coexist in one anomaly_scoring
    # Action without forking the action type itself.
    family: str = "foundation"
    # For `family == "classical"`: the ADModel enum value (e.g. "rx",
    # "mnf_rx", "thermal_grx"). Drives the detector_factory lookup.
    # None for foundation models.
    detector_key: str | None = None
    # Inference batch size — number of patches forwarded in parallel.
    # Conservative defaults; the dialog exposes this as an override since
    # it's the main memory dial when running on a CPU-only Mac VM.
    default_batch_size: int = 8
    # Valid patch sizes the dialog offers. Patch size must divide cleanly
    # into the model's token grid; 64/128/256 are the safe set for the
    # autoencoders and segformer at our scene sizes.
    valid_patch_sizes: tuple[int, ...] = (64, 128, 256)
    # Pixel-normalization stats file the inferencer's `build_model`
    # needs at construction time so its PixelNormalize layer exists and
    # the checkpoint's `normalize.*` / `denormalize.*` buffers can load
    # cleanly. None means the architecture doesn't have those layers
    # (e.g. the `_unnormalized` Asanskrita variant).
    #
    # Value is relative to the `app/` package — resolved at runtime via
    # `app.__file__` so it works in both the worker container (where
    # app/ sits at /srv/app/) and local dev.
    pixel_stats_relpath: str | None = None


_THERMAL_STATS = "constants/thermal_pixel_consts.json"
_HSI_STATS = "constants/hyperspectral.json"


_CAPABILITIES: dict[str, ModelCapabilities] = {
    "spatial_autoencoder": ModelCapabilities(
        architecture="spatial_autoencoder",
        foundation_model_name="spatial_autoencoder",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="MSE",
        default_patch_size=128,
        default_stride=64,
        pixel_stats_relpath=_THERMAL_STATS,
    ),
    # Antardhana — same SpatialAutoencoder class as Pratibimba (1
    # channel · has PixelNormalize), just trained with random masking
    # + MSE loss. Route to the SpatialAutoencoderInferencer.
    "spatial_masked_autoencoder": ModelCapabilities(
        architecture="spatial_masked_autoencoder",
        foundation_model_name="spatial_autoencoder",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="MSE",
        default_patch_size=128,
        default_stride=64,
        pixel_stats_relpath=_THERMAL_STATS,
    ),
    # Tirohita — same as Antardhana but L1 loss. Same model class.
    "spatial_masked_autoencoder_l1": ModelCapabilities(
        architecture="spatial_masked_autoencoder_l1",
        foundation_model_name="spatial_autoencoder",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="L1",
        default_patch_size=128,
        default_stride=64,
        pixel_stats_relpath=_THERMAL_STATS,
    ),
    "spatial_masked_autoencoder_l1_unnormalized": ModelCapabilities(
        architecture="spatial_masked_autoencoder_l1_unnormalized",
        foundation_model_name="spatial_masked_autoencoder_l1_unnormalized",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="L1",
        default_patch_size=128,
        default_stride=64,
        # No `_unnormalized` checkpoint carries normalize.* buffers.
        pixel_stats_relpath=None,
    ),
    "normalized_masked_autoencoder": ModelCapabilities(
        architecture="normalized_masked_autoencoder",
        foundation_model_name="normalized_masked_autoencoder",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="L1",
        default_patch_size=128,
        default_stride=64,
        pixel_stats_relpath=_THERMAL_STATS,
    ),
    "segformer_mae": ModelCapabilities(
        architecture="segformer_mae",
        foundation_model_name="segformer_mae",
        scoring_methods=("L1", "MSE"),
        default_scoring_method="L1",
        default_patch_size=256,
        default_stride=128,
        pixel_stats_relpath=_THERMAL_STATS,
    ),
    "hyperspectral_segformer_mae": ModelCapabilities(
        architecture="hyperspectral_segformer_mae",
        foundation_model_name="hyperspectral_segformer_mae",
        scoring_methods=("L1", "SAM", "combined"),
        default_scoring_method="combined",
        default_patch_size=128,
        default_stride=32,
        pixel_stats_relpath=_HSI_STATS,
    ),
    # --- Classical statistical detectors -----------------------------
    # Closed-form, no learned weights. The worker dispatches via
    # app.utils.anomaly_detection.detector_factory and skips the
    # foundation inferencer entirely. They produce the same (H, W)
    # score map, so the rest of the anomaly_scoring pipeline (TIFF +
    # PNG render, ROC, viewer panels) is unchanged.
    #
    # Plain RX on hyperspectral is intentionally not shipped (dropped
    # 2026-05-11). With 150+ surviving bands the covariance comes out
    # near-singular and Mahalanobis distances blow up to 1e11. Use
    # MNF-RX instead — same machinery, but RX runs in a 10-dim
    # noise-whitened subspace so the covariance is well-conditioned.
    "mnf_rx": ModelCapabilities(
        architecture="mnf_rx",
        foundation_model_name="",
        scoring_methods=("mahalanobis",),
        default_scoring_method="mahalanobis",
        default_patch_size=0,
        default_stride=0,
        family="classical",
        detector_key="mnf_rx",
    ),
    "thermal_grx": ModelCapabilities(
        architecture="thermal_grx",
        foundation_model_name="",
        scoring_methods=("mahalanobis",),
        default_scoring_method="mahalanobis",
        default_patch_size=0,
        default_stride=0,
        family="classical",
        detector_key="thermal_grx",
    ),
}


def resolve_pixel_stats_path(relpath: str | None) -> str | None:
    """Resolve a capability's `pixel_stats_relpath` to an absolute path
    inside the worker container. Anchored at the `app/` package so it
    works in both Docker (`/srv/app/`) and local dev (`<repo>/app/`).
    Returns None if the architecture doesn't carry stats."""
    if not relpath:
        return None
    import app  # type: ignore[import-not-found]

    pkg_root = Path(app.__file__).resolve().parent
    p = (pkg_root / relpath).resolve()
    return str(p) if p.exists() else None


@dataclass(frozen=True)
class ResolvedModel:
    """Everything the action_run handler needs to load + score one model."""

    architecture: str
    codename: str
    codename_script: str
    sensor: str                          # "thermal" | "hyperspectral"
    label: str                           # human-readable model label
    foundation_model_name: str
    # Foundation models: absolute path to the .pt; classical: "".
    checkpoint_abs_path: str
    checkpoint_filename: str
    model_config: dict                   # passed straight to InferenceConfig.model_config
    scoring_methods: tuple[ScoringMethod, ...]
    default_scoring_method: ScoringMethod
    default_patch_size: int
    default_stride: int
    default_batch_size: int
    valid_patch_sizes: tuple[int, ...]
    pixel_stats_abs_path: str | None     # absolute path; None when arch has no baked-in normalize
    family: str = "foundation"           # "foundation" | "classical"
    detector_key: str | None = None      # ADModel value when family == "classical"


_SCENE_SENSOR_FAMILY: dict[str, str] = {
    "prisma": "hyperspectral",
    "enmap": "hyperspectral",
    "landsat9": "thermal",
    # AVIRIS-NG comes out of band_filter_apply as the same 165-band
    # common-grid reflectance cube PRISMA/EnMAP produce, so the same
    # hyperspectral foundation models apply unchanged.
    "aviris_ng": "hyperspectral",
    # HotSat-1 L2 Visual is single-band thermal but ships uncalibrated
    # DN, not Kelvin. The anomaly_scoring action handler detects the
    # vendable's ``units == "DN_14bit_relative"`` and overrides the
    # foundation model's pixel-normalisation stats with per-scene
    # (mean, std) so the input distribution matches what the thermal
    # model was trained against. Scores are scene-relative — not
    # comparable across scenes — and the action diagnostics record
    # ``normalization_mode="per_scene_dn_zscore"`` so downstream
    # consumers know what they're looking at.
    "hotsat1": "thermal",
}


def sensor_family(scene_sensor_type: str) -> str:
    """Map a scene's `sensor_type` ('prisma' / 'enmap' / 'landsat9') to
    the model-side family ('hyperspectral' / 'thermal'). Raises
    ValueError on unknown sensors."""
    fam = _SCENE_SENSOR_FAMILY.get(scene_sensor_type)
    if fam is None:
        raise ValueError(
            f"unknown scene sensor_type {scene_sensor_type!r}; "
            f"expected one of {sorted(_SCENE_SENSOR_FAMILY)}"
        )
    return fam


def list_catalog(models_root: Path) -> list[ResolvedModel]:
    """Enumerate every model with a `current.json` manifest under
    `models_root`. Used by the api `/action-types` catalog + the
    NewActionDialog picker.

    Logs a WARNING if two manifests publish the same `codename.name`
    (case-insensitive). The picker uses the codename as a lookup key
    (`by_codename = {m.codename.lower(): m}` in
    `_anomaly_scoring_run.py`), so collisions would silently route to
    whichever manifest sorted last. Caught at catalog time means a new
    manifest with a bad name shows up in the api logs the moment the
    process boots — much better than a baffling routing bug at run
    time.
    """
    import logging as _logging
    _log = _logging.getLogger("allotrope.foundation_models.resolver")

    out: list[ResolvedModel] = []
    if not models_root.exists():
        return out
    for p in sorted(models_root.iterdir()):
        if not p.is_dir():
            continue
        manifest_path = p / "current.json"
        if not manifest_path.exists():
            continue
        try:
            out.append(_from_manifest(p, manifest_path))
        except (KeyError, ValueError):
            # Unknown architecture (no _CAPABILITIES entry) — skip.
            continue

    # Collision check — codename is the runtime routing key for
    # anomaly_scoring. Duplicate names would silently route to
    # whichever manifest sorts last.
    seen: dict[str, str] = {}  # lowered codename -> architecture
    for m in out:
        key = m.codename.strip().lower()
        if key in seen:
            _log.warning(
                "model catalog: codename collision — %r used by both "
                "architecture %r and %r; codename lookups will route to "
                "the last-loaded manifest. Rename one of them.",
                m.codename,
                seen[key],
                m.architecture,
            )
        else:
            seen[key] = m.architecture
    return out


def resolve_by_codename(models_root: Path, codename: str) -> ResolvedModel:
    """Look up the model whose `codename.name` matches `codename`
    (case-insensitive). Raises ValueError if absent."""
    needle = codename.strip().lower()
    for m in list_catalog(models_root):
        if m.codename.lower() == needle:
            return m
    raise ValueError(f"unknown model codename: {codename!r}")


def _from_manifest(arch_dir: Path, manifest_path: Path) -> ResolvedModel:
    data = json.loads(manifest_path.read_text())
    arch = data["architecture"]
    caps = _CAPABILITIES.get(arch)
    if caps is None:
        raise KeyError(f"no capabilities registered for architecture {arch!r}")

    current = data.get("current", {}) or {}
    # Classical detectors don't ship a checkpoint — tolerate a missing
    # or empty `current.file`.
    ckpt_filename = current.get("file", "") if isinstance(current, dict) else ""
    ckpt_path = arch_dir / ckpt_filename if ckpt_filename else arch_dir

    # Pull the saved model_config from the checkpoint manifest when it's
    # available. Falls back to a minimal reconstruction from `current_dims`
    # fields if needed; the saved .pt always carries the full one but
    # opening the pickle is expensive — manifests already snapshot the
    # key fields. Classical detectors get an empty dict.
    model_config = _model_config_from_manifest(arch, current) if caps.family == "foundation" else {}

    codename = data["codename"]
    return ResolvedModel(
        architecture=arch,
        codename=codename["name"],
        codename_script=codename["script"],
        sensor=data["sensor"],
        label=data["label"],
        foundation_model_name=caps.foundation_model_name,
        checkpoint_abs_path=str(ckpt_path) if ckpt_filename else "",
        checkpoint_filename=ckpt_filename,
        model_config=model_config,
        scoring_methods=caps.scoring_methods,
        default_scoring_method=caps.default_scoring_method,
        default_patch_size=caps.default_patch_size,
        default_stride=caps.default_stride,
        default_batch_size=caps.default_batch_size,
        valid_patch_sizes=caps.valid_patch_sizes,
        pixel_stats_abs_path=resolve_pixel_stats_path(caps.pixel_stats_relpath),
        family=caps.family,
        detector_key=caps.detector_key,
    )


def _model_config_from_manifest(arch: str, current: dict) -> dict:
    """Reconstruct the model_config dict the inferencer factory expects.

    Each architecture's config schema differs; we pull only the fields
    needed at inference time. The full canonical config can also be read
    from the .pt file under `config.model_config` — that's the deepest
    source of truth — but doing that here would require torch.load just
    to enumerate the catalog, which the api doesn't want.

    Strategy: read from `current.json`, which carries enough info to
    rebuild the model graph (channels, stages, embed dims, …).

    Subtlety on `in_channels` for the masked variants:
        UnNormalizedSpatialAutoencoder and NormalizedMaskedSpatialAutoencoder
        both take `in_channels` to mean the **thermal-only** channel count
        (i.e., 1 for Landsat B10). Inside the constructor they wire
        `SpatialEncoder(in_channels + 2, ...)` because validity + input
        masks are stacked into the encoder. The `encoder_dims` field in
        `current.json` shows the *encoder's* input count (which is 3
        for those variants); we subtract 2 to recover the constructor
        `in_channels`.
    """
    if arch in (
        "spatial_autoencoder",
        "spatial_masked_autoencoder",
        "spatial_masked_autoencoder_l1",
        "spatial_masked_autoencoder_l1_unnormalized",
        "normalized_masked_autoencoder",
    ):
        # encoder_dims string like "1→32→64→128→256" or "3→64→128"
        dims = _parse_dims(current.get("encoder_dims", ""))
        encoder_input = dims[0] if dims else 1
        base_channels = dims[1] if len(dims) >= 2 else 32
        num_stages = max(1, len(dims) - 1)

        # Map architecture → (Pydantic-config model_type literal,
        # constructor in_channels). The mask-aware variants take
        # thermal-only in_channels; encoder.input = in_channels + 2.
        if arch in ("spatial_autoencoder",):
            model_type = "spatial_autoencoder"
            constructor_in_channels = encoder_input
        elif arch in ("spatial_masked_autoencoder", "spatial_masked_autoencoder_l1"):
            # Antardhana / Tirohita route through SpatialAutoencoderInferencer
            # (same class as Pratibimba — 1-channel input, has PixelNormalize).
            model_type = "spatial_autoencoder"
            constructor_in_channels = encoder_input
        elif arch == "spatial_masked_autoencoder_l1_unnormalized":
            # Asanskrita — UnNormalizedSpatialAutoencoder, encoder takes
            # 3 channels but constructor's in_channels is the thermal-only count.
            model_type = "spatial_masked_autoencoder"
            constructor_in_channels = max(1, encoder_input - 2)
        elif arch == "normalized_masked_autoencoder":
            # Drashta — same shape as Asanskrita on the encoder side, but
            # with PixelNormalize on the thermal channel.
            model_type = "normalized_masked_autoencoder"
            constructor_in_channels = max(1, encoder_input - 2)
        else:
            raise KeyError(f"no builder branch for arch {arch!r}")

        cfg: dict = {
            "model_type": model_type,
            "in_channels": constructor_in_channels,
            "base_channels": base_channels,
            "num_stages": num_stages,
        }
        return cfg

    if arch == "segformer_mae":
        # encoder_dims "[16, 32, 64, 96]"
        embed_dims = _parse_int_list(current.get("encoder_dims", "[]"))
        decoder_dim = int(current.get("decoder_dim", 96))
        return {
            "model_type": arch,
            "in_channels": 1,
            "embed_dims": embed_dims,
            "num_heads": [1, 2, 4, 8][: len(embed_dims)] or [1, 2, 4, 8],
            "reduction_ratios": [8, 4, 2, 1][: len(embed_dims)] or [8, 4, 2, 1],
            "num_blocks": [1] * len(embed_dims) or [1, 1, 1, 1],
            "decoder_dim": decoder_dim,
            "expansion_ratio": 2,
            "drop_rate": 0.0,
        }

    if arch == "hyperspectral_segformer_mae":
        embed_dims = _parse_int_list(current.get("encoder_dims", "[]"))
        decoder_dim = int(current.get("decoder_dim", 256))
        spectral_dim = int(current.get("spectral_dim_D", 32))
        return {
            "model_type": arch,
            "in_channels": 165,
            "compressed_channels": spectral_dim,
            "embed_dims": embed_dims,
            "num_heads": [2, 2, 5, 8][: len(embed_dims)] or [2, 2, 5, 8],
            "reduction_ratios": [8, 4, 2, 1][: len(embed_dims)] or [8, 4, 2, 1],
            "num_blocks": [2] * len(embed_dims) or [2, 2, 2, 2],
            "decoder_dim": decoder_dim,
            "expansion_ratio": 4,
            "drop_rate": 0.0,
        }

    raise KeyError(f"no model_config builder for architecture {arch!r}")


def _parse_dims(s: str) -> list[int]:
    """Parse 'A→B→C' (with various arrow chars and optional 'ch') into ints.
    Tolerates trailing tokens like '· H/4' that some manifests use."""
    out: list[int] = []
    head = s.split("·")[0]  # strip detail suffix
    for token in head.replace("→", " ").replace("->", " ").split():
        token = token.strip().rstrip(",").lower().replace("ch", "")
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _parse_int_list(s: str) -> list[int]:
    """Parse '[16, 32, 64, 96]' into [16, 32, 64, 96]."""
    cleaned = s.strip().strip("[]")
    if not cleaned:
        return []
    out: list[int] = []
    for tok in cleaned.split(","):
        try:
            out.append(int(tok.strip()))
        except ValueError:
            continue
    return out
