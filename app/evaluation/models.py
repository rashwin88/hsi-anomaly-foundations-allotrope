"""
EvalModel: one checkpoint under evaluation - how it reads a scene, its reconstruction and, if it has a head, its
segmentation (docs/lld/evaluation.md).

Reconstruction goes through the product's full-scene inferencers with the product's Indradhanu defaults (128 px
patches, stride 32, two complementary token masks), so the numbers are the product's. Reading, unless set:
wavelength stems read native bands, keeping only bands their trained-wavelength record covers
(docs/lld/spectral-stem.md); fixed stems read se202; hyperspectral_segformer_mae (v0.2.0) reads product165.

Segmentation is tiled: 128 px windows at stride 64, sigmoid probabilities averaged where windows overlap; classes
by priority cloud (cloud or thin cloud) > snow > shadow > water > land (config.CLASS_IDS).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PATCH, STRIDE, BATCH = 128, 32, 8           # backend/allotrope/foundation_models/resolver.py defaults
REPO = Path(__file__).resolve().parents[2]


def local_checkpoint(uri: str, cache_dir: Path) -> Path:
    if not uri.startswith("s3://"):
        return Path(uri)
    from app.utils.general_utils import s3_config
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    path = cache_dir / "checkpoints" / key.replace("/", "__")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        s3_config.client().download_file(bucket, key, str(path))
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class EvalModel:
    def __init__(self, spec, cache_dir: Path):
        import torch
        self.spec, self.name = spec, spec.name
        self.path = local_checkpoint(spec.checkpoint, Path(cache_dir))
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.model_config = ck["config"]["model_config"]
        self.epoch = ck.get("epoch")
        self.kind = self.model_config["model_type"]
        self.mode = spec.reading if spec.reading != "auto" else self._default_mode()
        self._inferencer = None

    def _default_mode(self) -> str:
        from app.foundation_models.components.spectral_stem_seg_former_mae import WAVELENGTH_STEMS
        if self.kind == "hyperspectral_segformer_mae":
            return "product165"
        return "native" if self.model_config.get("stem") in WAVELENGTH_STEMS else "se202"

    @property
    def inferencer(self):
        if self._inferencer is None:
            from app.foundation_models.inferencers.hyperspectral_segformer_mae_inferencer import (
                HyperspectralSegFormerMAEInferencer)
            from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
            from app.models.training.inference_config import InferenceConfig
            common = dict(model_config=self.model_config, checkpoint_path=str(self.path), patch_size=PATCH,
                          stride=STRIDE, inference_batch_size=BATCH)
            if self.kind == "hyperspectral_segformer_mae":
                self._inferencer = HyperspectralSegFormerMAEInferencer(InferenceConfig(
                    foundation_model_name="hyperspectral_segformer_mae", **common,
                    pixel_stats_path=str(REPO / "app" / "constants" / "hyperspectral.json")))
            else:
                self._inferencer = SpectralStemInferencer(InferenceConfig(foundation_model_name="spectral_stem_mae", **common))
        return self._inferencer

    @property
    def is_stem(self) -> bool:
        return self.kind != "hyperspectral_segformer_mae"

    @property
    def has_segmentation(self) -> bool:
        return self.is_stem and getattr(self.inferencer.model.inner, "seg_head", None) is not None

    def _read(self, scene, mode: str):
        cube, wl, valid = scene.read(mode)
        if self.is_stem:
            cube, wl, _ = self.inferencer.read_bands(cube, wl)
        return cube, wl, valid

    @property
    def has_scale(self) -> bool:
        return self.is_stem and getattr(self.inferencer.model.inner, "scale_head", None) is not None

    def reconstruct(self, scene) -> tuple:
        """(cube the model read, its reconstruction, valid, log b or None) - cube and reconstruction (C, H, W) float32;
        log b (H, W) only for a checkpoint with an uncertainty head, from the same pass."""
        import torch
        cube, wl, valid = self._read(scene, self.mode)
        x, m = torch.from_numpy(cube), torch.from_numpy(valid[None].astype(np.float32))
        if self.spec.adapt_steps:
            return self._adapted_reconstruction(cube, wl, valid, x, m)
        with torch.no_grad():
            if self.has_scale:
                recon, log_b = self.inferencer.predict_full_scene_with_scale(x, m)
                log_b = log_b.detach().cpu().numpy().astype(np.float32)
            else:
                recon, log_b = self.inferencer.predict_full_scene(x, m), None
        return cube, recon.detach().cpu().numpy().astype(np.float32), valid, log_b

    def _adapted_reconstruction(self, cube, wl, valid, x, m) -> tuple:
        """Gated per-scene adaptation (app/foundation_models/scene_adaptation.py): gate on the unadapted model's
        residual Mahalanobis score, adapt, reconstruct, then restore the original weights for the next scene."""
        import torch
        from app.foundation_models.scene_adaptation import adapt, gate_mask
        from app.utils.anomaly_detection.residual_mahalanobis import residual_mahalanobis
        if not self.is_stem:
            raise ValueError(f"{self.name}: per-scene adaptation needs a wavelength stem")
        s = self.spec
        net = self.inferencer.model.inner
        original = {k: v.detach().clone() for k, v in net.state_dict().items()}
        try:
            with torch.no_grad():
                recon0 = self.inferencer.predict_full_scene(x, m).cpu().numpy().astype(np.float32)
            gate = gate_mask(residual_mahalanobis(cube, recon0, valid), valid, s.adapt_gate_pct, s.adapt_halo)
            del recon0
            self.last_adapt_losses = adapt(net, cube, wl, valid, gate, s.adapt_steps, lr=s.adapt_lr,
                                           device=self.inferencer.device)
            with torch.no_grad():
                recon = self.inferencer.predict_full_scene(x, m).cpu().numpy().astype(np.float32)
        finally:
            net.load_state_dict(original)
        return cube, recon, valid, None

    def segment(self, scene) -> np.ndarray:
        """(H, W) int class map (config.CLASS_IDS), 0 outside valid - SpectralStemInferencer.segment_full_scene,
        the same code the product's anomaly_scoring runs."""
        if not self.has_segmentation:
            raise ValueError(f"{self.name}: checkpoint has no segmentation head")
        cube, _, valid = self._read(scene, "native")
        return self.inferencer.segment_full_scene(cube, valid, thin_cloud=self.spec.cloud_rule != "thick")[0]

    def release(self) -> None:
        import torch
        self._inferencer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
