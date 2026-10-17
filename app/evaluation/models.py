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
SEG_TILE, SEG_STRIDE, SEG_BATCH = 128, 64, 16
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
            bm = self.inferencer.model.inner.trained_band_mask(wl).cpu().numpy()
            cube, wl = cube[bm], wl[bm]
        return cube, wl, valid

    def reconstruct(self, scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cube the model read, its reconstruction, valid) - both (C, H, W) float32."""
        import torch
        cube, wl, valid = self._read(scene, self.mode)
        if self.is_stem:
            self.inferencer.set_wavelengths(wl)
        with torch.no_grad():
            recon = self.inferencer.predict_full_scene(torch.from_numpy(cube), torch.from_numpy(valid[None].astype(np.float32)))
        return cube, recon.detach().cpu().numpy().astype(np.float32), valid

    def segment(self, scene) -> np.ndarray:
        """(H, W) int class map (config.CLASS_IDS), 0 outside valid."""
        import torch
        if not self.has_segmentation:
            raise ValueError(f"{self.name}: checkpoint has no segmentation head")
        cube, wl, valid = self._read(scene, "native")
        net = self.inferencer.model.inner.eval()
        dev = next(net.parameters()).device
        C, H, W = cube.shape
        pad = [(-(n - SEG_TILE) % SEG_STRIDE) if n > SEG_TILE else SEG_TILE - n for n in (H, W)]
        x = np.pad(cube, ((0, 0), (0, pad[0]), (0, pad[1])))
        acc, cnt = np.zeros((5,) + x.shape[1:], np.float32), np.zeros(x.shape[1:], np.float32)
        wl_t = torch.as_tensor(wl, dtype=torch.float32, device=dev)
        pos = [(y, z) for y in range(0, x.shape[1] - SEG_TILE + 1, SEG_STRIDE)
               for z in range(0, x.shape[2] - SEG_TILE + 1, SEG_STRIDE)]
        for i in range(0, len(pos), SEG_BATCH):
            chunk = pos[i:i + SEG_BATCH]
            xb = torch.from_numpy(np.stack([x[:, y:y + SEG_TILE, z:z + SEG_TILE] for y, z in chunk])).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                p = torch.sigmoid(net.segment(xb, wl_t).float()).cpu().numpy()
            for (y, z), pp in zip(chunk, p):
                acc[:, y:y + SEG_TILE, z:z + SEG_TILE] += pp
                cnt[y:y + SEG_TILE, z:z + SEG_TILE] += 1
        prob = (acc / np.maximum(cnt, 1))[:, :H, :W]          # water, cloud, thin_cloud, shadow, snow
        cls = np.zeros((H, W), np.int64)
        cls[prob[0] > 0.5] = 1
        cls[prob[3] > 0.5] = 3
        cls[prob[4] > 0.5] = 4
        cls[np.maximum(prob[1], prob[2]) > 0.5] = 2
        return cls * valid

    def release(self) -> None:
        import torch
        self._inferencer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
