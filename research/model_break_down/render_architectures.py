"""Render torchview block diagrams + torchinfo param tables for each foundation model.

Outputs:
  model_break_down/diagrams/<arch>.svg  (graphviz block diagram)
  model_break_down/diagrams/<arch>.txt  (torchinfo summary)

Re-run after architecture or checkpoint changes:
  .venv/bin/python model_break_down/render_architectures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.foundation_models.components.spatial_auto_encoder import SpatialAutoencoder
from app.foundation_models.components.unnormalized_spatial_auto_encoder import (
    UnNormalizedSpatialAutoencoder,
)
from app.foundation_models.components.normalized_masked_spatial_auto_encoder import (
    NormalizedMaskedSpatialAutoencoder,
)
from app.foundation_models.components.seg_former_mae import SegFormerMAE
from app.foundation_models.components.hyperspectral_seg_former_mae import (
    HyperspectralSegFormerMAE,
)

from torchinfo import summary
from torchview import draw_graph


OUT_DIR = ROOT / "model_break_down" / "diagrams"
OUT_DIR.mkdir(exist_ok=True)
CHECKPOINTS = ROOT / "checkpoints"


def _load_cfg(arch: str) -> dict:
    manifest = json.loads((CHECKPOINTS / arch / "current.json").read_text())
    pt_path = CHECKPOINTS / arch / manifest["current"]["file"]
    return torch.load(pt_path, map_location="cpu", weights_only=False)["config"][
        "model_config"
    ]


def build_spatial_autoencoder() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("spatial_autoencoder")
    model = SpatialAutoencoder(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_stages=cfg["num_stages"],
    )
    H = 1 << (cfg["num_stages"] + 4)  # ensure divisible by 2^stages
    x = torch.randn(1, cfg["in_channels"], H, H)
    return model, (x,), ["x"]


def build_spatial_masked_autoencoder() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("spatial_masked_autoencoder")
    model = SpatialAutoencoder(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_stages=cfg["num_stages"],
    )
    H = 1 << (cfg["num_stages"] + 5)
    x = torch.randn(1, cfg["in_channels"], H, H)
    mask = torch.ones(1, 1, H, H)
    return model, (x, mask), ["x", "mask"]


def build_spatial_masked_autoencoder_l1() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("spatial_masked_autoencoder_l1")
    model = SpatialAutoencoder(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_stages=cfg["num_stages"],
        kernel_size=cfg.get("kernel_size", 4),
    )
    H = 1 << (cfg["num_stages"] + 4)
    x = torch.randn(1, cfg["in_channels"], H, H)
    mask = torch.ones(1, 1, H, H)
    return model, (x, mask), ["x", "mask"]


def build_spatial_masked_autoencoder_l1_unnormalized() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("spatial_masked_autoencoder_l1_unnormalized")
    model = UnNormalizedSpatialAutoencoder(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_stages=cfg["num_stages"],
        kernel_size=cfg.get("kernel_size", 4),
    )
    H = 1 << (cfg["num_stages"] + 5)
    x = torch.randn(1, cfg["in_channels"], H, H)
    validity = torch.ones(1, 1, H, H)
    input_mask = torch.ones(1, 1, H, H)
    return model, (x, validity, input_mask), ["x", "validity_mask", "input_mask"]


def build_normalized_masked_autoencoder() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("normalized_masked_autoencoder")
    model = NormalizedMaskedSpatialAutoencoder(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        num_stages=cfg["num_stages"],
    )
    H = 1 << (cfg["num_stages"] + 5)
    x = torch.randn(1, cfg["in_channels"], H, H)
    validity = torch.ones(1, 1, H, H)
    input_mask = torch.ones(1, 1, H, H)
    return model, (x, validity, input_mask), ["x", "validity_mask", "input_mask"]


def build_segformer_mae() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("segformer_mae")
    model = SegFormerMAE(
        in_channels=cfg["in_channels"],
        embed_dims=cfg["embed_dims"],
        num_heads=cfg["num_heads"],
        reduction_ratios=cfg["reduction_ratios"],
        num_blocks=cfg["num_blocks"],
        decoder_dim=cfg["decoder_dim"],
        expansion_ratio=cfg["expansion_ratio"],
        drop_rate=cfg["drop_rate"],
    )
    H = 256
    x = torch.randn(1, cfg["in_channels"], H, H)
    return model, (x,), ["x"]


def build_hyperspectral_segformer_mae() -> tuple[torch.nn.Module, tuple, list[str]]:
    cfg = _load_cfg("hyperspectral_segformer_mae")
    model = HyperspectralSegFormerMAE(
        in_channels=cfg["in_channels"],
        compressed_channels=cfg["compressed_channels"],
        embed_dims=cfg["embed_dims"],
        num_heads=cfg["num_heads"],
        reduction_ratios=cfg["reduction_ratios"],
        num_blocks=cfg["num_blocks"],
        decoder_dim=cfg["decoder_dim"],
        expansion_ratio=cfg["expansion_ratio"],
        drop_rate=cfg["drop_rate"],
    )
    H = 128
    x = torch.randn(1, cfg["in_channels"], H, H)
    return model, (x,), ["x"]


BUILDERS = {
    "spatial_autoencoder": build_spatial_autoencoder,
    "spatial_masked_autoencoder": build_spatial_masked_autoencoder,
    "spatial_masked_autoencoder_l1": build_spatial_masked_autoencoder_l1,
    "spatial_masked_autoencoder_l1_unnormalized": build_spatial_masked_autoencoder_l1_unnormalized,
    "normalized_masked_autoencoder": build_normalized_masked_autoencoder,
    "segformer_mae": build_segformer_mae,
    "hyperspectral_segformer_mae": build_hyperspectral_segformer_mae,
}


def render(arch: str) -> None:
    print(f"\n=== {arch} ===")
    model, args, names = BUILDERS[arch]()
    model.eval()

    info = summary(
        model,
        input_data=args,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=3,
        verbose=0,
    )
    txt_path = OUT_DIR / f"{arch}.txt"
    txt_path.write_text(str(info))
    print(f"  wrote {txt_path.relative_to(ROOT)}")

    g = draw_graph(
        model,
        input_data=args,
        graph_name=arch,
        depth=3,
        expand_nested=True,
        save_graph=False,
    )
    g.visual_graph.format = "svg"
    out = OUT_DIR / arch
    g.visual_graph.render(filename=str(out), cleanup=True)
    print(f"  wrote {out.with_suffix('.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(BUILDERS.keys())
    for a in targets:
        try:
            render(a)
        except Exception as e:
            print(f"  FAILED {a}: {e}")
