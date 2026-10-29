"""
Write a model's `current.json` for the product's models volume (`<models>/<architecture>/current.json`), read from a
spectral-stem checkpoint. The api cannot torch.load, so the manifest embeds the checkpoint's full `model_config`
(backend/allotrope/foundation_models/resolver.py: _model_config_from_manifest).

    python scripts/make_model_manifest.py C:/es/models/FT04_best.pt --version v2.0.0 \
        --codename "Indradhanu V2" --out C:/es/ship/spectral_stem_finetune

Copies the checkpoint next to the manifest under `<architecture>_<version>_epoch<N>.pt` (docs/proposals/ship-ft04.md).
The label, meaning and notes default to the FT04 text; a later checkpoint behind the same codename passes its own, and
`--previous <old current.json>` moves the checkpoint it replaces (and that manifest's alternatives) into `alternatives`
so it stays reachable. Indradhanu Heavy (v2.1.0, 2026-10-29) was written with:

    python scripts/make_model_manifest.py C:/es/models/FTH3_ep5.pt --version v2.1.0 --codename "Indradhanu V2" \
        --label "Indradhanu Heavy: hyperspectral spectral-stem SegFormer MAE + segmentation (3x wide)" \
        --notes-source research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2 \
        --previous C:/es/ship/fth3/ft04_current.json --previous-note "Indradhanu V2 v2.0.0 (FT04), shipped 2026-10-19" \
        --out C:/es/ship/fth3/spectral_stem_finetune
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

import torch

SCRIPT = {"Indradhanu V2": "इंद्रधनु V2"}
INFERENCER = ("SpectralStemInferencer", "app.foundation_models.inferencers.spectral_stem_inferencer")


LABEL = "Hyperspectral spectral-stem SegFormer MAE + segmentation"
MEANING = "rainbow (literally 'Indra's bow'), second generation"
NOTES_SOURCE = "research/experiments/indradhanu-finetune/FT04-final-joint"


def previous_alternatives(path: Path, note: str) -> list[dict]:
    """The replaced manifest's current checkpoint (with `note`) followed by its own alternatives."""
    old = json.loads(path.read_text(encoding="utf-8"))
    return [{**old["current"], "note": note}] + list(old.get("alternatives", []))


def manifest(ck: dict, file: str, version: str, codename: str, why: str, label: str = LABEL, meaning: str = MEANING,
             notes_source: str = NOTES_SOURCE, alternatives: list[dict] | None = None) -> dict:
    cfg = ck["config"]["model_config"]
    sd = ck["model_state_dict"]
    wl = sd["stats.wl"]
    return {
        "architecture": cfg["model_type"],
        "label": label,
        "codename": {"name": codename, "script": SCRIPT.get(codename, codename),
                     "meaning": meaning, "why": why},
        "sensor": "hyperspectral",
        "family": "reconstruction",
        "current": {"file": file, "version": version, "epoch": int(ck["epoch"]),
                    "params": int(sum(v.numel() for k, v in sd.items() if not k.startswith(("stats.", "trained_wl")))),
                    "encoder_dims": str(cfg["embed_dims"]), "decoder_dim": int(cfg["decoder_dim"]),
                    "spectral_dim_D": int(cfg["compressed_channels"]), "val_loss": round(float(ck["avg_val_loss"]), 5),
                    "model_config": cfg},
        "alternatives": alternatives or [],
        "normalization": {
            "mode": "baked_in", "applied_in_forward": True, "stats_shape": [int(wl.numel())],
            "mean": f"per-wavelength curve ({wl.numel()} points, {float(wl.min()):.0f}-{float(wl.max()):.0f} nm)",
            "std": f"per-wavelength curve ({wl.numel()} points)",
            "source": "checkpoint buffers stats.wl / stats.mean / stats.std, interpolated to each scene's native "
                      "wavelengths. Inputs are reflectance (0..~1), at native bands - not the 165-band grid."},
        "doc": "docs/lld/spectral-stem.md",
        "inferencer": INFERENCER[0],
        "inferencer_module": INFERENCER[1],
        "notes": "Reads each sensor's native bands (needs band_filter_apply with use_common_wavelength_grid=false). "
                 "Segmentation head (water, cloud, thin cloud, shadow, snow); anomaly_scoring zeros its thick cloud "
                 "after scoring (thin cloud is not excluded). Fine-tuned from SpectralEarth pretraining on EnMAP, "
                 f"PRISMA and AVIRIS-NG ({notes_source}).",
        "updated_at": datetime.date.today().isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--version", required=True)
    p.add_argument("--codename", required=True)
    p.add_argument("--why", default="Native bands of every sensor, plus its own cloud / water / shadow masks.")
    p.add_argument("--label", default=LABEL)
    p.add_argument("--meaning", default=MEANING)
    p.add_argument("--notes-source", default=NOTES_SOURCE, help="the experiment folder the notes cite")
    p.add_argument("--previous", type=Path, help="the current.json this manifest replaces; its checkpoint becomes an "
                                                 "alternative (the .pt file must stay in the volume)")
    p.add_argument("--previous-note", default="previous checkpoint behind this codename")
    p.add_argument("--out", type=Path, required=True, help="the architecture's directory in the models volume")
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    arch = ck["config"]["model_config"]["model_type"]
    file = f"{arch}_{a.version}_epoch{int(ck['epoch'])}.pt"
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.checkpoint, a.out / file)
    alts = previous_alternatives(a.previous, a.previous_note) if a.previous else []
    m = manifest(ck, file, a.version, a.codename, a.why, a.label, a.meaning, a.notes_source, alts)
    (a.out / "current.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out / 'current.json'} and {file}")


if __name__ == "__main__":
    main()
