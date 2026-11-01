"""
Step 0 of the HyperSL anomaly bench: export each bench scene's validity mask and object map, so the GPU step
(step1, in WSL, which has no product environment) scores exactly the pixels the evaluator counts.

Runs on the Windows product venv:  C:/av/Scripts/python.exe step0_export_masks.py
Writes OUT/set{n}_masks.npz with valid (H, W) uint8 and objects (H, W) int32, plus OUT/bench_bands.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from app.evaluation.datasets import PrismaBench  # noqa: E402

ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT = ROOT / "hypersl_anomaly_bench"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bands = {}
    for scene in PrismaBench(str(ROOT)):
        np.savez_compressed(OUT / f"{scene.name}_masks.npz", valid=scene.valid.astype(np.uint8),
                            objects=scene.objects.astype(np.int32))
        bands[scene.name] = {"native_bands": int(len(scene.wavelengths_nm)), "objects": int(scene.objects.max()),
                             "valid_px": int(scene.valid.sum()), "shape": list(scene.valid.shape)}
        print(scene.name, bands[scene.name], flush=True)
    (OUT / "bench_bands.json").write_text(json.dumps(bands, indent=1))


if __name__ == "__main__":
    main()
