"""Dump the HyperSL checkpoint's tensor names, shapes and sizes to figdata/hypersl_ckpt_shapes.json (CPU only).
Run in WSL Ubuntu (Windows torch fails on this laptop):
  ~/hypersl_eval/.venv/bin/python tools/hypersl_ckpt_shapes.py
Checkpoint: C:/hypersl_eval/10_base_mask95_checkpoint.pt (sha256 3d5f1914...a74a68c, HF WeilKon/HyperSL)."""
import hashlib, json
from pathlib import Path
import torch
CK = Path("/mnt/c/hypersl_eval/10_base_mask95_checkpoint.pt")
OUT = Path(__file__).resolve().parent.parent / "figdata" / "hypersl_ckpt_shapes.json"
ck = torch.load(CK, map_location="cpu", weights_only=False)
sd = ck["model"]
rows = [{"name": k, "shape": list(v.shape), "numel": int(v.numel())} for k, v in sd.items()]
OUT.write_text(json.dumps({"sha256": hashlib.sha256(CK.read_bytes()).hexdigest(), "epoch": ck.get("epoch"),
                           "keys": sorted(ck.keys()), "tensors": rows, "total_numel": sum(r["numel"] for r in rows)}, indent=1))
print(len(rows), sum(r["numel"] for r in rows), ck.get("epoch"))
