"""Pull a few PRISMA test tiles (one per scene) from fine-tune shard 0 into one npz for H5 (README)."""
import io, tarfile, sys
import numpy as np
from app.utils.shard_codec import decode_i16dz

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tiles, seen = {}, set()
with tarfile.open(src) as tf:
    for m in tf:
        key, _, ext = m.name.partition(".")
        if ext not in ("pixels.i16dz", "pixels.npy", "wavelengths.npy"):
            continue
        scene = key.rsplit("_", 1)[0]
        if scene in seen and key not in tiles:
            continue
        data = tf.extractfile(m).read()
        d = tiles.setdefault(key, {}); seen.add(scene)
        d[ext] = decode_i16dz(data) if ext.endswith("i16dz") else np.load(io.BytesIO(data))
        if len(tiles) >= n and all(len(v) == 2 for v in tiles.values()):
            break
out = {}
for k, d in tiles.items():
    raw = d.get("pixels.i16dz", d.get("pixels.npy"))
    wl = d["wavelengths.npy"]
    print(k, raw.shape, wl.shape, float(wl.min()), float(wl.max()), "invalid band share", float((raw == -32768).all(axis=(1, 2)).mean()))
    out[f"{k}__pix"], out[f"{k}__wl"] = raw, wl
np.savez_compressed(dst, **out)
