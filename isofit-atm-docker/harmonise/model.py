"""
Model file: one .npz of plain arrays, read with allow_pickle=False. Scene keys: sensor, zmu, zsd, zlo, zhi
(standardised covariate range of training), gate (sd margin, NaN = off). Each part <p> has <p>.kind, <p>.wl
(target bands, nm) and: const <p>.value (bands,); own <p>.coef (bands, 2+2k) = [a, c, A, C],
y = (a + A.z) x + c + C.z; ridge <p>.in_wl, <p>.coef (inputs+k+1, bands), y = [x_in, z, 1] @ coef.
"""
import numpy as np

KINDS = {"identity": (), "nodata": (), "const": ("value",), "own": ("coef",), "ridge": ("in_wl", "coef")}


def load(path):
    """(scene arrays, {part: {"kind", "wl", ...}}). ValueError on an unknown kind or a missing array."""
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    parts = {}
    for p in sorted(k[:-5] for k in a if k.endswith(".kind")):
        kind = str(a[f"{p}.kind"])
        if kind not in KINDS or any(f"{p}.{n}" not in a for n in ("wl",) + KINDS[kind]):
            raise ValueError(f"{path}: part {p!r} of kind {kind!r} is unknown or incomplete")
        parts[p] = {k[len(p) + 1:]: v for k, v in a.items() if k.startswith(p + ".")} | {"kind": kind}
    return {k: v for k, v in a.items() if "." not in k}, parts
