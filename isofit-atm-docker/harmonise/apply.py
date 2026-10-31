"""Apply a harmonisation model (harmonise.model) to our reflectance: rows are pixels, columns our native bands."""
import numpy as np

MATCH_NM = 3.0   # a model band applies to our band within this distance


def cols(wl, target):
    """Index of our band for each model band. ValueError if the cube lacks one (never silently skipped)."""
    j = np.abs(np.asarray(wl)[None] - np.asarray(target)[:, None]).argmin(1)
    if np.any(np.abs(np.asarray(wl)[j] - target) > MATCH_NM):
        raise ValueError(f"cube has no band within {MATCH_NM} nm of a model band")
    return j


def standardise(scene, z):
    """(clamped, raw) standardised covariates. Clamped to the training range: a model never extrapolates in z."""
    s = (np.asarray(z, float) - scene["zmu"]) / scene["zsd"]
    return np.clip(s, scene["zlo"], scene["zhi"]), s


def gated(scene, s_raw):
    """H6 gate: True when the scene lies over `gate` sd outside the training range, so it is left as is."""
    g = float(scene["gate"])
    return bool(np.isfinite(g) and np.any((s_raw < scene["zlo"] - g) | (s_raw > scene["zhi"] + g)))


def fitted(part, x, wl, zs):
    """Model values (pixels, part bands) of an own or ridge part. x: (pixels, our bands); zs: clamped z."""
    c, k = part.get("coef"), len(zs)
    if part["kind"] == "own":        # per band: y = (a + A.z) x + c + C.z
        xb = x[:, cols(wl, part["wl"])]
        return (c[:, 0] + c[:, 2:2 + k] @ zs) * xb + (c[:, 1] + c[:, 2 + k:] @ zs)
    if part["kind"] == "ridge":      # spectral: y = [x_in, z, 1] @ coef
        xi = x[:, cols(wl, part["in_wl"])]
        return np.column_stack([xi, np.repeat(zs[None], len(xi), 0), np.ones(len(xi))]) @ c
    raise ValueError(f"fitted() takes an own or ridge part, not {part['kind']!r}")


def lacks_inputs(part, wl):
    """True for a ridge part with an input band the cube lacks (a scene whose SWIR is cut short): it is skipped."""
    return part["kind"] == "ridge" and bool((np.abs(np.asarray(wl)[None] - part["in_wl"][:, None]).min(1) > MATCH_NM).any())


def harmonise(scene, parts, x, wl, z, labels):
    """Harmonised copy of x (pixels, our bands). core_<class> parts apply to that class, others to all pixels;
    in a gated scene own/ridge are skipped, and so is a ridge part that lacks_inputs. Model bands the cube lacks
    are skipped; anything uncovered stays."""
    out, (zs, s_raw) = x.astype(np.float64, copy=True), standardise(scene, z)
    off, live = gated(scene, s_raw), ~np.isnan(x).all(1)   # a pixel with no valid band stays nodata
    for name, p in parts.items():
        rows = (labels == name[5:] if name.startswith("core_") else np.ones(len(x), bool)) & live
        if p["kind"] == "identity" or not rows.any() or (off and p["kind"] in ("own", "ridge")) or lacks_inputs(p, wl):
            continue
        have = np.abs(np.asarray(wl)[None] - p["wl"][:, None]).min(1) <= MATCH_NM
        if p["kind"] in ("nodata", "const"):     # use neither x nor z, so the gate does not apply
            val = np.nan if p["kind"] == "nodata" else p["value"][have]
        else:
            c = p["coef"][have] if p["kind"] == "own" else p["coef"][:, have]
            val = fitted({**p, "wl": p["wl"][have], "coef": c}, x[rows], wl, zs)
        out[np.ix_(rows, cols(wl, p["wl"][have]))] = val
    return out
