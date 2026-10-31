"""Which sensor delivered this cube, and the per-sensor facts a delivery may leave out.

Identification, strongest first: the file name (product naming conventions), then header / tag text, then the
spectral signature (band count and range). A signature match is "derived"; nothing matching is "unknown", which
runs with generic rules and says so.

Radiance units: from header or tag text when present (metadata); otherwise from magnitude (derived): the scale
among 10x-apart candidates whose implied TOA reflectance at 860 nm is plausible (scene 99th percentile 0.15-1.2).
Candidates are a decade apart, so a sun-zenith error of up to 2x in cos cannot pick the wrong one.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np

from .scene import Fact

# alt_km: nominal orbit (assumed when the delivery has no path length); isofit: apply_oe's sensor argument family
SENSORS = {
    "enmap": {"alt_km": 653.0},
    "prisma": {"alt_km": 615.0},
    "tanager": {"alt_km": 522.0},
    "ghost": {"alt_km": 500.0},       # Orbital Sidekick GHOSt: not published per delivery
    "aviris_ng": {"alt_km": None},    # airborne: altitude must come from obs / path length
    "unknown": {"alt_km": 500.0},
}
NAME_RULES = (
    (re.compile(r"ENMAP01-", re.I), "enmap"),
    (re.compile(r"PRS_L[12]", re.I), "prisma"),
    (re.compile(r"\bang\d{8}t\d{6}", re.I), "aviris_ng"),
    (re.compile(r"\d{8}_\d{6}_\d{2}_4001", re.I), "tanager"),
    (re.compile(r"ghost|orbital.?sidekick", re.I), "ghost"),
)
TEXT_RULES = (("enmap", "enmap"), ("prisma", "prisma"), ("aviris", "aviris_ng"), ("tanager", "tanager"),
              ("ghost", "ghost"), ("orbital sidekick", "ghost"))
TIME_RULES = (
    (re.compile(r"ang(\d{8})t(\d{6})", re.I), "%Y%m%d%H%M%S"),
    (re.compile(r"PRS_L\w*?_(\d{14})", re.I), "%Y%m%d%H%M%S"),
    (re.compile(r"_(\d{8})T(\d{6})Z", re.I), "%Y%m%d%H%M%S"),
    (re.compile(r"(?<!\d)(\d{8})_(\d{6})_\d{2}_4001"), "%Y%m%d%H%M%S"),
    (re.compile(r"(?<!\d)(20\d{6})[T_-]?(\d{6})(?!\d)"), "%Y%m%d%H%M%S"),
)
# scale to uW/cm^2/sr/nm (ISOFIT's radiance unit)
UNIT_SCALES = (
    (re.compile(r"(uw|µw|microw\w*)\s*[/ ]?\s*\(?cm\^?-?2.*nm", re.I), 1.0),
    (re.compile(r"mw\s*[/ ]?\s*\(?cm\^?-?2.*(um|µm|micro)", re.I), 1.0),
    (re.compile(r"\bw\s*[/ ]?\s*\(?m\^?-?2.*(um|µm|micro)", re.I), 0.1),
    (re.compile(r"mw\s*[/ ]?\s*\(?m\^?-?2.*nm", re.I), 0.1),
    (re.compile(r"\bw\s*[/ ]?\s*\(?m\^?-?2.*nm", re.I), 100.0),
    (re.compile(r"nw\s*[/ ]?\s*\(?cm\^?-?2.*nm", re.I), 0.001),
)


def identify(path: str, text: str, wl: np.ndarray, fwhm: np.ndarray | None) -> Fact:
    name = re.split(r"[\\/]", path)[-1]   # the file's own name: a folder called ghost_* proves nothing
    for rx, s in NAME_RULES:
        if rx.search(name):
            return Fact(s, "metadata", "product file name")
    low = text.lower()
    for key, s in TEXT_RULES:
        if key in low:
            return Fact(s, "metadata", f"'{key}' in the header / tags")
    n, w0, w1 = len(wl), float(wl[0]), float(wl[-1])
    fw = float(np.median(fwhm)) if fwhm is not None else float(np.median(np.diff(wl)))
    if 420 <= n <= 430 and w0 < 385:
        s = "aviris_ng" if n == 425 else "tanager" if n == 426 else None
        if s:
            return Fact(s, "derived", f"{n} bands from {w0:.1f} nm")
    if abs(w0 - 396.8) < 2 and abs(w1 - 2507.2) < 3:
        return Fact("ghost", "derived", f"{n} bands {w0:.1f}-{w1:.1f} nm (GHOSt grid)")
    if 210 <= n <= 230 and 412 < w0 < 426 and w1 < 2460:
        return Fact("enmap", "derived", f"{n} bands {w0:.1f}-{w1:.1f} nm")
    if 220 <= n <= 240 and 395 < w0 < 412 and fw > 8:
        return Fact("prisma", "derived", f"{n} bands {w0:.1f}-{w1:.1f} nm, FWHM ~{fw:.0f} nm")
    return Fact("unknown", "derived", f"{n} bands {w0:.1f}-{w1:.1f} nm match no known sensor; generic rules")


def acquisition_time(path: str, meta: dict) -> Fact:
    for k in ("acquisition time", "acquisition_time", "acquisition_date", "tifftag_datetime", "datetime",
              "start_time", "product_starttime"):
        v = meta.get(k)
        if v:
            t = _parse_time(str(v))
            if t:
                return Fact(t, "metadata", f"'{k}'")
    for rx, fmt in TIME_RULES:
        m = rx.search(path)
        if m:
            try:
                t = datetime.strptime("".join(m.groups()), fmt).replace(tzinfo=timezone.utc)
                return Fact(t, "derived", "parsed from the file name")
            except ValueError:
                continue
    return Fact(None, "missing", "no acquisition time in the header, tags or file name")


def _parse_time(s: str):
    s = s.strip().strip("{}").strip().replace("Z", "+00:00")
    for fmt in (None, "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%dT%H%M%S"):
        try:
            t = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def radiance_units(text_by_key: dict, sample: np.ndarray, wl: np.ndarray, toa_refl_per_unit) -> tuple[Fact, Fact]:
    """(units, radiance_scale). text_by_key: header / tag strings that may name the unit. toa_refl_per_unit(i):
    TOA reflectance of 1 uW/cm^2/sr/nm at band i for this scene's (known or assumed) sun."""
    for k, v in text_by_key.items():
        for rx, s in UNIT_SCALES:
            if v and rx.search(str(v)):
                return Fact(str(v), "metadata", f"'{k}'"), Fact(s, "metadata", f"from '{v}'")
    i = int(np.argmin(np.abs(wl - 860)))
    col = sample[:, i]
    p99 = float(np.nanpercentile(col[np.isfinite(col)], 99)) if np.isfinite(col).any() else np.nan
    best, cands = None, (100.0, 10.0, 1.0, 0.1, 0.01, 0.001)
    for s in cands:
        rho = p99 * s * toa_refl_per_unit(i)
        if 0.15 <= rho <= 1.2:
            best = s
    if best is None and np.isfinite(p99):
        best = min(cands, key=lambda s: abs(np.log(max(p99 * s * toa_refl_per_unit(i), 1e-12) / 0.45)))
    note = f"no unit in the delivery; scale {best} gives TOA reflectance p99 {p99 * best * toa_refl_per_unit(i):.2f} at 860 nm"
    return Fact("inferred", "derived", note), Fact(best, "derived", note)
