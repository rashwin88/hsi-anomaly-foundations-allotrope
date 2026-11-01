"""
Decisions the generic hyperspectral reader makes about a cube's numbers: band order, FWHM, no-data, scale.

Used by app/utils/files/generic_hsi_helper.py. Each function returns its
decision together with where it came from, because the pre-flight report
(generic_hsi_preflight.py) must show every one of them.

Why it exists: a wrong scale or a missing FWHM does not crash anything - it
quietly ruins the result. Reflectance stored x10000 read as 0-1 saturates
every model; radiance read as reflectance gives confident nonsense; FWHM 0
makes splib07 resampling skip every band and matching return nothing. So
each is detected, reported, and (for scale) overridable by the operator.

Gotchas:
  - Scale order of precedence: operator override > header factor
    (`reflectance scale factor`, GDAL scale/offset, `scale_factor` /
    `add_offset` attributes) > radiance units in the metadata > heuristic.
  - FWHM 1.95 x band spacing is the fallback (agreed 2026-11-01): GHOSt's
    4.48 nm spacing x 1.95 = 8.7 nm, OSK's stated width. The GHOSt hint
    (9 nm) applies only when the metadata names GHOSt / Orbital Sidekick.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GenericHsiHelper")

# FWHM when the file has none: OSK's data sheet says "~10 nm FWHM" and its header example 8.7 nm.
GHOST_FWHM_NM = 9.0
FWHM_PER_SPACING = 1.95

# Scale heuristic on a decimated sample (8 good bands, <= 256 x 256), ignoring no-data and zeros.
# 0-1 reflectance stays below ~1.5 even over snow and glint (p99.9, so a few hot pixels don't flip it).
_REFLECTANCE_MAX = 1.5
# x10000 integer reflectance: a land median of 0.003-0.6 -> 30-6000, and p99 at most 1.2 x 10000.
_SCALED_MEDIAN_RANGE = (30.0, 6000.0)
_SCALED_P99_MAX = 12000.0
_SAMPLE_BANDS = 8
_SAMPLE_PIXELS = 256
_RADIANCE_RE = re.compile(r"radiance|w\s*/?\s*\(?\s*(m|cm)\s*\^?2|\bsr\b|sr-1|sr\^-1|µw|uw\s*/|mw\s*/", re.IGNORECASE)


@dataclass
class SortedBands:
    file_index: List[int]
    wavelengths: List[float]
    fwhm: List[float]
    good: List[int]
    fwhm_source: str
    wavelength_source: str
    units_read: Optional[str]
    notes: List[str] = field(default_factory=list)


def sorted_band_table(tables, sensor_hint: bool) -> SortedBands:
    """Ascending, de-duplicated band table from the first source; FWHM from the first that has one."""
    first = tables[0]
    notes = list(first.notes)
    wl = first.wavelengths
    with_fwhm = [t for t in tables if t.fwhm is not None and np.allclose(t.wavelengths, wl, atol=0.5)]
    fw_table = next((t for t in with_fwhm if t.fwhm_exact), with_fwhm[0] if with_fwhm else None)
    fw = fw_table.fwhm if fw_table is not None else None
    good = first.good if first.good is not None else np.ones(wl.size, bool)
    keep = wl > 0
    if not keep.all():
        notes.append(f"{int((~keep).sum())} bands with wavelength <= 0 dropped")
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(wl[idx], kind="stable")]
    if np.all(np.diff(wl[idx]) < 0):
        notes.append("wavelengths are descending in the file; bands reversed")
    elif np.any(order != idx):
        notes.append(f"wavelengths are not in file order ({int((order != idx).sum())} bands moved); sorted")
    wl_s = wl[order]
    uniq = np.concatenate([[True], np.diff(wl_s) > 1e-6]) if wl_s.size else np.zeros(0, bool)
    if not uniq.all():
        notes.append(f"{int((~uniq).sum())} duplicate wavelengths dropped (first kept)")
    order, wl_s = order[uniq], wl_s[uniq]
    spacing = np.gradient(wl_s) if wl_s.size > 1 else np.full(wl_s.size, 10.0)
    if fw is not None:
        fw_s = fw[order].astype(np.float64)
        bad = ~np.isfinite(fw_s) | (fw_s <= 0)
        fw_s[bad] = FWHM_PER_SPACING * spacing[bad]
        source = fw_table.fwhm_source + (f" ({int(bad.sum())} bands derived from spacing)" if bad.any() else "")
    elif sensor_hint:
        fw_s, source = np.full(wl_s.size, GHOST_FWHM_NM), "derived: GHOSt / OSK hint 9 nm"
        notes.append("no FWHM in the file; 9 nm from the OSK data sheet (metadata names GHOSt / Orbital Sidekick)")
    else:
        fw_s, source = FWHM_PER_SPACING * spacing, f"derived: {FWHM_PER_SPACING} x band spacing"
        notes.append(f"no FWHM in the file; {FWHM_PER_SPACING} x band spacing assumed")
    return SortedBands(
        file_index=[int(i) for i in order], wavelengths=wl_s.tolist(), fwhm=fw_s.tolist(),
        good=[int(g) for g in np.asarray(good, bool)[order]], fwhm_source=source,
        wavelength_source=first.source, units_read=first.units_read, notes=notes,
    )


def declared_fill(src, tokens: Optional[Dict[str, str]]) -> Tuple[Optional[float], str]:
    """(no-data value, source): the ENVI `data ignore value`, else GDAL's nodata, else none."""
    if tokens and "data ignore value" in tokens:
        return float(tokens["data ignore value"]), "envi_header"
    if src.nodata is not None:
        return float(src.nodata), "gdal_nodata"
    return None, "none"


def _sample(src, band: SortedBands, fill: Optional[float]) -> np.ndarray:
    from rasterio.enums import Resampling

    good = [band.file_index[i] for i, g in enumerate(band.good) if g] or band.file_index
    pick = np.asarray(good)[np.linspace(0, len(good) - 1, min(_SAMPLE_BANDS, len(good))).astype(int)]
    shape = (pick.size, min(src.height, _SAMPLE_PIXELS), min(src.width, _SAMPLE_PIXELS))
    arr = src.read(indexes=[int(i) + 1 for i in pick], out_shape=shape, out_dtype="float64",
                   resampling=Resampling.nearest)
    ok = np.isfinite(arr) & (arr != -9999.0) & (arr != 0)
    if fill is not None and np.isfinite(fill):
        ok &= arr != fill
    return arr[ok]


def _header_factor(src, tokens, root_tags, var) -> Optional[Tuple[float, float, str]]:
    if tokens and "reflectance scale factor" in tokens:
        f = float(tokens["reflectance scale factor"])
        if f > 0 and f != 1:
            return 1.0 / f, 0.0, f"ENVI reflectance scale factor {f:g}"
    scale, offset = float(src.scales[0]), float(src.offsets[0])
    if scale != 1.0 or offset != 0.0:
        return scale, offset, f"GDAL scale {scale:g} / offset {offset:g}"
    leaf = (var or "").rsplit("/", 1)[-1].lower()
    tags = {k.lower(): v for k, v in root_tags.items() if leaf and leaf in k.lower()}
    tags.update({k.lower(): v for k, v in src.tags(1).items()})
    s = next((v for k, v in tags.items() if k.endswith("scale_factor")), None)
    o = next((v for k, v in tags.items() if k.endswith("add_offset")), None)
    if s is not None or o is not None:
        s_f, o_f = float(re.findall(r"[-+.\deE]+", str(s or "1"))[0]), float(re.findall(r"[-+.\deE]+", str(o or "0"))[0])
        if s_f != 1.0 or o_f != 0.0:
            return s_f, o_f, f"scale_factor {s_f:g} / add_offset {o_f:g}"
    return None


def _units_text(src, tokens, root_tags, var) -> str:
    leaf = (var or "").rsplit("/", 1)[-1].lower()
    parts = [leaf, (tokens or {}).get("data units", ""), str(src.units[0] or "")]
    parts += [v for k, v in src.tags(1).items() if "unit" in k.lower() and "wavel" not in k.lower()
              and "fwhm" not in k.lower()]
    parts += [v for k, v in root_tags.items() if "unit" in k.lower() and leaf and k.lower().endswith(leaf + "_unit")]
    return " ".join(p for p in parts if p)


def parse_override(options: Dict[str, Any]) -> Optional[Tuple[float, bool, str]]:
    """(factor, reflectance, tag) from the operator's options, or None for auto. ValueError on nonsense."""
    raw = options.get("scale", options.get("scale_override"))
    if raw is None and options.get("scale_factor") is not None:
        raw = f"x{float(options['scale_factor']):g}"
    if raw is None or str(raw).strip().lower() in ("", "auto"):
        return None
    text = str(raw).strip().lower()
    if text in ("radiance", "not_reflectance", "dn"):
        return 1.0, False, "radiance"
    m = re.fullmatch(r"x\s*([0-9.]+(?:e-?\d+)?)", text)
    if m is None or float(m.group(1)) <= 0:
        raise ValueError(f"scale override {raw!r} not understood: use auto, x1, x0.0001, x<factor> or radiance")
    return float(m.group(1)), True, f"x{float(m.group(1)):g}"


def decide_scale(src, band: SortedBands, fill, tokens, root_tags, var, options) -> Dict[str, Any]:
    """The scale decision with its source, reason and the sample statistics behind it."""
    vals = _sample(src, band, fill)
    stats = ({"n": int(vals.size), "p50": float(np.percentile(vals, 50)), "p99": float(np.percentile(vals, 99)),
              "p99_9": float(np.percentile(vals, 99.9)), "max": float(vals.max())} if vals.size else {"n": 0})
    out: Dict[str, Any] = {"factor": 1.0, "offset": 0.0, "sample": stats, "override_tag": None}
    units = _units_text(src, tokens, root_tags, var)
    radiance = bool(_RADIANCE_RE.search(units))
    override = parse_override(options)
    header = _header_factor(src, tokens, root_tags, var)
    if override is not None:
        factor, refl, tag = override
        out.update(factor=factor if refl else 1.0, reflectance=refl, source="operator", override_tag=tag,
                   decision="not_reflectance" if not refl else "reflectance_scaled" if factor != 1 else "reflectance_0_1",
                   reason=f"operator override {tag}")
    elif radiance:
        out.update(reflectance=False, source="metadata_units", decision="not_reflectance",
                   reason=f"metadata units say radiance ({units.strip()[:80]})")
    elif header is not None:
        out.update(factor=header[0], offset=header[1], reflectance=True, source="header",
                   decision="reflectance_scaled", reason=header[2])
        scaled_max = stats.get("p99_9", 0.0) * header[0] + header[1]
        if scaled_max > _REFLECTANCE_MAX:
            out["warning"] = f"after the header factor p99.9 is {scaled_max:.3g}, above 0-1 reflectance"
    elif not vals.size:
        out.update(reflectance=True, source="heuristic", decision="reflectance_0_1",
                   reason="no valid sample values; 0-1 assumed")
    elif stats["p99_9"] <= _REFLECTANCE_MAX:
        out.update(reflectance=True, source="heuristic", decision="reflectance_0_1",
                   reason=f"sample p99.9 {stats['p99_9']:.3g} <= {_REFLECTANCE_MAX}")
    elif _SCALED_MEDIAN_RANGE[0] <= stats["p50"] <= _SCALED_MEDIAN_RANGE[1] and stats["p99"] <= _SCALED_P99_MAX:
        out.update(factor=1e-4, reflectance=True, source="heuristic", decision="reflectance_x10000",
                   reason=f"sample median {stats['p50']:.4g}, p99 {stats['p99']:.4g}: reflectance x 10000")
    else:
        out.update(reflectance=False, source="heuristic", decision="not_reflectance",
                   reason=f"sample median {stats['p50']:.4g}, p99 {stats['p99']:.4g} fit neither 0-1 nor x10000 "
                          "reflectance (DN or radiance?)")
    logger.info("generic_hsi scale: %s x%g (%s: %s)", out["decision"], out["factor"], out["source"], out["reason"])
    return out
