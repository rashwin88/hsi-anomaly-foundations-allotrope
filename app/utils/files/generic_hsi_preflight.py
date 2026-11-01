"""
The pre-flight report of a generic_hsi scene: what the reader decided about the file, and every assumption.

Built by app/utils/files/generic_hsi_helper.py (`read_generic_hsi_metadata`),
stored by scene onboarding in `Scene.extra_metadata["preflight"]` and in the
onboarding job's payload, and shown on the Ingest panel and the scene page.

Why it exists: once evaluation starts the code is frozen, so the operator has
to see a surprise (micrometre wavelengths, radiance, a missing map) before
running anything, not after a bad submission. Plain JSON, one level of keys
per decision, so the frontend can print it without knowing its schema.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.georef.from_scene_file import GeorefUnavailable
from app.utils.files.generic_hsi_georef import scene_georef

SCHEMA = "allotrope.generic_hsi.preflight/1"


def _georef(data_path: str, upload: Sequence[Path], shape) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    notes: List[str] = []
    try:
        transform, crs, how = scene_georef(data_path, upload, shape)
    except GeorefUnavailable as exc:
        notes.append(f"georef: {exc}")
        return ({"georef_source": "none", "epsg": None, "bbox": None},
                {"source": "none", "exportable": False, "reason": str(exc)}, notes)
    epsg = crs.to_epsg()
    bbox = None
    try:
        b = transform_bounds(crs, "EPSG:4326", *array_bounds(shape[0], shape[1], transform))
        bbox = [float(v) for v in (b[0], b[1], b[2], b[3])]
    except Exception as exc:   # noqa: BLE001 - a CRS PROJ cannot invert still exports
        notes.append(f"georef: no WGS-84 bbox ({exc})")
    if how == "igm":
        notes.append("georef: EPSG:4326 grid fitted to the IGM swath's lat/lon bounding box (approximate for a "
                     "rotated swath, as PRISMA's lat/lon fallback)")
    if transform.b != 0 or transform.d != 0:
        notes.append("georef: the map grid is rotated")
    report = {"source": how, "exportable": True, "crs": crs.to_string(), "epsg": epsg,
              "transform": [float(v) for v in tuple(transform)[:6]], "bbox_wgs84": bbox}
    return {"georef_source": how, "epsg": epsg, "bbox": bbox}, report, notes


def assemble(*, source: Path, data_path: str, var, upload, info, band, fill, fill_source, scale,
             options) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(GenericHsiMetadata georef fields, report dict)."""
    shape = (info["height"], info["width"])
    md_georef, georef, geo_notes = _georef(data_path, upload, shape)
    wl = np.asarray(band.wavelengths)
    assumptions = list(band.notes)
    if fill is None:
        assumptions.append("no declared no-data value; NaN, -9999 and all-zero spectra are treated as no-data")
    if scale["source"] in ("heuristic",):
        assumptions.append(f"scale detected: {scale['reason']}")
    assumptions += geo_notes
    warnings = []
    if not scale["reflectance"]:
        warnings.append("not reflectance: Indradhanu V2 and the spectral namer expect reflectance; anomaly scoring "
                        "runs MNF-RX only on this scene")
    if scale.get("warning"):
        warnings.append(scale["warning"])
    if not georef["exportable"]:
        warnings.append("no map coordinates: the scene can be scored, but export will refuse")
    report = {
        "schema": SCHEMA,
        "file": source.name,
        "data_file": Path(data_path).name if var is None else data_path.rsplit("/", 1)[-1],
        "subdataset": var,
        "format": info["driver"],
        "interleave": (info.get("interleave") or "").lower() or None,
        "dtype": info["dtype"],
        "shape": {"bands_in_file": info["count"], "bands": int(wl.size), "height": shape[0], "width": shape[1]},
        "wavelengths": {
            "source": band.wavelength_source, "units_read": band.units_read,
            "min_nm": float(wl.min()), "max_nm": float(wl.max()),
            "mean_spacing_nm": float(np.mean(np.diff(wl))) if wl.size > 1 else None,
        },
        "fwhm": {"source": band.fwhm_source, "derived": band.fwhm_source.startswith("derived"),
                 "min_nm": float(min(band.fwhm)), "max_nm": float(max(band.fwhm))},
        "bad_bands": int(len(band.good) - sum(band.good)),
        "scale": {k: scale[k] for k in ("decision", "factor", "offset", "reflectance", "source", "reason", "sample")},
        "nodata": {"declared": fill, "source": fill_source, "also_invalid": ["NaN / inf", "-9999", "all-zero spectrum"]},
        "georef": georef,
        "options": options or {},
        "assumptions": assumptions,
        "warnings": warnings,
    }
    return md_georef, report
