"""The one record every reader fills: the cube, how to read it, and where each piece of metadata came from.

Every fact carries its source, so the run can say plainly what it knew and what it assumed:
  metadata  read from the delivery (header, tags, product XML/HDF5 attributes)
  derived   computed from other delivered facts (sun angles from time + location, FWHM from band spacing,
            radiance units from the signal's magnitude)
  assumed   a documented safe default because nothing delivered it (nadir view, sun zenith 35 deg)
  missing   not available and no default applies (the run stops if it needs it)

The grade summarises the checklist for the report and the HTML quick-look:
  A  everything the physics needs came from metadata
  B  some of it was derived, nothing assumed
  C  the view geometry, sensor altitude or noise model was assumed (small effect on spaceborne nadir-ish scenes)
  D  the acquisition time, the location or the radiance scale was assumed: the result is a best effort and is
     flagged as such everywhere it is reported
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np

SOURCES = ("metadata", "derived", "assumed", "missing")
# Facts whose being assumed or missing drops the grade to D (time and location set the sun and the DEM; the
# radiance scale sets every number); to C (view geometry, altitude, noise: second-order on a spaceborne scene).
D_FACTS = ("time", "location", "radiance_scale")
C_FACTS = ("view", "altitude_km", "noise", "dem")


@dataclass
class Fact:
    value: Any
    source: str
    note: str = ""

    def __post_init__(self):
        assert self.source in SOURCES, self.source


@dataclass
class Scene:
    """A hyperspectral cube plus its metadata, independent of sensor and file format.

    read_rows(r0, r1) returns lines r0..r1 as float32 (rows, bands, samples) in the delivered units, with every
    invalid value (nodata, NaN, inf, fill) already set to NaN. Geometry callables return per-pixel arrays for the
    same line range, so nothing larger than a row block is ever held in memory.
    """
    path: str
    fmt: str                       # enmap_l1c | prisma_l1 | tanager_h5 | aviris_ng_l1 | envi | geotiff
    sensor: Fact                   # enmap | prisma | aviris_ng | tanager | ghost | unknown
    shape: tuple[int, int, int]    # lines, bands, samples
    interleave: str                # as delivered: bil | bip | bsq
    wl: np.ndarray                 # band centres, nm, ascending
    fwhm: Fact                     # np.ndarray, nm
    read_rows: Callable[[int, int], np.ndarray]
    units: Fact                    # radiance unit string, "reflectance" or "dn"
    radiance_scale: Fact           # factor to uW/cm^2/sr/nm (ISOFIT's unit); None if not radiance
    time: Fact                     # datetime, UTC
    location: Fact                 # callable (r0, r1) -> (lon, lat) arrays, or None
    sun: Fact                      # callable (r0, r1) -> (zenith, azimuth) arrays, degrees
    view: Fact                     # callable (r0, r1) -> (zenith, azimuth) arrays, degrees
    altitude_km: Fact              # sensor altitude above sea level
    noise: Fact = field(default_factory=lambda: Fact(None, "assumed", "ISOFIT's generic noise model"))
    good_bands: Fact = field(default_factory=lambda: Fact(None, "derived", "detected from the data"))
    dem: Fact = field(default_factory=lambda: Fact(None, "missing", "not sampled yet"))
    path_length: Fact | None = None   # callable (r0, r1) -> sensor-to-ground metres, when delivered
    isofit_sensor: str = ""           # apply_oe's sensor argument: enmap | prisma | ang | tanager | NA-YYYYMMDD
    map_info: str | None = None    # ENVI map info for the outputs, when the delivery is georeferenced
    crs_wkt: str | None = None
    fid: str = "scene"             # file id for ISOFIT's cube names
    level: Fact | None = None      # input-level verdict (level.py)
    notes: list[str] = field(default_factory=list)

    @property
    def lines(self) -> int:
        return self.shape[0]

    @property
    def samples(self) -> int:
        return self.shape[2]

    def facts(self) -> dict[str, Fact]:
        names = ("sensor", "fwhm", "units", "radiance_scale", "time", "location", "sun", "view", "altitude_km",
                 "noise", "dem", "good_bands")
        return {n: getattr(self, n) for n in names} | ({"level": self.level} if self.level else {})

    def grade(self) -> str:
        weak = lambda n: self.facts()[n].source in ("assumed", "missing")
        if any(weak(n) for n in D_FACTS):
            return "D"
        if any(weak(n) for n in C_FACTS):
            return "C"
        return "B" if any(f.source == "derived" for n, f in self.facts().items() if n != "good_bands") else "A"

    def checklist(self) -> list[dict]:
        """One row per fact: what it is, where it came from, and a note. Callables are summarised, not dumped."""
        rows = []
        for n, f in self.facts().items():
            v = f.value
            if callable(v):
                v = "per pixel"
            elif isinstance(v, np.ndarray):
                v = f"{v.size} values, {np.nanmin(v):.2f}..{np.nanmax(v):.2f}"
            elif isinstance(v, datetime):
                v = v.isoformat()
            rows.append({"item": n, "value": v, "source": f.source, "note": f.note})
        return rows
