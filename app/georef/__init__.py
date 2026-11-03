"""Resolve scene-level CRS + affine at export time.

The vendable model doesn't carry spatial reference (today). For export
purposes we read it lazily from the raw scene file on disk: PRISMA HE5
carries per-pixel lat/lon arrays; EnMAP / AVIRIS-NG / Landsat 9 / HotSat
all ship as georeferenced raster files that rasterio reads directly.

Public API: ``resolve_scene_georef(scene_dir, sensor_type, target_shape) ->
(transform, crs)``. Raises ``GeorefUnavailable`` if no georef can be
recovered. The export endpoint then falls back, loudly, to ``pixel_grid()`` with no CRS
and records ``NO_GEOREF_NOTE`` in the MANIFEST (the object-coords endpoint stays strict).
"""

from app.georef.from_scene_file import (
    NO_GEOREF_NOTE,
    GeorefUnavailable,
    pixel_grid,
    resolve_scene_georef,
)

__all__ = ["NO_GEOREF_NOTE", "GeorefUnavailable", "pixel_grid", "resolve_scene_georef"]
