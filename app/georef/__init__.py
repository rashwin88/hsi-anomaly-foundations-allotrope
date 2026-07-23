"""Resolve scene-level CRS + affine at export time.

The vendable model doesn't carry spatial reference (today). For export
purposes we read it lazily from the raw scene file on disk: PRISMA HE5
carries per-pixel lat/lon arrays; EnMAP / AVIRIS-NG / Landsat 9 / HotSat
all ship as georeferenced raster files that rasterio reads directly.

Public API: ``resolve_scene_georef(scene_dir, sensor_type, target_shape) ->
(transform, crs)``. Raises ``GeorefUnavailable`` if no georef can be
recovered (export endpoint translates that to a 422 ``crs_missing``).
"""

from app.georef.from_scene_file import (
    GeorefUnavailable,
    resolve_scene_georef,
)

__all__ = ["GeorefUnavailable", "resolve_scene_georef"]
