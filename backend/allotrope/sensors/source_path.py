"""Resolve the on-disk path that a sensor's DatasetBuilder expects.

`Scene.raw_path` is the **directory** holding the uploaded raw files
(`scenes/<scene_id>/raw/`). Each builder wants something different:

  - PRISMA: the single ``.he5`` file inside the directory (possibly nested).
  - Landsat 9: the single ``.tif`` / ``.tiff`` file.
  - EnMAP: the **folder** containing the ``*-METADATA.XML`` (not the XML itself).
  - AVIRIS-NG: the **folder** containing the ``ang*_corr_*.bin`` / ``.hdr`` pair
    (the flight-line scene directory, e.g. ``ang20151219081738/``).
  - HotSat-1: the **folder** containing ``metadata.json`` + the ortho /
    stacked / UDM GeoTIFFs (SatVu L2 Visual scene bundle).
  - Tanager: the surface-reflectance ``.h5`` (an ``sr``-named one if the
    upload carries several HDF5 files), else an ENVI ``.hdr`` whose data
    file is beside it; ``pick_tanager_file``.
  - generic_hsi: the cube file - a GeoTIFF, an ENVI ``.hdr`` (its data file
    beside it) or an HDF5 / NetCDF container, the largest if several;
    ``locate_data`` (raises with the extensions it saw when there is none).

The user-uploaded folder structure is preserved at staging, so the file
may sit at either ``raw_root/<file>`` or ``raw_root/SOMEFOLDER/<file>``.
We ``rglob`` to be robust.

Originally lived inside `allotrope_worker/scene_onboard.py`. Lifted into
this shared module in Step 12d-fix so the action_run handler can reuse
it when re-vending a Scene from the action's `band_filter_apply` recipe.
Both onboarding and action handlers now share this single source of truth.
"""

from __future__ import annotations

from pathlib import Path


def _files_in(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def resolve_source_path(sensor_type: str, raw_root: Path) -> Path:
    """Return the file (PRISMA, Landsat 9) or folder (EnMAP) the
    sensor's DatasetBuilder expects.

    Raises:
        ValueError: if the directory is empty, the sensor is unknown, or
            the expected file/folder isn't present.
    """
    files = _files_in(raw_root)
    if not files:
        raise ValueError(f"raw dir is empty: {raw_root}")

    if sensor_type == "prisma":
        for p in files:
            if p.suffix.lower() == ".he5":
                return p
        raise ValueError(f"no .he5 file under {raw_root}")

    if sensor_type == "landsat9":
        for p in files:
            if p.suffix.lower() in (".tif", ".tiff"):
                return p
        raise ValueError(f"no .tif file under {raw_root}")

    if sensor_type == "enmap":
        # EnMAP source_path is the FOLDER containing METADATA.XML.
        for p in files:
            if p.name.upper().endswith("-METADATA.XML"):
                return p.parent
        raise ValueError(f"no *-METADATA.XML file under {raw_root}")

    if sensor_type == "aviris_ng":
        # AVIRIS-NG source_path is the FOLDER containing the primary
        # _corr_*.bin / _corr_*.hdr pair. Filename pattern:
        # ang<YYYYMMDD>t<HHMMSS>_corr_*.bin
        for p in files:
            name = p.name.lower()
            if name.endswith(".bin") and "_corr_" in name:
                return p.parent
        raise ValueError(f"no *_corr_*.bin file under {raw_root}")

    if sensor_type == "hotsat1":
        # HotSat-1 source_path is the FOLDER containing metadata.json
        # + the ortho / stacked / UDM GeoTIFFs.
        for p in files:
            if p.name == "metadata.json":
                return p.parent
        raise ValueError(f"no metadata.json under {raw_root}")

    if sensor_type == "tanager":
        # Imported here: this module is shared with the api, which must not load h5py at startup.
        from app.utils.files.tanager_helper import pick_tanager_file

        picked = pick_tanager_file(files)
        if picked is not None:
            return picked
        raise ValueError(f"no .h5 / .he5 / .hdf5 file, and no ENVI .hdr with its data file, under {raw_root}")

    if sensor_type == "generic_hsi":
        # Imported here for the same reason. The largest GeoTIFF / ENVI cube / HDF file wins.
        from app.utils.files.generic_hsi_helper import locate_data

        return locate_data(files)[0]

    raise ValueError(f"unknown sensor_type {sensor_type!r}")
