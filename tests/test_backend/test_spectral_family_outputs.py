"""
The spectral_library_match worker's family step writes a consistent set of files.

Runs _family_match on a synthetic natural scene with planted roofing pixels, against the
committed family library on the 165-band grid. Checks the contract the viewer and the export
read: parquet columns, raster codes, legend, library snapshot and the summary block. Needs
numpy / rasterio / pyarrow / PIL only, not the database.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[2] / "data" / "family_library"

# allotrope.config builds Settings at import and requires these; nothing here connects.
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
    from app.spectral_match.library import build_sensor_cache, load_slim_bundle

    wl = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, float)
    cache = tmp_path_factory.mktemp("cache")
    ents, _ = build_sensor_cache(bundle=load_slim_bundle(BUNDLE), sensor_id="t", target_wl_nm=wl,
                                 target_fwhm_nm=np.full(len(wl), 10.0), cache_dir=cache,
                                 progress=False)
    rng = np.random.default_rng(0)
    base = [np.nan_to_num(e.refl) for e in ents if e.chapter == "natural.vegetation"][:5] + \
           [np.nan_to_num(e.refl) for e in ents if e.chapter == "natural.soil"][:5]
    H = W = 90
    cube = np.einsum("hwk,kb->bhw", rng.dirichlet(np.ones(10), (H, W)), np.stack(base))
    cube = (cube + rng.normal(0, 0.003, cube.shape)).astype(np.float32)
    roof = next(e for e in ents if e.chapter == "manmade.roofing")
    # 150 pixels, so the one with nodata below is under common_band_mask's 1 % and its band
    # is kept; with fewer the band would be dropped and the pixel scored on the rest.
    r, c = np.divmod(rng.choice(H * W, 150, replace=False), W)
    r, c = np.clip(r, 8, H - 9), np.clip(c, 8, W - 9)
    cube[:, r, c] = 0.6 * cube[:, r, c] + 0.4 * np.nan_to_num(roof.refl)[:, None]
    mask = np.zeros((H, W), np.uint8)
    mask[r, c] = 1
    validity = np.ones(cube.shape, np.uint8)
    rr, cc = np.nonzero(mask)
    validity[40, rr[0], cc[0]] = 0                    # one pixel with nodata on a used band
    vendable = SimpleNamespace(band_cw_order=wl.tolist(), band_fwhm_order=[10.0] * len(wl),
                               band_validity_by_position=[1] * len(wl))
    return SimpleNamespace(cube=cube, validity=validity, mask=mask, rows=rr, cols=cc,
                           vendable=vendable, cache=cache)


def test_family_step_writes_the_viewer_and_export_contract(scene, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.transform import Affine

    from allotrope.action_types import _spectral_library_match_run as run_mod
    from allotrope.config import settings
    import app.spectral_match.family_match as fm

    monkeypatch.setattr(settings, "splib07_cache_dir", str(scene.cache))
    monkeypatch.setattr(fm, "FAMILY_BUNDLE_DIR", str(BUNDLE))
    monkeypatch.setattr(fm.ensure_family_cache, "__defaults__", (str(scene.cache), str(BUNDLE)))
    H, W = scene.mask.shape
    profile = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "uint8",
               "transform": Affine.identity()}
    block = run_mod._family_match(
        SimpleNamespace(sensor_type="t"), {"mode": "anomaly_pixels", "family_alpha": 0.05},
        vendable=scene.vendable, cube=scene.cube, validity=scene.validity,
        anomaly_mask=scene.mask, pixel_mask=scene.mask.astype(bool), rows=scene.rows,
        cols=scene.cols, profile=profile, output_dir=tmp_path,
    )
    assert block["status"] == "ok", block
    assert block["n_not_scored"] == 1 and block["n_scored"] == len(scene.rows) - 1
    assert block["p_star"] == pytest.approx(1 - 0.05 / block["n_manmade_families"])

    t = pq.read_table(tmp_path / "family_matches.parquet").to_pandas()
    assert list(t.columns) == ["row", "col", "scored", "family", "family_group", "family_code",
                               "confidence", "answered", "best_entry"]
    assert len(t) == len(scene.rows) and (~t.scored).sum() == 1
    assert (t.loc[~t.scored, "family_code"] == -1).all()
    assert (t.loc[t.answered, "family"] == "roofing").mean() >= 0.8

    with rasterio.open(tmp_path / "family_map.tif") as src:
        code_map = src.read(1)
        assert src.nodata == -2
    assert (code_map[scene.mask == 0] == -2).all()
    assert (code_map[t.row, t.col] == t.family_code).all()

    legend = json.loads((tmp_path / "family_legend.json").read_text())
    assert {"-1", "0"} <= set(legend["codes"])
    assert all(str(c) in legend["codes"] for c in t.family_code.unique())
    assert legend["codes"][str(block["families"].index("roofing") + 1)]["family"] == "roofing"

    lib = np.load(tmp_path / "family_library_refl.npz")
    assert all(v.dtype.kind in "fiu" for v in lib.values())      # the browser reader is numeric-only
    ids = [e["id"] for e in legend["library"]]
    assert set(t.best_entry[t.scored]) <= set(ids)
    assert lib["refl"].shape == (len(ids), scene.cube.shape[0])
    for png in ("family_map.png", "family_map_overlay.png"):
        assert (tmp_path / png).stat().st_size > 0


def _write_sam_outputs(art: Path, scene, n_matched: int) -> None:
    """The minimal SAM files build_hyper_bundle needs: rank-0 hits on the first n pixels."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.transform import Affine

    H, W = scene.mask.shape
    mm = np.full((H, W), -2, np.int32)
    mm[scene.rows, scene.cols] = -1
    mm[scene.rows[:n_matched], scene.cols[:n_matched]] = 7
    with rasterio.open(art / "match_map.tif", "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="int32", transform=Affine.identity()) as dst:
        dst.write(mm, 1)
    n = n_matched
    pq.write_table(pa.table({
        "row": scene.rows[:n].astype("int32"), "col": scene.cols[:n].astype("int32"),
        "rank": np.zeros(n, "int32"), "library_ix": np.full(n, 7, "int32"),
        "material_id": ["m7"] * n, "name": ["Brick GDS350"] * n, "chapter": ["artificial"] * n,
        "asd_subtype": [None] * n, "angle_deg": np.full(n, 9.5, "float32"),
        "n_bands_used": np.full(n, 150, "int32"),
    }), art / "matches.parquet")
    (art / "match_map_legend.json").write_text(json.dumps({"7": {"name": "Brick GDS350"}}))


def _bundle(art: Path):
    import io
    import zipfile

    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from app.spectral_match.export import ExportSpec, build_hyper_bundle

    spec = ExportSpec(action_id="action_11111111-2222", scene_id="scene_33333333-4444",
                      sensor_type="prisma", artifact_dir=art, splib_version="splib07a",
                      software_version="test", override_crs=CRS.from_epsg(32643),
                      override_transform=from_origin(500000, 1300000, 30, 30))
    blob, name = build_hyper_bundle(spec)
    return name, zipfile.ZipFile(io.BytesIO(blob))


def test_export_bundle_carries_family_codes_and_sam_alongside(scene, tmp_path, monkeypatch):
    import csv
    import io

    import geopandas as gpd
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    from allotrope.action_types import _spectral_library_match_run as run_mod
    from allotrope.config import settings
    import app.spectral_match.family_match as fm

    monkeypatch.setattr(settings, "splib07_cache_dir", str(scene.cache))
    monkeypatch.setattr(fm.ensure_family_cache, "__defaults__", (str(scene.cache), str(BUNDLE)))
    H, W = scene.mask.shape
    run_mod._family_match(
        SimpleNamespace(sensor_type="t"), {"mode": "anomaly_pixels"},
        vendable=scene.vendable, cube=scene.cube, validity=scene.validity,
        anomaly_mask=scene.mask, pixel_mask=scene.mask.astype(bool), rows=scene.rows,
        cols=scene.cols, output_dir=tmp_path,
        profile={"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "uint8",
                 "transform": Affine.identity()},
    )
    _write_sam_outputs(tmp_path, scene, n_matched=10)
    name, z = _bundle(tmp_path)
    folder = name[:-4]
    assert name.startswith("hyper_") and all(n.startswith(f"{folder}/hyper_") or
                                             n == f"{folder}/MANIFEST.json" for n in z.namelist())
    assert {f"{folder}/hyper_anomaly_materials.{e}" for e in ("shp", "shx", "dbf", "prj", "cpg")} \
        <= set(z.namelist())

    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["schema"] == "allotrope-export-hyper/2"
    assert manifest["n_matched_pixels"] == len(scene.rows)        # every anomaly pixel
    assert sum(manifest["family_counts"].values()) == len(scene.rows)

    with MemoryFile(z.read(f"{folder}/hyper_anomaly_materials.tif")) as mf, mf.open() as src:
        assert src.crs.to_epsg() == 32643 and src.nodata == -2
        tif = src.read(1)
    with rasterio.open(tmp_path / "family_map.tif") as src:
        assert (tif == src.read(1)).all()

    rows = list(csv.DictReader(io.StringIO(z.read(f"{folder}/hyper_match_summary.csv").decode())))
    assert list(rows[0])[:9] == ["row", "col", "latitude", "longitude", "family", "family_group",
                                 "family_confidence", "family_named", "best_entry"]
    assert "confident" not in rows[0]
    assert sum(1 for r in rows if r["material_1"] == "Brick GDS350") == 10

    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        (shp_dir / f"a.{ext}").write_bytes(z.read(f"{folder}/hyper_anomaly_materials.{ext}"))
    cols = list(gpd.read_file(shp_dir / "a.shp").columns)
    assert cols[:9] == ["row", "col", "lat", "lon", "family", "group", "fconf", "named", "best_ent"]
    assert "mat1" in cols and "scr1" in cols


def test_export_bundle_for_a_sam_only_action_is_unchanged(scene, tmp_path):
    import csv
    import io

    _write_sam_outputs(tmp_path, scene, n_matched=10)
    name, z = _bundle(tmp_path)
    folder = name[:-4]
    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["schema"] == "allotrope-export-hyper/1"
    assert manifest["confidence_threshold_deg"] == 15.0 and manifest["n_matched_pixels"] == 10
    rows = list(csv.DictReader(io.StringIO(z.read(f"{folder}/hyper_match_summary.csv").decode())))
    assert list(rows[0])[:5] == ["row", "col", "latitude", "longitude", "confident"]
    assert f"{folder}/hyper_sam_legend.json" not in z.namelist()


def test_family_step_skips_all_kept_and_records_failures(scene, tmp_path):
    from allotrope.action_types import _spectral_library_match_run as run_mod

    kw = dict(vendable=scene.vendable, cube=scene.cube, validity=scene.validity,
              anomaly_mask=scene.mask, pixel_mask=scene.mask.astype(bool), rows=scene.rows,
              cols=scene.cols, profile={}, output_dir=tmp_path)
    ctx = SimpleNamespace(sensor_type="t")
    assert run_mod._family_match(ctx, {"mode": "all_kept"}, **kw)["status"] == "skipped"
    # No family bundle at /srv/family_library on a dev box: recorded, not raised.
    failed = run_mod._family_match(ctx, {"mode": "anomaly_pixels"}, **kw)
    assert failed["status"] == "failed" and "family library cache" in failed["error"]
