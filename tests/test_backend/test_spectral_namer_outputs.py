"""
The spectral namer inside spectral_library_match: its files, the man-made-only matching, the export.

The namer model itself is replaced by a fake per-pixel result (tests/test_spectral_match/test_namer.py tests the
model); what is tested here is the product contract around it - the namer_* files the viewer reads, that SAM and
family matching only see man-made objects' pixels, the /3 export bundle and the config defaults. Needs numpy /
rasterio / pyarrow / geopandas / PIL only, not the database.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# allotrope.config builds Settings at import and requires these; nothing here connects.
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

CLASSES = ("oil_fuel", "plastic", "textile_fabric", "wood_paper", "paint_coatings", "metal",
           "concrete_asphalt")
H = W = 40


def _mask():
    """Two man-made objects (a 3 x 3 metal block, a tentative plastic pair) and a natural 2 x 2."""
    m = np.zeros((H, W), bool)
    m[5:8, 5:8] = True
    m[20, 20:22] = True
    m[30:32, 10:12] = True
    return m


def _fake_result(mask):
    from app.spectral_match.namer.infer import NamerResult

    rows, cols = np.nonzero(mask)
    P = len(rows)
    probs = np.full((P, len(CLASSES)), 0.05)
    p_mm = np.empty(P)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if r < 10:
            p_mm[i], probs[i, CLASSES.index("metal")] = 0.92, 0.7
        elif r < 25:
            p_mm[i], probs[i, CLASSES.index("plastic")] = 0.66, 0.4
        else:
            p_mm[i], probs[i, CLASSES.index("concrete_asphalt")] = 0.12, 0.6
    best = probs.argmax(1)
    return NamerResult(p_mm, best, probs.max(1), np.full(P, 0.4), CLASSES, probs)


@pytest.fixture
def fake_namer(monkeypatch, tmp_path):
    """resolve / load / name_pixels replaced: a manifest-shaped NamerModel and the fake result."""
    import app.spectral_match.namer.infer as infer
    from allotrope.foundation_models import resolver

    info = resolver.NamerModel(checkpoint_abs_path=str(tmp_path / "ep40.pt"),
                               checkpoint_filename="ep40.pt", codename="Test Namer",
                               label="test", version="N001b", epoch=40)
    monkeypatch.setattr(resolver, "resolve_spectral_namer", lambda root: info)
    monkeypatch.setattr(infer, "load_namer", lambda path: object())
    monkeypatch.setattr(infer, "name_pixels",
                        lambda model, cube, wl, pixels, *a, **k: _fake_result(pixels))
    return info


def _profile():
    from rasterio.transform import Affine

    return {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "uint8",
            "transform": Affine.identity()}


def _run_namer(out: Path, mask):
    from allotrope.action_types._spectral_namer_run import run_namer

    out.mkdir(parents=True, exist_ok=True)
    return run_namer(SimpleNamespace(), {"namer_confident_p": 0.55}, cube=np.zeros((3, H, W)),
                     validity=np.ones((3, H, W), np.int8), wavelengths=[500, 600, 700],
                     pixel_mask=mask, profile=_profile(), output_dir=out)


def test_namer_stage_writes_maps_tables_and_summary(fake_namer, tmp_path):
    import pyarrow.parquet as pq
    import rasterio

    mask = _mask()
    block, manmade = _run_namer(tmp_path, mask)
    assert block["status"] == "ok", block
    assert block["n_objects"] == 3 and block["n_objects_manmade"] == 2 and block["n_objects_natural"] == 1
    assert block["n_pixels"] == mask.sum() == block["n_pixels_manmade"] + block["n_pixels_natural"]
    assert block["class_counts"] == {"metal": 1, "plastic": 1}
    assert block["class_counts_confident"] == {"metal": 1}            # plastic 0.4 < 0.55: tentative
    assert block["checkpoint"] == "ep40.pt" and block["codename"] == "Test Namer"
    assert (manmade == (mask & (np.arange(H)[:, None] < 25))).all()

    with rasterio.open(tmp_path / "namer_level1_map.tif") as src:
        l1 = src.read(1)
        assert src.dtypes[0] == "int8" and src.nodata == -1 and src.transform.is_identity
    assert (l1[~mask] == -1).all() and (l1[5:8, 5:8] == 1).all() and (l1[30:32, 10:12] == 0).all()
    with rasterio.open(tmp_path / "namer_class_map.tif") as src:
        cm = src.read(1)
    assert cm[6, 6] == CLASSES.index("metal") + 1 and cm[20, 20] == CLASSES.index("plastic") + 11
    assert cm[30, 10] == 0

    px = pq.read_table(tmp_path / "namer_pixels.parquet").to_pandas()
    assert list(px.columns) == ["row", "col", "p_manmade", "cls", "p_cls", "fill", "obj_id"]
    assert len(px) == mask.sum() and str(px.row.dtype) == "int32"
    ob = pq.read_table(tmp_path / "namer_objects.parquet").to_pandas()
    assert ob.n_px.sum() == mask.sum() and ob.call.tolist() == ["manmade", "manmade", "natural"]
    assert ob.cls.tolist() == ["metal", "plastic", ""] and ob.confident.tolist() == [True, False, False]

    legend = json.loads((tmp_path / "namer_legend.json").read_text())
    assert legend["level1"]["1"]["colour"] == "#E8743B" and legend["level1"]["0"]["colour"] == "#2A9D8F"
    assert legend["classes"]["16"]["tentative"] and legend["classes"]["16"]["cls"] == "metal"
    from PIL import Image
    for png in ("namer_level1_map.png", "namer_level1_overlay.png", "namer_class_overlay.png"):
        assert Image.open(tmp_path / png).size == (W, H)
    rgba = np.asarray(Image.open(tmp_path / "namer_level1_overlay.png"))
    assert tuple(rgba[6, 6]) == (0xE8, 0x74, 0x3B, 255) and tuple(rgba[30, 10]) == (0x2A, 0x9D, 0x8F, 255)
    assert rgba[0, 0, 3] == 0
    cls_rgba = np.asarray(Image.open(tmp_path / "namer_class_overlay.png"))
    assert cls_rgba[6, 6, 3] == 255 and cls_rgba[20, 20, 3] < 255          # tentative faded
    assert cls_rgba[30, 10, 3] == 0                                          # natural not drawn


def test_namer_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    from allotrope.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "no-models"))
    block, manmade = _run_namer(tmp_path / "out", _mask())
    assert block["status"] == "failed" and "manifest" in block["error"] and manmade is None


def test_resolver_reads_the_manifest(tmp_path):
    from allotrope.foundation_models.resolver import resolve_spectral_namer

    d = tmp_path / "spectral_namer"
    d.mkdir()
    (d / "n.pt").write_bytes(b"x")
    (d / "current.json").write_text(json.dumps({
        "architecture": "spectral_namer", "label": "L", "codename": {"name": "N"},
        "current": {"file": "n.pt", "version": "N001b", "epoch": 40}}))
    m = resolve_spectral_namer(tmp_path)
    assert m.checkpoint_abs_path == str(d / "n.pt") and m.epoch == 40 and m.codename == "N"
    (d / "n.pt").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_spectral_namer(tmp_path)


# --- the whole run: SAM and family see man-made pixels only ----------------


def _library(B):
    rng = np.random.default_rng(0)
    return [SimpleNamespace(refl=rng.uniform(0.05, 0.5, B).astype(np.float32),
                            valid=np.ones(B, np.uint8), name=f"lib{i}", material_id=f"m{i}",
                            chapter="artificial", asd_subtype=None) for i in range(4)]


def _ctx(tmp_path, cfg):
    import rasterio

    B = 30
    mask = _mask()
    up = tmp_path / "up"
    up.mkdir()
    with rasterio.open(up / "anomaly_mask.tif", "w", **_profile()) as dst:
        dst.write(mask.astype(np.uint8), 1)
    vdir = tmp_path / "data" / "scenes" / "s1" / "vendable"
    vdir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    vendable = SimpleNamespace(
        normalized_hyperspectral_cube=rng.uniform(0.05, 0.5, (B, H, W)).astype(np.float32),
        validity_cube=np.ones((B, H, W), np.int8), band_cw_order=list(np.linspace(450, 2400, B)),
        band_fwhm_order=[10.0] * B, band_validity_by_position=[1] * B)
    with (vdir / "vendable.pkl").open("wb") as f:
        pickle.dump(vendable, f)
    full = {"input_anomaly_detection_output_id": "output_x", "chapters": ["artificial"], "top_k": 3,
            "min_coverage": 0.7, "min_band_count": 5, "sg_window_length": 7, "sg_polyorder": 2,
            "mode": "anomaly_pixels", "family_alpha": 0.05, **cfg}
    return SimpleNamespace(configuration=full, on_step=lambda step: None,
                           resolve_action_output=lambda oid: up, data_dir=tmp_path / "data",
                           scene_id="s1", output_dir=tmp_path / "out", sensor_type="prisma"), mask


def _run(tmp_path, monkeypatch, cfg):
    import app.spectral_match as sm
    from allotrope.action_types import _spectral_library_match_run as run_mod

    seen = {}
    real_match = sm.match_pixels

    def spy_match(**kw):
        seen["sam_pixels"] = len(kw["pixels"])
        return real_match(**kw)

    def spy_family(ctx, cfg, *, rows, cols, natural_mask=None, **kw):
        seen["family_rows"], seen["family_cols"], seen["natural_mask"] = rows, cols, natural_mask
        return {"status": "skipped", "reason": "test"}

    monkeypatch.setattr(sm, "match_pixels", spy_match)
    monkeypatch.setattr(sm, "load_sensor_cache", lambda path: _library(30))
    monkeypatch.setattr(run_mod, "_resolve_cache_path", lambda ctx, cfg, v: tmp_path / "c.npz")
    monkeypatch.setattr(run_mod, "_family_match", spy_family)
    # The SAM heatmap calls matplotlib.cm.get_cmap: the worker image's matplotlib 3.10 has it,
    # a newer laptop venv may not.
    import matplotlib
    from matplotlib import cm
    if not hasattr(cm, "get_cmap"):
        monkeypatch.setattr(cm, "get_cmap", lambda name, n: matplotlib.colormaps[name].resampled(n),
                            raising=False)
    ctx, mask = _ctx(tmp_path, cfg)
    run_mod.run(ctx)
    return seen, mask, json.loads((ctx.output_dir / "summary.json").read_text()), ctx.output_dir


def test_sam_and_family_get_only_manmade_pixels(fake_namer, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    import rasterio

    seen, mask, summary, out = _run(tmp_path, monkeypatch, {"namer": True, "namer_confident_p": 0.55})
    manmade = mask & (np.arange(H)[:, None] < 25)
    natural = mask & ~manmade
    assert summary["namer"]["status"] == "ok"
    assert seen["sam_pixels"] == manmade.sum() == len(seen["family_rows"])
    assert manmade[seen["family_rows"], seen["family_cols"]].all()
    assert (seen["natural_mask"] == natural).all()
    assert summary["pixels_matched"] == manmade.sum()
    assert summary["pixels_natural_skipped"] == natural.sum()
    assert summary["n_pixels"] == mask.sum()
    assert summary["n_pixels_matched"] + summary["n_pixels_no_match"] == summary["pixels_matched"]

    m = pq.read_table(out / "matches.parquet").to_pandas()
    assert manmade[m.row, m.col].all()
    with rasterio.open(out / "match_map.tif") as src:
        mm = src.read(1)
    assert (mm[natural] == -3).all() and (mm[~mask] == -2).all() and (mm[manmade] >= -1).all()
    spectra = np.load(out / "anomaly_pixel_spectra.npz")
    assert len(spectra["rows"]) == mask.sum()                     # the viewer inspects natural too


def test_without_the_namer_every_pixel_is_matched(tmp_path, monkeypatch):
    seen, mask, summary, out = _run(tmp_path, monkeypatch, {"namer": False})
    assert summary["namer"] == {"status": "disabled"}
    assert seen["sam_pixels"] == mask.sum() and seen["natural_mask"] is None
    assert summary["pixels_natural_skipped"] == 0 and not (out / "namer_level1_map.tif").exists()


def test_a_failed_namer_falls_back_to_matching_every_pixel(tmp_path, monkeypatch):
    from allotrope.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "no-models"))
    seen, mask, summary, _ = _run(tmp_path, monkeypatch, {"namer": True})
    assert summary["namer"]["status"] == "failed"
    assert seen["sam_pixels"] == mask.sum() and summary["pixels_matched"] == mask.sum()


# --- export ---------------------------------------------------------------


def _write_family_and_sam(art: Path, rows, cols):
    """Minimal family and SAM files for the man-made pixels (rows, cols)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio

    n = len(rows)
    pq.write_table(pa.table({
        "row": rows.astype("int32"), "col": cols.astype("int32"), "scored": np.ones(n, bool),
        "family": ["metal"] * n, "family_group": ["manmade"] * n,
        "family_code": np.full(n, 1, "int32"), "confidence": np.full(n, 0.999, "float32"),
        "answered": np.ones(n, bool), "best_entry": ["m1"] * n}), art / "family_matches.parquet")
    fmap = np.full((H, W), -2, np.int16)
    fmap[_mask()] = -3
    fmap[rows, cols] = 1
    with rasterio.open(art / "family_map.tif", "w", **{**_profile(), "dtype": "int16"}) as dst:
        dst.write(fmap, 1)
    (art / "family_legend.json").write_text(json.dumps({
        "alpha": 0.05, "p_star": 0.99, "n_manmade_families": 5,
        "codes": {"-3": {"family": "namer_natural", "group": "natural"},
                  "1": {"family": "metal", "group": "manmade"}}}))
    with rasterio.open(art / "match_map.tif", "w", **{**_profile(), "dtype": "int32"}) as dst:
        dst.write(np.where(fmap == 1, 2, -2).astype(np.int32), 1)
    pq.write_table(pa.table({
        "row": rows.astype("int32"), "col": cols.astype("int32"), "rank": np.zeros(n, "int32"),
        "library_ix": np.full(n, 2, "int32"), "material_id": ["m2"] * n, "name": ["Steel"] * n,
        "chapter": ["artificial"] * n, "asd_subtype": [None] * n,
        "angle_deg": np.full(n, 7.0, "float32"), "n_bands_used": np.full(n, 30, "int32")}),
        art / "matches.parquet")
    (art / "match_map_legend.json").write_text(json.dumps({"2": {"name": "Steel"}}))


def test_export_bundle_carries_the_namer(fake_namer, tmp_path):
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    from app.spectral_match.export import ExportSpec, build_hyper_bundle

    mask = _mask()
    _run_namer(tmp_path, mask)
    manmade = mask & (np.arange(H)[:, None] < 25)
    _write_family_and_sam(tmp_path, *np.nonzero(manmade))
    transform = from_origin(589050, 3653460, 30, 30)
    blob, name = build_hyper_bundle(ExportSpec(
        action_id="action_11111111-2222", scene_id="scene_33333333-4444", sensor_type="tanager",
        artifact_dir=tmp_path, splib_version=None, software_version="test",
        override_crs=CRS.from_epsg(32613), override_transform=transform))
    z = zipfile.ZipFile(io.BytesIO(blob))
    folder = name[:-4]

    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["schema"] == "allotrope-export-hyper/3"
    assert manifest["n_matched_pixels"] == mask.sum()                    # natural pixels too
    assert manifest["namer"]["n_objects_manmade"] == 2 and manifest["namer"]["n_pixels_natural"] == 4

    with MemoryFile(z.read(f"{folder}/hyper_namer_level1.tif")) as mf, mf.open() as src:
        assert src.crs.to_epsg() == 32613 and src.transform == transform and src.nodata == -1
        assert (src.read(1)[mask] >= 0).all() and (src.read(1)[~mask] == -1).all()

    rows = list(csv.DictReader(io.StringIO(z.read(f"{folder}/hyper_match_summary.csv").decode())))
    assert list(rows[0])[:10] == ["row", "col", "latitude", "longitude", "nm_mm", "nm_call", "nm_cls",
                                  "nm_p", "nm_conf", "nm_obj"]
    nat = [r for r in rows if r["nm_call"] == "natural"]
    assert len(nat) == 4 and all(r["family"] == "namer_natural" and r["material_1"] == "" for r in nat)
    metal = [r for r in rows if r["nm_cls"] == "metal"]
    assert len(metal) == 9 and all(r["nm_conf"] == "1" and r["material_1"] == "Steel" for r in metal)
    assert all(r["nm_conf"] == "0" for r in rows if r["nm_cls"] == "plastic")

    objs = list(csv.DictReader(io.StringIO(z.read(f"{folder}/hyper_namer_objects.csv").decode())))
    assert [o["call"] for o in objs] == ["manmade", "manmade", "natural"]
    assert 32.0 < float(objs[0]["latitude"]) < 33.1 and -104.2 < float(objs[0]["longitude"]) < -103.9

    shp = tmp_path / "shp"
    shp.mkdir()
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        (shp / f"a.{ext}").write_bytes(z.read(f"{folder}/hyper_anomaly_materials.{ext}"))
    cols = list(gpd.read_file(shp / "a.shp").columns)
    assert cols[:10] == ["row", "col", "lat", "lon", "nm_mm", "nm_call", "nm_cls", "nm_p", "nm_conf",
                         "nm_obj"]
    assert "family" in cols and "mat1" in cols


def test_export_bundle_without_georeference_is_in_pixel_coordinates(fake_namer, tmp_path):
    """v1.0.1: no georeference still exports - pixel grid, no CRS, no .prj, lat / lon blank, a MANIFEST note."""
    import geopandas as gpd
    from rasterio.io import MemoryFile

    from app.georef import NO_GEOREF_NOTE, pixel_grid
    from app.spectral_match.export import ExportSpec, MissingCRSError, build_hyper_bundle

    mask = _mask()
    _run_namer(tmp_path, mask)
    _write_family_and_sam(tmp_path, *np.nonzero(mask & (np.arange(H)[:, None] < 25)))
    spec = ExportSpec(action_id="action_11111111-2222", scene_id="scene_33333333-4444", sensor_type="generic_hsi",
                      artifact_dir=tmp_path, splib_version=None, software_version="test")
    with pytest.raises(MissingCRSError):                                  # no override at all: still refused
        build_hyper_bundle(spec)
    for subset in ("all", "manmade"):
        blob, name = build_hyper_bundle(dataclasses.replace(
            spec, override_transform=pixel_grid(), override_crs=None, georef_note=NO_GEOREF_NOTE, subset=subset))
        z = zipfile.ZipFile(io.BytesIO(blob))
        folder = name[:-4]
        assert not any(n.endswith(".prj") for n in z.namelist())
        tif = next(n for n in z.namelist() if n.endswith(".tif"))
        with MemoryFile(z.read(tif)) as mf, mf.open() as src:
            assert src.crs is None and tuple(src.transform)[:6] == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        assert json.loads(z.read(f"{folder}/MANIFEST.json"))["georeference"] == NO_GEOREF_NOTE
        shp = tmp_path / f"shp_{subset}"
        shp.mkdir()
        for n in z.namelist():
            if "/hyper_anomaly_materials" in n:
                (shp / f"a.{n.rsplit('.', 1)[1]}").write_bytes(z.read(n))
        pts = gpd.read_file(shp / "a.shp")
        assert pts.crs is None and len(pts) == (mask.sum() if subset == "all" else 11)
        assert np.allclose(pts.geometry.x, pts["col"] + 0.5) and np.allclose(pts.geometry.y, -(pts["row"] + 0.5))
        assert pts["lat"].isna().all()


def _bundle(art: Path, subset: str):
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from app.spectral_match.export import ExportSpec, build_hyper_bundle

    transform = from_origin(589050, 3653460, 30, 30)
    blob, name = build_hyper_bundle(ExportSpec(
        action_id="action_11111111-2222", scene_id="scene_33333333-4444", sensor_type="tanager",
        artifact_dir=art, splib_version=None, software_version="test", subset=subset,
        override_crs=CRS.from_epsg(32613), override_transform=transform))
    return zipfile.ZipFile(io.BytesIO(blob)), name, transform


@pytest.mark.parametrize("subset", ["manmade", "natural"])
def test_subset_export_is_a_boolean_mask_of_that_call(fake_namer, tmp_path, subset):
    import geopandas as gpd
    from rasterio.io import MemoryFile

    mask = _mask()
    _run_namer(tmp_path, mask)
    manmade = mask & (np.arange(H)[:, None] < 25)
    _write_family_and_sam(tmp_path, *np.nonzero(manmade))
    want = manmade if subset == "manmade" else mask & ~manmade
    z, name, transform = _bundle(tmp_path, subset)

    folder = f"hyper_33333333_11111111_{subset}"
    assert name == f"{folder}.zip"
    names = {n.split("/", 1)[1] for n in z.namelist()}
    assert all(n.split("/")[0] == folder for n in z.namelist())
    assert f"hyper_anomaly_mask_{subset}.tif" in names
    assert {f"hyper_match_summary_{subset}.csv", f"hyper_namer_objects_{subset}.csv",
            f"hyper_anomaly_materials_{subset}.shp", "MANIFEST.json"} <= names
    # No heatmaps or coded rasters in a subset bundle: the boolean mask is the only raster.
    assert [n for n in names if n.endswith(".tif")] == [f"hyper_anomaly_mask_{subset}.tif"]

    with MemoryFile(z.read(f"{folder}/hyper_anomaly_mask_{subset}.tif")) as mf, mf.open() as src:
        assert src.crs.to_epsg() == 32613 and src.transform == transform
        assert src.dtypes[0] == "uint8" and src.nodata is None
        m = src.read(1)
    assert set(np.unique(m)) <= {0, 1}
    assert (m.astype(bool) == want).all() and not (m.astype(bool) & ~mask).any()  # subset of mask

    rows = list(csv.DictReader(io.StringIO(
        z.read(f"{folder}/hyper_match_summary_{subset}.csv").decode())))
    assert len(rows) == want.sum() and {r["nm_call"] for r in rows} == {subset}
    assert all(want[int(r["row"]), int(r["col"])] for r in rows)
    objs = list(csv.DictReader(io.StringIO(
        z.read(f"{folder}/hyper_namer_objects_{subset}.csv").decode())))
    assert {o["call"] for o in objs} == {subset} and len(objs) == (2 if subset == "manmade" else 1)

    shp = tmp_path / "shp"
    shp.mkdir()
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        (shp / f"a.{ext}").write_bytes(z.read(f"{folder}/hyper_anomaly_materials_{subset}.{ext}"))
    pts = gpd.read_file(shp / "a.shp")
    assert len(pts) == want.sum() and set(pts.nm_call) == {subset} and pts.crs.to_epsg() == 32613

    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["subset"] == subset and manifest["n_matched_pixels"] == want.sum()


def test_subset_all_is_the_full_bundle(fake_namer, tmp_path):
    mask = _mask()
    _run_namer(tmp_path, mask)
    _write_family_and_sam(tmp_path, *np.nonzero(mask & (np.arange(H)[:, None] < 25)))
    z, name, _ = _bundle(tmp_path, "all")
    assert name == "hyper_33333333_11111111.zip"
    names = {n.split("/", 1)[1] for n in z.namelist()}
    assert {"hyper_anomaly_materials.tif", "hyper_namer_level1.tif",
            "hyper_match_summary.csv"} <= names
    assert json.loads(z.read(f"{name[:-4]}/MANIFEST.json"))["subset"] == "all"


def test_subset_without_namer_results_is_refused(tmp_path):
    from app.spectral_match.export import SubsetUnavailableError

    _write_family_and_sam(tmp_path, *np.nonzero(_mask()))         # a pre-namer action's files
    with pytest.raises(SubsetUnavailableError):
        _bundle(tmp_path, "manmade")


def test_subset_without_namer_results_is_a_422(tmp_path):
    pytest.importorskip("fastapi")                                  # api image, not every venv
    from fastapi import HTTPException

    from allotrope.api.action_export import check_subset

    _write_family_and_sam(tmp_path, *np.nonzero(_mask()))
    with pytest.raises(HTTPException) as exc:
        check_subset("spectral_library_match", tmp_path, "manmade")
    assert exc.value.status_code == 422 and "spectral-namer" in exc.value.detail
    with pytest.raises(HTTPException) as exc:
        check_subset("anomaly_detection_prep", tmp_path, "natural")
    assert exc.value.status_code == 422
    check_subset("spectral_library_match", tmp_path, "all")         # the old export still works


# --- config ----------------------------------------------------------------


def test_config_defaults_turn_the_namer_on():
    from allotrope.action_types import spectral_library_match as slm

    cfg = slm.validate_config({"input_anomaly_detection_output_id":
                               "output_00000000-0000-0000-0000-000000000000"}, "prisma")
    assert cfg["namer"] is True and cfg["namer_confident_p"] == 0.55
    for sensor in slm.META.accepted_sensor_types:
        d = slm.META.default_config_per_sensor[sensor]
        assert d["namer"] is True and d["namer_confident_p"] == 0.55
    with pytest.raises(ValueError):
        slm.validate_config({"input_anomaly_detection_output_id":
                             "output_00000000-0000-0000-0000-000000000000",
                             "namer_confident_p": 1.5}, "prisma")
    with pytest.raises(ValueError):
        slm.validate_config({"input_anomaly_detection_output_id":
                             "output_00000000-0000-0000-0000-000000000000"}, "landsat9")
