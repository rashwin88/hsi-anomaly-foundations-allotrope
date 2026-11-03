"""
The class-driven output pack of spectral_library_match: tokens, the man-made cut, the selection and the zip.

Uses the fake namer of test_spectral_namer_outputs.py: three objects - a 3 x 3 metal block (P(man-made) 0.92,
confident), a plastic pair (0.66, tentative) and a natural 2 x 2 (0.12, top class concrete_asphalt at 0.6). Needs
numpy / rasterio / pyarrow / geopandas / PIL, not the database.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from test_spectral_namer_outputs import (  # noqa: F401  (fake_namer is a fixture)
    H, W, _mask, _run_namer, _write_family_and_sam, fake_namer,
)

TRANSFORM_ARGS = (589050, 3653460, 30, 30)


@pytest.fixture
def art(fake_namer, tmp_path):
    _run_namer(tmp_path, _mask())
    _write_family_and_sam(tmp_path, *np.nonzero(_mask() & (np.arange(H)[:, None] < 25)))
    from PIL import Image
    Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(tmp_path / "rgb.png")
    return tmp_path


def _select(art, classes, **kw):
    from app.spectral_match.class_selection import parse_classes, select

    return select(art, parse_classes(classes), confident_only=kw.get("confident_only", False),
                  manmade_p=kw.get("manmade_p"), confident_p=0.55)


def test_tokens_are_checked_against_the_namer_classes(monkeypatch):
    from app.spectral_match import class_selection as cs
    from app.spectral_match.namer import classes

    assert cs.parse_classes("plastic, oil_fuel,natural") == ("natural", "oil_fuel", "plastic")
    with pytest.raises(cs.SelectionError, match="unknown class glass_solar"):
        cs.parse_classes("glass_solar")
    with pytest.raises(cs.SelectionError, match="empty"):
        cs.parse_classes(" , ")
    # Read at call time: a class added to namer/classes.py is a valid token with no change here.
    monkeypatch.setattr(classes, "MANMADE_CLASSES", classes.MANMADE_CLASSES + ("glass_solar",))
    assert cs.parse_classes("glass_solar") == ("glass_solar",)


def test_selection_at_the_actions_cut(art):
    assert _select(art, "manmade").counts() == {"objects": 2, "pixels": 11,
                                                "per_class": {"metal": 1, "plastic": 1}}
    assert _select(art, "natural").counts()["pixels"] == 4
    assert _select(art, "metal,plastic").n_objects == 2
    assert _select(art, "manmade", confident_only=True).counts()["per_class"] == {"metal": 1}
    assert _select(art, "natural,plastic", confident_only=True).n_objects == 1    # natural kept, plastic dropped
    assert _select(art, "oil_fuel").n_objects == 0


def test_a_lower_cut_names_the_newly_manmade_object(art):
    low = _select(art, "manmade", manmade_p=0.1)
    assert low.n_objects == 3 and low.class_source == "prob_columns"
    o = low.objects
    k = list(o["obj_id"]).index(3)
    assert o["call"][k] == "manmade" and o["cls"][k] == "concrete_asphalt"
    assert o["p_cls"][k] == pytest.approx(0.6) and bool(o["confident"][k])
    assert _select(art, "natural", manmade_p=0.1).n_objects == 0
    high = _select(art, "manmade", manmade_p=0.8)
    assert high.counts()["per_class"] == {"metal": 1}
    assert low.slug() == "manmade_p01" and _select(art, "oil_fuel,plastic").slug() == "oil_fuel-plastic"


def test_without_prob_columns_the_class_comes_from_the_pixels(art):
    import pyarrow.parquet as pq

    t = pq.read_table(art / "namer_objects.parquet")
    pq.write_table(t.drop([c for c in t.column_names if c.startswith("prob_")]), art / "namer_objects.parquet")
    low = _select(art, "concrete_asphalt", manmade_p=0.1)
    assert low.class_source == "pixels" and low.n_objects == 1
    k = list(low.objects["obj_id"]).index(3)
    assert low.objects["p_cls"][k] == pytest.approx(0.6)     # one class over all four pixels: exact


def _pack(art, classes, georef=True, **kw):
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from allotrope.version import __version__
    from app.georef import NO_GEOREF_NOTE, pixel_grid
    from app.spectral_match.class_pack import ClassPackSpec, build_class_pack
    from app.spectral_match.class_selection import parse_classes

    grid = ({"crs": CRS.from_epsg(32613), "transform": from_origin(*TRANSFORM_ARGS)} if georef
            else {"crs": None, "transform": pixel_grid(), "georef_note": NO_GEOREF_NOTE})
    spec = ClassPackSpec(action_id="action_11111111-2222", scene_id="scene_33333333-4444", sensor_type="tanager",
                         artifact_dir=art, software_version=f"allotrope/{__version__}",
                         tokens=parse_classes(classes), chain={"scene": {"id": "scene_33333333-4444"}}, **grid, **kw)
    blob, name, sel = build_class_pack(spec)
    return zipfile.ZipFile(io.BytesIO(blob)), name, sel, spec


@pytest.mark.parametrize("classes,kw", [("manmade", {}), ("oil_fuel,plastic", {}), ("natural", {}),
                                        ("manmade", {"manmade_p": 0.1}), ("oil_fuel", {})])
def test_pack_contents(art, tmp_path, classes, kw):
    from rasterio.io import MemoryFile

    z, name, sel, spec = _pack(art, classes, **kw)
    slug = sel.slug()
    folder = f"hyper_33333333_11111111_{slug}"
    assert name == f"{folder}.zip"
    names = {n.split("/", 1)[1] for n in z.namelist()}
    # v1.0.1: exactly the mask, the pixel shapefile and the manifest.
    assert names == {f"hyper_anomaly_mask_{slug}.tif", "MANIFEST.json"} | {
        f"hyper_anomaly_pixels_{slug}.{ext}" for ext in ("shp", "shx", "dbf", "prj", "cpg")}

    with MemoryFile(z.read(f"{folder}/hyper_anomaly_mask_{slug}.tif")) as mf, mf.open() as src:
        assert src.crs == spec.crs and src.transform == spec.transform and src.dtypes[0] == "uint8"
        m = src.read(1)
    assert set(np.unique(m)) <= {0, 1} and int(m.sum()) == sel.n_pixels
    assert not (m.astype(bool) & ~_mask()).any()

    pts = _read_pixels(z, folder, slug, tmp_path)
    assert len(pts) == sel.n_pixels and pts.crs.to_epsg() == 32613
    assert all(32.0 < float(v) < 33.1 for v in pts["lat"])
    assert {"material", "mat_lib", "mat_ang", "mat1", "scr1", "family", "nm_cls"} <= set(pts.columns)
    assert set(pts.loc[pts["nm_call"] == "natural", "nm_cls"]) <= {"natural"}

    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["software_version"] == "allotrope/1.0.2"
    assert manifest["files"] == sorted(names - {"MANIFEST.json"}) and "georeference" not in manifest
    assert manifest["selection"]["classes"] == list(sel.tokens)
    assert manifest["selection"]["n_pixels"] == sel.n_pixels and manifest["selection"]["n_objects"] == sel.n_objects
    assert manifest["selection"]["manmade_p"] == kw.get("manmade_p", 0.5)
    assert manifest["chain"] == {"scene": {"id": "scene_33333333-4444"}}


def _read_pixels(z, folder, slug, tmp_path):
    import geopandas as gpd

    shp = tmp_path / f"shp_{slug}"
    shp.mkdir(exist_ok=True)
    for n in z.namelist():
        if f"/hyper_anomaly_pixels_{slug}." in n:
            (shp / f"pixels.{n.rsplit('.', 1)[1]}").write_bytes(z.read(n))
    return gpd.read_file(shp / "pixels.shp")


def _write_library_and_spectra(art):
    """A 3-entry family library (two metal, one natural) and pixel spectra: the metal block is a scaled copy of
    the ECOSTRESS metal entry, so that entry is its closest metal material at 0 degrees."""
    B = 6
    rng = np.random.default_rng(3)
    refl = rng.uniform(0.1, 0.6, (3, B)).astype(np.float32)
    valid = np.ones((3, B), np.uint8)
    valid[0, 5] = 0                                           # a band the USGS entry does not cover
    np.savez(art / "family_library_refl.npz", refl=refl, valid=valid, wavelengths=np.arange(B, dtype=float))
    legend = json.loads((art / "family_legend.json").read_text())
    legend["library"] = [{"id": "usgs:steel", "label": "Steel USGS", "cls": "metal"},
                         {"id": "ecostress:alu", "label": "Aluminium ECOSTRESS", "cls": "metal"},
                         {"id": "usgs:grass", "label": "Grass", "cls": "natural"}]
    (art / "family_legend.json").write_text(json.dumps(legend))
    rows, cols = np.nonzero(_mask())
    spectra = np.tile(rng.uniform(0.1, 0.6, B).astype(np.float32), (len(rows), 1))
    spectra[rows < 10] = 2.0 * refl[1]
    np.savez(art / "anomaly_pixel_spectra.npz", rows=rows.astype(np.int32), cols=cols.astype(np.int32),
             spectra=spectra, valid=np.ones_like(spectra, np.uint8), wavelengths=np.arange(B, dtype=float))


def test_pixel_material_is_the_objects_closest_entry_in_its_class(art, tmp_path):
    _write_library_and_spectra(art)
    z, name, sel, _ = _pack(art, "manmade,natural")
    pts = _read_pixels(z, name[:-4], sel.slug(), tmp_path)
    metal = pts[pts["nm_cls"] == "metal"]
    assert len(metal) == 9 and set(metal["material"]) == {"Aluminium ECOSTRESS"}
    assert set(metal["mat_lib"]) == {"ECOSTRESS"} and np.allclose(metal["mat_ang"], 0.0, atol=0.01)
    natural = pts[pts["nm_call"] == "natural"]
    assert len(natural) == 4 and set(natural["material"]) == {"Grass"} and set(natural["nm_cls"]) == {"natural"}
    plastic = pts[pts["nm_cls"] == "plastic"]                 # no plastic entry in the library: no material
    assert len(plastic) == 2 and set(plastic["material"].fillna("")) == {""}


def test_pack_without_georeference_is_in_pixel_coordinates(art, tmp_path):
    from rasterio.io import MemoryFile

    z, name, sel, _ = _pack(art, "manmade", georef=False)
    folder, slug = name[:-4], sel.slug()
    names = {n.split("/", 1)[1] for n in z.namelist()}
    assert f"hyper_anomaly_pixels_{slug}.prj" not in names and f"hyper_anomaly_pixels_{slug}.shp" in names
    with MemoryFile(z.read(f"{folder}/hyper_anomaly_mask_{slug}.tif")) as mf, mf.open() as src:
        assert src.crs is None and tuple(src.transform)[:6] == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        assert int(src.read(1).sum()) == sel.n_pixels
    pts = _read_pixels(z, folder, slug, tmp_path)
    assert pts.crs is None and len(pts) == sel.n_pixels
    assert np.allclose(pts.geometry.x, pts["col"] + 0.5) and np.allclose(pts.geometry.y, -(pts["row"] + 0.5))
    assert pts["lat"].isna().all()
    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["georeference"].startswith("no georeference: pixel coordinates")
    assert manifest["grid"]["crs_wkt"] is None and manifest["grid"]["transform"] == [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]


def test_pack_without_namer_results_is_refused(tmp_path):
    from app.spectral_match.class_selection import SelectionError

    _write_family_and_sam(tmp_path, *np.nonzero(_mask()))
    with pytest.raises(SelectionError, match="spectral-namer"):
        _pack(tmp_path, "manmade")


def test_version_is_one():
    from allotrope.version import APP_NAME, __version__

    assert (APP_NAME, __version__) == ("Allotrope", "1.0.2")


def test_api_rejects_unknown_tokens():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from allotrope.api.action_class_pack import parse_tokens

    with pytest.raises(HTTPException) as exc:
        parse_tokens("oil_fuel,gold")
    assert exc.value.status_code == 422 and "gold" in exc.value.detail
    assert parse_tokens("oil_fuel") == ("oil_fuel",)
