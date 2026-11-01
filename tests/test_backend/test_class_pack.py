"""
The class-driven output pack of spectral_library_match: tokens, the man-made cut, the selection and the zip.

Uses the fake namer of test_spectral_namer_outputs.py: three objects - a 3 x 3 metal block (P(man-made) 0.92,
confident), a plastic pair (0.66, tentative) and a natural 2 x 2 (0.12, top class concrete_asphalt at 0.6). Needs
numpy / rasterio / pyarrow / geopandas / PIL, not the database.
"""
from __future__ import annotations

import csv
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


def _pack(art, classes, **kw):
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from allotrope.version import __version__
    from app.spectral_match.class_pack import ClassPackSpec, build_class_pack
    from app.spectral_match.class_selection import parse_classes

    spec = ClassPackSpec(action_id="action_11111111-2222", scene_id="scene_33333333-4444", sensor_type="tanager",
                         artifact_dir=art, software_version=f"allotrope/{__version__}",
                         crs=CRS.from_epsg(32613), transform=from_origin(*TRANSFORM_ARGS),
                         tokens=parse_classes(classes), chain={"scene": {"id": "scene_33333333-4444"}}, **kw)
    blob, name, sel = build_class_pack(spec)
    return zipfile.ZipFile(io.BytesIO(blob)), name, sel, spec


@pytest.mark.parametrize("classes,kw", [("manmade", {}), ("oil_fuel,plastic", {}), ("natural", {}),
                                        ("manmade", {"manmade_p": 0.1}), ("oil_fuel", {})])
def test_pack_contents(art, tmp_path, classes, kw):
    import geopandas as gpd
    from rasterio.io import MemoryFile

    z, name, sel, spec = _pack(art, classes, **kw)
    slug = sel.slug()
    folder = f"hyper_33333333_11111111_{slug}"
    assert name == f"{folder}.zip"
    names = {n.split("/", 1)[1] for n in z.namelist()}
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        assert f"hyper_anomaly_pixels_{slug}.{ext}" in names and f"hyper_anomaly_objects_{slug}.{ext}" in names
    assert {f"hyper_anomaly_mask_{slug}.tif", "hyper_namer_level1.tif", "hyper_namer_class.tif",
            f"hyper_quicklook_{slug}.png", f"hyper_anomaly_objects_{slug}.csv", "MANIFEST.json"} <= names
    assert all(n.startswith("hyper_") or n == "MANIFEST.json" for n in names)

    with MemoryFile(z.read(f"{folder}/hyper_anomaly_mask_{slug}.tif")) as mf, mf.open() as src:
        assert src.crs == spec.crs and src.transform == spec.transform and src.dtypes[0] == "uint8"
        m = src.read(1)
    assert set(np.unique(m)) <= {0, 1} and int(m.sum()) == sel.n_pixels
    assert not (m.astype(bool) & ~_mask()).any()
    with MemoryFile(z.read(f"{folder}/hyper_namer_level1.tif")) as mf, mf.open() as src:
        l1 = src.read(1)
    assert int((l1 == 1).sum()) == _select(art, "manmade", **kw).n_pixels

    text = z.read(f"{folder}/hyper_anomaly_objects_{slug}.csv").decode()
    assert text.splitlines()[0] == "obj_id,call,class,confident,p_manmade,p_class,n_px,lat,lon"
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == sel.n_objects
    assert all(32.0 < float(r["lat"]) < 33.1 for r in rows)

    shp = tmp_path / f"shp_{slug}"
    shp.mkdir()
    for base in ("pixels", "objects"):
        for ext in ("shp", "shx", "dbf", "prj", "cpg"):
            (shp / f"{base}.{ext}").write_bytes(z.read(f"{folder}/hyper_anomaly_{base}_{slug}.{ext}"))
    pts, objs = gpd.read_file(shp / "pixels.shp"), gpd.read_file(shp / "objects.shp")
    assert len(pts) == sel.n_pixels and len(objs) == sel.n_objects and pts.crs.to_epsg() == 32613

    manifest = json.loads(z.read(f"{folder}/MANIFEST.json"))
    assert manifest["software_version"] == "allotrope/1.0.0"
    assert manifest["selection"]["classes"] == list(sel.tokens)
    assert manifest["selection"]["n_pixels"] == sel.n_pixels and manifest["selection"]["n_objects"] == sel.n_objects
    assert manifest["selection"]["manmade_p"] == kw.get("manmade_p", 0.5)
    assert manifest["chain"] == {"scene": {"id": "scene_33333333-4444"}}

    from PIL import Image
    assert Image.open(io.BytesIO(z.read(f"{folder}/hyper_quicklook_{slug}.png"))).size[0] >= W


def test_quicklook_marks_selected_pixels(art):
    from PIL import Image

    z, name, sel, _ = _pack(art, "manmade")
    png = np.asarray(Image.open(io.BytesIO(z.read(f"{name[:-4]}/hyper_quicklook_{sel.slug()}.png"))).convert("RGB"))
    scale = png.shape[1] / W
    y, x = int(6.5 * scale), int(6.5 * scale)                 # inside the metal block
    assert tuple(png[y, x]) == (255, 230, 0)
    y, x = int(31 * scale), int(11 * scale)                   # the natural object is not selected
    assert tuple(png[y, x]) != (255, 230, 0)


def test_pack_without_namer_results_is_refused(tmp_path):
    from app.spectral_match.class_selection import SelectionError

    _write_family_and_sam(tmp_path, *np.nonzero(_mask()))
    with pytest.raises(SelectionError, match="spectral-namer"):
        _pack(tmp_path, "manmade")


def test_version_is_one():
    from allotrope.version import APP_NAME, __version__

    assert (APP_NAME, __version__) == ("Allotrope", "1.0.0")


def test_api_rejects_unknown_tokens():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from allotrope.api.action_class_pack import parse_tokens

    with pytest.raises(HTTPException) as exc:
        parse_tokens("oil_fuel,gold")
    assert exc.value.status_code == 422 and "gold" in exc.value.detail
    assert parse_tokens("oil_fuel") == ("oil_fuel",)
