"""readers.sensors (identification, time, units) and readers.scene (facts, grade, checklist)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import WL
from readers import sensors
from readers.scene import Fact, Scene


@pytest.mark.parametrize("name,expect", [
    ("ENMAP01-____L1C-DT0000056279_x-SPECTRAL_IMAGE.TIF", "enmap"),
    ("PRS_L1_STD_OFFL_20260305054623_20260305054628_0001.he5", "prisma"),
    ("ang20151219t081738_rdn_v2m2_img", "aviris_ng"),
    ("20250712_100036_58_4001_basic_radiance_hdf5.h5", "tanager"),
    ("ghost_dataset2.tif", "ghost"),
])
def test_identify_by_name(name, expect):
    f = sensors.identify(f"/data/x/{name}", "", WL, None)
    assert (f.value, f.source) == (expect, "metadata")


def test_folder_name_proves_nothing():
    f = sensors.identify("/data/ghost_run/scene.tif", "", WL, None)
    assert f.value != "ghost" or f.source != "metadata"


def test_identify_by_text_and_signature():
    assert sensors.identify("a.tif", "sensor = Tanager-1", WL, None).value == "tanager"
    ang = np.linspace(377, 2500, 425)
    assert sensors.identify("a.tif", "", ang, None).value == "aviris_ng"
    enmap = np.linspace(418, 2445, 224)
    assert sensors.identify("a.tif", "", enmap, None).value == "enmap"
    f = sensors.identify("a.tif", "", WL, None)
    assert (f.value, f.source) == ("unknown", "derived")


def test_acquisition_time_from_meta_and_name():
    t = sensors.acquisition_time("x.tif", {"acquisition_time": "2025-06-01T06:00:00Z"})
    assert t.source == "metadata" and t.value == datetime(2025, 6, 1, 6, tzinfo=timezone.utc)
    t = sensors.acquisition_time("ang20151219t081738_rdn", {})
    assert t.source == "derived" and t.value.hour == 8
    t = sensors.acquisition_time("/x/PRS_L1_STD_OFFL_20260305054623_x.he5", {})
    assert t.value == datetime(2026, 3, 5, 5, 46, 23, tzinfo=timezone.utc)
    assert sensors.acquisition_time("nothing.tif", {}).source == "missing"
    assert sensors._parse_time("2025:06:01 06:00:00").year == 2025
    assert sensors._parse_time("garbage") is None


@pytest.mark.parametrize("unit,scale", [("uW/(cm^2 sr nm)", 1.0), ("W/(m^2 sr um)", 0.1), ("W/m2/sr/nm", 100.0),
                                        ("mW/(cm^2 sr um)", 1.0)])
def test_radiance_units_from_text(unit, scale):
    u, s = sensors.radiance_units({"units": unit}, np.ones((10, len(WL))), WL, lambda i: 1.0)
    assert s.value == scale and s.source == "metadata"


def test_radiance_units_inferred_from_magnitude():
    smp = np.full((100, len(WL)), 9000.0)   # W/m^2/sr/nm-ish numbers: needs x0.001 to reach ~0.3 TOA
    u, s = sensors.radiance_units({}, smp, WL, lambda i: 0.033)
    assert s.source == "derived"
    assert 0.15 <= 9000 * s.value * 0.033 <= 1.2


def _scene(**over):
    f = lambda v, src="metadata": Fact(v, src)
    kw = dict(path="x", fmt="geotiff", sensor=f("enmap"), shape=(2, 3, 4), interleave="bil", wl=WL[:3],
              fwhm=f(np.ones(3)), read_rows=lambda a, b: None, units=f("u"), radiance_scale=f(1.0),
              time=f(datetime(2025, 1, 1)), location=f(lambda a, b: None), sun=f(lambda a, b: None),
              view=f(lambda a, b: None), altitude_km=f(650.0), noise=f(None), dem=f("dem.vrt"))
    kw.update(over)
    return Scene(**kw)


def test_grades():
    assert _scene().grade() == "A"
    assert _scene(sun=Fact(lambda a, b: None, "derived")).grade() == "B"
    assert _scene(view=Fact(None, "assumed")).grade() == "C"
    assert _scene(dem=Fact(None, "assumed")).grade() == "C"
    assert _scene(time=Fact(None, "assumed")).grade() == "D"
    assert _scene(radiance_scale=Fact(None, "missing")).grade() == "D"


def test_checklist_summarises_values():
    rows = {r["item"]: r for r in _scene().checklist()}
    assert rows["sun"]["value"] == "per pixel"
    assert rows["fwhm"]["value"].startswith("3 values")
    assert rows["time"]["value"].startswith("2025-01-01")
    assert _scene().lines == 2 and _scene().samples == 4


def test_fact_rejects_unknown_source():
    with pytest.raises(AssertionError):
        Fact(1, "guessed")
