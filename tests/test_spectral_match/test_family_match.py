"""
Tests for app/spectral_match/family_library.py and family_match.py.

Synthetic checks where the right answer is known, plus two end-to-end checks on the committed
family library (data/family_library) resampled to the 165-band common grid through the same
cache builder the worker uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.spectral_match.family_library import balanced_subset, common_band_mask, parse_tag, tag_library
from app.spectral_match.family_match import ace, calibrate, family_scores, match_families, p_star
from app.spectral_match.library import LibraryEntry, build_sensor_cache, load_slim_bundle

BUNDLE = Path(__file__).resolve().parents[2] / "data" / "family_library"


def _entry(name, chapter, refl):
    refl = np.asarray(refl, np.float32)
    return LibraryEntry(name, name, chapter, "ASDFR", refl, np.isfinite(refl).astype(np.uint8), 1.0)


def test_p_star_bounds_false_claims_by_alpha():
    assert p_star(0.05, 10) == pytest.approx(0.995)
    assert p_star(0.02, 15) == pytest.approx(1 - 0.02 / 15)
    assert p_star(0.05, 0) == pytest.approx(0.95)          # no man-made families: no division by 0


def test_calibrate_is_share_of_null_strictly_below():
    null = np.arange(100, dtype=float)[:, None]
    pct = calibrate(np.array([[-1.0], [50.0], [500.0]]), null)
    assert pct[:, 0].tolist() == [0.0, 0.5, 1.0]


def test_parse_tag_and_untagged_entries_are_excluded():
    assert parse_tag("manmade.roofing") == ("manmade", "roofing")
    assert parse_tag("minerals") is None and parse_tag("other.x") is None and parse_tag("natural.") is None
    lib = tag_library([_entry("a", "natural.soil", [0.2] * 5), _entry("b", "minerals", [0.3] * 5)])
    assert lib.group.tolist() == ["natural", "exclude"]


def test_cap_and_band_mask():
    rng = np.random.default_rng(0)
    ents = [_entry(f"n{i}", "natural.mineral", rng.uniform(0.1, 0.6, 20)) for i in range(55)]
    gappy = rng.uniform(0.1, 0.6, 20)
    gappy[4] = np.nan
    ents += [_entry(f"m{i}", "manmade.metal", rng.uniform(0.1, 0.6, 20)) for i in range(5)]
    ents += [_entry("gap", "manmade.metal", gappy)]
    lib = tag_library(ents)
    pv = np.ones((100, 20), np.uint8)
    pv[:3, 7] = 0                                          # band 7 bad in 3 % of pixels
    bands, entries = common_band_mask(pv, lib)
    assert not bands[7] and bands[4]                       # band 4 lacks 1 entry only: kept
    assert not entries[-1] and entries.sum() == 60         # the gappy entry goes instead
    keep = balanced_subset(lib.subset(entries))
    fam = lib.subset(entries).family[keep]
    assert (fam == "mineral").sum() == 40 and (fam == "metal").sum() == 5


def test_simplex_ace_equals_ace_for_one_spectrum_and_bounds_members():
    rng = np.random.default_rng(1)
    B = 30
    x, m = rng.uniform(0.1, 0.5, (50, B)), rng.uniform(0.1, 0.5, (50, B))
    E = rng.uniform(0.1, 0.5, (3, B))
    W = np.eye(B)
    one = family_scores(x, m, E[:1], W, [np.array([0])], np.array([True]))[:, 0]
    assert np.allclose(one, np.maximum(ace(x, m, E[:1], W)[:, 0], 0), atol=1e-9)
    cone = family_scores(x, m, E, W, [np.arange(3)], np.array([True]))[:, 0]
    assert np.all(cone >= np.maximum(ace(x, m, E, W), 0).max(1) - 1e-9)


@pytest.fixture(scope="module")
def grid_library(tmp_path_factory):
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
    wl = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, float)
    ents, _ = build_sensor_cache(bundle=load_slim_bundle(BUNDLE), sensor_id="test", target_wl_nm=wl,
                                 target_fwhm_nm=np.full(len(wl), 10.0),
                                 cache_dir=tmp_path_factory.mktemp("cache"), progress=False)
    return ents


def _scene(ents, rng, H=110, W=110):
    """Natural background: random mixtures of 10 vegetation / soil spectra plus noise."""
    base = [np.nan_to_num(e.refl) for e in ents if e.chapter == "natural.vegetation"][:5] + \
           [np.nan_to_num(e.refl) for e in ents if e.chapter == "natural.soil"][:5]
    cube = np.einsum("hwk,kb->bhw", rng.dirichlet(np.ones(10), (H, W)), np.stack(base))
    return (cube + rng.normal(0, 0.003, cube.shape)).astype(np.float32)


def test_ordinary_pixels_get_a_manmade_name_at_most_alpha(grid_library):
    rng = np.random.default_rng(2)
    cube = _scene(grid_library, rng)
    B, H, W = cube.shape
    r, c = np.divmod(rng.choice(H * W, 400, replace=False), W)
    res = match_families(cube, np.ones((B, H, W), np.uint8), np.zeros((H, W), bool), r, c,
                         grid_library, alpha=0.05)
    assert ((res.group == "manmade") & res.answered).mean() <= 0.05 + 0.02   # 400 px: binomial slack


def test_pixels_with_nodata_on_a_kept_band_are_not_scored(grid_library):
    rng = np.random.default_rng(4)
    cube = _scene(grid_library, rng)
    B, H, W = cube.shape
    r, c = np.divmod(rng.choice(H * W, 400, replace=False), W)
    validity = np.ones((B, H, W), np.uint8)
    validity[50, r[:2], c[:2]] = 0                         # 0.5 % of pixels: band 50 still kept
    cube[50, r[:2], c[:2]] = -9999.0                       # AVIRIS-NG nodata fill
    res = match_families(cube, validity, np.zeros((H, W), bool), r, c, grid_library)
    assert res.bands_used[50]
    assert res.scored.tolist() == [False, False] + [True] * 398
    assert res.family[:2].tolist() == ["", ""] and not res.answered[:2].any()
    assert (res.family[2:] != "").all()
    assert len(res.family_names) == len(res.family_groups) >= 2


def test_cache_builds_for_a_vendable_with_null_fwhm(tmp_path):
    from types import SimpleNamespace
    from app.spectral_match.family_match import ensure_family_cache
    wl = np.arange(450.0, 2400.0, 10.0)
    fwhm = [10.0] * len(wl)
    fwhm[3] = None                                          # PRISMA ships some null FWHMs
    v = SimpleNamespace(band_cw_order=wl.tolist(), band_fwhm_order=fwhm,
                        band_validity_by_position=[1] * len(wl))
    ents = ensure_family_cache(v, "test", cache_dir=str(tmp_path), bundle_dir=str(BUNDLE))
    assert ents and not any(e.valid[3] for e in ents)      # the null-FWHM band is invalid, not a crash


def test_held_out_roofing_target_is_named_roofing(grid_library):
    rng = np.random.default_rng(3)
    cube = _scene(grid_library, rng)
    B, H, W = cube.shape
    roof = [e for e in grid_library if e.chapter == "manmade.roofing"]
    target, library = roof[0], [e for e in grid_library if e is not roof[0]]   # hold it out
    r, c = np.divmod(rng.choice(H * W, 40, replace=False), W)
    r, c = np.clip(r, 8, H - 9), np.clip(c, 8, W - 9)
    cube[:, r, c] = 0.6 * cube[:, r, c] + 0.4 * np.nan_to_num(target.refl)[:, None]
    mask = np.zeros((H, W), bool)
    mask[r, c] = True
    res = match_families(cube, np.ones((B, H, W), np.uint8), mask, r, c, library)
    assert res.answered.mean() >= 0.8
    assert (res.family[res.answered] == "roofing").mean() >= 0.8
