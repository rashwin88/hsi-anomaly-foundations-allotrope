"""Story frames (app/story/frames.py) from a synthetic spectral_library_match chain on disk, and the
endpoint's 422 checks (backend/allotrope/api/action_story.py) without the database."""
from __future__ import annotations

import io
import json
import os
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from app.story.chain import SelectionParams, Step, StoryChain, StoryError
from app.story.frames import build_story
from app.story.render import render_gif

# allotrope.config builds Settings at import and requires these; nothing here connects.
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

H, W = 60, 80


def _tif(path, arr):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype) as dst:
        dst.write(arr, 1)


def _objects(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [  # obj_id, n_px, row, col, p_manmade, call, cls, p_cls, confident
        (0, 4, 10, 10, 0.9, "manmade", "oil_fuel", 0.8, True),
        (1, 2, 20, 30, 0.7, "manmade", "plastic", 0.5, False),
        (2, 6, 40, 50, 0.2, "natural", None, float("nan"), False),
        (3, 1, 50, 70, 0.4, "natural", None, float("nan"), False),
    ]
    names = ["obj_id", "n_px", "row", "col", "p_manmade", "call", "cls", "p_cls", "confident"]
    pq.write_table(pa.table(dict(zip(names, [list(c) for c in zip(*rows)]))), path)


def _chain(tmp_path, *, namer=True, seg=True, exclude_at_scoring=None):
    d = {k: tmp_path / k for k in ("match", "prep", "scoring", "seg", "bf")}
    for p in d.values():
        p.mkdir()
    rgb = np.full((H, W, 3), 120, np.uint8)
    rgb[:, :8] = 0                                                # no-data strip
    Image.fromarray(rgb).save(d["match"] / "rgb.png")
    mask = np.zeros((H, W), np.uint8)
    mask[9:11, 9:11] = 1
    mask[20, 30:32] = 1
    mask[39:42, 49:51] = 1
    mask[50, 70] = 1
    _tif(d["prep"] / "anomaly_mask.tif", mask)
    score = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    score[:, :8] = 0
    _tif(d["prep"] / "composite_score.tif", score)
    codes = np.zeros((H, W), np.uint8)
    codes[:, :8] = 255
    codes[:10, 20:40] = 2                                         # cloud
    codes[50:, 10:30] = 1                                         # water
    codes[30:35, 60:70] |= 0x10                                   # thin cloud on land
    codes.tofile(d["seg"] / "class_codes.bin")
    _tif(d["seg"] / "class_map.tif", np.where(codes == 255, 255, codes & 0x0F).astype(np.uint8))
    legend = [{"key": "cloud", "pct_of_valid": 2.9}, {"key": "water", "pct_of_valid": 15.0}]
    if namer:
        _objects(d["match"] / "namer_objects.parquet")
        (d["match"] / "namer_legend.json").write_text(json.dumps({"classes": {
            "1": {"cls": "oil_fuel", "label": "oil / fuel"}, "2": {"cls": "plastic", "label": "plastic"}}}))
    scoring_cfg = {"model_codenames": ["Indradhanu V2"]}
    if exclude_at_scoring is not None:
        scoring_cfg["exclude_classes"] = exclude_at_scoring
    return StoryChain(
        sensor="prisma", scene_name="PRS_TEST", acquired=datetime(2025, 11, 27), scene_bands=234,
        match=Step(d["match"], {"namer_confident_p": 0.55}),
        prep=Step(d["prep"], {"committed_threshold_mode": "percentile", "committed_threshold": 92.0},
                  {"active_codenames": ["Indradhanu V2"]}),
        scoring=Step(d["scoring"], scoring_cfg),
        segmentation=Step(d["seg"], {"exclude_classes": ["cloud"]},
                          {"scene_shape": [H, W], "class_names": ["land", "water", "cloud", "shadow", "snow"],
                           "legend": legend}) if seg else None,
        band_filter=Step(d["bf"], {}, {"band_count": 186, "spatial_validity_pct": 90.0}),
    )


def test_seven_frames_with_captions(tmp_path):
    frames, title = build_story(_chain(tmp_path), SelectionParams(), size=400)
    caps = [f.caption for f in frames]
    assert len(frames) == 7 and title == "Allotrope · PRS_TEST"
    assert caps[0] == "PRISMA · 2025-11-27 · 80×60 px · 234 bands"
    assert caps[1] == "186 of 234 bands usable; no-data removed (10.0% of the frame)"
    assert caps[2] == "Cloud left out (2.9%)"
    assert caps[3] == "Indradhanu V2 scores every pixel"
    assert caps[4] == "Top 8% → 13 anomalous pixels"
    assert caps[5] == "4 objects · 2 man-made"
    assert caps[6] == "Man-made → 2 objects"
    assert {f.image.shape for f in frames} == {(300, 400, 3)}       # one extent, display grid


def test_scoring_exclude_classes_win_over_segmentation(tmp_path):
    frames, _ = build_story(_chain(tmp_path, exclude_at_scoring=["cloud", "water"]), SelectionParams(), 400)
    assert frames[2].caption == "Cloud (2.9%) and water (15.0%) left out"


def test_selection_follows_class_selection(tmp_path):
    chain = _chain(tmp_path)
    f, _ = build_story(chain, SelectionParams(tokens=("oil_fuel", "plastic")), 400)
    assert f[-1].caption == "Oil / fuel + plastic → 2 objects"
    f, _ = build_story(chain, SelectionParams(tokens=("oil_fuel", "plastic"), confident_only=True), 400)
    assert f[-1].caption == "Oil / fuel + plastic, confident only → 1 object"
    f, _ = build_story(chain, SelectionParams(tokens=("manmade",), manmade_p=0.35), 400)
    assert f[-2].caption == "4 objects · 3 man-made"
    assert f[-1].caption == "Man-made (man-made above P 0.35) → 3 objects"
    f, _ = build_story(chain, SelectionParams(tokens=("natural",)), 400)
    assert f[-1].caption == "Natural → 2 objects"


def test_markers_are_drawn_in_their_colours(tmp_path):
    from app.story import render

    frames, _ = build_story(_chain(tmp_path), SelectionParams(), size=400)
    pts, objs = frames[4].overlay, frames[5].overlay
    assert (pts[..., :3] == render.OCHRE).all(axis=-1).any()
    assert (pts[..., :3] == render.OUTLINE).all(axis=-1).any()
    assert (objs[..., :3] == render.MANMADE_RED).all(axis=-1).any()
    assert (objs[..., :3] == render.NATURAL_GREEN).all(axis=-1).any()


def test_missing_steps_are_skipped(tmp_path):
    chain = _chain(tmp_path, namer=False, seg=False)
    chain.band_filter = None
    frames, _ = build_story(chain, SelectionParams(), 400)
    assert [f.caption.split()[0] for f in frames] == ["PRISMA", "Indradhanu", "Top"]
    (chain.prep.dir / "anomaly_mask.tif").unlink()
    (chain.prep.dir / "composite_score.tif").unlink()
    assert len(build_story(chain, SelectionParams(), 400)[0]) == 1
    (chain.match.dir / "rgb.png").unlink()
    with pytest.raises(StoryError):
        build_story(chain, SelectionParams(), 400)


def test_whole_story_renders(tmp_path):
    frames, title = build_story(_chain(tmp_path), SelectionParams(), size=400)
    data = render_gif(frames, title=title, size=400)
    im = Image.open(io.BytesIO(data))
    assert im.n_frames == 7 + 6 * 3 and im.info["loop"] == 0


def test_endpoint_checks_are_422():
    pytest.importorskip("fastapi")                                  # api image, not every venv
    from fastapi import HTTPException

    from allotrope.api.action_story import check_selection_possible, check_story_action

    with pytest.raises(HTTPException) as exc:
        check_story_action("anomaly_detection_prep")
    assert exc.value.status_code == 422
    check_story_action("spectral_library_match")
    with pytest.raises(HTTPException) as exc:
        check_selection_possible(namer_present=False, explicit_selection=True)
    assert exc.value.status_code == 422 and "spectral-namer" in exc.value.detail
    check_selection_possible(namer_present=False, explicit_selection=False)  # frames stop at the threshold
    check_selection_possible(namer_present=True, explicit_selection=True)


def test_unknown_class_is_a_selection_error():
    from app.spectral_match.class_selection import SelectionError, parse_classes

    with pytest.raises(SelectionError):
        parse_classes("oil_fuel,granite")


def test_a_capped_rgb_keeps_the_scene_grid(tmp_path):
    """rgb.png is capped at 4,096 px upstream; captions and markers stay on the scene grid."""
    from app.story import render

    chain = _chain(tmp_path)
    Image.fromarray(np.full((H // 2, W // 2, 3), 120, np.uint8)).save(chain.match.dir / "rgb.png")
    chain.band_filter.summary.update(height=H, width=W)
    frames, _ = build_story(chain, SelectionParams(tokens=("natural",)), size=400)
    assert frames[0].caption.endswith("80×60 px · 234 bands")
    green = (frames[-1].overlay[..., :3] == render.NATURAL_GREEN).all(axis=-1)
    rows, cols = np.nonzero(green)
    scale = 300 / H                                              # the object at (50, 70) -> display (250, 350)
    assert abs(rows.max() - 50 * scale) < 12 and abs(cols.max() - 70 * scale) < 12
