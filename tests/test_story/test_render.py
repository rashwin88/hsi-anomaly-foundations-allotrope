"""The story GIF renderer (app/story/render.py) on synthetic frames: frame count, size, loop, palette, bytes."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.story import render
from app.story.render import Frame, fit_shape, render_gif


def _frames(n=4, h=300, w=500):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        img = (rng.random((h, w, 3)) * 60 + 40 * i).astype(np.uint8)
        ov = np.zeros((h, w, 4), np.uint8)
        ov[50:60, 50:60] = (*render.OCHRE, 255)
        out.append(Frame(img, f"step {i + 1} caption", overlay=ov if i % 2 else None))
    return out


def _decode(data):
    im = Image.open(io.BytesIO(data))
    frames, durations = 0, []
    try:
        while True:
            frames += 1
            durations.append(im.info.get("duration"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return Image.open(io.BytesIO(data)), frames, durations


def test_frame_count_loop_and_holds():
    data = render_gif(_frames(4), title="Allotrope · test", size=400, fades=3)
    im, n, durations = _decode(data)
    assert data[:6] == b"GIF89a"
    assert n == 4 + 3 * 3                                   # keys + 3 blends between each pair
    assert im.info.get("loop") == 0                         # forever
    holds = [d for d in durations if d >= 1000]
    assert len(holds) == 4 and durations[-1] == int(render.LAST_HOLD_S * 1000) > holds[0]
    assert all(d == int(render.FADE_FRAME_S * 1000) for d in durations if d < 1000)


def test_size_and_layout():
    data = render_gif(_frames(2, h=300, w=600), title="t", size=700)
    im, _, _ = _decode(data)
    mh, mw = fit_shape(300, 600, 700)
    assert (mh, mw) == (350, 700)
    assert im.size[0] == mw + 2 * render.MARGIN and im.size[1] > mh
    # a tall, narrow scene keeps a readable caption width
    tall = render_gif(_frames(2, h=800, w=100), title="t", size=480)
    assert Image.open(io.BytesIO(tall)).size[0] == render.MIN_TEXT_WIDTH + 2 * render.MARGIN


def test_palette_reserves_brand_colours():
    data = render_gif(_frames(2), title="t", size=300)
    im, _, _ = _decode(data)
    pal = im.getpalette()
    colours = {tuple(pal[i:i + 3]) for i in range(0, len(pal), 3)}
    for c in (render.PAPER, render.INK, render.OCHRE, render.MANMADE_RED, render.NATURAL_GREEN):
        assert c in colours
    assert len(pal) // 3 <= 256
    # the paper margin is exactly the paper colour
    first = im.convert("RGB")
    assert first.getpixel((2, 2)) == render.PAPER


def test_file_size_budget_is_enforced():
    big = render_gif(_frames(5, 600, 600), title="t", size=600, fades=3)
    small = render_gif(_frames(5, 600, 600), title="t", size=600, fades=3, max_bytes=len(big) // 2)
    assert len(small) < len(big)
    _, n, _ = _decode(small)
    assert n < 5 + 4 * 3                                    # fades were dropped to fit


def test_960_story_stays_under_8_mb():
    rng = np.random.default_rng(1)
    base = (rng.random((1200, 1200, 3)) * 255).astype(np.uint8)  # worst case: noise compresses badly
    frames = [Frame(render.blend(base, render.PAPER, 0.1 * i), f"step {i}") for i in range(7)]
    data = render_gif(frames, title="t")
    assert len(data) <= render.DEFAULT_MAX_BYTES


def test_mismatched_extent_is_refused():
    f = _frames(2)
    f[1] = Frame(np.zeros((300, 300, 3), np.uint8), "square")
    with pytest.raises(ValueError):
        render_gif(f, title="t")
    with pytest.raises(ValueError):
        render_gif([], title="t")


def test_a_serif_font_is_found():
    font = render.load_font(20)
    assert hasattr(font, "getbbox")
