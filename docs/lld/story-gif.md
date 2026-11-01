# The story GIF (`GET /actions/{id}/story.gif`)

**Purpose.** One looping GIF that shows how a scene was narrowed down to the selected anomalies, step
by step, for the 2026-11-04 committee demo and for slides. It plays anywhere (slides, chat, a
browser) and needs no viewer. It is drawn from a `spectral_library_match` action and the actions
upstream of it, using the files they already wrote. Nothing is recomputed except the class-driven
selection.

**Status.** Built 2026-11-01 on the local stack. Not committed, and not deployed to prod or the
air-gapped VM.

| What | Where |
|---|---|
| Renderer (frames -> GIF bytes; numpy + PIL) | `app/story/render.py` |
| The chain as data, readers (rgb, grid, masks, codes) | `app/story/chain.py` |
| Frames 1-7 from the chain's files | `app/story/frames.py` |
| Endpoint, lineage walk, cache | `backend/allotrope/api/action_story.py` (mounted in `main.py`) |
| Selection (shared with the class-driven export) | `app/spectral_match/class_selection.py` `parse_classes`, `select` |
| Frontend helper | `frontend/src/api/story.ts` `storyGifUrl`, `downloadStoryGif` |
| Tests | `tests/test_story/test_render.py`, `tests/test_story/test_frames.py` |

## Never in a submission pack

Frame 4 is a heatmap, and the submission rules disqualify heatmap outputs. So the GIF:

- is not written by any export builder (`app/spectral_match/export.py`, `app/anomaly_detection/export.py`);
- is cached **beside** the action's output directory, in `projects/<pid>/actions/<aid>/story/`, and
  not inside `output/`. The project export (`allotrope_worker/project_export.py`) zips every file under
  each action's `output/`, so a GIF cached there would end up in that zip. The action delete removes
  `actions/<aid>/` as a whole, so the cache is removed with it.

Do not "tidy" the cache into `output/`.

## Lineage

The lineage is followed through the `input_*_output_id` fields of each action's stored configuration.
Each field holds an `output_<uuid>`, which resolves to an `action_outputs` row and then to its action.
A link is accepted only if it belongs to the same project and its output directory exists:

```
spectral_library_match.input_anomaly_detection_output_id
  -> anomaly_detection_prep.input_anomaly_scoring_output_id
     -> anomaly_scoring.input_scene_segmentation_output_id     (optional)
     -> anomaly_scoring.input_band_filter_output_id            (else the segmentation's)
```

A link that does not resolve leaves that step out. The scene (sensor, name, acquisition date and raw
band count) comes from the project's bound scene.

## Frames

All frames share one map extent, scaled so that its longer side is `size` (default 960). Only
`rgb.png` is resized to that display grid.

Masks and namer centroids are on the **scene grid** (`chain.scene_grid`): the band filter summary's
`height` × `width`, or else the prep's `composite_shape`. `rgb.png` is capped at 4,096 px upstream:
on AVIRIS-NG it is 687 × 4096 for a 691 × 4118 scene. Placing markers by the RGB's size would put
them in the wrong place.

| # | from | picture | caption |
|---|---|---|---|
| 1 | scene row, `rgb.png` | the RGB | `PRISMA · 2025-02-14 · 1237×1202 px · 239 bands` |
| 2 | band filter summary | no-data washed to paper and hatched | `186 of 239 bands usable; no-data removed (30.3% of the frame)` |
| 3 | segmentation `class_codes.bin` (else `class_map.tif`, else the index method's `keep_mask.tif`) | the excluded classes greyed and hatched | `Cloud left out (4.9%)`; several: `Cloud (2.9%) and water (15.0%) left out`; 0 %: `none in this scene` |
| 4 | prep `composite_score.tif` (else the first model's `anomaly_score.tif`) | score p25..p99.5 on a muted paper -> straw -> rust -> oxblood ramp, over a dimmed, desaturated RGB | `Indradhanu V2 + MNF-RX score every pixel` |
| 5 | prep `anomaly_mask.tif`, config `committed_threshold_mode` / `committed_threshold` | committed pixels as ochre `#d4a017` points with a dark outline, at least 3 display px across | `Top 0.5% → 3,303 anomalous pixels` |
| 6 | `namer_objects.parquet`, re-decided at `manmade_p` | squares `#c0392b` for man-made objects, circles `#2e7d6b` for natural ones, at their centroids; side √n_px × scale, 6-14 px | `5,544 objects · 405 man-made` |
| 7 | the selection | the selected objects enlarged and ringed; all other objects faint, on a faded RGB | `Oil / fuel + plastic → 141 objects` |

The excluded classes in frame 3 are those of the scoring step (`anomaly_scoring.exclude_classes`, from
part A of the class-driven flow) when that field is set, and the segmentation's own `exclude_classes`
when it is not. Percentages come from the segmentation summary's `legend[].pct_of_valid`, or are
counted from the codes. The valid area in frames 2 onward comes from the segmentation's class map
(255 = outside valid), or else from the RGB's non-black pixels.

A step whose action or file is missing is skipped. Without namer results the story ends at frame 5.
The step counter ("3 / 7") counts the frames actually present.

## Selection (frame 7)

The query parameters mean the same as in the class-driven export (`docs/proposals/class-driven-flow.md`).
They are parsed and applied by `class_selection.parse_classes` and `select`, the same helper the export
and `selection_counts` use, so the GIF's last caption counts exactly the objects the pack would
contain:

- `classes`: a comma list of `natural`, `manmade` and the material keys (the namer's
  `MANMADE_CLASSES`). The default is `manmade`.
- `confident_only`: drops the tentative man-made objects.
- `manmade_p`: the cut on an object's mean P(man-made). It defaults to the action's own 0.5. A
  non-default cut re-decides every object, and frame 6's man-made count follows it.

Labels come from `namer_legend.json` (`oil / fuel`). A non-default cut and `confident_only` are named in
the caption.

## Endpoint

`GET /actions/{id}/story.gif?classes=&confident_only=&manmade_p=&size=` is a sync `def` with the
usual cookie auth (401 without it). Its heavy imports are inside the handler.

- 200 `image/gif`, `Content-Disposition: attachment; filename="allotrope_story_<scene8>_<action8>.gif"`.
- 422 when the action is not a `spectral_library_match`.
- 422 when `classes`, `confident_only` or `manmade_p` is given for an action without namer results
  (`namer_objects.parquet`). If none of them is given, such an action gets the story up to frame 5.
- 422 for an unknown class token, or when the chain has no `rgb.png` at all.
- 404 `output_not_ready` or `scene_not_bound`.
- `size` is 320-1600.

**Cache**: `actions/<aid>/story/story_<sha1[:16]>.gif`. The key is built from `STORY_VERSION`, the
selection, `size`, and the mtimes of the match's `namer_objects.parquet` / `summary.json` and the
prep's `anomaly_mask.tif` / `summary.json`. Bump `STORY_VERSION` whenever the frames or the renderer
change. The write is best effort: on a read-only volume the GIF is re-rendered on the next request.

## Renderer

- Tufte style:
  - paper `#fffff8`, ink `#111`;
  - a 15 pt title (`Allotrope · <scene name>`) and step counter in muted ink;
  - a 26 pt serif caption.
- Font:
  - ET Book if a copy is found (none ships today);
  - else STIX General from matplotlib's `mpl-data`, which is in the api image;
  - else DejaVu Serif;
  - else Pillow's built-in face.
- Canvas width is at least 560 px plus margins, so a tall flight line keeps a readable caption.
- Timing:
  - each step is held for 2.0 s, the last for 4.5 s;
  - 3 cross-fade frames of 80 ms go between steps;
  - the GIF loops forever (`loop=0`).
- Palette:
  - one global palette, built by median cut from a sample of all frames, with **no dithering**;
  - paper, ink, a 10-step ink -> paper ramp and the marker colours are reserved;
  - learned colours within 4 levels of a reserved one are dropped, so paper stays exactly `#fffff8`.
- **Size bound**: when the GIF is over 8 MB it is re-encoded with one fewer cross-fade frame, down
  to 0, and then with half the colours.

## Verified (2026-11-01, local stack, through nginx at :3010)

| action | GIF (default) | frames | bytes | first / cached | `classes=oil_fuel,plastic` |
|---|---|---|---|---|---|
| PRISMA 56b746f0 (1237 × 1202) | 1000 × 1042 | 7 keys + 2 fades each | 6,916,716 | 1.27 s / 0.27 s | 6,831,709; `Oil / fuel + plastic → 141 objects` (116 + 25, = summary `class_counts`) |
| Tanager 9824dc98 (895 × 1020) | 882 × 1069 | 7 + 2 each | 7,558,157 | 1.08 s / 0.28 s | 7,563,083 |
| AVIRIS-NG fae2bccf (691 × 4118) | 600 × 1104 | 7 + 3 each | 3,071,282 | 0.49 s / 0.11 s | 3,102,790 |
| Tanager pre-namer 2f025cc8 | 882 × 1069 | 4 (no segmentation upstream, no namer) | 5,441,290 | 0.63 s | 422 (no namer results) |

PRISMA and Tanager came in over 8 MB with 3 fades and were re-encoded with 2. The other checks:

- a prep action gives 422;
- `classes=granite` gives 422 and lists the allowed tokens;
- with no cookie the endpoint returns 401;
- the cache files appear under `actions/<aid>/story/`;
- `tests/test_story` passes 16 of 16 in the worker image. The 422 test needs fastapi, so it is
  skipped in the Windows venv.

## Gotchas

1. **Frame 5 can be a wall of ochre.** It shows the committed threshold as it is. The PRISMA prep
   above committed the top 8 % (118,950 px), and the frame looks like that.
2. **rgb.png is capped at 4,096 px.** Scale to the scene grid, never to the RGB's size.
3. **The index-method segmentation** has no class codes. Frame 3 then uses `keep_mask.tif` and names
   `classes_to_mask`. This path is covered by reading the code only: no index-method chain was on the
   stack.
