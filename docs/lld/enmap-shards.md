# EnMAP native-band shards

**Purpose** - training shards from the 210 EnMAP L2A scenes in the inventory, at **native bands**
(all 224, each scene's own wavelengths) and carrying the provider's quality layers as labels, so
the reduced-attention model can be fine-tuned on EnMAP and a cloud/water segmentation head can be
trained from the same shards. Nothing is resampled, masked away or chosen at shard time; the
trainer decides at load time.

## Interfaces

```python
# app/utils/enmap/
EnmapScene(folder)                         # scene.py: .wavelengths_nm (224,), .band_index (202,), .facts
find_member(folder, stem) -> path          # case-insensitive (-METADATA.XML / -METADATA.xml)
select_bands(scene_wl, wanted_wl) -> idx   # by wavelength, 0.5 nm tolerance
read_scene(scene) -> SceneArrays           # tiles.py: whole cube once; QL_PIXELMASK -> nodata; 6 label layers
tiles(scene, arrays) -> (key, sample)...   # 128 px tiles, >= 10% valid; key en_<datatake>_<tile>_r<row>c<col>
split_datatakes(scenes, 0.10, "en-v1")     # split.py: test datatakes, stratified on cloud
# app/utils/spectralearth/sharding.py
write_samples((key, {name: array|dict|bytes}), fileobj) -> (keys, BandStats)
```

```bash
python scripts/shard_enmap.py plan     s3://allotrope-raw-inventory-india/enmap/L2A/ <work>
python scripts/shard_enmap.py tiles    s3://allotrope-raw-inventory-india/enmap/L2A/ <work> --part K/N
python scripts/shard_enmap.py shards   <work> s3://allotrope-training-data-prod-india/patches/enmap/native-v1/ --part K/N
python scripts/shard_enmap.py finalize <work> s3://allotrope-training-data-prod-india/patches/enmap/native-v1/
```

## Data flow

```
inventory scene dataset (files.jsonl.gz: sha256 per file)
  plan     METADATA.XML of every scene -> facts -> split by datatake        -> <work>/plan.json
  tiles    scene payload (verified) -> 224 x H x W int16, pixel mask applied
           -> 128 px tiles -> <work>/tiles/<split>/<scene>.tar + .json (keys, stats by wavelength)
  shards   all tiles of a split, shuffled (seed) -> 150-tile shards        -> <out>/<split>/shard-NNNNNN.tar + .json
  finalize manifest.json (scenes, test datatakes, shard sha256s), stats.json keyed by wavelength
```

Per sample: `pixels.npy` int16 (224, 128, 128) (x 0.0001 = reflectance, -32768 = nodata),
`wavelengths.npy` float32 (224,), `label_cloud|cirrus|haze|cloud_shadow|snow|classes.npy` uint8
(1, 128, 128) exactly as delivered (cirrus 0-3; classes 0 and 3 no-data, 1 land, 2 water),
`meta.json`: scene id, datatake, tile, date, processor, mean ground elevation, scene water vapour
and aerosol, sun elevation/azimuth, scene cover %, tile row/col, centre lon/lat, CRS, affine.

Plan of the full run (2026-10-14): 210 scenes, 66 datatakes; test 6 datatakes / 13 scenes, train
197 scenes; ~66-78 tiles per scene, ~0.5 GB of tiles per scene. Final counts and sha256s are in
`native-v1/manifest.json`.

## Invariants

1. **Bands are matched by wavelength, never by number.** 5 of the 210 scenes (early 2023, V010502)
   number their SWIR one band off: band 142 is 1530.29 nm there, 1518.87 nm elsewhere. Every scene
   contains all 202 SpectralEarth wavelengths exactly (0.0 nm), so `band_index` selects them for
   both maps; `wavelengths.npy` travels with every sample.
2. **Labels are never applied to the pixels.** Validity reflects nodata and the pixel mask only, so
   cloudy tiles are kept - they are the segmentation head's training data.
3. **A datatake lies wholly on one side of the split** (its tiles border each other), and both a
   cloudy and a clear stratum are represented in test.
4. **Every input byte is checked** against the inventory's sha256 before it is decoded.
5. **Stats are keyed by wavelength** (0.01 nm), so the 5 renumbered scenes add to the right bands;
   bands with no valid voxel (L2A blanks 1343-1390 nm) have no stats rather than a fake zero.

## Failure modes

| What | Seen as |
|---|---|
| a file's bytes differ from the inventory | `ValueError` in `tiles`; the scene is retried, then the part fails |
| cube band count differs from METADATA.XML | `ValueError` naming the scene |
| a SpectralEarth wavelength missing from a scene | `SceneError` from `band_index` (not raised by tiling itself) |
| a part killed mid-scene | no sidecar, so the scene is redone; a partial tile tar is overwritten |
| finalize finds shard keys that differ from the plan | `SystemExit`; nothing is written |

## Decisions

- **Native bands over SpectralEarth's 202** (decided 2026-10-14): the wavelength stems read any
  list; ~11% more bytes; the fixed-band stems and SpectralEarth-mixed batches select 202 at load
  time via `band_list_path`.
- **Whole-scene reads, then shuffle from local tile tars**: the rasters are 512 x 512-block COGs, so
  tile-by-tile reads in shuffled order would decode each block up to 16 times.
- **Label keys and values identical to the `enmap_seg` lane**, so one reader serves both; pixels here
  stay int16 (exact) where that lane used float16.
- **Runs on the prod box** (inventory -> EC2 -> training bucket in one region, no egress); the box
  role may write `patches/enmap/*`.
- **Rejected:** splitting by scene (neighbouring tiles of one datatake would straddle the split);
  applying quality masks at shard time (consumes the labels, and the 0.5 validity filter would drop
  every mostly-cloudy tile - the trap documented in `segmentation-sharding.md`).
