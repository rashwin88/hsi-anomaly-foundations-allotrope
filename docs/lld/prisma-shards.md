# PRISMA native-band shards

**Purpose** - training shards from the 223 PRISMA L2D scenes in the inventory, at **native bands**
(all 234 that PRISMA flags valid), in the same sample layout as the EnMAP shards, so the fine-tuning
of the SpectralEarth-pretrained Indradhanu can read EnMAP and PRISMA with one trainer. Nothing is
resampled or filtered at shard time.

## Interfaces

```python
# app/utils/prisma/
read_scene(path) -> PrismaScene      # scene.py: .pixels int16 (234,H,W), .wavelengths_nm, .lat/.lon (H,W), .facts, .centre
tiles(scene) -> (key, sample)...     # tiles.py: 128 px tiles, >= 10% valid; key pr_<acquisition start>_r<row>c<col>
sites(centres, site_km=15)           # split.py: scenes grouped by place (single-link on centre distance)
split_scenes(centres, 0.10, "pr-v1") # split.py: scene -> train | test | excluded (benchmark sites)
```

```bash
python scripts/shard_prisma.py plan     s3://allotrope-raw-inventory-india/prisma/L2D/ <work>
python scripts/shard_prisma.py tiles    s3://allotrope-raw-inventory-india/prisma/L2D/ <work> --part K/N
python scripts/shard_prisma.py split    - <work>
python scripts/shard_prisma.py shards   <work> s3://allotrope-training-data-prod-india/patches/prisma/native-v1/ --part K/N
python scripts/shard_prisma.py finalize <work> s3://allotrope-training-data-prod-india/patches/prisma/native-v1/
```

Full runs go through `/opt/se-shard/prisma_all.sh` on the prod box (logs and `DONE` in
`/var/log/prisma-shard/`), in the `se-recompress:1` container with `/opt/se-shard` as the code.

## Data flow

```
inventory scene dataset (files.jsonl.gz: sha256 per file; no footprint)
  plan      list scene datasets                                                 -> <work>/plan.json
  tiles     .he5 (verified, ~2 GB) -> read_scene -> 128 px tiles -> <work>/tiles/all/<scene>.tar
            + .json (keys, scene centre, stats by wavelength); the .he5 is deleted
  split     centres -> sites -> train / test / excluded; tile files moved to <work>/tiles/<side>/
  shards    tiles of each side, shuffled (seed 0) -> 150-tile shards              (scripts/shard_enmap.py)
  finalize  manifest.json (scenes, split record, shard sha256s), stats.json keyed by wavelength
```

Per sample: `pixels.npy` int16 (234, 128, 128) (x 0.0001 = reflectance, -32768 = nodata),
`wavelengths.npy` float32 (234,) in file order, `meta.json` (scene id, date, sun zenith/azimuth, EPSG,
L2 scale factors, tile row/col/origin, valid fraction, tile-centre lat/lon). **No label keys.**

Full run, 2026-10-15 (`/opt/se-shard/prisma_all.sh`, ~2 h on the prod box): 223 scenes; split 202 train /
21 test / 0 excluded (no inventory scene within 30 km of a benchmark site). Train 99 shards, 14,802 tiles,
105.8 GiB; test 11 shards, 1,570 tiles, 11.2 GiB; stats over the 234 wavelengths. Shard sha256s in
`native-v1/manifest.json`.

## Invariants

1. **The product's reading, minus its pipeline.** DN -> reflectance by
   `DN / 65535 x (L2Scale<Det>Max - L2Scale<Det>Min) + L2Scale<Det>Min`; nodata where DN = 0 or the
   detector's `PIXEL_L2_ERR_MATRIX` != 0; bands with `List_Cw_<Det>_Flags` = 0 dropped - the rules
   `PrismaDatasetBuilder` applies before its band filter. Checked on set1: max |difference| from the
   builder's native read at 860 nm is 0.00005 reflectance (int16 rounding).
2. **One band list for every PRISMA tile.** 234 centres, identical to 0.000 nm across 8 inventory
   scenes (2023-2026) and the four 2020-2024 benchmark scenes, so PRISMA batches need no alignment.
   File order is kept (VNIR from ~980 nm down, then SWIR; 920-980 nm overlap); wavelengths travel with
   every sample.
3. **No split leakage, and the benchmark stays unseen.** Scenes within 15 km of each other form a site
   and land on one side. Any site within 30 km of a benchmark scene (`BENCH_SITES`: set1, set2,
   set3+set4) is `excluded` from both train and test, though the four scenes themselves are not in the
   inventory.
4. **Every input byte is checked** against the inventory's sha256 before it is read.

## Failure modes

| What | Seen as |
|---|---|
| a file's bytes differ from the inventory | `ValueError` in `tiles`; retried, then the part fails |
| a cube's band count differs from its wavelength list | `ValueError` naming the file |
| `split` run before every scene is tiled | `SystemExit` naming the missing scenes; nothing moved |
| a part killed mid-scene | no sidecar, so the scene is redone |

## Decisions

- **Read the HE5 directly, not through `PrismaDatasetBuilder`:** the builder's band pipeline excludes,
  interpolates and resamples; shards must keep the sensor's bands untouched.
- **No segmentation labels.** PRISMA L2D has no cloud or water mask and the inventory holds no L1
  (whose `Cloud_Mask` would need its own geometry). The cloud/water head learns from EnMAP's labels;
  PRISMA batches contribute reconstruction only. A spectral pseudo-label (e.g. NDWI water, 98%
  precision against EnMAP's labels) can be computed from the pixels at load time if wanted.
- **Split after tiling:** the inventory has no PRISMA footprints, so a scene's place is known only once
  it is read; `split` moves the finished tile files rather than planning ahead.
- **Shared code with the EnMAP sharder:** `shards`, `finalize`, `fetch_scene` and `list_scenes` come from
  `scripts/shard_enmap.py` (generalised by sensor and manifest wording, 2026-10-15).
- **Permissions:** the prod box role `allotrope-prod-ec2` may list, read and write `patches/prisma/*`
  (statements `TrainingShardsList` / `TrainingShardsWrite`, added 2026-10-15). The GPU training role
  does not read `patches/enmap/*` or `patches/prisma/*` yet - fine-tuning will need that.
