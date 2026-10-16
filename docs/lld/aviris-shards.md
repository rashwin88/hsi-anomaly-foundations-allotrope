# AVIRIS-NG native-band shards

**Purpose** - training shards from the 79 AVIRIS-NG L2 flight lines of the ISRO-NASA India campaign
(Dec 2015 - Mar 2016, Feb - May 2018) at **native bands and native pixel size**, in the EnMAP / PRISMA
sample layout, so fine-tuning sees airborne data (3-8 m pixels, 5 nm bands) alongside the 30 m
satellites. Nothing is resampled at shard time.

## How the data got here

Google Drive (`My Drive/PS_11/aviris_datasets`, a shared folder - addressed by folder ID because
`rclone` cannot follow the shortcut by path) -> a one-off c7i.xlarge running `rclone` with a
read-only Drive token (`research/experiments/aviris-ng-ingest/copy_userdata.sh`; 770.9 GiB, 638
files, ~2.5 h, every file MD5-checked against Drive: 0 differences) -> staging prefix in the inventory
bucket -> `import-s3` per flight line (`aviris_import.sh`) as `aviris_ng` / `L2` scene datasets, every
byte re-hashed. 2026-10-15.

## Interfaces

```python
# app/utils/aviris/
open_scene(folder) -> AvirisScene   # scene.py: the _corr_*_img pair; .strip(y0, h) int16 (372,h,W); .lonlat(); .centre
tiles(scene, stats) -> (key, sample)...  # tiles.py: 128 px tiles strip by strip, >= 10% valid; key av_<yyyymmdd>t<hhmmss>_r<row>c<col>
```

```bash
python scripts/shard_aviris.py plan     s3://allotrope-raw-inventory-india/aviris_ng/L2/ <work>
python scripts/shard_aviris.py tiles    - <work> --part K/N
python scripts/shard_aviris.py split    - <work>
python scripts/shard_aviris.py shards   <work> s3://allotrope-training-data-prod-india/patches/aviris_ng/native-v1/ --part K/N
python scripts/shard_aviris.py finalize <work> s3://allotrope-training-data-prod-india/patches/aviris_ng/native-v1/
```

Full runs: `/opt/se-shard/aviris_all.sh` on the prod box (logs and `DONE` in `/var/log/aviris-shard/`).

## Data flow

```
inventory dataset (one per flight line)
  tiles   stream the _corr_*_img .bin/.hdr to disk, hashing (sha256 vs the inventory) - not the h2o
          layer or preview; memory-map the BIL cube; per 128-line strip: good bands -> int16 -> tiles
          -> <work>/tiles/all/<line>.tar (+ .json: keys, centre, stats by wavelength); delete the download
  split   scripts/shard_prisma.py's split with seed av-v1: 15 km sites, benchmark sites held out
  shards  / finalize   as scripts/shard_enmap.py
```

Per sample: `pixels.i16dz` (band-delta + zstd of int16 (372, 128, 128), x 0.0001 = reflectance,
-32768 = nodata; app/utils/shard_codec.py), `wavelengths.npy` float32 (372,), `meta.json` (line id,
date, product version, pixel size, tile row/col/origin, valid fraction, tile-centre lat/lon). No label keys.

Smoke run (2 lines, 2026-10-15): 215 and 221 tiles, ~1.1 GB each (~5.3 MB per compressed tile), 2-2.5 min
per line; a read-back tile: 372 bands 381-2500 nm, 91% valid, median 0.067 at 550 nm.

Full run, 2026-10-15 (`/opt/se-shard/aviris_all.sh`, ~3.5 h on the prod box): 79 flight lines; split 73
train / 6 test / 0 excluded. Train 150 shards, 22,377 tiles, 116.0 GiB; test 16 shards, 2,270 tiles,
11.2 GiB (compressed). `stats.json` lists 1,116 distinct wavelengths: the 372 bands appear at up to three
slightly different centres across product versions, all within 0.5 nm of each other (372 groups), so one
aligned band list serves every line. Shard sha256s in `native-v1/manifest.json`; local tiles deleted after
the shards were checked against it.

Provenance check, 2026-10-16: all 316 imported data files (79 lines x 4) match the provider's `.sha256`
sidecars from the Drive delivery (0 mismatches), so the inventory hashes the shards were verified against are
the provider's own bytes.

## Invariants

1. **The product's reading, version-agnostic.** The `_corr_` cube is surface reflectance (the header's
   "Measured Radiances" text is wrong for it - median 0.08 at 550 nm); nodata = `data ignore value`
   (-9999); bands with `bbl` = 0 dropped: **372 of 425 in every line**. `AvirisNgDatasetBuilder` looks
   only for `_corr_v2m2`; the campaign has v2m2, v2m3, v2s1 and v2s2, so files are found by pattern.
2. **One band list, nearly.** Centres differ by at most 0.42 nm between product versions - inside the
   0.5 nm tolerance of `app/utils/band_alignment.py`, so lines share batches once aligned onto one list.
3. **Never whole in memory.** Flight lines reach ~11 GB; the cube is memory-mapped and read 128 lines at
   a time; downloads stream to disk while hashing.
4. **Native pixel size.** 3.2-8.2 m depending on the line (in `meta.json`); not resampled to 30 m
   (decided 2026-10-15: fine-tuning should see airborne texture).

## Failure modes

| What | Seen as |
|---|---|
| a file's bytes differ from the inventory | `ValueError` in `tiles`; retried, then the part fails |
| not exactly one `_corr_*_img.hdr` in a line | `FileNotFoundError` |
| .bin size disagrees with its header | `ValueError` naming the file |
| a part killed mid-line | no sidecar, so the line is redone |

## Decisions

- **Compressed pixels in the tiles and shards** (`pixels.i16dz`, exact, ~1.9x): 372 bands x 128 x 128
  int16 is 12 MB raw; ~17,000 tiles would be ~200 GB raw against 257 GB free on the prod box. The trainer
  already decodes `.i16dz` in its loader workers; stats are counted from the int16 tile before encoding.
- **No labels:** AVIRIS-NG L2 has no cloud or water mask; the segmentation head learns from EnMAP.
- **Fine-tuning will need a per-epoch cap per sensor** so the airborne tiles, ~200 per line, do not
  swamp EnMAP and PRISMA - a trainer decision, not a shard one.
- **Permissions:** `allotrope-prod-ec2` may write `patches/aviris_ng/*` (added 2026-10-15).
