# SpectralEarth data layer

**Files:** `app/constants/spectralearth_bands.json`, `scripts/build_spectralearth_band_table.py`,
`app/models/dataset/spectralearth.py`, `app/utils/spectralearth/{catalog,source,screens,torch_dataset}.py`,
`app/utils/data_transformations/common_grid_pipeline.py`, `scripts/spectralearth_sanity.py`
**Tests:** `tests/test_models/test_spectralearth.py`, `tests/test_utils/test_spectralearth/`
(25 offline; one `network_access` test against the real inventory)

## Purpose

Make DLR's 539,755 SpectralEarth patches usable for pretraining the next Indradhanu: load
any patch with the spectral meaning its file doesn't carry, split it reproducibly, screen
it, and hand it to torch either on its own 202 EnMAP bands or on the 165-band common grid
— without deciding anything about the model.

## Interfaces

```python
load_band_table() -> SpectralEarthBands            # 202: wavelengths_nm, fwhm_nm, families, enmap_bands
SpectralEarthPatch.from_geotiff_bytes(data, path)  # .raw int16 (202,128,128), .key "se_<site>_<patch>"
  .reflectance() -> float32 (nodata -> 0.0)   .validity() -> int8   .to_vendable(config, keep)
SpectralEarthCatalog.from_files_manifest(uri)      # .split(f, seed) .by_site() .sample(n, seed) .filter(fn)
SpectralEarthSource(cache_dir, s3_client, workers) # .load(entry) .load_many(entries)
ScreenChain([NodataScreen(), ...]).apply(patch)    # -> ScreenReport(keep (H,W), rejected_by, usable())
SpectralEarthTorchDataset(catalog, source, mode="native"|"common_grid", screens, band_filter_config)
to_common_grid(cube, validity, wavelengths, families, fwhm, config, pixel_reject) -> VendableHyperspectralDataset
shard_plan(entries, seed, per_shard=150) -> list[list[CatalogEntry]]
write_shard(patches, fileobj) -> (keys, BandStats)  # BandStats: .add(raw, scale, nodata) .merge() .mean_std()
```

```bash
python scripts/shard_spectralearth.py <collection s3://...> <out s3://... | dir> [--part K/N] [--limit N]
python scripts/shard_spectralearth.py <collection> <out> --finalize    # manifest.json + stats.json
```

## Data flow

```
inventory: <collection>/files.jsonl.gz ──▶ SpectralEarthCatalog (539,755 entries, one GET)
           <collection>/payload/<path>  ──▶ SpectralEarthSource (GET or cache, sha256 check)
                                              └▶ SpectralEarthPatch (+ band table)
                                                   └▶ ScreenChain → keep mask
                                                        ├▶ native: reflectance, mask, (202,128,128)
                                                        └▶ common_grid: to_common_grid → (165,128,128)
```

Nothing is written anywhere except the optional cache directory.

## Invariants

**1. A patch's bands mean what the table says, and the table is derived, not typed.**
The 202 planes are EnMAP L2A bands 1–224 minus 127–141 and 161–167 (1-based) — the drop
list published with HySpecNet and stated for SpectralEarth. The paper and DLR's page say
only "water-absorption bands removed"; the indices came from the related literature and
were then checked three ways (2026-10-14):
- the authors' own loader uses RGB indices 43/28/10, which this table puts at 641/556/468 nm;
- EnMAP band centres were identical (0.0 nm spread) across six scenes' `METADATA.XML`;
- the median spectrum of 128,000 real pixels shows the vegetation red edge between 670 and
  740 nm, where it must be.

The build script refuses anything but 224 bands in and 202 out. **Never hard-code a
SpectralEarth wavelength**; read the table.

**2. Family comes from the detector, never from wavelength.** EnMAP's VNIR and SWIR
detectors overlap (VNIR runs to 993 nm, SWIR starts at 902), so band order is not
wavelength order. The table takes family from `METADATA.XML`'s channel lists, as the EnMAP
builder does; anything interpolating along wavelength must sort first.

**3. Every byte used was verified against the inventory.** The source checks each patch's
SHA-256 against the catalogue entry before decoding or caching, whether it came from S3
or from the cache; a corrupt cache file is discarded and re-fetched.

**4. The split is by site and stable.** A site's side is `sha256("<seed>:<site>") < f`.
Every date of a site lands on one side; the same seed gives the same split on any machine;
adding or removing patches never moves a site. `se-v1` at 5%: 512,849 / 26,906 patches.

**5. The common-grid path is the EnMAP path.** `to_common_grid` composes the EnMAP
builder's own components in its order, with the same `BandFilterConfig`. The screen mask
enters where the builder applies provider quality masks. The EnMAP builder itself was left
untouched — it is the production onboarding path — so the two can drift only if someone
changes one and not the other; this note is the reminder.

**6. Dataset samples are model-agnostic.** `pixels` float32 (C,128,128), `validity` uint8
(1,128,128), `keep_fraction`, `key`. Wavelengths live on the dataset, out of collation.
In native mode a pixel needs its complete spectrum; in common-grid mode small gaps are
interpolated (the 40% rule decides).

**7. Worker-safe.** The source rebuilds its S3 client per process and drops it when
pickled, so forked and spawned DataLoader workers each get their own.

## Shards

`app/utils/spectralearth/sharding.py` + `scripts/shard_spectralearth.py` pack the collection
into WebDataset tars for the trainer's `hot_storage` path.

- **Where:** `s3://allotrope-training-data-prod-india/patches/spectralearth/se-v1/`
  `{train,test}/shard-NNNNNN.tar`, each with a `.json` sidecar (keys, sha256, bytes, per-band
  count / sum / sum of squares), plus `manifest.json` and `stats.json` at the top.
- **Sample:** `<key>.pixels.npy` raw int16 `(202, 128, 128)` and `<key>.wavelengths.npy`
  float32 `(202,)` in band order; the WebDataset `__key__` is `se_<site>_<patch>`. No
  validity cube - the reader derives it from `-32768`.
- **Plan:** each split is shuffled with `--seed` (0) and cut into 150-patch shards (~0.99 GB),
  so shard n is fixed by the seed. A shard counts as done when its sidecar exists - written
  only after its tar - so reruns resume, and `--part K/N` spreads shards over processes.
- **`stats.json`:** per-band mean and population std of valid reflectance over the train
  split (float64 sums, merged from the sidecars) - the wavelength-normalisation curve of
  `docs/proposals/spectral-attention-bottleneck.md`, decision 5.
- **Writes with stdlib `tarfile`**, members at mtime 0: byte-identical tars for the same
  patches, and the sharding host needs no `webdataset`.
- **Not in the inventory.** The inventory registers only objects in its own bucket; shards are
  derived training data, and `manifest.json` (collection, split, seed, every shard's sha256)
  is their record.
- **Permissions:** the prod box role `allotrope-prod-ec2` may list, read and write
  `patches/spectralearth/*` and `patches/enmap/*` in that bucket (statements
  `TrainingShardsList`/`Write`); the GPU training role `allotrope-gpu-train` reads
  `patches/spectralearth/*` and writes `experiments/indradhanu-se-pretrain/*`.

## Compressed shards (`se-v1z`)

For shipping off AWS (S3 egress $0.109/GB) and for disk-bound readers, a copy of the first 1,000
train and 20 test shards is re-encoded with `app/utils/shard_codec.py`: each pixel stored as its
difference from the previous band (int16, wrapping), then zstd-3, as member `pixels.i16dz`.
Exact (round trip and training loss identical to raw), 1.89-1.91x smaller (1,000 shards: 486.6 GiB
vs ~925 GiB), decode ~300 MiB/s per core. `scripts/recompress_shards.py` runs on the prod box (S3 ->
EC2 -> S3 in-region), checks each source shard's sha256 against the source manifest first, and
writes `se-v1z/manifest.json` with both sha256s. The spectral-stem trainer decodes `.i16dz` in the
DataLoader workers (`_sample_decoders`).

## Numbers (measured 2026-10-14)

| | |
|---|---|
| Catalogue load | 4.3 s from a laptop (one ~tens-of-MB GET) |
| Fetch + verify + decode | 37 patches/s with 32 threads (EC2, same region) |
| DataLoader, native, 8 workers | 18 samples/s from S3 — far below what a GPU needs |
| `to_common_grid` | median 59 ms per patch per core (max 433 ms when gaps need filling) |
| Native batch of 32 | ~424 MB (202 × 128 × 128 × 4 bytes each) — loader containers need `--shm-size` ≥ 16 GB at 8 workers |
| Nodata | 0 of 500 sampled patches have any |
| Reflectance | 0.1st pct −0.032 · median 0.19 · 99th 0.71 · 99.9th 1.11 |
| Sharding, prod box (m7i.4xlarge) | ~115 MiB/s per process to local disk; ~64 MiB/s per process incl. S3 upload, ~510 MiB/s with 8 processes |

## Failure modes

| What | Seen as |
|---|---|
| Patch bytes differ from the inventory | `PatchIntegrityError` (S3) or silent re-fetch (cache) |
| A file that isn't 202 × 128 × 128 int16 | `ValueError` from `from_geotiff_bytes` |
| Two screens with the same name | counts add up (an earlier version overwrote — caught by a test) |
| Too little shared memory in a loader container | `unable to allocate shared memory(shm)` from collate |
| `split()` with a fraction outside (0, 1) | `ValueError` |
| Sharder process dies mid-shard | no sidecar, so the shard is redone on the next run; a stale `.tar` is overwritten |
| Finalize finds a missing shard or keys that differ from the plan | stops (`SystemExit`); nothing is written |

## Decisions

- **Shards for training, random access for everything else.** S3 alone gives 18 samples/s,
  so training reads WebDataset shards (below); the torch dataset and the cache remain for
  sampling, sanity checks and anything that needs one patch by name.
- **Native 202 bands as a first-class mode.** The next Indradhanu may take SpectralEarth's
  bands directly rather than resampling to 165; the layer doesn't pre-empt that.
- **Read the inventory's bucket layout directly** (`files.jsonl.gz`, `payload/`), not its
  API — `app/` must not import the backend. The cost: `app/` knows that layout.
- **Screens as a chain with reasons**, because the real screening problem is cloud (no
  provider masks, DLR filtered only per scene at < ~10% by eye) and a run should be able
  to say what it discarded and why.
- **Compose, don't refactor, the EnMAP pipeline** — see invariant 5.
- **Rejected:** typing the wavelength list by hand; splitting by patch (leaks a site's
  other dates into test); a separate members table for patches (the inventory's
  `dataset_files` already holds them).
