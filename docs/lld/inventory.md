# Raw-dataset inventory

**Files:** `backend/allotrope_inventory/` — `s3.py`, `api/{datasets,uploads,imports,bulk,tags,stats,sensors}.py`,
`cli/{__main__,transfer,s3import,unpack}.py`; `backend/alembic_inventory/`;
`backend/Dockerfile.inventory`; `scripts/inventory/create_bucket.sh`
**Tests:** none — see `09-known-issues.md`. Every path below was exercised against live S3
and Supabase on 2026-10-13/14, and the numbers quoted are from those runs.

## Purpose

Raw data was scattered across the Beta bucket, Google Drive and local disks, with no record
of what exists, where it came from, or whether a copy is intact. This service gives every
holding one home in `s3://allotrope-raw-inventory-india` and one queryable row, and
guarantees the bytes in the bucket are the bytes that were hashed at the source.

## The three kinds

| kind | One row is | Files | Example |
|---|---|---|---|
| `scene` | one acquisition, as delivered | a handful | a PRISMA `.he5`; an EnMAP scene folder |
| `archive` | a packaged delivery kept as shipped | a few large parts | SpectralEarth's 9 × 300 GiB `.tar.gz` |
| `collection` | many small items unpacked from an archive | up to ~10⁶ | 539,755 SpectralEarth patches |

The kind decides nothing in the code paths — it exists so a filter for scenes never returns
half a million patches. Status is `pending` → `available` / `failed`, and `retired` once a
dataset's payload has been deleted.

## Three ways bytes get in

| Path | For | Bytes travel | Verified by |
|---|---|---|---|
| **Upload** — `POST /uploads` | files on a laptop / Drive | client → S3 via presigned multipart PUTs | per-part SHA-256 signed into each URL; composite on finalise |
| **Import** — `POST /imports` | objects already in S3 | inside S3 (`UploadPartCopy`) | S3's SHA-256 of each copied part vs the client's hash of the source |
| **Bulk** — `POST /files/bulk` | collections (many small files) | client → S3 as single PUTs | S3's true whole-file SHA-256, checked by HEAD before the row exists |

All three end in the same `finalize`, and all three are resumable by re-sending the same
request.

```
upload:  hash file (whole + 64 MiB parts) ─▶ /uploads ─▶ PUT parts (header = part hash) ─▶ /finalize
import:  ranged GETs of the source, hash ─▶ /imports ─▶ UploadPartCopy ×N, compare ─▶ /finalize
bulk:    stream member, hash ─▶ PUT with ChecksumSHA256 ─▶ /files/bulk (HEAD each) ─▶ /finalize
```

## Invariants

**1. A stored object's bytes equal what was hashed, part by part.**
- *Upload:* the part's SHA-256 is a signed header of its presigned URL
  (`X-Amz-SignedHeaders=host;x-amz-checksum-sha256`). Verified 2026-10-13: correct body →
  200; tampered body → **400** *"does not match the calculated checksum"*; header omitted →
  **403** `SignatureDoesNotMatch`.
- *Import:* the multipart upload is opened with `ChecksumAlgorithm=SHA256`, so S3 computes
  each copied part's hash itself and returns it in `CopyPartResult`. Verified 2026-10-13:
  returned hashes equal ours, part for part. A mismatch aborts the upload (409).
- *Bulk:* a single PUT with `ChecksumSHA256` makes S3 store the **whole-file** SHA-256 —
  the strongest of the three. The endpoint HEADs every object before inserting its row.

**2. The composite S3 stores matches the declared parts.** For multipart objects S3 keeps
`b64(sha256(raw_part_1 ‖ … ‖ raw_part_N))-N`, not a whole-file hash; `complete_upload`
recomputes it from `part_sha256` and compares, with `ContentLength`. On disagreement the
object version is **deleted** and the file reported `checksum_mismatch`.

**3. The whole-file SHA-256 of a multipart file is client-declared.** S3 has none for
multipart objects (its full-object checksums cover only the CRC family). What is guaranteed
is invariant 1 — every byte matches a hash computed in the same read as the whole-file
digest. Bulk-registered files don't have this gap.

**4. The bucket alone can rebuild the database.** Each dataset's prefix holds
`manifest.json` (every dataset column, tags, counts, sources, `part_size`) and
`files.jsonl.gz` (one line per file: path, size, sha256, part hashes, version id). Both are
written on finalise and retire, before the DB commit, so an S3 failure rolls the change
back. Tag changes rewrite only `manifest.json` — the file list didn't change, and
rewriting 539,755 lines per tag would be waste. The bucket is versioned.
(`rebuild-index` itself is not written yet.)

**5. One holding, one row.** `(sensor_type, sensor_scene_id, processing_level)` is unique,
enforced by the constraint (`IntegrityError` → 409), so concurrent registrations can't both
win.

**6. Writes to one dataset serialise on its row lock** (`SELECT … FOR UPDATE`) — but the
lock is held only while declaring. `/imports` commits before its minutes-long copy and
`/files/bulk` runs its HEADs before taking the lock, so many files move at once.

**7. Supabase's public Data API sees nothing.** Supabase publishes `public` tables to the
`anon` role, whose key is public by design. Migration `20261013_02` enables RLS with no
policies on every table; the `postgres` owner the api connects as bypasses it. Verified:
owner sees 5 sensors, `SET ROLE anon` sees 0. **Any new table needs the same
`ENABLE ROW LEVEL SECURITY`.**

**8. Nothing is deleted unless verified and confirmed.** `retire` requires
`status=available` and `confirm` equal to the scene id, deletes every version and delete
marker under `payload/`, and keeps the row, file rows, both manifests and the audit trail.
The SpectralEarth archive was retired only after its collection passed: rows = S3 objects =
manifest lines (539,755), identical path sets, 300/300 random patches re-read and matched.

## Moving big data fast — what was measured

The SpectralEarth unpack is the reference workload: 2.62 TiB of gzip in, 3.25 TiB / 539,755
members out, on an m7i.4xlarge. Each fix was measured on the box before it was kept:

| Bottleneck | Symptom | Fix | Rate |
|---|---|---|---|
| one S3 GET stream per part | pigz at 80%, idle cores | 8 × 64 MiB ranged GETs in flight, fed in order | 94 → 113 MiB/s |
| zlib inflate | pigz pinned at 100% | ISA-L `igzip` (apt `isal`): 261 vs pigz 129, gzip 114 MiB/s | → 144 MiB/s |
| 64 KiB Linux pipes | neither side busy | `F_SETPIPE_SZ` to 1 MiB on both pipes | → **240–280 MiB/s** |

Final run: 3.9 h end to end. gzip inflate is single-core by nature, so ~260 MiB/s is the
ceiling for one `.tar.gz` stream on this hardware. Registration runs on its own thread and
uploads are capped at 256 in flight (~1.7 GB of patches), so neither stalls the stream.

`import-s3` hashing of the same 2.62 TiB ran at ~410 MB/s (9 objects in parallel, 6 ranged
GETs in flight each, CPU 77% idle); the server-side copy of 41,756 parts took ~1 h at 16
`UploadPartCopy` calls in flight per object, 3 objects at a time.

## Numbers

| | |
|---|---|
| Multipart part size | 64 MiB (`s3.PART_SIZE`; the CLI checks the plan's `part_size`) |
| Largest multipart file | 10,000 parts × 64 MiB = **625 GiB** |
| Largest bulk file | 5 GiB (one PUT) |
| Files per `/uploads` call, per `/files/bulk` call | ≤ 1,000; ≤ 5,000 (the unpacker sends 2,000) |
| URL lifetime | 24 h nominal; on EC2 bounded by the instance role's credentials (hours) |
| Abandoned uploads | aborted by the bucket lifecycle after 7 days |
| Storage class | Intelligent-Tiering from day 0; objects < 128 KiB (manifests) stay Standard |

## Failure modes

| What happens | Caller sees | Recovery |
|---|---|---|
| Upload interrupted | finalize → 409 `upload_incomplete` + missing parts | re-call `/uploads` with the same declaration; `ingest` does this on re-run |
| Declaration changed mid-upload | — | old multipart upload aborted, new one opened |
| Stored size/checksum ≠ declared | 409 `checksum_mismatch` | object version deleted, row removed; declare again |
| Import source ≠ declared size | 422 `source_size_mismatch` | nothing written |
| Provider `.md5` ≠ computed MD5 | `import-s3` refuses **before registering** | nothing written; investigate the source |
| Bulk row whose object isn't stored / differs | listed in `failed`, not inserted | re-upload and re-send; the unpacker stops on any |
| Unpack interrupted | — | re-run: the stream is re-read from the start (gzip can't seek), members already in the collection are skipped |
| S3 refuses a manifest on finalise | 5xx | dataset stays `pending`, files stay verified; retry |
| Request dies after `start_upload`, before commit | 5xx | lifecycle aborts the orphan in 7 days |

## Decisions

- **A separate service, not a router in the main api.** Two databases in one process tangle
  sessions and migrations, and the inventory must outlive any one deployment of the app.
- **Supabase as plain Postgres.** Its REST and auth layers are unused and locked out
  (invariant 7). Moving to Neon or RDS is an env change.
- **Patches are files of a collection, not datasets.** A dataset is a holding with its own
  prefix and manifests; 539,755 of those would swamp every list. `dataset_files` already had
  the right shape, so collections reuse it; `/files` pages it.
- **Retire, don't delete.** The provenance of what was derived from an archive outlives the
  archive's bytes.
- **Bulk registration writes directly to S3, then registers.** A presigned URL per patch
  would mean 539,755 extra round trips; the HEAD check keeps the api the authority anyway.
- **Server-side copy for S3 sources.** Reading once to hash is unavoidable; writing through
  the box isn't.
- **Fixed part size.** The server validates part count against size before signing.
- **Tags in Postgres, not S3 object tags** (S3 caps at 10 and can't query them).
- **`q` escapes LIKE wildcards rather than stripping them** — product ids are full of `_`
  (the main app strips them; `09-known-issues.md` #8).
- **Rejected: MD5/ETag verification.** Multipart ETags aren't content hashes. Provider MD5s
  are checked when supplied, as evidence about the source — not as our integrity record.
