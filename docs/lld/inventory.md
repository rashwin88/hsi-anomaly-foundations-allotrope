# Raw-dataset inventory

**Files:** `backend/allotrope_inventory/` (`s3.py`, `api/uploads.py`, `api/datasets.py`,
`api/tags.py`, `cli/`), `backend/alembic_inventory/`, `backend/Dockerfile.inventory`,
`scripts/inventory/create_bucket.sh`
**Tests:** none — see `09-known-issues.md`.

## Purpose

Raw scene deliveries were scattered across the Beta bucket, Google Drive and local disks,
with no record of what exists, where it came from, or whether a copy is intact. This
service gives every delivery one home in `s3://allotrope-raw-inventory-india` and one
queryable row, and guarantees the bytes in the bucket are the bytes the client hashed.

## Interfaces

HTTP, `X-Inventory-Key` on every route except `/healthz*`. Full table in
[`06-backend.md`](../06-backend.md#the-raw-dataset-inventory--a-third-separate-service).
The ones whose behaviour is not obvious:

```
POST /datasets/{id}/uploads   {files: [{relative_path, size_bytes, sha256 (hex),
                                        part_sha256: [b64, …], source_uri?}]}
  → {part_size, files: [{relative_path, parts: [{part_number, url, headers}],
                         uploaded_parts: [n, …], duplicate_of: [ds_…]}]}
POST /datasets/{id}/finalize  → DatasetPublic | 409 {code: upload_incomplete, missing: {path: [n]}}
                                              | 409 {code: checksum_mismatch, paths: [...]}
```

Error `detail` is a snake_case string everywhere (`dataset_not_found`, `dataset_exists`,
`dataset_not_pending`, `file_exists`, `unknown_sensor`, `bad_bbox`, …) **except** the two
finalize conflicts above, which are objects because the client needs the data to retry.

## Data flow

```
client                         inventory api                    S3 / Supabase
------                         -------------                    -------------
hash file once:
  sha256(whole), sha256(each
  64 MiB part)  ─────────────▶ POST /uploads ─── create_multipart_upload(SHA256) ─▶ S3
                               row: dataset_files(part_sha256, upload_id)  ──────▶ DB
               ◀── one presigned PUT per part, each with the part's
                   x-amz-checksum-sha256 signed into the URL
PUT parts  ─────────────────────────────────────────────────────────────────────▶ S3
                                                    (S3 rejects a body ≠ its hash)
               ──────────────▶ POST /finalize ─ list_parts → complete → head_object ▶ S3
                               check size + composite checksum
                               write manifest.json ──────────────────────────────▶ S3
                               status=available, commit ─────────────────────────▶ DB
```

Bytes never pass through the api — it signs, the client moves the data.

## Invariants

**1. A stored object's bytes equal what the client hashed, part by part.** The part's
SHA-256 is a *signed header* of its presigned URL (`X-Amz-SignedHeaders=host;x-amz-checksum-sha256`).
Verified against real S3 on 2026-10-13: correct body + header → 200; tampered body →
**400** *"does not match the calculated checksum"*; header omitted → **403**
`SignatureDoesNotMatch`. A client cannot skip the check.

**2. The composite checksum S3 stores equals the one recomputed from `part_sha256`.**
For a multipart object S3 stores `b64(sha256(raw_part_1 ‖ … ‖ raw_part_N))-N`, not a
whole-file hash. `s3.complete_upload` recomputes it from the declared part hashes and
compares, along with `ContentLength`. On disagreement it **deletes that object version**
and reports `checksum_mismatch` — a bad file never sits in the bucket looking finished.

**3. `dataset_files.sha256` (whole-file) is client-declared, not server-verified.** S3 has
no whole-file SHA-256 for multipart objects (its full-object checksums cover only the CRC
family). What *is* guaranteed is invariant 1: every byte matches a hash the same client
computed in the same read as the whole-file digest (`cli/transfer.py:hash_file`). The
whole-file hash is what duplicate detection and downloads compare against. A buggy or
malicious client could declare a wrong one; an honest one cannot store wrong bytes.

**4. The bucket alone can rebuild the database.** `manifest.json` carries every column of
the dataset, its tags, files (with `part_sha256` and `s3_version_id`), sources and
`part_size`. It is written on finalise **and** on every tag change of an available
dataset, both *before* the DB commit — an S3 failure rolls the change back rather than
leaving the manifest stale. The bucket is versioned, so every prior manifest is kept.
(`rebuild-index` itself is not written yet.)

**5. One delivery, one row.** `(sensor_type, sensor_scene_id, processing_level)` is
unique — L1 and L2D of one capture are two datasets. The constraint, not a pre-check, is
what enforces it (`register_dataset` catches `IntegrityError` → 409), so two concurrent
registrations cannot both win.

**6. Uploads and finalise on one dataset are serialised** by `SELECT … FOR UPDATE` on the
dataset row (`_pending_or_409`). Finalise commits **per file** and re-takes the lock, so a
verified file stays verified even if a later one fails, and a retry only redoes the rest.

**7. Supabase's public Data API sees nothing.** Supabase exposes every `public` table to
the `anon` role through PostgREST, and the anon key is public by design. Migration
`20261013_02` enables RLS on all seven tables with no policies: `anon` and `authenticated`
get zero rows; the `postgres` role the api connects as owns the tables and bypasses RLS.
Verified: owner sees 5 sensors, `SET ROLE anon` sees 0. **Any new table needs the same
`ENABLE ROW LEVEL SECURITY`**, or it is world-readable.

## Numbers

| | |
|---|---|
| Part size | 64 MiB (`s3.PART_SIZE`; the CLI's must match — `ingest` checks the plan's `part_size`) |
| Largest file | 10,000 parts × 64 MiB = **625 GiB** (S3's part-count limit) |
| URL lifetime | 24 h; calling `/uploads` or `/download` again issues fresh ones |
| Files per `/uploads` call | ≤ 1,000 (the CLI batches) |
| Abandoned uploads | aborted by the bucket lifecycle after 7 days |
| Storage class | Intelligent-Tiering from day 0; objects < 128 KiB (manifests) stay Standard |

## Failure modes

| What happens | What the caller sees | Recovery |
|---|---|---|
| Upload interrupted | finalize → 409 `upload_incomplete` + missing parts | re-call `/uploads` with the same declaration: same upload id, URLs only for missing parts. `cli ingest` does this — re-run the command |
| Declaration changed for a file still uploading | — | old multipart upload aborted, new one started |
| Stored size/checksum ≠ declared | finalize → 409 `checksum_mismatch` | object version deleted, row removed; declare and upload again |
| S3 refuses the manifest at finalise | 5xx | dataset stays `pending`, files stay verified; retry finalize |
| Request fails after `start_upload`, before commit | 5xx | DB rolls back; the orphaned multipart upload is aborted by lifecycle in 7 days |
| Same bytes already in another available dataset | `duplicate_of` in the plan | informational only — providers ship identical aux files |

## Decisions

- **A separate service, not a router in the main api.** Two databases in one process
  tangles sessions and migrations, and the inventory must outlive any one deployment of the
  app stack. Cost: one more container.
- **Supabase as plain Postgres.** Its REST and auth layers are unused (and locked out,
  invariant 7) — they would let clients bypass validation. Swapping to Neon or RDS is an
  env-file change.
- **Tags in Postgres, not S3 object tags.** S3 caps objects at 10 tags and cannot query
  them.
- **Fixed part size, not client-chosen.** The server must know the boundaries to validate
  the part count against `size_bytes` before signing anything.
- **Sensors keyed by slug**, the same strings as `scenes.sensor_type`, so a future
  "import from inventory" in the main app needs no mapping table.
- **Footprint as PostGIS `GEOMETRY(4326)`**, any type — scene footprints are often
  multipolygons. Validated with shapely on the way in; self-intersecting rings are 422.
- **`q` escapes LIKE wildcards rather than stripping them.** The main app strips them, which
  breaks every product-id search (`09-known-issues.md` #8).
- **Rejected: MD5/ETag verification.** Multipart ETags are not content hashes, and MD5 is
  not something to build a provenance record on.
