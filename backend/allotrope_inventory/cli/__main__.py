"""`python -m allotrope_inventory.cli <command>` — run from backend/.

Commands:
    sensors                      list sensors
    ls [filters]                 list datasets
    get DS_ID                    one dataset as JSON (files, sources, tags)
    tag DS_ID TAG...             add tags        untag DS_ID TAG...  remove tags
    ingest PATH --sensor --level --scene-id [...]
                                 register, hash, upload (resumable), finalise
    import-s3 s3://bucket/prefix/ --sensor --level --scene-id [...]
                                 hash in place, copy server-side, finalise
    download DS_ID DEST          fetch every file and verify its sha256
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import typer

from . import transfer
from .client import Client, InventoryError

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


@app.callback()
def _main() -> None:
    """Allotrope raw-dataset inventory CLI (INVENTORY_URL, INVENTORY_KEY)."""


@app.command("sensors")
def sensors() -> None:
    """List the sensors datasets can belong to."""
    for s in Client().get("/sensors"):
        typer.echo(f"{s['slug']:<12} {s['family']:<14} {s['name']}")


@app.command("ls")
def ls(
    sensor: str | None = typer.Option(None, help="sensor slug, e.g. prisma"),
    level: str | None = typer.Option(None, help="processing level, e.g. L2D"),
    status: str | None = typer.Option(None, help="pending | available | failed"),
    tag: list[str] = typer.Option([], help="repeatable; all must match"),
    q: str | None = typer.Option(None, help="substring of the scene id"),
    bbox: str | None = typer.Option(None, help="min_lon,min_lat,max_lon,max_lat"),
    since: str | None = typer.Option(None, help="acquired at or after (ISO date)"),
    until: str | None = typer.Option(None, help="acquired before (ISO date)"),
    limit: int = typer.Option(50, min=1, max=500),
    offset: int = typer.Option(0, min=0),
) -> None:
    """List datasets, newest-registered first."""
    page = Client().get(
        "/datasets", sensor_type=sensor, processing_level=level, status=status, tag=tag,
        q=q, bbox=bbox, acquired_from=since, acquired_to=until, limit=limit, offset=offset,
    )
    for d in page["items"]:
        when = (d["acquisition_at"] or "")[:10]
        typer.echo(
            f"{d['id']}  {d['status']:<9} {d['sensor_type']:<9} {d['processing_level']:<5} "
            f"{when:<10}  {d['sensor_scene_id']}  {','.join(d['tags'])}"
        )
    shown = len(page["items"])
    typer.secho(f"{offset + 1 if shown else 0}-{offset + shown} of {page['total']}", dim=True, err=True)


@app.command("get")
def get(dataset_id: str) -> None:
    """Print one dataset as JSON."""
    typer.echo(json.dumps(Client().get(f"/datasets/{dataset_id}"), indent=2))


@app.command("tag")
def tag(dataset_id: str, tags: list[str]) -> None:
    """Add one or more tags (`name` or `key:value`)."""
    c = Client()
    for t in tags:
        result = c.put(f"/datasets/{dataset_id}/tags/{t}")
    typer.echo(", ".join(result["tags"]))


@app.command("untag")
def untag(dataset_id: str, tags: list[str]) -> None:
    """Remove one or more tags."""
    c = Client()
    for t in tags:
        result = c.delete(f"/datasets/{dataset_id}/tags/{t}")
    typer.echo(", ".join(result["tags"]) or "(no tags)")


def _find_existing(c: Client, sensor: str, level: str, scene_id: str) -> dict:
    """The dataset a 409 dataset_exists refers to (exact scene id match)."""
    page = c.get("/datasets", sensor_type=sensor, processing_level=level, q=scene_id, limit=500)
    for d in page["items"]:
        if d["sensor_scene_id"] == scene_id:
            return d
    _fail(f"api says {sensor}/{level}/{scene_id} exists but it can't be found")


def register_or_resume(c: Client, body: dict) -> dict | None:
    """Register the dataset, or return the pending one from an earlier run.

    None when it is already available (nothing to do). Registration fields
    are fixed at first registration; a re-run passing different ones gets a
    warning rather than having them silently ignored.
    """
    try:
        d = c.post("/datasets", body)
        typer.echo(f"registered {d['id']}")
        return d
    except InventoryError as e:
        if e.code != "dataset_exists":
            raise
    d = _find_existing(c, body["sensor_type"], body["processing_level"], body["sensor_scene_id"])
    if d["status"] == "available":
        typer.echo(f"{d['id']} is already available; nothing to do")
        return None
    typer.echo(f"resuming {d['id']}")
    asked = {k: body.get(k) for k in ("metadata", "footprint", "acquisition_at")}
    differs = [k for k, v in asked.items() if v and v != d[k] and not (
        k == "acquisition_at" and d[k] and d[k][:19] == v[:19])]
    if differs:
        typer.secho(f"warning: kept the registered {', '.join(differs)}; "
                    "this run's values were not applied", fg=typer.colors.YELLOW)
    return d


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., exists=True, help="a delivery file, or a folder (uploaded recursively)"),
    sensor: str = typer.Option(..., help="sensor slug, e.g. prisma"),
    level: str = typer.Option(..., help="processing level, e.g. L2D"),
    scene_id: str = typer.Option(..., help="the provider's product id"),
    acquired_at: str | None = typer.Option(None, help="ISO timestamp, UTC"),
    footprint: Path | None = typer.Option(None, exists=True, help="GeoJSON geometry (or Feature) file"),
    meta: list[str] = typer.Option([], help="repeatable key=value; values parsed as JSON when they can be"),
    source: list[str] = typer.Option([], help="repeatable origin URI (gdrive://, s3://, https://)"),
    tag: list[str] = typer.Option([], help="repeatable"),
    workers: int = typer.Option(4, min=1, max=32, help="parallel part uploads"),
) -> None:
    """Register, hash, upload and finalise one delivery. Re-run to resume."""
    c = Client()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    rel = {f: (f.relative_to(path).as_posix() if path.is_dir() else f.name) for f in files}
    if not files:
        _fail(f"{path}: no files")

    geom = json.loads(footprint.read_text()) if footprint else None
    if geom and geom.get("type") == "Feature":
        geom = geom["geometry"]
    metadata = {}
    for kv in meta:
        k, _, v = kv.partition("=")
        try:
            metadata[k] = json.loads(v)
        except ValueError:
            metadata[k] = v

    # 1. Register, or pick up the pending dataset from an earlier run.
    d = register_or_resume(c, {
        "sensor_type": sensor, "sensor_scene_id": scene_id, "processing_level": level,
        "acquisition_at": acquired_at, "footprint": geom, "metadata": metadata,
        "sources": source,
    })
    if d is None:
        return
    ds = d["id"]

    # 2. Hash everything not already verified in an earlier run.
    done = {f["relative_path"]: f["sha256"] for f in c.get(f"/datasets/{ds}")["files"] if f["uploaded"]}
    decls, total = [], 0
    for f in files:
        size, sha, parts = transfer.hash_file(f)
        if rel[f] in done:
            if done[rel[f]] != sha:
                _fail(f"{rel[f]}: already uploaded with different contents")
            continue
        decls.append({"relative_path": rel[f], "size_bytes": size, "sha256": sha, "part_sha256": parts})
        total += size
    typer.echo(f"{len(files)} files, {len(decls)} not yet verified ({_human(total)})")

    # 3. Upload, finalise; one retry round covers parts lost to a flaky link.
    by_rel = {rel[f]: f for f in files}
    for _ in range(2):
        jobs = []
        for i in range(0, len(decls), 1000):
            plan = c.post(f"/datasets/{ds}/uploads", {"files": decls[i:i + 1000]})
            if plan["part_size"] != transfer.PART_SIZE:
                _fail(f"api part_size {plan['part_size']} != CLI {transfer.PART_SIZE}; update the CLI")
            for fu in plan["files"]:
                if fu["duplicate_of"]:
                    typer.secho(f"note: {fu['relative_path']} is identical to a file in {', '.join(fu['duplicate_of'])}", dim=True)
                jobs += [(by_rel[fu["relative_path"]], part) for part in fu["parts"]]
        sent = 0
        def progress(n: int) -> None:
            nonlocal sent
            sent += n
            typer.echo(f"\r  uploaded {_human(sent)}", nl=False)
        transfer.upload_parts(jobs, workers, progress)
        if jobs:
            typer.echo()
        try:
            d = c.post(f"/datasets/{ds}/finalize")
            break
        except InventoryError as e:
            if e.code != "upload_incomplete":
                raise
            typer.secho(f"incomplete, retrying: {e.detail['missing']}", fg=typer.colors.YELLOW)
    else:
        _fail(f"{ds}: still incomplete; re-run the same command to resume")

    for t in tag:
        c.put(f"/datasets/{ds}/tags/{t}")
    typer.secho(f"{ds} available  {d['s3_uri']}", fg=typer.colors.GREEN)


@app.command("import-s3")
def import_s3(
    uri: str = typer.Argument(..., help="s3://bucket/prefix/ — every object under it becomes a file"),
    sensor: str = typer.Option(..., help="sensor slug, e.g. enmap"),
    level: str = typer.Option(..., help="processing level, e.g. L2A"),
    scene_id: str = typer.Option(..., help="the provider's product id"),
    acquired_at: str | None = typer.Option(None, help="ISO timestamp, UTC"),
    meta: list[str] = typer.Option([], help="repeatable key=value"),
    tag: list[str] = typer.Option([], help="repeatable"),
    hash_workers: int = typer.Option(8, min=1, max=64, help="objects hashed at once"),
    copy_workers: int = typer.Option(3, min=1, max=16, help="objects copied server-side at once"),
) -> None:
    """Hash objects already in S3 (checking provider .md5 sidecars), then have
    the api copy them in server-side and finalise. Re-run to resume."""
    from . import s3import

    c = Client()
    if not uri.startswith("s3://"):
        _fail("URI must be s3://bucket/prefix/")
    bucket, prefix, objs, provider_md5 = s3import.list_objects(uri)
    if not objs:
        _fail(f"{uri}: no objects")
    total = sum(o["Size"] for o in objs)
    typer.echo(f"{len(objs)} objects, {_human(total)}; {len(provider_md5)} with provider .md5")

    rel = {o["Key"]: o["Key"][len(prefix):].lstrip("/") or o["Key"].rsplit("/", 1)[-1] for o in objs}

    # 1. Hash everything (cached per ETag, so re-runs are cheap) and check the
    #    provider's MD5s *before* registering, so a bad source leaves nothing behind.
    seen, lock = 0, threading.Lock()
    def progress(n: int) -> None:
        nonlocal seen
        with lock:
            seen += n
            typer.echo(f"\r  hashed {_human(seen)} / {_human(total)}", nl=False)
    hashes = s3import.hash_all(bucket, objs, hash_workers, progress)
    typer.echo()
    bad_md5 = [k for k, h in hashes.items() if k in provider_md5 and h["md5"] != provider_md5[k]]
    if bad_md5:
        _fail(f"MD5 differs from the provider's for: {', '.join(bad_md5)}; refusing to import")
    for k in sorted(provider_md5):
        typer.echo(f"  md5 ok  {rel[k]}")

    metadata = {}
    for kv in meta:
        k, _, v = kv.partition("=")
        try:
            metadata[k] = json.loads(v)
        except ValueError:
            metadata[k] = v
    d = register_or_resume(c, {
        "sensor_type": sensor, "sensor_scene_id": scene_id, "processing_level": level,
        "acquisition_at": acquired_at, "metadata": metadata, "sources": [uri],
    })
    if d is None:
        return
    ds = d["id"]
    done = {f["relative_path"] for f in c.get(f"/datasets/{ds}")["files"] if f["uploaded"]}
    todo = [o for o in objs if rel[o["Key"]] not in done]
    typer.echo(f"{len(done)} already verified, {len(todo)} to copy")

    # 2. Server-side copies, several at once; each call returns when its object is in.
    def copy(o: dict) -> str:
        h = hashes[o["Key"]]
        note = f"md5 {h['md5']} matches provider .md5" if o["Key"] in provider_md5 else None
        r = c.post(f"/datasets/{ds}/imports", {
            "relative_path": rel[o["Key"]], "size_bytes": h["size"], "sha256": h["sha256"],
            "part_sha256": h["part_sha256"], "source_uri": f"s3://{bucket}/{o['Key']}", "note": note,
        }, timeout=None)
        return f"  copied  {r['relative_path']}  ({r['parts_copied']} parts, {r['parts_already_held']} already held)"
    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        for f in as_completed([pool.submit(copy, o) for o in todo]):
            typer.echo(f.result())

    # 3. Finalise and tag.
    d = c.post(f"/datasets/{ds}/finalize", timeout=None)
    for t in tag:
        c.put(f"/datasets/{ds}/tags/{t}")
    typer.secho(f"{ds} available  {d['s3_uri']}", fg=typer.colors.GREEN)


@app.command("download")
def download(
    dataset_id: str,
    dest: Path = typer.Argument(..., help="folder to write into (created if missing)"),
) -> None:
    """Fetch every file plus manifest.json, verifying each sha256."""
    plan = Client().get(f"/datasets/{dataset_id}/download")
    for f in plan["files"]:
        target = dest / f["relative_path"]
        if target.exists() and transfer.hash_file(target)[1] == f["sha256"]:
            continue                  # already here and intact
        transfer.download_file(f["url"], target, f["sha256"])
        typer.echo(f"  {f['relative_path']}  {_human(f['size_bytes'])}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_bytes(httpx.get(plan["manifest_url"]).content)
    typer.secho(f"{len(plan['files'])} files verified in {dest}", fg=typer.colors.GREEN)


def main() -> None:
    try:
        app()
    except InventoryError as e:
        # Outside typer's context, so print and exit directly (no traceback).
        typer.secho(f"inventory api: {e}", fg=typer.colors.RED, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
