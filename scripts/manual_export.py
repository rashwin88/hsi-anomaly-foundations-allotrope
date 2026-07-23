"""Manually build a spectral-library-match export bundle, bypassing the api.

Usage (run on the host that has the docker volumes mounted, e.g. inside the
api or worker container so /data + /artifacts are visible):

    # List candidate scenes + their latest spectral_library_match action
    python scripts/manual_export.py --list

    # Build a bundle for one action by id
    python scripts/manual_export.py --action-id <action_uuid> --out /tmp/

    # Or by scene id (picks the most-recent completed spectral_library_match)
    python scripts/manual_export.py --scene-id <scene_uuid> --out /tmp/

Bypasses the FastAPI route entirely — reads the artifact dir + raw scene
file directly, resolves georef, and builds the zip. Useful when the /export
endpoint is throwing 500s.

Output: <out>/hyper_<short_scene>_<short_action>.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

# Allow running from inside a container where /srv is the python path.
sys.path.insert(0, "/srv")


def _connect_db():
    """Open a sqlalchemy session against the postgres container."""
    import sqlalchemy as sa
    from sqlalchemy.orm import Session

    pg_user = os.environ.get("POSTGRES_USER", "allotrope")
    pg_pass = os.environ.get("POSTGRES_PASSWORD", "")
    pg_host = os.environ.get("POSTGRES_HOST", "postgres")
    pg_port = os.environ.get("POSTGRES_PORT", "5432")
    pg_db = os.environ.get("POSTGRES_DB", "allotrope")
    url = f"postgresql+psycopg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    engine = sa.create_engine(url)
    return Session(engine), sa


def list_actions():
    session, sa = _connect_db()
    try:
        rows = session.execute(
            sa.text("""
                SELECT a.id, a.status, a.type, ao.artifact_path,
                       s.id, s.sensor_type, s.name, s.raw_path
                FROM actions a
                JOIN action_outputs ao ON ao.action_id = a.id
                JOIN projects p ON p.id = a.project_id
                JOIN scenes s ON s.id = p.scene_id
                WHERE a.type = 'spectral_library_match'
                  AND a.status = 'complete'
                ORDER BY a.created_at DESC
            """)
        ).fetchall()
    finally:
        session.close()

    if not rows:
        print("No completed spectral_library_match actions in the DB.")
        return

    print(f"{'action_id':<36}  {'sensor':<10}  scene_name (raw_path)")
    print("-" * 100)
    for r in rows:
        action_id, status, _type, art_path, scene_id, sensor, scene_name, raw_path = r
        print(f"{action_id}  {sensor:<10}  {scene_name}   →  {raw_path}")


def find_action(action_id: str | None, scene_id: str | None):
    """Return (action_id, scene_id, sensor_type, artifact_path, raw_path)."""
    session, sa = _connect_db()
    try:
        if action_id:
            row = session.execute(
                sa.text("""
                    SELECT a.id, s.id, s.sensor_type, ao.artifact_path, s.raw_path
                    FROM actions a
                    JOIN action_outputs ao ON ao.action_id = a.id
                    JOIN projects p ON p.id = a.project_id
                    JOIN scenes s ON s.id = p.scene_id
                    WHERE a.id = :aid AND a.type = 'spectral_library_match'
                """), {"aid": uuid.UUID(action_id)},
            ).fetchone()
        elif scene_id:
            row = session.execute(
                sa.text("""
                    SELECT a.id, s.id, s.sensor_type, ao.artifact_path, s.raw_path
                    FROM actions a
                    JOIN action_outputs ao ON ao.action_id = a.id
                    JOIN projects p ON p.id = a.project_id
                    JOIN scenes s ON s.id = p.scene_id
                    WHERE s.id = :sid
                      AND a.type = 'spectral_library_match'
                      AND a.status = 'complete'
                    ORDER BY a.created_at DESC LIMIT 1
                """), {"sid": uuid.UUID(scene_id)},
            ).fetchone()
        else:
            raise ValueError("Provide --action-id or --scene-id")

        if not row:
            raise SystemExit("No matching action found.")
        return row
    finally:
        session.close()


def build(action_id: str | None, scene_id: str | None, out_dir: str,
          confidence_deg: float):
    from app.spectral_match.export import ExportSpec, build_hyper_bundle
    from app.georef import GeorefUnavailable, resolve_scene_georef
    import rasterio

    action_uuid, scene_uuid, sensor_type, art_path_rel, raw_path = find_action(
        action_id, scene_id,
    )

    artifacts_root = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    data_root = Path(os.environ.get("DATA_DIR", "/data"))
    artifact_dir = artifacts_root / art_path_rel
    scene_raw_dir = data_root / raw_path
    if not scene_raw_dir.is_dir() and scene_raw_dir.suffix:
        scene_raw_dir = scene_raw_dir.parent

    print(f"action      : {action_uuid}")
    print(f"scene       : {scene_uuid} ({sensor_type})")
    print(f"artifact_dir: {artifact_dir}")
    print(f"raw_dir     : {scene_raw_dir}")

    # Resolve georef from the raw scene file.
    ref_path = artifact_dir / "match_map.tif"
    with rasterio.open(ref_path) as src:
        target_shape = (src.height, src.width)
    print(f"target_shape: {target_shape}")

    try:
        transform, crs = resolve_scene_georef(
            scene_dir=scene_raw_dir,
            sensor_type=sensor_type,
            target_shape=target_shape,
        )
    except GeorefUnavailable as exc:
        raise SystemExit(f"georef resolve failed: {exc}") from exc

    print(f"crs         : {crs}")
    print(f"transform   : {transform}")

    spec = ExportSpec(
        action_id=f"action_{action_uuid}",
        scene_id=f"scene_{scene_uuid}",
        sensor_type=sensor_type,
        artifact_dir=artifact_dir,
        splib_version=None,
        software_version="allotrope/manual",
        confidence_threshold_deg=confidence_deg,
        override_transform=transform,
        override_crs=crs,
    )
    blob, filename = build_hyper_bundle(spec)

    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"\nwrote {out}  ({len(blob)/1e6:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true",
                   help="List all completed spectral_library_match actions.")
    p.add_argument("--action-id", help="UUID of a specific action.")
    p.add_argument("--scene-id",
                   help="UUID of a scene (picks its latest completed match).")
    p.add_argument("--out", default="/tmp",
                   help="Output directory (default /tmp).")
    p.add_argument("--confidence-deg", type=float, default=15.0,
                   help="SAM-angle threshold for the `conf` shapefile column.")
    args = p.parse_args()

    if args.list:
        list_actions()
        return
    if not (args.action_id or args.scene_id):
        p.error("provide --list, --action-id, or --scene-id")
    build(args.action_id, args.scene_id, args.out, args.confidence_deg)


if __name__ == "__main__":
    main()
