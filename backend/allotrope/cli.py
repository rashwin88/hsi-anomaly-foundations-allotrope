"""Allotrope api — administrative CLI.

Run via:
    docker compose -f docker/docker-compose.yml run --rm api \\
        python -m allotrope.cli <command>

Commands:
    seed-admin             create the initial admin user from ADMIN_* env vars
    seed-action-templates  upsert one ActionTemplate row per (action type × sensor)
                           from each action_types module's META.default_config_per_sensor

Both are idempotent and safe to re-run as part of a bootstrap script.
"""

import typer
from sqlalchemy import select

from . import action_types as action_types_registry
from .auth.password import hash_password
from .config import settings
from .db import SessionLocal
from .models import Action, ActionTemplate, User

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _main() -> None:
    """Allotrope api administrative CLI.

    The callback forces typer into subcommand mode (otherwise, with only
    one command defined, typer treats that command as the root and rejects
    the command name as an extra argument).
    """


@app.command("seed-admin")
def seed_admin() -> None:
    """Create the initial admin user from env vars (idempotent).

    Reads ADMIN_USERNAME, ADMIN_EMAIL, ADMIN_PASSWORD. Exits 0 if an admin
    already exists (so it's safe to re-run as part of a bootstrap script).
    Exits 1 if the env vars are missing.
    """
    if not (settings.admin_username and settings.admin_email and settings.admin_password):
        typer.secho(
            "ADMIN_USERNAME, ADMIN_EMAIL, and ADMIN_PASSWORD must all be "
            "set in docker/.env",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    with SessionLocal() as session:
        existing = session.scalar(
            select(User).where(User.is_admin.is_(True)).limit(1)
        )
        if existing is not None:
            typer.secho(
                f"Admin already exists: {existing.username} <{existing.email}> "
                f"(id={existing.id})",
                fg=typer.colors.YELLOW,
            )
            return

        user = User(
            username=settings.admin_username,
            email=settings.admin_email,
            password_hash=hash_password(settings.admin_password),
            is_admin=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        typer.secho(
            f"Created admin: {user.username} <{user.email}> (id={user.id})",
            fg=typer.colors.GREEN,
        )


@app.command("seed-action-templates")
def seed_action_templates() -> None:
    """Upsert one ActionTemplate per (action type × applicable sensor).

    Reads each action_types module's `META.default_config_per_sensor` and
    materialises a row marked `is_system=True` for each entry. Idempotent
    and safe to re-run: existing system rows are updated in place when
    the META payload drifts (so a code change to a default config
    propagates without manually clearing the DB).

    User-saved templates (is_system=False) are never touched.

    Naming convention:
        "<META.label> · default · <SENSOR_LABEL>"
    e.g. "Band filter · default · PRISMA"

    A system row is found by type and its " · default · <SENSOR_LABEL>" suffix, not by the
    full name, so a shorter META.label renames the existing row in place instead of adding a
    second one beside it. Actions do not reference templates by name, so a rename never
    touches existing projects. Stale system duplicates from earlier relabels are deleted
    when no Action references them.

    Exit codes:
        0  one or more rows upserted (or no work to do)
        1  registry empty (programming error)

    Sequence diagram: final design/diagrams/cli-seed-action-templates.drawio
    """
    sensor_labels = {"prisma": "PRISMA", "enmap": "EnMAP", "landsat9": "Landsat 9", "aviris_ng": "AVIRIS-NG", "hotsat1": "HotSAT-1", "tanager": "Tanager", "generic_hsi": "Other hyperspectral"}

    if not action_types_registry.REGISTRY:
        typer.secho(
            "action_types registry is empty — nothing to seed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    inserted = 0
    updated = 0
    unchanged = 0
    removed = 0

    with SessionLocal() as session:
        for kind, spec in sorted(action_types_registry.REGISTRY.items()):
            meta = spec.META
            for sensor, default_cfg in sorted(
                meta.default_config_per_sensor.items()
            ):
                sensor_pretty = sensor_labels.get(sensor, sensor)
                name = f"{meta.label} · default · {sensor_pretty}"

                # Identity for upsert: (type, " · default · <sensor>" suffix,
                # is_system=True), so a relabel renames in place. Keep the row that
                # already has the name, else the oldest; earlier relabels (before this
                # matching) left stale duplicates, removed here unless an Action
                # still points at one. User templates are never matched.
                suffix = f" · default · {sensor_pretty}"
                rows = sorted(
                    session.scalars(
                        select(ActionTemplate).where(
                            ActionTemplate.type == kind,
                            ActionTemplate.name.endswith(suffix),
                            ActionTemplate.is_system.is_(True),
                        )
                    ),
                    key=lambda t: (t.name != name, t.created_at),
                )
                existing = rows[0] if rows else None
                for stale in rows[1:]:
                    in_use = session.scalar(
                        select(Action.id).where(Action.action_template_id == stale.id).limit(1)
                    )
                    if in_use is None:
                        session.delete(stale)
                        removed += 1
                        typer.secho(f"  - {kind:25s} · {sensor_pretty:9s} → {stale.id} "
                                    f"(stale duplicate '{stale.name}')", fg=typer.colors.YELLOW)
                session.flush()

                description = (
                    f"System default for {sensor_pretty} scenes. "
                    f"{meta.short_description}"
                )

                if existing is None:
                    row = ActionTemplate(
                        type=kind,
                        name=name,
                        description=description,
                        configuration=default_cfg,
                        is_system=True,
                    )
                    session.add(row)
                    session.flush()
                    typer.secho(
                        f"  + {kind:25s} · {sensor_pretty:9s} → {row.id}",
                        fg=typer.colors.GREEN,
                    )
                    inserted += 1
                elif (
                    existing.configuration != default_cfg
                    or existing.description != description
                    or existing.name != name
                ):
                    existing.configuration = default_cfg
                    existing.description = description
                    existing.name = name
                    typer.secho(
                        f"  ~ {kind:25s} · {sensor_pretty:9s} → {existing.id} (config, name or description drift)",
                        fg=typer.colors.YELLOW,
                    )
                    updated += 1
                else:
                    typer.secho(
                        f"  · {kind:25s} · {sensor_pretty:9s} → {existing.id} (unchanged)",
                        fg=typer.colors.WHITE,
                    )
                    unchanged += 1

        session.commit()

    typer.secho(
        f"Seeded action templates: {inserted} new · {updated} updated · {unchanged} unchanged"
        f" · {removed} stale removed",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
