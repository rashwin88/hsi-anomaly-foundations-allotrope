"""Alembic environment for the INVENTORY database.

Same shape as alembic/env.py, pointed at allotrope_inventory's settings and
metadata. The URL is passed as a kwarg, never via set_main_option, because
configparser chokes on the `%` escapes in an encoded password.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from allotrope_inventory.config import settings
from allotrope_inventory.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Keep autogenerate to our tables — Supabase ships its own (auth, storage, PostGIS)."""
    return not (type_ == "table" and reflected and compare_to is None)


def run_migrations_offline() -> None:
    """Emit SQL without connecting to the DB."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the live DB."""
    connectable = create_engine(settings.database_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
