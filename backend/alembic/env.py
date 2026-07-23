"""Alembic environment.

Connects Alembic to the same database URL and metadata that the api uses
at runtime, so future `--autogenerate` runs can diff our SQLAlchemy models
against the live DB.

Note: we DO NOT call `config.set_main_option("sqlalchemy.url", url)`. The
url often contains URL-encoded characters (e.g. `%40` for `@` in passwords),
and alembic's underlying configparser interprets `%` as interpolation syntax,
raising "invalid interpolation syntax". Reading the URL directly from
settings.database_url and passing it as a kwarg avoids that entirely.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Importing the models package registers every model class on Base.metadata.
# Alembic's autogenerate walks Base.metadata to know which tables exist.
from allotrope.config import settings
from allotrope.models import Base

# Alembic Config object — used only for logging config and offline-mode flag.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without connecting to the DB.

    Useful for inspecting the SQL alembic would run, or applying it via a
    different tool.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations directly against a live DB."""
    connectable = create_engine(
        settings.database_url,
        poolclass=pool.NullPool,  # don't pool — alembic runs once and exits
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # compare_type=True helps autogenerate detect column-type changes;
            # default behavior misses some.
            compare_type=True,
            # compare_server_default=True helps it detect default-value
            # changes (boolean defaults, etc.).
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
