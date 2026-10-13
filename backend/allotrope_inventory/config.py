"""Inventory service — typed configuration, read from INVENTORY_* env vars.

The URL is built in Python, as in allotrope/config.py, so a password with
@ : / # in it is escaped correctly.
"""

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    # Local runs read backend/.env.inventory (gitignored); containers get env vars.
    model_config = SettingsConfigDict(
        env_prefix="inventory_", env_file=".env.inventory", case_sensitive=False
    )

    # --- Postgres (Supabase, Mumbai) -------------------------------------
    db_user: str = "postgres"
    db_password: str  # required
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "postgres"

    # --- S3 ---------------------------------------------------------------
    bucket: str = "allotrope-raw-inventory-india"
    region: str = "ap-south-1"
    # Profile for local runs; None inside a container that gets env creds.
    aws_profile: str | None = None

    # --- Auth -------------------------------------------------------------
    # Comma-separated `name:key` pairs accepted in X-Inventory-Key. The name
    # is what lands in datasets.created_by and the audit log.
    api_keys: str  # required

    @cached_property
    def database_url(self) -> str:
        """SQLAlchemy URL with all components properly URL-encoded."""
        return URL.create(
            "postgresql+psycopg",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        ).render_as_string(hide_password=False)


settings = Settings()
