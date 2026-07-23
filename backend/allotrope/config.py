"""Allotrope api — typed configuration.

Loads Postgres connection components from the environment and builds a
properly URL-encoded SQLAlchemy URL inside Python. Constructing the URL in
code (instead of via compose substitution into a string) means special
characters in the password (@ : / # % ?) get escaped correctly — without
that, you get "failed to resolve host '@postgres'" or similar parser errors.
"""

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    # --- Postgres ---------------------------------------------------------
    # Each component is read raw; URL encoding happens in `database_url`.
    postgres_user: str = "allotrope"
    postgres_password: str  # required
    postgres_host: str = "postgres"  # service name on the compose network
    postgres_port: int = 5432
    postgres_db: str = "allotrope"

    # --- Admin seed -------------------------------------------------------
    # Only required by the `seed-admin` CLI; the api ignores these.
    admin_username: str | None = None
    admin_email: str | None = None
    admin_password: str | None = None

    # --- JWT --------------------------------------------------------------
    # The signing secret for HS256 JWTs. REQUIRED.
    # Suggested: `openssl rand -base64 32`. Rotating this invalidates every
    # token in flight (our nuclear-option revocation per final design.md).
    jwt_secret: str

    # Token lifetime in seconds (default 24 hours). Cookie Max-Age matches.
    jwt_lifetime_seconds: int = 86400

    # Cookie name carrying the JWT.
    jwt_cookie_name: str = "allotrope_jwt"

    # Cookie `Secure` flag. False in dev (HTTP localhost); True in prod
    # (HTTPS). Browsers ignore Set-Cookie with Secure over plain HTTP.
    cookie_secure: bool = False

    # --- Volume mount paths ---------------------------------------------
    # Where the named volumes are mounted inside the container. The
    # compose file decides the actual mount point; these defaults match
    # docker-compose.yml.
    #
    # `data_dir`      → allotrope_data       (raw scene files + staging)
    # `artifacts_dir` → allotrope_artifacts  (thumbnails, exports)
    # `models_dir`    → allotrope_models     (foundation model checkpoints + manifests)
    #
    # The api reads the per-architecture `current.json` manifests under
    # `<models_dir>/<architecture>/current.json` to power the /models
    # catalog. .pt weight files are owned by the worker; the api never
    # touches them.
    data_dir: str = "/data"
    artifacts_dir: str = "/artifacts"
    models_dir: str = "/models"
    # Per-sensor USGS splib07 caches (`splib07_<key>.npz` + `.json` pairs)
    # built by `scripts/build_splib_sensor_cache.py`. Mounted as the
    # `allotrope_splib07` named volume; the cache-build CLI on the host
    # writes here, and the worker container reads it at action time.
    splib07_cache_dir: str = "/splib07_cache"

    @cached_property
    def database_url(self) -> str:
        """SQLAlchemy URL with all components properly URL-encoded."""
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)


settings = Settings()
