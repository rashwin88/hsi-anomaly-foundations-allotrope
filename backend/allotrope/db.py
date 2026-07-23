"""Allotrope api — SQLAlchemy engine + session factory.

Sync engine (psycopg v3 sync driver). At our scale, sync is enough; switching
to async later is a contained change if needed.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

# When you call create_engine(...), SQLAlchemy doesn't open a connection. 
# It creates a pool manager. 
# The first time someone runs a query, the engine opens a TCP connection to Postgres, hands it out, 
# and (when finished) returns it to the pool rather than closing it. 
# Subsequent queries reuse that connection. By default the pool keeps up to 5 idle connections plus 10 overflow.
# The engine is long-lived — one per process. We create it once at import time and never throw it away.



engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,  # detect dropped connections (free, cheap, prevents stale-conn bugs)
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=Session,
    expire_on_commit=False,
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a Session and closes it on request end.
    When the request comes in, FastAPI calls next(get_db()). The generator runs up to yield db and pauses, handing db to your endpoint.
    Your endpoint runs, does its query, returns a response.
    FastAPI then resumes the generator. It runs the finally block. db.close() returns the connection to the pool."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
