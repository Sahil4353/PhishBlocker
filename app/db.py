import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Project root (…/PhishBlocker)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default to a file-based SQLite DB in the project root, but allow env override
default_sqlite_path = (PROJECT_ROOT / "phishblocker.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_sqlite_path}")

IS_SQLITE = DATABASE_URL.startswith("sqlite")

# SQLite-specific connection args
connect_args = {}
if IS_SQLITE:
    connect_args = {
        "check_same_thread": False,  # required for FastAPI threadpool
        "timeout": 30,  # reduce "database is locked" errors
    }

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "0") == "1",  # set SQL_ECHO=1 to see SQL
    future=True,  # SQLAlchemy 2.x style
    pool_pre_ping=not IS_SQLITE,  # useful for Postgres/MySQL
    connect_args=connect_args,
)

# Apply useful SQLite PRAGMAs on every new DB-API connection
if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        # Enforce FKs (SQLite defaults this off)
        cur.execute("PRAGMA foreign_keys=ON")
        # WAL = better concurrency; persists at DB level but safe to re-assert
        cur.execute("PRAGMA journal_mode=WAL")
        # Balanced durability vs performance (FULL is safest; NORMAL is common)
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,  # keep ORM objects usable after commit()
    future=True,
)
