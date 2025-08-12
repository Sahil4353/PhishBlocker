# app/db.py
from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)


def _configure_sqlalchemy_logging() -> None:
    """
    Configure SQLAlchemy loggers to honor env settings.
      - SQL_ECHO=1  -> show SQL statements (approx INFO)
      - SQL_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (overrides)
    """
    # default level
    level_name = os.getenv("SQL_LOG_LEVEL")
    if not level_name:
        level_name = "INFO" if os.getenv("SQL_ECHO", "0") == "1" else "WARNING"
    level = getattr(logging, level_name.upper(), logging.WARNING)

    for name in ("sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.dialects"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True  # use root handlers (filtered by ENABLE_LOGGING)


_configure_sqlalchemy_logging()

# -----------------------------------------------------------------------------
# DSN / Engine
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
default_sqlite_path = (PROJECT_ROOT / "phishblocker.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_sqlite_path}")
IS_SQLITE = DATABASE_URL.startswith("sqlite")


def _redact_dsn(dsn: str) -> str:
    """
    Hide credentials in logs: postgres://user:pass@host -> postgres://***:***@host
    """
    return re.sub(r"(://)([^:/@]+)(:([^@]*))?@", r"\1***:***@", dsn)


# SQLite-specific connection args
connect_args: dict = {}
if IS_SQLITE:
    connect_args = {
        "check_same_thread": False,  # required for FastAPI threadpool
        "timeout": 30,  # reduce "database is locked" errors
    }

# Create the engine (defensive)
try:
    engine: Engine = create_engine(
        DATABASE_URL,
        echo=os.getenv("SQL_ECHO", "0") == "1",  # optional SQL echo
        future=True,  # SQLAlchemy 2.x style
        pool_pre_ping=not IS_SQLITE,  # useful for Postgres/MySQL
        connect_args=connect_args,
    )
    logger.info(
        "DB engine created",
        extra={"dsn": _redact_dsn(DATABASE_URL), "sqlite": IS_SQLITE},
    )
except Exception as e:
    logger.exception(
        "Failed to create DB engine for %s: %s", _redact_dsn(DATABASE_URL), e
    )
    raise

# -----------------------------------------------------------------------------
# SQLite PRAGMAs and engine events
# -----------------------------------------------------------------------------
if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        try:
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
        except Exception as e:
            # Don't crash the app if PRAGMAs fail; just log it.
            try:
                cur.close()
            except Exception:
                pass
            logger.exception("Failed to apply SQLite PRAGMAs: %s", e)


# Pool/connection diagnostics (low-verbosity by default)
@event.listens_for(engine, "engine_connect")
def _on_engine_connect(conn, branch):
    logger.debug("Engine connect (branch=%s)", bool(branch))


@event.listens_for(engine, "checkout")
def _on_checkout(dbapi_con, con_record, con_proxy):
    logger.debug("Connection checked OUT id=%s", id(dbapi_con))


@event.listens_for(engine, "checkin")
def _on_checkin(dbapi_con, con_record):
    logger.debug("Connection checked IN id=%s", id(dbapi_con))


@event.listens_for(engine, "handle_error")
def _on_handle_error(exception_context):
    # Avoid logging full SQL or parameters (privacy); keep it high-level.
    try:
        stmt = (exception_context.statement or "").strip().split("\n")[0]
        stmt_head = stmt[:80] + ("…" if len(stmt) > 80 else "")
    except Exception:
        stmt_head = "<unavailable>"
    logger.error(
        "DB error (disconnect=%s): %s | orig=%s",
        exception_context.is_disconnect,
        stmt_head,
        repr(exception_context.original_exception),
    )
    # Default behavior (re-raise) continues.


# -----------------------------------------------------------------------------
# Session factory + optional context manager
# -----------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,  # keep ORM objects usable after commit()
    future=True,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Optional context manager for scripts/one-offs:

        from app.db import get_session
        with get_session() as db:
            db.add(obj)

    Routes should keep using FastAPI's dependency pattern.
    """
    db: Session = SessionLocal()
    try:
        yield db
        # Leave commit control to caller; uncomment to auto-commit:
        # db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("Session error; rolled back: %s", e)
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass
