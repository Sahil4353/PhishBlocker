from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine.url import make_url

# --- Make project importable ---
ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of "migrations")
sys.path.append(str(ROOT))

# (Optional explicit import to be extra-safe about side effects)
import app.models  # noqa: F401

# --- Import SQLAlchemy metadata and DB URL ---
# IMPORTANT: import from app.models so Email/Scan classes are loaded into Base.metadata
from app import db as app_db  # app.db defines DATABASE_URL
from app.models import Base  # <-- imports __init__.py which imports email.py & scan.py

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DATABASE_URL = getattr(app_db, "DATABASE_URL", "sqlite:///phishblocker.db")
config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = Base.metadata
render_as_batch = make_url(DATABASE_URL).get_backend_name() == "sqlite"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=render_as_batch,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        {"sqlalchemy.url": DATABASE_URL},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=render_as_batch,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
