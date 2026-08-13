# backend/migrations/env.py
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from tandemista.db.base import Base
import tandemista.db.models  # noqa: F401  (register all mappers on Base.metadata)
from tandemista.config import get_settings

config = context.config
target_metadata = Base.metadata

db_url = os.environ.get("TANDEMISTA_DATABASE_URL") or get_settings().database_url
config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    context.configure(
        url=db_url, target_metadata=target_metadata, literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": db_url}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
