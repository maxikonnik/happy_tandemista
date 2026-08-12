# backend/tests/cloud/test_migrations.py
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


BACKEND = Path(__file__).resolve().parents[2]


def test_single_head_revision():
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    assert len(script.get_heads()) == 1


def test_migration_metadata_matches_models():
    # Every mapped table must appear in the initial migration's create_table calls.
    from tandemista.db.base import Base
    import tandemista.db.models  # noqa: F401  (register mappers)

    initial = (BACKEND / "migrations" / "versions" / "0001_initial.py").read_text()
    for table in Base.metadata.tables:
        assert f'"{table}"' in initial or f"'{table}'" in initial
