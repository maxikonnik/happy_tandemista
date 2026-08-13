# Cloud Scaffold — Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the cloud backend foundation for happy_tandemista — configuration, PostgreSQL data model, storage abstraction, a FastAPI API for the mini-manifest and manual uploads, and a Celery/Redis worker skeleton — so later plans can attach the analysis pipeline (wrapping the existing `tandemista.engine`) and a Next.js frontend.

**Architecture:** A single installable `tandemista` package gains cloud modules alongside the untouched `engine`: `config` (env-driven settings), `db` (SQLAlchemy 2.0 ORM + Alembic migrations), `storage` (a `StorageBackend` Protocol with local-folder and S3/MinIO implementations), `api` (FastAPI app: dropzones, loads, jumps, devices, manual media upload), and `worker` (a Celery app with a pipeline task **stub** that only transitions `MediaFile`/`JumpCut` status — the real engine call lands in the next plan). SQLAlchemy sessions are **synchronous** so the API and the Celery workers share one models layer. Local development runs on Docker Compose (Postgres, Redis, MinIO).

**Tech Stack:** Python 3.12+, FastAPI, Uvicorn, SQLAlchemy 2.0 (sync), Alembic, Pydantic v2 + pydantic-settings, Celery + Redis, boto3 (S3/MinIO), pytest. Existing engine: numpy + opencv-python-headless + system ffmpeg. Full design: `docs/superpowers/specs/2026-08-11-happy-tandemista-design.md`.

## Global Constraints

- Python ≥ 3.12.
- **Engine runtime dependencies stay exactly `numpy` + `opencv-python-headless`.** All cloud dependencies go in a new optional-dependency group `[cloud]`; `pip install -e .` alone must still install only the engine. No cloud vision-model SDKs anywhere in the codebase (owner decision 2026-08-11: CV is a local pipeline, no LLM at inference).
- Single-tenant deployment, but **every domain entity carries `dropzone_id`** and nothing blocks future multi-tenancy.
- Code, identifiers and comments in English; user-facing copy is out of scope for this plan.
- All public functions have type hints; dataclasses are `frozen=True` where not mutated.
- Conventional commits (`feat:`, `test:`, `chore:`, `docs:`).
- Configuration comes from environment variables via `pydantic-settings`; **no secrets committed** — only `.env.example` with placeholder values.
- **The pipeline never loses material silently** (spec §Error handling): an unreadable or failed item moves to an explicit `needs_attention`/`failed` status with a reason, it is never dropped.
- Times persisted for media are UTC-aware `datetime`; durations and offsets are seconds as `float`.
- New tests live under `backend/tests/cloud/` and must pass without any network service by defaulting to SQLite + a local-folder storage backend; tests that require Postgres-only features or MinIO are guarded with `pytest.mark.skipif`.

---

## File Structure

```
backend/
  pyproject.toml                 # + [project.optional-dependencies].cloud, + [tool] entries
  alembic.ini                    # Alembic config (script_location = migrations)
  docker-compose.yml             # postgres + redis + minio for local dev
  .env.example                   # placeholder settings
  tandemista/
    config.py                    # Settings (pydantic-settings): db, redis, storage
    db/
      __init__.py
      base.py                    # Base(DeclarativeBase), make_engine, SessionLocal, get_session
      types.py                   # GUID, JSONColumn portable type helpers
      models.py                  # ORM entities + enums
    storage/
      __init__.py                # re-exports StorageBackend, build_storage
      base.py                    # StorageBackend Protocol, StoredObject, StorageError
      local.py                   # LocalStorageBackend (network_drive / local folder)
      s3.py                      # S3StorageBackend (boto3, S3/MinIO)
      factory.py                 # build_storage(settings) -> StorageBackend
    api/
      __init__.py
      app.py                     # create_app() FastAPI factory + /health
      deps.py                    # get_db, get_storage, get_settings dependencies
      schemas.py                 # Pydantic request/response DTOs
      routes/
        __init__.py
        dropzones.py             # CRUD-lite: create/list dropzones, devices
        jumps.py                 # loads, customers, tandem jumps
        media.py                 # POST manual_web upload -> storage + MediaFile + enqueue
    worker/
      __init__.py
      celery_app.py              # Celery application, config from Settings
      tasks.py                   # analyze_media stub: status transitions only
  migrations/
    env.py                       # Alembic environment (imports Base.metadata)
    script.py.mako
    versions/
      0001_initial.py            # initial schema
  tests/
    cloud/
      __init__.py
      conftest.py                # settings override, in-memory db, TestClient, tmp storage
      test_config.py
      test_models.py
      test_migrations.py
      test_storage_local.py
      test_storage_s3.py         # skipif no MinIO / moto
      test_api_health.py
      test_api_manifest.py
      test_api_media_upload.py
      test_worker_stub.py
```

Engine files (`tandemista/engine/*`, `tandemista/cli.py`) are **not touched** by this plan.

---

### Task 1: Cloud dependency group and settings

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/tandemista/config.py`, `backend/.env.example`, `backend/tests/cloud/__init__.py`, `backend/tests/cloud/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields:
  - `database_url: str` (default `"sqlite+pysqlite:///:memory:"`)
  - `redis_url: str` (default `"redis://localhost:6379/0"`)
  - `storage_backend: str` (default `"local"`; one of `"local" | "s3"`)
  - `storage_local_root: str` (default `"./_storage"`)
  - `s3_endpoint_url: str | None`, `s3_region: str`, `s3_access_key: str | None`, `s3_secret_key: str | None`, `s3_bucket: str` (default `"tandemista"`)
  - `celery_task_always_eager: bool` (default `False`)
  - env prefix `TANDEMISTA_`, reads a `.env` file, ignores extra keys.
  - `get_settings() -> Settings` cached with `functools.lru_cache`.

- [ ] **Step 1: Add the cloud extra to pyproject**

In `backend/pyproject.toml`, after the existing `[project.optional-dependencies]` `dev` line, add a `cloud` group and extend `dev` to include it. Result of that table:

```toml
[project.optional-dependencies]
cloud = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "celery>=5.4",
    "redis>=5.0",
    "boto3>=1.34",
    "python-multipart>=0.0.9",
]
dev = ["pytest>=8", "httpx>=0.27", "moto[s3]>=5.0"]
```

(Engine `dependencies` stay `["numpy>=1.26", "opencv-python-headless>=4.10"]` — do not touch that line. `httpx` is needed by FastAPI's `TestClient`; `moto[s3]` mocks S3 in tests.)

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/cloud/test_config.py
from tandemista.config import Settings, get_settings


def test_defaults_are_offline_friendly():
    s = Settings()
    assert s.database_url.startswith("sqlite")
    assert s.storage_backend == "local"
    assert s.s3_bucket == "tandemista"
    assert s.celery_task_always_eager is False


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("TANDEMISTA_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("TANDEMISTA_S3_BUCKET", "jumps")
    s = Settings()
    assert s.storage_backend == "s3"
    assert s.s3_bucket == "jumps"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 2b: Install cloud deps and verify the test fails**

Run: `cd backend && pip install -e ".[cloud,dev]" -q && pytest tests/cloud/test_config.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.config)

- [ ] **Step 3: Implement config**

```python
# backend/tandemista/config.py
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for the cloud layer."""

    model_config = SettingsConfigDict(
        env_prefix="TANDEMISTA_", env_file=".env", extra="ignore"
    )

    database_url: str = "sqlite+pysqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/0"

    storage_backend: str = "local"
    storage_local_root: str = "./_storage"

    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str = "tandemista"

    celery_task_always_eager: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Create `.env.example`**

```dotenv
# backend/.env.example — copy to .env and fill in; never commit .env
TANDEMISTA_DATABASE_URL=postgresql+psycopg://tandemista:tandemista@localhost:5432/tandemista
TANDEMISTA_REDIS_URL=redis://localhost:6379/0
TANDEMISTA_STORAGE_BACKEND=local
TANDEMISTA_STORAGE_LOCAL_ROOT=./_storage
# S3 / MinIO (used when TANDEMISTA_STORAGE_BACKEND=s3)
TANDEMISTA_S3_ENDPOINT_URL=http://localhost:9000
TANDEMISTA_S3_REGION=us-east-1
TANDEMISTA_S3_ACCESS_KEY=minioadmin
TANDEMISTA_S3_SECRET_KEY=minioadmin
TANDEMISTA_S3_BUCKET=tandemista
```

Also add `backend/.env` and `backend/_storage/` to the repo root `.gitignore` (append two lines).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/cloud/test_config.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/tandemista/config.py backend/.env.example backend/tests/cloud/__init__.py backend/tests/cloud/test_config.py .gitignore
git commit -m "feat: cloud dependency group and env-driven settings"
```

---

### Task 2: Database base and portable column types

**Files:**
- Create: `backend/tandemista/db/__init__.py`, `backend/tandemista/db/base.py`, `backend/tandemista/db/types.py`
- Test: covered indirectly by Task 3; add `backend/tests/cloud/test_db_base.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces:
  - `Base` — SQLAlchemy 2.0 `DeclarativeBase` subclass shared by all models.
  - `make_engine(database_url: str) -> Engine` — creates a sync `Engine`; for SQLite URLs it enables `check_same_thread=False` and a `StaticPool` so an in-memory DB survives across sessions in tests.
  - `make_session_factory(engine) -> sessionmaker[Session]`.
  - `SessionLocal` — a module-level `scoped`/plain `sessionmaker` bound lazily via `configure_session(engine)`; and `get_session() -> Iterator[Session]` context-manager yielding a session and closing it.
  - `types.GUID` — a `TypeDecorator` storing a `uuid.UUID` as native `UUID` on Postgres and `CHAR(32)` hex elsewhere.
  - `types.JSONColumn` — `JSONB` on Postgres, generic `JSON` elsewhere.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_db_base.py
import uuid

from sqlalchemy import Column, String, select
from sqlalchemy.orm import Session

from tandemista.db.base import Base, configure_session, make_engine, make_session_factory
from tandemista.db.types import GUID, JSONColumn


class _Row(Base):
    __tablename__ = "t_row"
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    meta = Column(JSONColumn, nullable=False, default=dict)


def test_guid_and_json_roundtrip_on_sqlite():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[_Row.__table__])
    factory = make_session_factory(engine)
    rid = uuid.uuid4()
    with factory() as s:  # type: Session
        s.add(_Row(id=rid, name="a", meta={"k": [1, 2]}))
        s.commit()
    with factory() as s:
        row = s.execute(select(_Row).where(_Row.id == rid)).scalar_one()
        assert row.name == "a"
        assert row.meta == {"k": [1, 2]}
        assert isinstance(row.id, uuid.UUID)


def test_configure_session_binds_global_factory():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    configure_session(engine)
    from tandemista.db.base import SessionLocal

    with SessionLocal() as s:
        assert s.bind is engine
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_db_base.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.db.base)

- [ ] **Step 3: Implement `types.py`**

```python
# backend/tandemista/db/types.py
from __future__ import annotations

import uuid

from sqlalchemy import CHAR, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.types import TypeDecorator


class GUID(TypeDecorator):
    """UUID stored natively on PostgreSQL, as 32-char hex elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return (value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))).hex

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class JSONColumn(TypeDecorator):
    """JSONB on PostgreSQL, generic JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())
```

- [ ] **Step 4: Implement `base.py`**

```python
# backend/tandemista/db/base.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    return create_engine(database_url, pool_pre_ping=True, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# Global factory, bound at app/worker startup via configure_session().
SessionLocal: sessionmaker[Session] = sessionmaker(autoflush=False, expire_on_commit=False)


def configure_session(engine: Engine) -> None:
    SessionLocal.configure(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

`backend/tandemista/db/__init__.py`:

```python
from .base import Base, configure_session, get_session, make_engine, make_session_factory

__all__ = ["Base", "configure_session", "get_session", "make_engine", "make_session_factory"]
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_db_base.py -q`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/tandemista/db/__init__.py backend/tandemista/db/base.py backend/tandemista/db/types.py backend/tests/cloud/test_db_base.py
git commit -m "feat: db base (declarative Base, engine/session factories) and portable GUID/JSON types"
```

---

### Task 3: ORM domain model

**Files:**
- Create: `backend/tandemista/db/models.py`
- Test: `backend/tests/cloud/test_models.py`

**Interfaces:**
- Consumes: `Base` (Task 2), `GUID`, `JSONColumn` (Task 2).
- Produces string enums and mapped classes. Enums (subclass `str, enum.Enum`):
  - `DeviceKind`: `GOPRO, INSTA360, DSLR, PHONE, OTHER`.
  - `DeviceRole`: `HANDCAM, OUTSIDE, GROUND_INTERVIEW, GROUND_LANDING, MIXED`.
  - `UploadSourceKind`: `UPLOADER_AGENT, CLOUD_FOLDER, CAMERA_WIFI, MANUAL_WEB`.
  - `MediaStatus`: `REGISTERED, ANALYZING, ANALYZED, NEEDS_ATTENTION, FAILED`.
  - `CutVariant`: `FULL_16X9, EMOTIONS_16X9, EMOTIONS_9X16, HIGHLIGHTS_9X16`.
  - `CutStatus`: `DRAFT, IN_REVIEW, PUBLISHED, FAILED`.
  - `OrderStatus`: `PENDING, PAID, REFUNDED, CANCELLED`.
- Mapped classes, all with `id: uuid.UUID` PK (`GUID`, default `uuid4`), and `created_at`/`updated_at` timestamps via a shared `TimestampMixin`:
  - `Dropzone(name, currency="USD", payment_provider=None, storage_config: dict (JSONColumn, default {}), enabled_variants: list (JSONColumn, default all four))`.
  - `User(dropzone_id FK, name, role: str, email=None)`.
  - `Customer(dropzone_id FK, name, contact=None)`.
  - `Load(dropzone_id FK, name, takeoff_at: datetime|None)`.
  - `Device(dropzone_id FK, name, kind: DeviceKind, role: DeviceRole, owner_user_id FK|None, clock_offset_seconds: float=0.0, fingerprints: dict (JSONColumn, default {}), upload_source_id FK|None, active: bool=True)`.
  - `UploadSource(dropzone_id FK, kind: UploadSourceKind, config: dict (JSONColumn, default {}))`.
  - `TandemJump(dropzone_id FK, customer_id FK|None, load_id FK|None, instructor_user_id FK|None, outside_user_id FK|None, video_package: bool=False)`.
  - `MediaFile(dropzone_id FK, jump_id FK|None, device_id FK|None, upload_source_id FK|None, filename, locations: list (JSONColumn, default []), sha256=None, started_at: datetime|None, ended_at: datetime|None, status: MediaStatus=REGISTERED, telemetry: dict (JSONColumn, default {}))`.
  - `JumpCut(dropzone_id FK, jump_id FK, variant: CutVariant, status: CutStatus=DRAFT, edl: dict (JSONColumn, default {}), proxy_location=None, render_location=None)` with unique constraint `(jump_id, variant)`.
  - `Order(dropzone_id FK, jump_id FK, amount_cents: int, currency, provider=None, status: OrderStatus=PENDING, external_id=None)`.
- All FKs are `GUID`. Relationships are optional for this plan; only define the ones the tests use (`Dropzone.devices`, `TandemJump.cuts`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_models.py
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from tandemista.db.base import Base, make_engine, make_session_factory
from tandemista.db import models as m


@pytest.fixture()
def factory():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)


def test_create_dropzone_defaults(factory):
    with factory() as s:
        dz = m.Dropzone(name="Skyranch")
        s.add(dz)
        s.commit()
        assert isinstance(dz.id, uuid.UUID)
        assert dz.currency == "USD"
        assert set(dz.enabled_variants) == {v.value for v in m.CutVariant}
        assert dz.created_at is not None


def test_device_enum_and_defaults(factory):
    with factory() as s:
        dz = m.Dropzone(name="DZ")
        s.add(dz)
        s.flush()
        dev = m.Device(
            dropzone_id=dz.id, name="Hero 12 #1",
            kind=m.DeviceKind.GOPRO, role=m.DeviceRole.HANDCAM,
        )
        s.add(dev)
        s.commit()
        assert dev.active is True
        assert dev.clock_offset_seconds == 0.0
        assert dev.fingerprints == {}


def test_media_status_defaults_registered(factory):
    with factory() as s:
        dz = m.Dropzone(name="DZ")
        s.add(dz)
        s.flush()
        mf = m.MediaFile(dropzone_id=dz.id, filename="handcam_001.mp4")
        s.add(mf)
        s.commit()
        assert mf.status == m.MediaStatus.REGISTERED
        assert mf.locations == []


def test_jumpcut_variant_unique_per_jump(factory):
    with factory() as s:
        dz = m.Dropzone(name="DZ")
        s.add(dz)
        s.flush()
        jump = m.TandemJump(dropzone_id=dz.id)
        s.add(jump)
        s.flush()
        s.add(m.JumpCut(dropzone_id=dz.id, jump_id=jump.id, variant=m.CutVariant.FULL_16X9))
        s.commit()
        s.add(m.JumpCut(dropzone_id=dz.id, jump_id=jump.id, variant=m.CutVariant.FULL_16X9))
        with pytest.raises(IntegrityError):
            s.commit()


def test_all_tables_created(factory):
    expected = {
        "dropzones", "users", "customers", "loads", "devices",
        "upload_sources", "tandem_jumps", "media_files", "jump_cuts", "orders",
    }
    assert expected <= set(Base.metadata.tables)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_models.py -q`
Expected: FAIL (ImportError / missing attributes)

- [ ] **Step 3: Implement `models.py`**

```python
# backend/tandemista/db/models.py
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .types import GUID, JSONColumn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DeviceKind(str, enum.Enum):
    GOPRO = "gopro"
    INSTA360 = "insta360"
    DSLR = "dslr"
    PHONE = "phone"
    OTHER = "other"


class DeviceRole(str, enum.Enum):
    HANDCAM = "handcam"
    OUTSIDE = "outside"
    GROUND_INTERVIEW = "ground_interview"
    GROUND_LANDING = "ground_landing"
    MIXED = "mixed"


class UploadSourceKind(str, enum.Enum):
    UPLOADER_AGENT = "uploader_agent"
    CLOUD_FOLDER = "cloud_folder"
    CAMERA_WIFI = "camera_wifi"
    MANUAL_WEB = "manual_web"


class MediaStatus(str, enum.Enum):
    REGISTERED = "registered"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"


class CutVariant(str, enum.Enum):
    FULL_16X9 = "full_16x9"
    EMOTIONS_16X9 = "emotions_16x9"
    EMOTIONS_9X16 = "emotions_9x16"
    HIGHLIGHTS_9X16 = "highlights_9x16"


class CutStatus(str, enum.Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    FAILED = "failed"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(GUID, primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Dropzone(TimestampMixin, Base):
    __tablename__ = "dropzones"
    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String, nullable=False)
    currency: Mapped[str] = mapped_column(String, default="USD", nullable=False)
    payment_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    storage_config: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    enabled_variants: Mapped[list] = mapped_column(
        JSONColumn, default=lambda: [v.value for v in CutVariant], nullable=False
    )
    devices: Mapped[list["Device"]] = relationship(back_populates="dropzone")


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str | None] = mapped_column(String, nullable=True)


class Customer(TimestampMixin, Base):
    __tablename__ = "customers"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    contact: Mapped[str | None] = mapped_column(String, nullable=True)


class Load(TimestampMixin, Base):
    __tablename__ = "loads"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    takeoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UploadSource(TimestampMixin, Base):
    __tablename__ = "upload_sources"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    kind: Mapped[UploadSourceKind] = mapped_column(String, nullable=False)
    config: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)


class Device(TimestampMixin, Base):
    __tablename__ = "devices"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[DeviceKind] = mapped_column(String, nullable=False)
    role: Mapped[DeviceRole] = mapped_column(String, nullable=False)
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    clock_offset_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fingerprints: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("upload_sources.id"), nullable=True
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    dropzone: Mapped["Dropzone"] = relationship(back_populates="devices")


class TandemJump(TimestampMixin, Base):
    __tablename__ = "tandem_jumps"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    customer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("customers.id"), nullable=True)
    load_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("loads.id"), nullable=True)
    instructor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    outside_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    video_package: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cuts: Mapped[list["JumpCut"]] = relationship(back_populates="jump")


class MediaFile(TimestampMixin, Base):
    __tablename__ = "media_files"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=True)
    device_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("devices.id"), nullable=True)
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("upload_sources.id"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    locations: Mapped[list] = mapped_column(JSONColumn, default=list, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[MediaStatus] = mapped_column(
        String, default=MediaStatus.REGISTERED, nullable=False
    )
    telemetry: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)


class JumpCut(TimestampMixin, Base):
    __tablename__ = "jump_cuts"
    __table_args__ = (UniqueConstraint("jump_id", "variant", name="uq_jumpcut_jump_variant"),)
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=False)
    variant: Mapped[CutVariant] = mapped_column(String, nullable=False)
    status: Mapped[CutStatus] = mapped_column(String, default=CutStatus.DRAFT, nullable=False)
    edl: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    proxy_location: Mapped[str | None] = mapped_column(String, nullable=True)
    render_location: Mapped[str | None] = mapped_column(String, nullable=True)
    jump: Mapped["TandemJump"] = relationship(back_populates="cuts")


class Order(TimestampMixin, Base):
    __tablename__ = "orders"
    id: Mapped[uuid.UUID] = _pk()
    dropzone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dropzones.id"), nullable=False)
    jump_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tandem_jumps.id"), nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[OrderStatus] = mapped_column(String, default=OrderStatus.PENDING, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String, nullable=True)
```

Note: enums are stored via their string values in a `String` column (not SQLAlchemy `Enum`) to avoid Postgres native-enum migration churn; SQLAlchemy coerces the `str`-enum members to their `.value` on write and returns the enum on read because the column type is `String` and the attribute annotation is the enum — verify the equality assertions in the test pass (they compare against enum members, which equal their `str` value).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_models.py -q`
Expected: `5 passed`. If enum attributes come back as raw strings and an assertion `mf.status == m.MediaStatus.REGISTERED` fails, that is still true because `MediaStatus` subclasses `str` (member equals its value); do not change the assertion.

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/db/models.py backend/tests/cloud/test_models.py
git commit -m "feat: ORM domain model (dropzone, jump, device, media, cut, order) with enums"
```

---

### Task 4: Alembic migrations

**Files:**
- Create: `backend/alembic.ini`, `backend/migrations/env.py`, `backend/migrations/script.py.mako`, `backend/migrations/versions/0001_initial.py`
- Test: `backend/tests/cloud/test_migrations.py`

**Interfaces:**
- Consumes: `Base.metadata` (Task 2/3), `get_settings()` (Task 1).
- Produces: a runnable Alembic environment whose `0001_initial` migration creates the full schema; `env.py` reads the URL from `TANDEMISTA_DATABASE_URL` (falling back to `get_settings().database_url`) and targets `Base.metadata`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_migrations.py -q`
Expected: FAIL (alembic.ini missing)

- [ ] **Step 3: Create `alembic.ini`**

```ini
# backend/alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 4: Create `migrations/env.py`**

```python
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
```

`migrations/script.py.mako` (standard Alembic template):

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 5: Autogenerate the initial migration against a scratch Postgres, then commit it**

The initial migration must be generated (not hand-written) so it matches the models exactly. With Docker Compose Postgres up (Task 10 provides it; for now a throwaway container or local Postgres works):

Run:
```bash
cd backend
TANDEMISTA_DATABASE_URL=postgresql+psycopg://tandemista:tandemista@localhost:5432/tandemista \
  alembic revision --autogenerate -m "initial" --rev-id 0001_initial
```
Then open `migrations/versions/0001_initial.py` and confirm it contains `op.create_table("dropzones", ...)` through `op.create_table("orders", ...)` for all ten tables and the `uq_jumpcut_jump_variant` constraint. If Postgres is not available in the execution environment, generate against SQLite instead (`TANDEMISTA_DATABASE_URL=sqlite+pysqlite:///./_scratch.db`) — the portable `GUID`/`JSONColumn` types render per-dialect, and the metadata-matching test only checks table names.

- [ ] **Step 6: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_migrations.py -q`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add backend/alembic.ini backend/migrations
git commit -m "feat: alembic migrations with initial schema"
```

---

### Task 5: Storage abstraction and local backend

**Files:**
- Create: `backend/tandemista/storage/__init__.py`, `backend/tandemista/storage/base.py`, `backend/tandemista/storage/local.py`
- Test: `backend/tests/cloud/test_storage_local.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `StoredObject(backend: str, location: str, size: int)` — frozen dataclass; `location` is a backend-scoped URI (`local://<root>/<key>` or `s3://<bucket>/<key>`).
  - `StorageError(Exception)`.
  - `StorageBackend` Protocol: `put(key: str, data: BinaryIO) -> StoredObject`, `get(key: str) -> bytes`, `open(key: str) -> BinaryIO`, `exists(key: str) -> bool`, `delete(key: str) -> None`, `url(key: str) -> str`, property `name: str`.
  - `LocalStorageBackend(root: str | Path)` — stores under `root`, creating parent dirs; `location` is `local://<absroot>/<key>`; `url` returns a `file://` URI.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_storage_local.py
import io

import pytest

from tandemista.storage.base import StorageError
from tandemista.storage.local import LocalStorageBackend


def test_put_get_roundtrip(tmp_path):
    st = LocalStorageBackend(tmp_path)
    obj = st.put("jumps/j1/handcam.mp4", io.BytesIO(b"hello"))
    assert obj.size == 5
    assert obj.backend == "local"
    assert st.exists("jumps/j1/handcam.mp4")
    assert st.get("jumps/j1/handcam.mp4") == b"hello"


def test_open_streams_content(tmp_path):
    st = LocalStorageBackend(tmp_path)
    st.put("a.bin", io.BytesIO(b"abc"))
    with st.open("a.bin") as fh:
        assert fh.read() == b"abc"


def test_delete_and_missing_get(tmp_path):
    st = LocalStorageBackend(tmp_path)
    st.put("a.bin", io.BytesIO(b"abc"))
    st.delete("a.bin")
    assert not st.exists("a.bin")
    with pytest.raises(StorageError):
        st.get("a.bin")


def test_key_cannot_escape_root(tmp_path):
    st = LocalStorageBackend(tmp_path)
    with pytest.raises(StorageError):
        st.put("../evil.bin", io.BytesIO(b"x"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_storage_local.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `base.py`**

```python
# backend/tandemista/storage/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol, runtime_checkable


class StorageError(Exception):
    """Raised when a storage operation fails or a key is invalid/missing."""


@dataclass(frozen=True)
class StoredObject:
    backend: str
    location: str
    size: int


@runtime_checkable
class StorageBackend(Protocol):
    @property
    def name(self) -> str: ...

    def put(self, key: str, data: BinaryIO) -> StoredObject: ...

    def get(self, key: str) -> bytes: ...

    def open(self, key: str) -> BinaryIO: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

    def url(self, key: str) -> str: ...
```

- [ ] **Step 4: Implement `local.py`**

```python
# backend/tandemista/storage/local.py
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .base import StorageError, StoredObject


class LocalStorageBackend:
    """Files under a root directory. Models a network drive or local folder."""

    name = "local"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        target = (self._root / key).resolve()
        if self._root != target and self._root not in target.parents:
            raise StorageError(f"key escapes storage root: {key!r}")
        return target

    def put(self, key: str, data: BinaryIO) -> StoredObject:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = data.read()
        path.write_bytes(content)
        return StoredObject("local", f"local://{path}", len(content))

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as e:
            raise StorageError(str(e)) from e

    def open(self, key: str) -> BinaryIO:
        try:
            return self._path(key).open("rb")
        except FileNotFoundError as e:
            raise StorageError(str(e)) from e

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def url(self, key: str) -> str:
        return self._path(key).as_uri()
```

`storage/__init__.py`:

```python
from .base import StorageBackend, StorageError, StoredObject
from .local import LocalStorageBackend

__all__ = ["StorageBackend", "StorageError", "StoredObject", "LocalStorageBackend"]
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_storage_local.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/tandemista/storage/__init__.py backend/tandemista/storage/base.py backend/tandemista/storage/local.py backend/tests/cloud/test_storage_local.py
git commit -m "feat: StorageBackend protocol and local-folder backend"
```

---

### Task 6: S3/MinIO backend and storage factory

**Files:**
- Create: `backend/tandemista/storage/s3.py`, `backend/tandemista/storage/factory.py`
- Modify: `backend/tandemista/storage/__init__.py` (re-export `S3StorageBackend`, `build_storage`)
- Test: `backend/tests/cloud/test_storage_s3.py`

**Interfaces:**
- Consumes: `StorageBackend`, `StoredObject`, `StorageError` (Task 5); `Settings` (Task 1).
- Produces:
  - `S3StorageBackend(bucket, endpoint_url=None, region="us-east-1", access_key=None, secret_key=None)` — same interface as `LocalStorageBackend`, backed by boto3; creates the bucket if missing; `location` is `s3://<bucket>/<key>`; `url` returns a presigned GET URL (1 hour).
  - `build_storage(settings: Settings) -> StorageBackend` — returns `LocalStorageBackend(settings.storage_local_root)` or `S3StorageBackend(...)` per `settings.storage_backend`; raises `StorageError` on an unknown backend name.

- [ ] **Step 1: Write the failing test** (mock S3 with `moto`)

```python
# backend/tests/cloud/test_storage_s3.py
import io

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from tandemista.config import Settings
from tandemista.storage.base import StorageError
from tandemista.storage.factory import build_storage
from tandemista.storage.s3 import S3StorageBackend


@pytest.fixture()
def s3_backend():
    with moto.mock_aws():
        yield S3StorageBackend(bucket="test-bucket", region="us-east-1")


def test_put_get_roundtrip(s3_backend):
    obj = s3_backend.put("jumps/j1/handcam.mp4", io.BytesIO(b"hello"))
    assert obj.size == 5
    assert obj.location == "s3://test-bucket/jumps/j1/handcam.mp4"
    assert s3_backend.get("jumps/j1/handcam.mp4") == b"hello"
    assert s3_backend.exists("jumps/j1/handcam.mp4")


def test_missing_get_raises(s3_backend):
    with pytest.raises(StorageError):
        s3_backend.get("nope.bin")


def test_presigned_url(s3_backend):
    s3_backend.put("a.bin", io.BytesIO(b"x"))
    assert "a.bin" in s3_backend.url("a.bin")


def test_factory_selects_backend(tmp_path):
    local = build_storage(Settings(storage_backend="local", storage_local_root=str(tmp_path)))
    assert local.name == "local"
    with pytest.raises(StorageError):
        build_storage(Settings(storage_backend="nope"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_storage_s3.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.storage.s3)

- [ ] **Step 3: Implement `s3.py`**

```python
# backend/tandemista/storage/s3.py
from __future__ import annotations

import io
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from .base import StorageError, StoredObject


class S3StorageBackend:
    """S3-compatible object storage (AWS S3, Yandex Object Storage, MinIO)."""

    name = "s3"

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self._bucket)

    def put(self, key: str, data: BinaryIO) -> StoredObject:
        content = data.read()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=content)
        return StoredObject("s3", f"s3://{self._bucket}/{key}", len(content))

    def get(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            raise StorageError(str(e)) from e

    def open(self, key: str) -> BinaryIO:
        return io.BytesIO(self.get(key))

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def url(self, key: str) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=3600
        )
```

- [ ] **Step 4: Implement `factory.py`**

```python
# backend/tandemista/storage/factory.py
from __future__ import annotations

from ..config import Settings
from .base import StorageBackend, StorageError
from .local import LocalStorageBackend
from .s3 import S3StorageBackend


def build_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorageBackend(settings.storage_local_root)
    if settings.storage_backend == "s3":
        return S3StorageBackend(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
    raise StorageError(f"unknown storage backend: {settings.storage_backend!r}")
```

Extend `storage/__init__.py` exports to include `S3StorageBackend` and `build_storage`.

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_storage_s3.py -q`
Expected: `4 passed` (or skipped if boto3/moto unavailable)

- [ ] **Step 6: Commit**

```bash
git add backend/tandemista/storage/s3.py backend/tandemista/storage/factory.py backend/tandemista/storage/__init__.py backend/tests/cloud/test_storage_s3.py
git commit -m "feat: S3/MinIO storage backend and storage factory"
```

---

### Task 7: FastAPI app factory, dependencies, and health

**Files:**
- Create: `backend/tandemista/api/__init__.py`, `backend/tandemista/api/app.py`, `backend/tandemista/api/deps.py`
- Test: `backend/tests/cloud/conftest.py`, `backend/tests/cloud/test_api_health.py`

**Interfaces:**
- Consumes: `get_settings`, `Settings` (Task 1); `make_engine`, `configure_session`, `SessionLocal`, `Base` (Task 2); `build_storage` (Task 6).
- Produces:
  - `create_app(settings: Settings | None = None) -> FastAPI` — builds the engine, calls `configure_session`, mounts routers (added in Tasks 8–9), stores `settings` and a built `StorageBackend` on `app.state`, exposes `GET /health -> {"status": "ok"}`.
  - `deps.get_db() -> Iterator[Session]` — yields a `SessionLocal()`, commits on success, rolls back on exception, always closes.
  - `deps.get_storage(request) -> StorageBackend` — returns `request.app.state.storage`.
  - `deps.get_app_settings(request) -> Settings` — returns `request.app.state.settings`.
- `conftest.py` provides fixtures: `settings` (SQLite in-memory + local storage in a tmp dir), `app`, `client` (`fastapi.testclient.TestClient`), `db` (a session bound to the app's engine with tables created).

- [ ] **Step 1: Write `conftest.py` and the failing health test**

```python
# backend/tests/cloud/conftest.py
import pytest
from fastapi.testclient import TestClient

from tandemista.api.app import create_app
from tandemista.config import Settings
from tandemista.db.base import Base, SessionLocal


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_backend="local",
        storage_local_root=str(tmp_path / "storage"),
        celery_task_always_eager=True,
    )


@pytest.fixture()
def app(settings):
    application = create_app(settings)
    # create_app configured SessionLocal against the app engine; build the schema.
    Base.metadata.create_all(SessionLocal().get_bind())
    return application


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def db(app):
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

```python
# backend/tests/cloud/test_api_health.py
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_api_health.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.api.app)

- [ ] **Step 3: Implement `deps.py`**

```python
# backend/tandemista/api/deps.py
from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ..config import Settings
from ..db.base import SessionLocal
from ..storage.base import StorageBackend


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_storage(request: Request) -> StorageBackend:
    return request.app.state.storage


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
```

- [ ] **Step 4: Implement `app.py`**

```python
# backend/tandemista/api/app.py
from __future__ import annotations

from fastapi import FastAPI

from ..config import Settings, get_settings
from ..db.base import configure_session, make_engine
from ..storage.factory import build_storage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    configure_session(engine)

    app = FastAPI(title="happy_tandemista API")
    app.state.settings = settings
    app.state.engine = engine
    app.state.storage = build_storage(settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    from .routes import dropzones, jumps, media

    app.include_router(dropzones.router)
    app.include_router(jumps.router)
    app.include_router(media.router)
    return app
```

Note: the `from .routes import ...` line will fail until Tasks 8–9 exist. To keep Task 7 independently green, first implement `app.py` **without** the router imports/includes (health only), commit, and re-add the three include lines in Task 9's final step. Mark this explicitly:

- In Task 7, `app.py` ends after `app.state.storage = build_storage(settings)` and the `/health` route, then `return app` — **no router imports yet**.

`api/__init__.py` is empty.

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_api_health.py -q`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/tandemista/api/__init__.py backend/tandemista/api/app.py backend/tandemista/api/deps.py backend/tests/cloud/conftest.py backend/tests/cloud/test_api_health.py
git commit -m "feat: FastAPI app factory, DB/storage dependencies, health endpoint"
```

---

### Task 8: Manifest API — dropzones, devices, loads, customers, jumps

**Files:**
- Create: `backend/tandemista/api/schemas.py`, `backend/tandemista/api/routes/__init__.py`, `backend/tandemista/api/routes/dropzones.py`, `backend/tandemista/api/routes/jumps.py`
- Test: `backend/tests/cloud/test_api_manifest.py`

**Interfaces:**
- Consumes: `get_db` (Task 7), models (Task 3).
- Produces Pydantic v2 DTOs in `schemas.py` (all with `model_config = ConfigDict(from_attributes=True)` on read models):
  - `DropzoneCreate(name, currency="USD")`, `DropzoneRead(id, name, currency, enabled_variants)`.
  - `DeviceCreate(name, kind, role, clock_offset_seconds=0.0)`, `DeviceRead(id, dropzone_id, name, kind, role, active)`.
  - `LoadCreate(name, takeoff_at: datetime | None = None)`, `LoadRead(id, name, takeoff_at)`.
  - `CustomerCreate(name, contact: str | None = None)`, `CustomerRead(id, name, contact)`.
  - `JumpCreate(customer_id: UUID | None = None, load_id: UUID | None = None, instructor_user_id: UUID | None = None, outside_user_id: UUID | None = None, video_package: bool = False)`, `JumpRead(id, dropzone_id, customer_id, load_id, video_package)`.
- Routers:
  - `dropzones.router` (prefix `/dropzones`): `POST /` create dropzone; `GET /` list; `POST /{dropzone_id}/devices` create device; `GET /{dropzone_id}/devices` list devices.
  - `jumps.router` (prefix `/dropzones/{dropzone_id}`): `POST /loads`, `GET /loads`, `POST /customers`, `POST /jumps`, `GET /jumps`. Each validates the dropzone exists (404 otherwise).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_api_manifest.py
def _make_dropzone(client, name="Skyranch"):
    return client.post("/dropzones/", json={"name": name}).json()


def test_create_and_list_dropzone(client):
    created = client.post("/dropzones/", json={"name": "Skyranch"})
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Skyranch"
    assert body["currency"] == "USD"
    assert len(body["enabled_variants"]) == 4

    listed = client.get("/dropzones/")
    assert listed.status_code == 200
    assert any(d["id"] == body["id"] for d in listed.json())


def test_create_device_under_dropzone(client):
    dz = _make_dropzone(client)
    resp = client.post(
        f"/dropzones/{dz['id']}/devices",
        json={"name": "Hero12 #1", "kind": "gopro", "role": "handcam"},
    )
    assert resp.status_code == 201
    assert resp.json()["active"] is True

    devices = client.get(f"/dropzones/{dz['id']}/devices").json()
    assert len(devices) == 1


def test_create_jump_and_load(client):
    dz = _make_dropzone(client)
    load = client.post(f"/dropzones/{dz['id']}/loads", json={"name": "Load 1"})
    assert load.status_code == 201
    cust = client.post(f"/dropzones/{dz['id']}/customers", json={"name": "Jane"}).json()
    jump = client.post(
        f"/dropzones/{dz['id']}/jumps",
        json={"customer_id": cust["id"], "load_id": load.json()["id"], "video_package": True},
    )
    assert jump.status_code == 201
    assert jump.json()["video_package"] is True
    assert len(client.get(f"/dropzones/{dz['id']}/jumps").json()) == 1


def test_unknown_dropzone_404(client):
    import uuid

    resp = client.post(f"/dropzones/{uuid.uuid4()}/loads", json={"name": "x"})
    assert resp.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_api_manifest.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.api.routes.dropzones)

- [ ] **Step 3: Implement `schemas.py`**

```python
# backend/tandemista/api/schemas.py
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..db.models import DeviceKind, DeviceRole


class _Read(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DropzoneCreate(BaseModel):
    name: str
    currency: str = "USD"


class DropzoneRead(_Read):
    id: uuid.UUID
    name: str
    currency: str
    enabled_variants: list[str]


class DeviceCreate(BaseModel):
    name: str
    kind: DeviceKind
    role: DeviceRole
    clock_offset_seconds: float = 0.0


class DeviceRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    name: str
    kind: DeviceKind
    role: DeviceRole
    active: bool


class LoadCreate(BaseModel):
    name: str
    takeoff_at: datetime | None = None


class LoadRead(_Read):
    id: uuid.UUID
    name: str
    takeoff_at: datetime | None


class CustomerCreate(BaseModel):
    name: str
    contact: str | None = None


class CustomerRead(_Read):
    id: uuid.UUID
    name: str
    contact: str | None


class JumpCreate(BaseModel):
    customer_id: uuid.UUID | None = None
    load_id: uuid.UUID | None = None
    instructor_user_id: uuid.UUID | None = None
    outside_user_id: uuid.UUID | None = None
    video_package: bool = False


class JumpRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    customer_id: uuid.UUID | None
    load_id: uuid.UUID | None
    video_package: bool
```

- [ ] **Step 4: Implement `routes/dropzones.py`**

```python
# backend/tandemista/api/routes/dropzones.py
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as m
from ..deps import get_db
from ..schemas import DeviceCreate, DeviceRead, DropzoneCreate, DropzoneRead

router = APIRouter(prefix="/dropzones", tags=["dropzones"])


def _get_dropzone(db: Session, dropzone_id: uuid.UUID) -> m.Dropzone:
    dz = db.get(m.Dropzone, dropzone_id)
    if dz is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")
    return dz


@router.post("/", response_model=DropzoneRead, status_code=status.HTTP_201_CREATED)
def create_dropzone(payload: DropzoneCreate, db: Session = Depends(get_db)) -> m.Dropzone:
    dz = m.Dropzone(name=payload.name, currency=payload.currency)
    db.add(dz)
    db.flush()
    return dz


@router.get("/", response_model=list[DropzoneRead])
def list_dropzones(db: Session = Depends(get_db)) -> list[m.Dropzone]:
    return list(db.execute(select(m.Dropzone)).scalars())


@router.post(
    "/{dropzone_id}/devices", response_model=DeviceRead, status_code=status.HTTP_201_CREATED
)
def create_device(
    dropzone_id: uuid.UUID, payload: DeviceCreate, db: Session = Depends(get_db)
) -> m.Device:
    _get_dropzone(db, dropzone_id)
    dev = m.Device(
        dropzone_id=dropzone_id, name=payload.name, kind=payload.kind,
        role=payload.role, clock_offset_seconds=payload.clock_offset_seconds,
    )
    db.add(dev)
    db.flush()
    return dev


@router.get("/{dropzone_id}/devices", response_model=list[DeviceRead])
def list_devices(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.Device]:
    _get_dropzone(db, dropzone_id)
    return list(
        db.execute(select(m.Device).where(m.Device.dropzone_id == dropzone_id)).scalars()
    )
```

- [ ] **Step 5: Implement `routes/jumps.py`**

```python
# backend/tandemista/api/routes/jumps.py
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as m
from ..deps import get_db
from ..schemas import (
    CustomerCreate, CustomerRead, JumpCreate, JumpRead, LoadCreate, LoadRead,
)

router = APIRouter(prefix="/dropzones/{dropzone_id}", tags=["manifest"])


def _require_dropzone(db: Session, dropzone_id: uuid.UUID) -> None:
    if db.get(m.Dropzone, dropzone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")


@router.post("/loads", response_model=LoadRead, status_code=status.HTTP_201_CREATED)
def create_load(
    dropzone_id: uuid.UUID, payload: LoadCreate, db: Session = Depends(get_db)
) -> m.Load:
    _require_dropzone(db, dropzone_id)
    load = m.Load(dropzone_id=dropzone_id, name=payload.name, takeoff_at=payload.takeoff_at)
    db.add(load)
    db.flush()
    return load


@router.get("/loads", response_model=list[LoadRead])
def list_loads(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.Load]:
    _require_dropzone(db, dropzone_id)
    return list(db.execute(select(m.Load).where(m.Load.dropzone_id == dropzone_id)).scalars())


@router.post("/customers", response_model=CustomerRead, status_code=status.HTTP_201_CREATED)
def create_customer(
    dropzone_id: uuid.UUID, payload: CustomerCreate, db: Session = Depends(get_db)
) -> m.Customer:
    _require_dropzone(db, dropzone_id)
    cust = m.Customer(dropzone_id=dropzone_id, name=payload.name, contact=payload.contact)
    db.add(cust)
    db.flush()
    return cust


@router.post("/jumps", response_model=JumpRead, status_code=status.HTTP_201_CREATED)
def create_jump(
    dropzone_id: uuid.UUID, payload: JumpCreate, db: Session = Depends(get_db)
) -> m.TandemJump:
    _require_dropzone(db, dropzone_id)
    jump = m.TandemJump(
        dropzone_id=dropzone_id, customer_id=payload.customer_id, load_id=payload.load_id,
        instructor_user_id=payload.instructor_user_id, outside_user_id=payload.outside_user_id,
        video_package=payload.video_package,
    )
    db.add(jump)
    db.flush()
    return jump


@router.get("/jumps", response_model=list[JumpRead])
def list_jumps(dropzone_id: uuid.UUID, db: Session = Depends(get_db)) -> list[m.TandemJump]:
    _require_dropzone(db, dropzone_id)
    return list(
        db.execute(select(m.TandemJump).where(m.TandemJump.dropzone_id == dropzone_id)).scalars()
    )
```

`routes/__init__.py` is empty. To run this task's tests before Task 9 wires `app.py`, temporarily include the two routers by adding, at the end of `create_app` (before `return app`):

```python
    from .routes import dropzones, jumps
    app.include_router(dropzones.router)
    app.include_router(jumps.router)
```

(Task 9 adds the `media` router alongside these.)

- [ ] **Step 6: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_api_manifest.py -q`
Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
git add backend/tandemista/api/schemas.py backend/tandemista/api/routes/__init__.py backend/tandemista/api/routes/dropzones.py backend/tandemista/api/routes/jumps.py backend/tandemista/api/app.py backend/tests/cloud/test_api_manifest.py
git commit -m "feat: manifest API (dropzones, devices, loads, customers, jumps)"
```

---

### Task 9: Manual media upload endpoint

**Files:**
- Create: `backend/tandemista/api/routes/media.py`
- Modify: `backend/tandemista/api/app.py` (include the `media` router; ensure all three routers wired)
- Modify: `backend/tandemista/api/schemas.py` (add `MediaRead`)
- Test: `backend/tests/cloud/test_api_media_upload.py`

**Interfaces:**
- Consumes: `get_db`, `get_storage` (Task 7); models (Task 3); `StorageBackend` (Task 5); the Celery task `analyze_media` (Task 10) — imported lazily inside the handler so this route module does not hard-depend on a running broker.
- Produces:
  - `MediaRead(id, dropzone_id, jump_id, filename, status, locations)` in `schemas.py`.
  - `media.router` (prefix `/dropzones/{dropzone_id}`): `POST /media` — multipart form: `file: UploadFile`, optional form fields `jump_id`, `device_id`. It streams the file to storage under key `dropzones/{dropzone_id}/media/{media_id}/{filename}`, computes sha256, creates a `MediaFile` row with `upload_source_kind = MANUAL_WEB` semantics (record the `StoredObject.location` in `locations`), sets status `REGISTERED`, enqueues `analyze_media.delay(str(media_id))`, and returns `MediaRead` with 201. `GET /media/{media_id}` returns the row (404 if missing).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_api_media_upload.py
import io


def _dropzone(client):
    return client.post("/dropzones/", json={"name": "DZ"}).json()


def test_upload_registers_media_and_stores_file(client, app):
    dz = _dropzone(client)
    resp = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("handcam_001.mp4", io.BytesIO(b"video-bytes"), "video/mp4")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "handcam_001.mp4"
    assert body["status"] == "registered"
    assert len(body["locations"]) == 1

    # The bytes actually landed in the app's storage backend.
    storage = app.state.storage
    key = f"dropzones/{dz['id']}/media/{body['id']}/handcam_001.mp4"
    assert storage.get(key) == b"video-bytes"


def test_get_media_by_id(client):
    dz = _dropzone(client)
    created = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    ).json()
    got = client.get(f"/dropzones/{dz['id']}/media/{created['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == created["id"]


def test_upload_enqueues_analysis(client, monkeypatch):
    calls = {}

    def fake_delay(media_id):
        calls["media_id"] = media_id

    import tandemista.worker.tasks as tasks
    monkeypatch.setattr(tasks.analyze_media, "delay", staticmethod(fake_delay))

    dz = _dropzone(client)
    created = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    ).json()
    assert calls["media_id"] == created["id"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_api_media_upload.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.api.routes.media / tandemista.worker.tasks)

> This task depends on Task 10's `analyze_media`. Implement Task 10 first if executing strictly in order, or stub `analyze_media` now and finish it in Task 10. The recommended order is **Task 10 then Task 9**; the plan lists them in dependency-friendly reading order but they share one commit boundary. If executing 9 before 10, create a minimal `worker/tasks.py` with `analyze_media` (Celery task) returning `None` first.

- [ ] **Step 3: Add `MediaRead` to `schemas.py`**

```python
class MediaRead(_Read):
    id: uuid.UUID
    dropzone_id: uuid.UUID
    jump_id: uuid.UUID | None
    filename: str
    status: str
    locations: list
```

- [ ] **Step 4: Implement `routes/media.py`**

```python
# backend/tandemista/api/routes/media.py
from __future__ import annotations

import hashlib
import io
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ...db import models as m
from ...storage.base import StorageBackend
from ..deps import get_db, get_storage
from ..schemas import MediaRead

router = APIRouter(prefix="/dropzones/{dropzone_id}", tags=["media"])


@router.post("/media", response_model=MediaRead, status_code=status.HTTP_201_CREATED)
def upload_media(
    dropzone_id: uuid.UUID,
    file: UploadFile = File(...),
    jump_id: uuid.UUID | None = Form(default=None),
    device_id: uuid.UUID | None = Form(default=None),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
) -> m.MediaFile:
    if db.get(m.Dropzone, dropzone_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dropzone not found")

    media_id = uuid.uuid4()
    content = file.file.read()
    key = f"dropzones/{dropzone_id}/media/{media_id}/{file.filename}"
    stored = storage.put(key, io.BytesIO(content))

    media = m.MediaFile(
        id=media_id,
        dropzone_id=dropzone_id,
        jump_id=jump_id,
        device_id=device_id,
        filename=file.filename or "upload.bin",
        locations=[stored.location],
        sha256=hashlib.sha256(content).hexdigest(),
        status=m.MediaStatus.REGISTERED,
    )
    db.add(media)
    db.flush()

    from ...worker.tasks import analyze_media

    analyze_media.delay(str(media_id))
    return media


@router.get("/media/{media_id}", response_model=MediaRead)
def get_media(
    dropzone_id: uuid.UUID, media_id: uuid.UUID, db: Session = Depends(get_db)
) -> m.MediaFile:
    media = db.get(m.MediaFile, media_id)
    if media is None or media.dropzone_id != dropzone_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "media not found")
    return media
```

- [ ] **Step 5: Finalize `app.py` router wiring**

Ensure `create_app` includes all three routers exactly once:

```python
    from .routes import dropzones, jumps, media
    app.include_router(dropzones.router)
    app.include_router(jumps.router)
    app.include_router(media.router)
```

- [ ] **Step 6: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_api_media_upload.py -q`
Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add backend/tandemista/api/routes/media.py backend/tandemista/api/schemas.py backend/tandemista/api/app.py backend/tests/cloud/test_api_media_upload.py
git commit -m "feat: manual media upload endpoint (store, register, enqueue analysis)"
```

---

### Task 10: Celery app and analysis task stub

**Files:**
- Create: `backend/tandemista/worker/__init__.py`, `backend/tandemista/worker/celery_app.py`, `backend/tandemista/worker/tasks.py`
- Test: `backend/tests/cloud/test_worker_stub.py`

**Interfaces:**
- Consumes: `get_settings` (Task 1); `configure_session`, `make_engine`, `SessionLocal` (Task 2); models (Task 3).
- Produces:
  - `celery_app.celery` — a `Celery` instance; broker/result backend from `settings.redis_url`; `task_always_eager = settings.celery_task_always_eager`.
  - `tasks.analyze_media(media_id: str) -> str` — a `@celery.task` that loads the `MediaFile`, transitions `REGISTERED -> ANALYZING -> ANALYZED`, and returns the final status string. **Stub only:** no engine call yet (that arrives in the pipeline plan). If the media row is missing it returns `"missing"`; on any exception it sets status `FAILED` and re-raises (nothing is lost silently). The worker binds its own DB session via `configure_session` at import using `get_settings()`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/cloud/test_worker_stub.py
import uuid

import pytest

from tandemista.config import Settings
from tandemista.db.base import Base, configure_session, make_engine, SessionLocal
from tandemista.db import models as m


@pytest.fixture()
def eager_db():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    configure_session(engine)
    return engine


def _make_media(status=m.MediaStatus.REGISTERED) -> uuid.UUID:
    with SessionLocal() as s:
        dz = m.Dropzone(name="DZ")
        s.add(dz)
        s.flush()
        media = m.MediaFile(dropzone_id=dz.id, filename="a.mp4", status=status)
        s.add(media)
        s.commit()
        return media.id


def test_analyze_media_transitions_to_analyzed(eager_db):
    from tandemista.worker.tasks import analyze_media

    media_id = _make_media()
    result = analyze_media.run(str(media_id))
    assert result == "analyzed"
    with SessionLocal() as s:
        assert s.get(m.MediaFile, media_id).status == m.MediaStatus.ANALYZED


def test_analyze_media_missing_row(eager_db):
    from tandemista.worker.tasks import analyze_media

    assert analyze_media.run(str(uuid.uuid4())) == "missing"


def test_celery_configured_eager_from_settings():
    from tandemista.worker.celery_app import make_celery

    celery = make_celery(Settings(celery_task_always_eager=True))
    assert celery.conf.task_always_eager is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && pytest tests/cloud/test_worker_stub.py -q`
Expected: FAIL (ModuleNotFoundError: tandemista.worker.celery_app)

- [ ] **Step 3: Implement `celery_app.py`**

```python
# backend/tandemista/worker/celery_app.py
from __future__ import annotations

from celery import Celery

from ..config import Settings, get_settings
from ..db.base import configure_session, make_engine


def make_celery(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    celery = Celery(
        "tandemista",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    celery.conf.task_always_eager = settings.celery_task_always_eager
    celery.conf.task_serializer = "json"
    celery.conf.result_serializer = "json"
    celery.conf.accept_content = ["json"]
    # Bind the worker's DB session factory to the configured database.
    configure_session(make_engine(settings.database_url))
    return celery


celery = make_celery()
```

- [ ] **Step 4: Implement `tasks.py`**

```python
# backend/tandemista/worker/tasks.py
from __future__ import annotations

from ..db.base import SessionLocal
from ..db import models as m
from .celery_app import celery


@celery.task(name="tandemista.analyze_media")
def analyze_media(media_id: str) -> str:
    """Stub pipeline: mark media ANALYZING then ANALYZED.

    The real engine call (telemetry -> timeline -> EDL -> render) lands in the
    pipeline plan. Failures set status FAILED and re-raise: material is never
    dropped silently.
    """
    with SessionLocal() as session:
        media = session.get(m.MediaFile, media_id)
        if media is None:
            return "missing"
        try:
            media.status = m.MediaStatus.ANALYZING
            session.commit()
            # placeholder for analysis work
            media.status = m.MediaStatus.ANALYZED
            session.commit()
            return media.status.value
        except Exception:
            media.status = m.MediaStatus.FAILED
            session.commit()
            raise
```

`worker/__init__.py` is empty.

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && pytest tests/cloud/test_worker_stub.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/tandemista/worker/__init__.py backend/tandemista/worker/celery_app.py backend/tandemista/worker/tasks.py backend/tests/cloud/test_worker_stub.py
git commit -m "feat: celery app and analyze_media task stub (status transitions)"
```

---

### Task 11: Docker Compose, run docs, and full-suite green

**Files:**
- Create: `backend/docker-compose.yml`
- Modify: `README.md` (add a "Cloud backend (local dev)" section)
- Test: run the whole suite; no new test file, but add `backend/tests/cloud/test_app_smoke.py` exercising create_app end-to-end.

**Interfaces:**
- Consumes: everything above.
- Produces: a `docker compose up` that starts Postgres, Redis, MinIO; documented commands to run migrations, the API, and a worker.

- [ ] **Step 1: Write the smoke test**

```python
# backend/tests/cloud/test_app_smoke.py
import io


def test_end_to_end_upload_flow(client, app):
    dz = client.post("/dropzones/", json={"name": "DZ"}).json()
    client.post(
        f"/dropzones/{dz['id']}/devices",
        json={"name": "H12", "kind": "gopro", "role": "handcam"},
    )
    load = client.post(f"/dropzones/{dz['id']}/loads", json={"name": "L1"}).json()
    cust = client.post(f"/dropzones/{dz['id']}/customers", json={"name": "Jane"}).json()
    jump = client.post(
        f"/dropzones/{dz['id']}/jumps",
        json={"customer_id": cust["id"], "load_id": load["id"]},
    ).json()
    media = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("handcam_001.mp4", io.BytesIO(b"bytes"), "video/mp4")},
        data={"jump_id": jump["id"]},
    )
    assert media.status_code == 201
    assert media.json()["jump_id"] == jump["id"]
```

- [ ] **Step 2: Run to verify it passes** (all prior tasks make this green immediately)

Run: `cd backend && pytest tests/cloud/test_app_smoke.py -q`
Expected: `1 passed`

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
# backend/docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: tandemista
      POSTGRES_PASSWORD: tandemista
      POSTGRES_DB: tandemista
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - miniodata:/data

volumes:
  pgdata:
  miniodata:
```

- [ ] **Step 4: Update `README.md`**

Add this section after the engine usage section:

```markdown
## Cloud backend (local dev)

    cd backend
    pip install -e ".[cloud,dev]"
    cp .env.example .env            # adjust as needed
    docker compose up -d            # postgres + redis + minio
    alembic upgrade head            # create schema
    uvicorn tandemista.api.app:create_app --factory --reload   # API on :8000
    celery -A tandemista.worker.celery_app.celery worker -l info   # worker

The manual upload endpoint is `POST /dropzones/{dropzone_id}/media` (multipart).
Analysis currently runs a stub task; the real engine pipeline lands in a follow-up plan.
```

- [ ] **Step 5: Run the whole cloud suite**

Run: `cd backend && pytest tests/cloud -q`
Expected: all pass (S3 tests skip if MinIO/moto absent). Then run the full suite to confirm no engine regressions: `pytest -q` → prior engine tests still green.

- [ ] **Step 6: Commit**

```bash
git add backend/docker-compose.yml README.md backend/tests/cloud/test_app_smoke.py
git commit -m "feat: docker-compose dev stack, run docs, end-to-end smoke test"
```

---

## Self-Review

**Spec coverage (design §Components, §Data model, §Стек):**
- FastAPI API server — Tasks 7–9. ✅
- PostgreSQL data model (Dropzone, User, Load, Customer, TandemJump, Device, UploadSource, MediaFile, JumpCut, Order) — Task 3. ✅
- StorageBackend abstraction with s3 + network_drive/local — Tasks 5–6. ✅ (`yandex_disk` REST backend deferred to a follow-up; the Protocol makes it additive.)
- Celery + Redis queue — Task 10 (pipeline steps as real engine calls deferred to the pipeline plan). ✅ scaffold.
- Manual `manual_web` upload path (Этап 1 entry point) — Task 9. ✅
- Docker Compose (Postgres, Redis, MinIO) — Task 11. ✅
- Single-tenant with `dropzone_id` everywhere — Task 3 (every table has `dropzone_id`). ✅
- **Deferred to follow-up plans (explicitly out of scope here):** analysis pipeline wiring to `tandemista.engine`, JumpCut draft generation, proxy/final render tasks, review admin UI + client page (Next.js), matching, delivery, payments. Listed so the reader knows these are intentional gaps, not misses.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every step has concrete code. The one intentional stub (`analyze_media`) is labelled as such and has a real, tested behaviour (status transitions).

**Type consistency:** `Settings` fields used identically across config/factory/celery; `MediaStatus`/`CutVariant` enums used consistently in models, schemas, tasks; `StorageBackend.put(key, data) -> StoredObject` signature matches in local, s3, and the upload route; `analyze_media(media_id: str)` matches its `.delay(str(media_id))` call site in Task 9.

**Cross-task ordering note:** Tasks 9 and 10 are mutually referencing (upload enqueues `analyze_media`). Execute **Task 10 before Task 9**, or create the `analyze_media` stub during Task 9 step 2 as noted. All other tasks are strictly linear.
