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
