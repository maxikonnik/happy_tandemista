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
