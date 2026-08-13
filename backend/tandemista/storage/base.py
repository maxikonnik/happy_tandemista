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
