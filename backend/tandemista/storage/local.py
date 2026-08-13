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
