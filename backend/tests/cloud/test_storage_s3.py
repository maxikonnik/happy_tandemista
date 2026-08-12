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
