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
