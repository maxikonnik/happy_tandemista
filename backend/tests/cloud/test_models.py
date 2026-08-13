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
