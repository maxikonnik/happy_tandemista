import uuid

from sqlalchemy import Column, String, select
from sqlalchemy.orm import DeclarativeBase, Session

from tandemista.db.base import Base, configure_session, make_engine, make_session_factory
from tandemista.db.types import GUID, JSONColumn


class _TestBase(DeclarativeBase):
    pass


class _Row(_TestBase):
    __tablename__ = "t_row"
    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    meta = Column(JSONColumn, nullable=False, default=dict)


def test_guid_and_json_roundtrip_on_sqlite():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    _TestBase.metadata.create_all(engine, tables=[_Row.__table__])
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
