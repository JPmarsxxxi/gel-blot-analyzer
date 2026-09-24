from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from app.config import DB_URI
from app.models.schema import Base

_engine = create_engine(DB_URI, connect_args={"check_same_thread": False})
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
_ScopedSession = scoped_session(_SessionFactory)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def get_session():
    return _ScopedSession()


def remove_session() -> None:
    _ScopedSession.remove()


@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
