from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from app.config import DB_URI
from app.models.schema import Band, Base, GelImage, Lane, Project, _now

_engine = create_engine(DB_URI, connect_args={"check_same_thread": False})
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
_ScopedSession = scoped_session(_SessionFactory)


def _project_of(obj):
    if isinstance(obj, Project):
        return obj
    if isinstance(obj, GelImage):
        return obj.project
    if isinstance(obj, Lane):
        return obj.image.project if obj.image else None
    if isinstance(obj, Band):
        return obj.lane.image.project if obj.lane and obj.lane.image else None
    return None


@event.listens_for(Session, "before_flush")
def _touch_projects(session, _flush_context, _instances):
    """Edits change lanes and bands, not the project row, so project.updated_at
    wouldn't move; retention deletes by it, so bump it on any related change."""
    for obj in list(session.new) + list(session.dirty) + list(session.deleted):
        project = _project_of(obj)
        if project is not None and project not in session.deleted:
            project.updated_at = _now()


# create_all makes missing tables but never adds columns to existing ones,
# and there is no migration tool: columns added after launch go here.
_ADDED_COLUMNS = {"gel_images": {"auto_cropped": "BOOLEAN NOT NULL DEFAULT 0"}}


def init_db() -> None:
    Base.metadata.create_all(_engine)
    with _engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {c["name"] for c in inspect(conn).get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def get_session():
    return _ScopedSession()


def new_session() -> Session:
    """A session independent of the request's scoped one, so usage tracking
    can commit or fail without touching the request's transaction."""
    return _SessionFactory()


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
