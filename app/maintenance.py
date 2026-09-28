"""Deletes projects (rows and uploaded files) untouched for RETENTION_DAYS."""
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone

from app.config import RETENTION_DAYS, UPLOADS_DIR
from app.models.db import session_scope
from app.models.schema import Project


def purge_stale_projects(now: datetime | None = None) -> int:
    # SQLite hands back naive datetimes, so compare in naive UTC.
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    cutoff = now - timedelta(days=RETENTION_DAYS)
    with session_scope() as session:
        stale = [p for p in session.query(Project).all() if p.updated_at.replace(tzinfo=None) < cutoff]
        for project in stale:
            shutil.rmtree(os.path.join(UPLOADS_DIR, project.id), ignore_errors=True)
            session.delete(project)
        return len(stale)


def start_purge_thread(interval_seconds: int = 3600) -> None:
    def loop():
        while True:
            purge_stale_projects()
            stop.wait(interval_seconds)

    stop = threading.Event()
    threading.Thread(target=loop, name="purge-stale-projects", daemon=True).start()
