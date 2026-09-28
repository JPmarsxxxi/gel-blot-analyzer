import io
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.config import MAX_FILE_BYTES, MAX_FILES_PER_UPLOAD, UPLOAD_RATE_LIMIT, UPLOADS_DIR
from app.maintenance import purge_stale_projects
from app.models.db import session_scope
from app.models.schema import Project
from app.server import create_app
from tests.conftest import synth_png_bytes


def test_rejects_more_files_than_the_limit(client):
    files = [(io.BytesIO(synth_png_bytes(1)), f"g{i}.png") for i in range(MAX_FILES_PER_UPLOAD + 1)]
    resp = client.post("/api/projects", data={"images": files}, content_type="multipart/form-data")
    assert resp.status_code == 413
    assert "at most" in resp.get_json()["error"]


def test_rejects_a_file_over_the_size_limit(client):
    big = io.BytesIO(b"\0" * (MAX_FILE_BYTES + 1))
    resp = client.post("/api/projects", data={"images": (big, "huge.png")}, content_type="multipart/form-data")
    assert resp.status_code == 413
    assert "MB" in resp.get_json()["error"]


def test_upload_rate_limit_returns_429_with_retry():
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        limit = UPLOAD_RATE_LIMIT[0]
        codes = [c.post("/api/projects", environ_base={"REMOTE_ADDR": "203.0.113.7"}).status_code for _ in range(limit + 1)]
        assert 429 not in codes[:limit]
        assert codes[-1] == 429
        other = c.post("/api/projects", environ_base={"REMOTE_ADDR": "203.0.113.8"})
        assert other.status_code != 429


def test_stale_projects_are_purged_and_edits_keep_projects_alive(client):
    data = {"images": (io.BytesIO(synth_png_bytes(4)), "gel.png")}
    stale_id = client.post("/api/projects", data=data, content_type="multipart/form-data").get_json()["project_id"]
    data = {"images": (io.BytesIO(synth_png_bytes(5)), "gel.png")}
    live_id = client.post("/api/projects", data=data, content_type="multipart/form-data").get_json()["project_id"]

    # A Core UPDATE bypasses the before_flush hook that would re-stamp the rows.
    long_ago = datetime.now(timezone.utc) - timedelta(days=45)
    with session_scope() as session:
        session.execute(update(Project).where(Project.id.in_([stale_id, live_id])).values(updated_at=long_ago))

    lane = client.get(f"/api/projects/{live_id}").get_json()["images"][0]["lanes"][0]
    client.patch(f"/api/lanes/{lane['id']}", json={"label": "still in use"})

    purge_stale_projects()
    assert client.get(f"/api/projects/{stale_id}").status_code == 404
    assert not os.path.exists(os.path.join(UPLOADS_DIR, stale_id))
    assert client.get(f"/api/projects/{live_id}").status_code == 200


def test_production_app_has_debug_off():
    from app.wsgi import app

    assert not app.debug
