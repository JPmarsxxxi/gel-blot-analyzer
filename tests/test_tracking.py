import io
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text, update

from app import tracking
from app.config import BASE_DIR, UPLOAD_RATE_LIMIT
from app.models.db import _engine, init_db, session_scope
from app.models.schema import Project, UsageDaily, UsageEvent, UsageSalt
from app.server import create_app
from app.stats import report
from tests.conftest import synth_png_bytes

IP = "198.51.100.23"
UA = "Mozilla/5.0 (TestBrowser 1.0)"
ENV = {"REMOTE_ADDR": IP, "HTTP_USER_AGENT": UA}
DAY1 = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


@pytest.fixture(autouse=True)
def empty_usage_tables():
    init_db()
    with session_scope() as session:
        for table in (UsageEvent, UsageDaily, UsageSalt):
            session.query(table).delete()
    yield


def events(kind=None):
    with session_scope() as session:
        q = session.query(UsageEvent)
        rows = q.filter_by(kind=kind).all() if kind else q.all()
        session.expunge_all()
        return rows


def upload(client, n=1, **kw):
    files = [(io.BytesIO(synth_png_bytes(i + 1)), f"g{i}.png") for i in range(n)]
    return client.post("/api/projects", data={"images": files}, content_type="multipart/form-data", environ_base=ENV, **kw)


def beacon(client, **body):
    return client.post("/api/visit", json=body, environ_base=ENV)


# T2
def test_pages_send_a_visit_beacon(client):
    for page, script in (("upload", "app/static/js/upload.js"), ("project", "app/static/js/project.js")):
        src = open(os.path.join(BASE_DIR, script), encoding="utf-8").read()
        assert src.count('fetch("/api/visit"') == 1
        assert f'page: "{page}"' in src
    assert "js/upload.js" in client.get("/").get_data(as_text=True)
    assert "js/project.js" in client.get("/p/abc").get_data(as_text=True)


def test_beacon_records_one_visit_and_ignores_bots(client):
    resp = beacon(client, page="upload", referrer="")
    assert resp.status_code == 204 and resp.data == b""
    assert len(events("visit")) == 1
    bot = client.post("/api/visit", json={"page": "upload"}, environ_base={"HTTP_USER_AGENT": "Mozilla/5.0 (compatible; Googlebot/2.1)"})
    assert bot.status_code == 204
    assert len(events("visit")) == 1


def test_bad_beacon_body_still_records_unknown_visit(client):
    resp = client.post("/api/visit", data="not json", content_type="application/json", environ_base=ENV)
    assert resp.status_code == 204
    assert events("visit")[0].detail == "unknown|"


# T3
def test_daily_unique_visitors(client):
    with client.application.test_request_context(environ_base=ENV):
        tracking.record_visit({"page": "upload"}, now=DAY1)
        tracking.record_visit({"page": "upload"}, now=DAY1)
        tracking.record_visit({"page": "upload"}, now=DAY2)
    with session_scope() as session:
        assert tracking.day_totals(session, "2026-03-01")["visitors", ""] == 1
        assert tracking.day_totals(session, "2026-03-01")["visits", "upload"] == 2
    day1, _, day2 = events("visit")
    assert day1.visitor_hash != day2.visitor_hash


# T4
def test_no_raw_ip_user_agent_or_project_id_is_stored(client):
    pid = upload(client).get_json()["project_id"]
    with session_scope() as session:
        session.execute(update(Project).where(Project.id == pid).values(created_at=datetime.now(timezone.utc) - timedelta(days=2)))
    beacon(client, page="project", referrer="https://example.org/x", project_id=pid)
    for fmt in ("csv", "png", "pdf"):
        client.get(f"/api/projects/{pid}/export/{fmt}", environ_base=ENV)
    client.post("/api/projects", data={"images": (io.BytesIO(b"nope"), "x.png")}, content_type="multipart/form-data", environ_base=ENV)
    assert {e.kind for e in events()} == {"visit", "upload", "export", "upload_failed", "project_return"}
    tracking.rollup(now=datetime.now(timezone.utc) + timedelta(days=1))

    with _engine.connect() as conn:
        for table in ("usage_events", "usage_daily", "usage_salt"):
            for row in conn.execute(text(f"select * from {table}")):
                blob = " ".join(str(v) for v in row)
                assert IP not in blob and UA not in blob and pid not in blob


# T5
def test_referrer_is_cleaned():
    assert tracking.clean_referrer("https://www.reddit.com/r/labrats/comments/abc?utm=x#top", "gels.example") == "www.reddit.com"
    assert tracking.clean_referrer("https://gels.example/p/123", "gels.example") == ""
    assert tracking.clean_referrer("http://localhost:5000/", "localhost:5000") == ""
    assert tracking.clean_referrer("javascript:alert(1)", "gels.example") == ""
    assert tracking.clean_referrer("https://user:pw@site.org/a", "gels.example") == "site.org"
    assert tracking.clean_referrer("http://[bad", "gels.example") == ""
    assert tracking.clean_referrer(None, "gels.example") == ""
    assert tracking.clean_referrer("https://a.org/reset/secret-token", "gels.example") == "a.org"


def test_beacon_referrer_on_own_host_is_empty(client):
    beacon(client, page="upload", referrer="http://localhost/p/x")
    beacon(client, page="upload", referrer="https://www.reddit.com/r/labrats/?q=1")
    assert [e.detail for e in events("visit")] == ["upload|", "upload|www.reddit.com"]


# T6
def test_upload_and_exports_are_recorded(client):
    resp = upload(client, n=3)
    assert resp.status_code == 201
    [up] = events("upload")
    assert up.detail == "3"
    pid = resp.get_json()["project_id"]
    for fmt in ("csv", "png", "pdf"):
        assert client.get(f"/api/projects/{pid}/export/{fmt}", environ_base=ENV).status_code == 200
    assert sorted(e.detail for e in events("export")) == ["csv", "pdf", "png"]
    client.get("/api/projects/missing/export/csv", environ_base=ENV)
    assert len(events("export")) == 3


# T7
def test_failed_uploads_are_recorded_with_reason(client):
    resp = client.post("/api/projects", data={"images": (io.BytesIO(b"not an image"), "fake.png")}, content_type="multipart/form-data", environ_base=ENV)
    assert resp.status_code == 415 and "not a valid image" in resp.get_json()["error"]
    resp = client.post("/api/projects", data={"images": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data", environ_base=ENV)
    assert resp.status_code == 415
    resp = client.post("/api/projects", data={}, environ_base=ENV)
    assert resp.status_code == 400
    assert [e.detail for e in events("upload_failed")] == ["not_image", "not_image", "bad_request"]


def test_too_large_upload_is_recorded():
    app = create_app()
    app.testing = True
    app.config["RATE_LIMITS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 1000
    with app.test_client() as c:
        resp = c.post("/api/projects", data={"images": (io.BytesIO(b"\0" * 5000), "big.png")}, content_type="multipart/form-data", environ_base=ENV)
    assert resp.status_code == 413 and "too large" in resp.get_json()["error"]
    assert [e.detail for e in events("upload_failed")] == ["too_large"]


def test_rate_limited_upload_is_recorded():
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        env = {"REMOTE_ADDR": "203.0.113.99"}
        codes = [c.post("/api/projects", environ_base=env).status_code for _ in range(UPLOAD_RATE_LIMIT[0] + 1)]
    assert codes[-1] == 429
    reasons = [e.detail for e in events("upload_failed")]
    assert reasons.count("rate_limited") == 1 and reasons.count("bad_request") == UPLOAD_RATE_LIMIT[0]


# T8
def test_returning_projects(client):
    old = upload(client).get_json()["project_id"]
    new = upload(client).get_json()["project_id"]
    with session_scope() as session:
        session.execute(update(Project).where(Project.id == old).values(created_at=datetime.now(timezone.utc) - timedelta(days=3)))
    beacon(client, page="project", project_id=old)
    beacon(client, page="project", project_id=old)
    beacon(client, page="project", project_id=new)
    beacon(client, page="project", project_id="does-not-exist")
    returns = events("project_return")
    assert len(returns) == 2 and len(events("visit")) == 4
    with session_scope() as session:
        assert tracking.day_totals(session, tracking.day_of())["returning_projects", ""] == 1


# T9
def test_notrack_cookie_stops_all_recording(client):
    resp = client.get("/?notrack=1", environ_base=ENV)
    cookie = resp.headers["Set-Cookie"]
    assert "gel_notrack=1" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Max-Age=31536000" in cookie
    beacon(client, page="upload")
    pid = upload(client).get_json()["project_id"]
    client.get(f"/api/projects/{pid}/export/csv", environ_base=ENV)
    client.post("/api/projects", data={"images": (io.BytesIO(b"no"), "x.png")}, content_type="multipart/form-data", environ_base=ENV)
    assert events() == []

    resp = client.get("/?notrack=0", environ_base=ENV)
    assert "gel_notrack=;" in resp.headers["Set-Cookie"]
    beacon(client, page="upload")
    assert len(events("visit")) == 1


# T10
def test_rollup_is_correct_idempotent_and_prunes():
    app = create_app()
    old = DAY1 - timedelta(days=40)
    with app.test_request_context(environ_base=ENV):
        tracking.record_visit({"page": "upload", "referrer": "https://site.org/a"}, now=DAY1)
        tracking.record_visit({"page": "project"}, now=DAY1)
        tracking.record("upload", detail="2", now=DAY1)
        tracking.record("export", detail="pdf", now=DAY1)
        tracking.record("upload_failed", detail="not_image", now=DAY1)
        tracking.record("visit", detail="upload|", now=old)
        tracking.record("visit", detail="upload|", now=DAY2)
    expected = {
        ("visits", "upload"): 1, ("visits", "project"): 1, ("visitors", ""): 1, ("referrer", "site.org"): 1,
        ("uploads", ""): 1, ("images_uploaded", ""): 2, ("exports", "pdf"): 1, ("upload_failed", "not_image"): 1,
    }

    def daily(day):
        with session_scope() as session:
            return {(r.metric, r.key): r.count for r in session.query(UsageDaily).filter_by(day=day)}

    tracking.rollup(now=DAY2)
    tracking.rollup(now=DAY2)
    assert daily("2026-03-01") == expected
    assert daily(tracking.day_of(old)) == {("visits", "upload"): 1, ("visitors", ""): 1}
    assert daily("2026-03-02") == {}
    with session_scope() as session:
        assert session.query(UsageEvent).filter_by(day=tracking.day_of(old)).count() == 0
        assert session.query(UsageEvent).filter_by(day="2026-03-01").count() == 5
        assert [s.day for s in session.query(UsageSalt)] == ["2026-03-02"]
        tracking.rollup_day(session, "2026-03-01")
    assert daily("2026-03-01") == expected


# T11
def test_tracking_failure_never_breaks_requests(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("usage table missing")

    monkeypatch.setattr(tracking, "_record", boom)
    resp = upload(client)
    assert resp.status_code == 201
    pid = resp.get_json()["project_id"]
    assert client.get(f"/api/projects/{pid}/export/csv").status_code == 200
    assert client.get("/").status_code == 200
    assert beacon(client, page="project", project_id=pid).status_code == 204

    monkeypatch.undo()
    with _engine.begin() as conn:
        conn.execute(text("drop table usage_events"))
    try:
        assert upload(client).status_code == 201
        assert beacon(client, page="upload").status_code == 204
    finally:
        init_db()


# T12
def test_stats_cli(client):
    assert report(30) == "No usage recorded yet."
    beacon(client, page="upload", referrer="https://www.reddit.com/r/labrats/")
    upload(client, n=2)
    out = report(30)
    today = tracking.day_of()
    assert re.search(rf"{today}  visitors 1  visits 1  uploads 1  images 2", out)
    assert "www.reddit.com" in out
    assert "All time: visitor-days 1, visits 1, uploads 1, images 2" in out

    run = subprocess.run([sys.executable, "-m", "app.stats", "--days", "7"], cwd=BASE_DIR, capture_output=True, text=True,
                         env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
    assert run.returncode == 0 and "All time:" in run.stdout


# T13
def test_no_route_returns_usage_data(client):
    assert client.get("/stats").status_code == 404
    assert client.get("/api/stats").status_code == 404
    assert client.get("/api/visit").status_code == 405
    for name in os.listdir(os.path.join(BASE_DIR, "app", "templates")):
        src = open(os.path.join(BASE_DIR, "app", "templates", name), encoding="utf-8").read()
        assert "usage" not in src.lower()


# T14
def test_no_third_party_requests_in_frontend():
    url = re.compile(r"https?://[^\s\"'<>)]+")
    for folder in ("templates", os.path.join("static", "js")):
        root = os.path.join(BASE_DIR, "app", folder)
        for name in os.listdir(root):
            src = open(os.path.join(root, name), encoding="utf-8").read()
            for found in url.findall(src):
                assert found.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com", "http://www.w3.org", "https://github.com")), (name, found)


# T15
def test_privacy_note_and_docs(client):
    page = client.get("/").get_data(as_text=True)
    assert "anonymous daily visits" in page and "no IP addresses are stored" in page and "opt-out" in page
    doc = open(os.path.join(BASE_DIR, "deploy", "oracle.md"), encoding="utf-8").read()
    assert "sudo docker exec gel-app python -m app.stats" in doc and "?notrack=1" in doc


def test_usage_tables_exist():
    tables = set(inspect(_engine).get_table_names())
    assert {"usage_events", "usage_daily", "usage_salt"} <= tables


def test_gpc_and_dnt_headers_stop_recording(client):
    for header in ("Sec-GPC", "DNT"):
        client.post("/api/visit", json={"page": "upload"}, headers={header: "1"}, environ_base=ENV)
    assert events() == []


def test_new_day_salt_deletes_past_salts():
    from app.models.db import new_session

    with new_session() as session:
        tracking._salt(session, "2026-03-01")
        tracking._salt(session, "2026-03-02")
        assert [s.day for s in session.query(UsageSalt)] == ["2026-03-02"]
