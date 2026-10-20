"""First-party usage tracking (SPEC.md "Usage tracking").

Stores no raw IPs, user agents or project IDs: visitors and projects are
hashed with a random salt per UTC day, and past salts are deleted, so hashes
can't be linked across days or reversed. Recording never affects a response.
"""
import hashlib
import logging
import re
import secrets
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from flask import request
from sqlalchemy.exc import IntegrityError

from app.config import RETENTION_DAYS
from app.models.db import new_session
from app.models.schema import Project, UsageDaily, UsageEvent, UsageSalt

log = logging.getLogger(__name__)

NOTRACK_COOKIE = "gel_notrack"
BOT_UA = re.compile(r"bot|crawl|spider|slurp|headless|preview|monitor|curl|wget|python-requests", re.I)
EXPORT_PATH = re.compile(r"^/api/projects/[^/]+/export/(csv|png|pdf)$")
FAILURE_REASONS = {413: "too_large", 415: "not_image", 429: "rate_limited"}
PAGES = ("upload", "project")
MAX_REFERRER = 300


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).replace(tzinfo=None)


def day_of(now: datetime | None = None) -> str:
    return _now(now).strftime("%Y-%m-%d")


def _salt(session, day: str) -> str:
    row = session.get(UsageSalt, day)
    if row is None:
        # Rotate here so past salts never outlive midnight, whether or not the
        # maintenance thread is running.
        session.query(UsageSalt).filter(UsageSalt.day < day).delete()
        try:
            session.add(UsageSalt(day=day, salt=secrets.token_hex(32)))
            session.commit()
        except IntegrityError:
            session.rollback()
        row = session.get(UsageSalt, day)
    return row.salt


def _hash(salt: str, *parts: str) -> str:
    return hashlib.sha256("\n".join((salt, *parts)).encode()).hexdigest()


def opted_out() -> bool:
    return (
        request.cookies.get(NOTRACK_COOKIE) == "1"
        or request.args.get("notrack") == "1"
        or request.headers.get("Sec-GPC") == "1"
        or request.headers.get("DNT") == "1"
    )


def clean_referrer(referrer, own_host: str) -> str:
    """Host only: paths, queries and fragments can carry tokens, usernames or
    document IDs. Internal navigation is stored as empty."""
    try:
        parts = urlsplit(referrer if isinstance(referrer, str) else "")
        host = parts.hostname
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not host:
        return ""
    if host == urlsplit("//" + own_host).hostname:
        return ""
    return host[:MAX_REFERRER]


def _record(kind: str, detail: str | None = None, project_id: str | None = None, now: datetime | None = None) -> None:
    if opted_out():
        return
    day = day_of(now)
    with new_session() as session:
        salt = _salt(session, day)
        visitor = _hash(salt, request.remote_addr or "", request.user_agent.string or "")
        session.add(
            UsageEvent(
                created_at=_now(now),
                day=day,
                kind=kind,
                visitor_hash=visitor,
                detail=detail,
                project_hash=_hash(salt, project_id) if project_id else None,
            )
        )
        session.commit()


def record(kind: str, detail: str | None = None, project_id: str | None = None, now: datetime | None = None) -> None:
    try:
        _record(kind, detail, project_id, now)
    except Exception:
        log.exception("usage tracking failed")


def record_visit(payload, now: datetime | None = None) -> None:
    """Beacon from the upload or project page. The beacon request's own Referer
    is our page, so the page sends document.referrer in the body."""
    try:
        if BOT_UA.search(request.user_agent.string or ""):
            return
        payload = payload if isinstance(payload, dict) else {}
        page = payload.get("page") if payload.get("page") in PAGES else "unknown"
        referrer = clean_referrer(payload.get("referrer"), request.host)
        # One detail column holds both: page names never contain "|".
        record("visit", detail=f"{page}|{referrer}", now=now)

        project_id = payload.get("project_id")
        if page != "project" or not isinstance(project_id, str):
            return
        with new_session() as session:
            project = session.get(Project, project_id)
            created = project.created_at.strftime("%Y-%m-%d") if project else None
        if created and created < day_of(now):
            record("project_return", project_id=project_id, now=now)
    except Exception:
        log.exception("usage tracking failed")


def track_response(response):
    """after_request hook: sees the final status of uploads (including 413 and
    429 produced outside the route) and exports, and handles ?notrack=."""
    try:
        status = response.status_code
        if request.method == "POST" and request.path == "/api/projects":
            if status == 201:
                record("upload", detail=str(len(request.files.getlist("images"))))
            elif status >= 400:
                record("upload_failed", detail=FAILURE_REASONS.get(status, "server_error" if status >= 500 else "bad_request"))
        elif request.method == "GET" and status == 200:
            match = EXPORT_PATH.match(request.path)
            if match:
                record("export", detail=match.group(1))

        notrack = request.args.get("notrack")
        if notrack == "1":
            response.set_cookie(NOTRACK_COOKIE, "1", max_age=365 * 24 * 3600, samesite="Lax", httponly=True, secure=request.is_secure)
        elif notrack == "0":
            response.delete_cookie(NOTRACK_COOKIE, samesite="Lax", httponly=True, secure=request.is_secure)
    except Exception:
        log.exception("usage tracking failed")
    return response


# ------------------------------------------------------------------ rollup --

def day_totals(session, day: str) -> dict[tuple[str, str], int]:
    totals: Counter = Counter()
    visitors, returning = set(), set()
    for event in session.query(UsageEvent).filter_by(day=day):
        detail = event.detail or ""
        if event.kind == "visit":
            page, _, referrer = detail.partition("|")
            totals["visits", page] += 1
            visitors.add(event.visitor_hash)
            if referrer:
                totals["referrer", referrer] += 1
        elif event.kind == "upload":
            totals["uploads", ""] += 1
            totals["images_uploaded", ""] += int(detail or 0)
        elif event.kind == "upload_failed":
            totals["upload_failed", detail] += 1
        elif event.kind == "export":
            totals["exports", detail] += 1
        elif event.kind == "project_return":
            returning.add(event.project_hash)
    if visitors:
        totals["visitors", ""] = len(visitors)
    if returning:
        totals["returning_projects", ""] = len(returning)
    return dict(totals)


def rollup_day(session, day: str) -> None:
    session.query(UsageDaily).filter_by(day=day).delete()
    for (metric, key), count in day_totals(session, day).items():
        session.add(UsageDaily(day=day, metric=metric, key=key, count=count))


def rollup(now: datetime | None = None) -> None:
    """Totals for every completed UTC day not yet rolled up, then drop raw rows
    past retention and salts for past days. Safe to run any number of times."""
    today = day_of(now)
    cutoff = day_of(_now(now) - timedelta(days=RETENTION_DAYS))
    with new_session() as session:
        done = {d for (d,) in session.query(UsageDaily.day).distinct()}
        pending = {d for (d,) in session.query(UsageEvent.day).filter(UsageEvent.day < today).distinct()}
        for day in sorted(pending - done):
            rollup_day(session, day)
        session.query(UsageEvent).filter(UsageEvent.day < cutoff).delete()
        session.query(UsageSalt).filter(UsageSalt.day < today).delete()
        session.commit()
