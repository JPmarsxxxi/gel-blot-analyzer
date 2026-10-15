"""Print usage stats. Run: python -m app.stats [--days N]

On the Oracle server: sudo docker exec gel-app python -m app.stats
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

from app.models.db import init_db, new_session
from app.models.schema import UsageDaily, UsageEvent
from app.tracking import day_of, day_totals


def collect() -> dict[str, dict[tuple[str, str], int]]:
    """Rolled-up totals, plus totals computed from raw rows for days not yet
    rolled up (today, at least), so the numbers are current."""
    by_day: dict[str, dict[tuple[str, str], int]] = {}
    with new_session() as session:
        for row in session.query(UsageDaily):
            by_day.setdefault(row.day, {})[row.metric, row.key] = row.count
        raw_days = {d for (d,) in session.query(UsageEvent.day).distinct()}
        for day in raw_days - by_day.keys():
            by_day[day] = day_totals(session, day)
    return by_day


def _by_key(totals: dict, metric: str) -> str:
    parts = [f"{key} {count}" for (m, key), count in sorted(totals.items()) if m == metric]
    return ", ".join(parts) or "0"


def _sum(totals: dict, metric: str) -> int:
    return sum(count for (m, _), count in totals.items() if m == metric)


def report(days: int, now: datetime | None = None) -> str:
    by_day = collect()
    if not by_day:
        return "No usage recorded yet."
    first = day_of((now or datetime.now(timezone.utc)) - timedelta(days=days - 1))
    recent = sorted((d for d in by_day if d >= first), reverse=True)

    lines = [f"Last {days} days (UTC), newest first:"]
    for day in recent:
        t = by_day[day]
        lines.append(
            f"{day}  visitors {_sum(t, 'visitors')}  visits {_sum(t, 'visits')}  uploads {_sum(t, 'uploads')}"
            f"  images {_sum(t, 'images_uploaded')}  exports {_by_key(t, 'exports')}"
            f"  failed uploads {_by_key(t, 'upload_failed')}  returning projects {_sum(t, 'returning_projects')}"
        )
    if not recent:
        lines.append("(nothing in this period)")

    referrers: Counter = Counter()
    for day in recent:
        for (metric, key), count in by_day[day].items():
            if metric == "referrer":
                referrers[key] += count
    lines.append("")
    lines.append("Top referrers:")
    lines += [f"  {count:5d}  {ref}" for ref, count in referrers.most_common(10)] or ["  (none)"]

    total: Counter = Counter()
    for t in by_day.values():
        for (metric, _), count in t.items():
            total[metric] += count
    lines.append("")
    lines.append(
        f"All time: visitor-days {total['visitors']}, visits {total['visits']}, uploads {total['uploads']},"
        f" images {total['images_uploaded']}, exports {total['exports']}, failed uploads {total['upload_failed']},"
        f" returning project-days {total['returning_projects']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    init_db()
    print(report(args.days))
