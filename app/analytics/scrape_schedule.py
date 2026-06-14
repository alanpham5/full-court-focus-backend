
from __future__ import annotations

from datetime import datetime, timezone

SCHEDULE_MONTHS = (11, 12, 1, 2, 3, 4)
SCHEDULE_DAYS = (1, 15)
SCHEDULE_HOUR_UTC = 6
FIRST_SCHEDULE_YEAR = 1996


def _season_year_datetimes(season_start_year: int) -> list[datetime]:
    out: list[datetime] = []
    y = season_start_year
    for month in SCHEDULE_MONTHS:
        year = y if month >= 11 else y + 1
        for day in SCHEDULE_DAYS:
            out.append(
                datetime(year, month, day, SCHEDULE_HOUR_UTC, 0, tzinfo=timezone.utc)
            )
    return out


def latest_scheduled_run(before: datetime | None = None) -> datetime | None:
    before = (before or datetime.now(timezone.utc)).astimezone(timezone.utc)
    latest: datetime | None = None
    for y in range(FIRST_SCHEDULE_YEAR, before.year + 2):
        for dt in _season_year_datetimes(y):
            if dt <= before and (latest is None or dt > latest):
                latest = dt
    return latest


def is_stale(last_updated: datetime, now: datetime | None = None) -> bool:
    last_updated = last_updated.astimezone(timezone.utc)
    due = latest_scheduled_run(now)
    if due is None:
        return False
    return last_updated < due
