"""Helpers for the current NBA draft *class* year.

The current draft class is the *upcoming* draft. RealGM's prospect stats page
(``/nba/draft/prospects/stats``) always points at that upcoming class and rolls
over to the next one shortly after each draft (held in late June). We persist the
class year inside the prospects file so the rest of the app (and the frontend)
read a single source of truth instead of guessing from the wall clock.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Top-level key holding the prospect records inside the wrapped prospects file.
PROSPECTS_KEY = "prospects"

# Earliest historical class with prospect data on disk.
EARLIEST_DRAFT_YEAR = 2007


def current_draft_year_from_date(now: datetime | None = None) -> int:
    """Date-based estimate of the current (upcoming) draft class year.

    The draft is held in late June; once it concludes attention shifts to the
    next class. We treat July 1 as the cutover: Jan–Jun the current calendar
    year's draft is still upcoming; Jul–Dec the next year's class is current.
    """
    now = now or datetime.now(timezone.utc)
    return now.year + 1 if now.month >= 7 else now.year


def extract_draft_year_from_html(html: str) -> int | None:
    """Best-effort parse of the draft class year from a RealGM prospects page.

    RealGM renders headings/titles such as "2026 NBA Draft" / "NBA Draft 2026".
    Returns ``None`` when no plausible year is found.
    """
    if not html:
        return None
    patterns = (
        r"(\d{4})\s+NBA\s+Draft",
        r"NBA\s+Draft\s+(\d{4})",
        r"(\d{4})\s+Draft\s+Prospects",
    )
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if EARLIEST_DRAFT_YEAR <= year <= 2100:
                return year
    return None


def season_for_draft_class_year(year: int) -> str:
    return f"{year - 1}-{str(year)[-2:]}"


def draft_class_year_for_season(season: str) -> int | None:
    try:
        return int(str(season)[:4]) + 1
    except (TypeError, ValueError):
        return None


def normalize_prospects_payload(data) -> tuple[list, dict]:
    """Split a loaded prospects file into ``(records, meta)``.

    Accepts both the wrapped object shape (``{"draft_class_year": ...,
    "prospects": [...]}``) and the legacy bare-list shape. ``meta`` holds the
    non-record top-level keys (empty for the legacy list).
    """
    if isinstance(data, dict):
        records = data.get(PROSPECTS_KEY, [])
        meta = {k: v for k, v in data.items() if k != PROSPECTS_KEY}
        return records, meta
    return data, {}
