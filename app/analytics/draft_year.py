"""Helpers for the current NBA draft *class* year.

The current draft class is the *upcoming* draft. RealGM's prospect stats page
(``/nba/draft/prospects/stats``) always points at that upcoming class and rolls
over to the next one shortly after each draft (held in late June). The scraper
parses that year off the page and persists it inside the prospects file, so the
rest of the app (and the frontend) read a single source of truth from stored
data rather than guessing from the wall clock.

Likewise, which *historical* draft classes are selectable is determined by the
prospect files the historical scraper has actually written to disk — not by a
computed ``EARLIEST..current`` range.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

# Top-level key holding the prospect records inside the wrapped prospects file.
PROSPECTS_KEY = "prospects"

# Earliest plausible class year, used purely as a sanity bound when parsing a
# year off a RealGM page. It is *not* used to enumerate available classes.
EARLIEST_DRAFT_YEAR = 2007


def available_historical_draft_years(draft_dir: Path | str) -> list[int]:
    """Historical draft classes with prospect data on disk, ascending.

    The set of selectable past drafts is whatever the historical scraper has
    written to ``static/draft/`` (one ``prospects_<year>.json`` per class) — the
    single source of truth for "which past classes exist", read from stored data
    rather than assumed from a contiguous year range.
    """
    years: set[int] = set()
    directory = Path(draft_dir)
    if directory.exists():
        for path in directory.glob("prospects_*.json"):
            match = re.match(r"prospects_(\d{4})$", path.stem)
            if match:
                years.add(int(match.group(1)))
    return sorted(years)


def current_draft_year_from_history(historical_years) -> int | None:
    """Infer the current (upcoming) class as one past the newest completed draft.

    Driven by the historical classes actually on disk (RealGM-sourced), so it
    needs no wall-clock cutover. Returns ``None`` when no history is available.
    """
    years = [int(y) for y in historical_years if y is not None]
    return max(years) + 1 if years else None


def current_draft_year_from_date(now: datetime | None = None) -> int:
    """Absolute last-resort estimate of the current (upcoming) draft class year.

    Only used when there is no scraped page and no stored data at all to read a
    year from (e.g. a brand-new deployment before the first scrape). The draft is
    held in late June; we treat July 1 as the cutover.
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
