"""NBA season → decade helpers for era-filtered queries."""

from __future__ import annotations

VALID_ERAS = frozenset({"1990s", "2000s", "2010s", "2020s"})


def season_start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def season_to_decade(season: str) -> str:
    year = season_start_year(season)
    return f"{year // 10 * 10}s"


def normalize_era(era: str) -> str:
    e = era.strip().lower()
    if not e.endswith("s"):
        e = f"{e}s"
    if e in {"90s", "1990s"}:
        return "1990s"
    if e in {"00s", "2000s"}:
        return "2000s"
    if e in {"10s", "2010s"}:
        return "2010s"
    if e in {"20s", "2020s"}:
        return "2020s"
    return e


def is_valid_era(era: str) -> bool:
    return normalize_era(era) in VALID_ERAS
