from __future__ import annotations

from datetime import date


FIRST_PLAYER_PROFILE_SEASON = 1996


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start_year(season: str) -> int:
    return int(str(season).split("-", 1)[0])


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def seasons_since_1996(end_start_year: int | None = None) -> list[str]:
    end = end_start_year if end_start_year is not None else current_season_start()
    return [season_label(y) for y in range(FIRST_PLAYER_PROFILE_SEASON, end + 1)]

