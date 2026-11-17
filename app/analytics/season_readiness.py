from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

logger = logging.getLogger(__name__)

MIN_GAMES_PER_TEAM = 10
EXPECTED_TEAM_COUNT = 30

Fetcher = Callable[[str], pd.DataFrame]


def _default_fetch(season: str, timeout: int = 30) -> pd.DataFrame:
    return LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=timeout,
    ).get_data_frames()[0]


def min_team_games(season: str, fetch: Fetcher | None = None) -> int | None:
    fetch = fetch or _default_fetch
    try:
        df = fetch(season)
    except Exception as exc:
        logger.warning("Season readiness fetch failed for %s: %s", season, exc)
        return None
    if df is None or "GP" not in df.columns or len(df) < EXPECTED_TEAM_COUNT:
        return None
    gp = pd.to_numeric(df["GP"], errors="coerce").fillna(0)
    return int(gp.min())


def season_ready(
    season: str,
    min_games: int = MIN_GAMES_PER_TEAM,
    fetch: Fetcher | None = None,
) -> bool:
    lowest = min_team_games(season, fetch=fetch)
    if lowest is None:
        return False
    return lowest >= min_games
