from __future__ import annotations

import logging
import random
import time
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

logger = logging.getLogger(__name__)

MIN_GAMES_PER_TEAM = 10
EXPECTED_TEAM_COUNT = 30
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 4

Fetcher = Callable[[str], pd.DataFrame]


def _default_fetch(season: str, timeout: int = DEFAULT_TIMEOUT) -> pd.DataFrame:
    return LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=timeout,
    ).get_data_frames()[0]


def _fetch_with_retry(season: str, fetch: Fetcher, retries: int) -> pd.DataFrame | None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetch(season)
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Season readiness fetch for %s failed on attempt %s/%s: %s",
                season,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(2.0 * (2 ** (attempt - 1)) + random.random() * 0.5)
    logger.error("Season readiness fetch for %s failed after %s attempts: %s", season, retries, last_err)
    return None


def min_team_games(
    season: str,
    fetch: Fetcher | None = None,
    retries: int = DEFAULT_RETRIES,
) -> int | None:
    fetch = fetch or _default_fetch
    df = _fetch_with_retry(season, fetch, retries)
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
