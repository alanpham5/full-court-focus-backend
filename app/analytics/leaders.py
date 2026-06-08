import json
import time
from functools import lru_cache

import pandas as pd
from nba_api.stats.endpoints import TeamPlayerDashboard
from nba_api.stats.library.http import STATS_HEADERS

MIN_GP = 8
MIN_MPG = 12.0
MIN_FG3A_PER_GAME = 3.0

_MAX_ATTEMPTS = 3
_BACKOFF_SEC = 1.25
_REQUEST_TIMEOUT_SEC = 22


@lru_cache(maxsize=128)
def _fetch_league_players_season_totals_cached(season: str, timeout: int = 22, retries: int = 3) -> pd.DataFrame:
    headers = dict(STATS_HEADERS)
    last_err: BaseException | None = None
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    for attempt in range(retries):
        try:
            dashboard = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                headers=headers,
                timeout=timeout,
            )
            return dashboard.get_data_frames()[0]
        except Exception as e:
            last_err = e

        if attempt < retries - 1:
            time.sleep(_BACKOFF_SEC * (attempt + 1))

    if last_err is None:
        raise RuntimeError("LeagueDashPlayerStats failed with no error captured")
    raise last_err


@lru_cache(maxsize=2048)
def _fetch_players_season_totals_cached(team_id: int, season: str, timeout: int = 22, retries: int = 3) -> pd.DataFrame:
    league_df = _fetch_league_players_season_totals_cached(season, timeout, retries)
    team_df = league_df[league_df["TEAM_ID"] == team_id].copy()
    return team_df


def _fetch_players_season_totals(team_id: int, season: str, timeout: int = 22, retries: int = 3) -> pd.DataFrame:
    """Fetch per-game player stats for a team-season, cached within one scrape run."""
    return _fetch_players_season_totals_cached(team_id, season, timeout, retries).copy()


def get_stat_leaders(team_id: int, season: str) -> dict:
    try:
        data = _fetch_players_season_totals(team_id, season)
    except Exception:
        return {}

    if data.empty:
        return {}

    def leader_volume(col: str) -> dict:
        row = data.loc[data[col].idxmax()]
        return {
            "player_id": int(row["PLAYER_ID"]),
            "name": row["PLAYER_NAME"],
            "value": round(float(row[col]), 1),
        }

    def leader_fg3_pct() -> dict:
        df = data.copy()
        gp = pd.to_numeric(df.get("GP", 0), errors="coerce").fillna(0)
        mpg = pd.to_numeric(df.get("MIN", 0), errors="coerce").fillna(0)
        fg3a = pd.to_numeric(df.get("FG3A", 0), errors="coerce").fillna(0)
        pct = pd.to_numeric(df.get("FG3_PCT", 0), errors="coerce")

        tiers = [
            (MIN_GP, MIN_MPG, MIN_FG3A_PER_GAME),
            (5, 10.0, 2.0),
            (3, 8.0, 1.5),
        ]
        pool = df.iloc[0:0]
        for min_gp, min_mpg, min_fg3a in tiers:
            mask = (gp >= min_gp) & (mpg >= min_mpg) & (fg3a >= min_fg3a) & pct.notna()
            cand = df.loc[mask]
            if not cand.empty:
                pool = cand
                break

        if pool.empty:
            mask = pct.notna() & (fg3a >= 1.0)
            pool = df.loc[mask]
        if pool.empty:
            pool = df

        pool = pool.assign(_fg3_pct=pd.to_numeric(pool["FG3_PCT"], errors="coerce"))
        pool = pool.dropna(subset=["_fg3_pct"])
        if pool.empty:
            pool = df.assign(_fg3_pct=pd.to_numeric(df["FG3_PCT"], errors="coerce")).dropna(
                subset=["_fg3_pct"]
            )
        row = pool.loc[pool["_fg3_pct"].idxmax()]
        return {
            "player_id": int(row["PLAYER_ID"]),
            "name": row["PLAYER_NAME"],
            "value": round(float(row["_fg3_pct"]), 3),
        }

    return {
        "ppg": leader_volume("PTS"),
        "apg": leader_volume("AST"),
        "rpg": leader_volume("REB"),
        "spg": leader_volume("STL"),
        "bpg": leader_volume("BLK"),
        "fg3_pct": leader_fg3_pct(),
    }
