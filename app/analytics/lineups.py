
from __future__ import annotations

import re
import time

import pandas as pd
from nba_api.stats.endpoints import LeagueDashLineups
from nba_api.stats.library.http import STATS_HEADERS

from analytics.leaders import _fetch_players_season_totals
from analytics.player_roles import assign_player_roles

_MIN_LINEUP_GP = 10
_MIN_PLAYER_GP = 20
_MIN_LINEUP_PLAYER_MPG = 18.0
_REQUEST_TIMEOUT = 22
_BACKOFF = 1.25
_MAX_ATTEMPTS = 3


def _parse_lineup_label(label: str) -> list[str]:
    return [p.strip() for p in str(label).split(" - ") if p.strip()]


def _match_short_name(short: str, roster: pd.DataFrame) -> pd.Series | None:
    short = short.strip()
    exact = roster[roster["PLAYER_NAME"].str.casefold() == short.casefold()]
    if len(exact) == 1:
        return exact.iloc[0]
    m = re.match(r"^([A-Za-z])\.?\s+(.+)$", short)
    if m:
        initial, last = m.group(1).lower(), m.group(2).lower()
        for _, row in roster.iterrows():
            parts = str(row["PLAYER_NAME"]).split()
            if len(parts) < 2:
                continue
            if parts[-1].lower() == last and parts[0][0].lower() == initial:
                return row
    last = short.split()[-1].lower()
    hits = roster[roster["PLAYER_NAME"].str.lower().str.endswith(last)]
    if len(hits) == 1:
        return hits.iloc[0]
    return None


from functools import lru_cache

@lru_cache(maxsize=128)
def _fetch_league_lineups_cached(season: str, timeout: int = 15, retries: int = 2) -> pd.DataFrame:
    headers = dict(STATS_HEADERS)
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            df = LeagueDashLineups(
                season=season,
                group_quantity=5,
                per_mode_detailed="Totals",
                headers=headers,
                timeout=timeout,
            ).get_data_frames()[0]
            return df
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(_BACKOFF * (attempt + 1))
    if last_err is None:
        raise RuntimeError("LeagueDashLineups failed")
    raise last_err


def _fetch_lineup_from_api(team_id: int, season: str, timeout: int = 15, retries: int = 2) -> tuple[list[str], float] | None:
    try:
        df = _fetch_league_lineups_cached(season, timeout=timeout, retries=retries)
        if df.empty or "GROUP_NAME" not in df.columns:
            return None
        team_df = df[df["TEAM_ID"] == team_id] if "TEAM_ID" in df.columns else df
        if team_df.empty:
            return None
        if "GP" in team_df.columns:
            gp = pd.to_numeric(team_df["GP"], errors="coerce").fillna(0)
            qualified = team_df.loc[gp >= _MIN_LINEUP_GP]
            if not qualified.empty:
                team_df = qualified
        team_df = team_df.assign(_MIN=pd.to_numeric(team_df["MIN"], errors="coerce"))
        team_df = team_df.dropna(subset=["_MIN"]).sort_values("_MIN", ascending=False)
        if team_df.empty:
            return None
        top = team_df.iloc[0]
        names = _parse_lineup_label(top["GROUP_NAME"])
        if len(names) < 5:
            return None
        gp = _num(top.get("GP", 0))
        total_min = _num(top.get("_MIN", 0))
        mpg = total_min / gp if gp > 0 else 0.0
        return names[:5], mpg
    except Exception:
        return None


def _row_player_id(row: pd.Series) -> int | None:
    pid = pd.to_numeric(row.get("PLAYER_ID"), errors="coerce")
    if pd.isna(pid):
        return None
    return int(pid)


def _num(value: object, default: float = 0.0) -> float:
    n = pd.to_numeric(value, errors="coerce")
    if pd.isna(n):
        return default
    return float(n)


def _fallback_top_minutes(roster: pd.DataFrame) -> list[str]:
    df = roster.copy()
    gp = pd.to_numeric(df.get("GP", 0), errors="coerce").fillna(0)
    df = df.loc[gp >= _MIN_PLAYER_GP].sort_values("MIN", ascending=False)
    return [str(n) for n in df["PLAYER_NAME"].head(5).tolist()]


def _matched_roster_rows(lineup_labels: list[str], roster: pd.DataFrame) -> list[pd.Series]:
    matched_rows: list[pd.Series] = []
    seen_player_ids: set[int] = set()
    for label in lineup_labels:
        row = _match_short_name(label, roster)
        if row is None:
            continue
        pid = _row_player_id(row)
        if pid is None or pid in seen_player_ids:
            continue
        seen_player_ids.add(pid)
        matched_rows.append(row)
    return matched_rows


def _lineup_has_rotation_minutes(rows: list[pd.Series]) -> bool:
    if len(rows) < 5:
        return False
    mpg = pd.to_numeric(
        pd.Series([row.get("MIN", 0) for row in rows]),
        errors="coerce",
    ).fillna(0)
    return bool((mpg >= _MIN_LINEUP_PLAYER_MPG).all())


def build_starting_lineup(team_id: int, season: str, timeout: int = 15, retries: int = 2) -> dict | None:
    try:
        roster = _fetch_players_season_totals(team_id, season, timeout=timeout, retries=retries)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Roster fetch failed for team %s in %s: %s", team_id, season, e)
        return None

    if roster.empty:
        return None

    lineup_labels: list[str] | None = None
    lineup_mpg: float | None = None
    api_lineup = _fetch_lineup_from_api(team_id, season, timeout=timeout, retries=retries)
    if api_lineup is not None:
        lineup_labels, lineup_mpg = api_lineup

    if not lineup_labels:
        lineup_labels = _fallback_top_minutes(roster)
        lineup_mpg = None

    matched_rows = _matched_roster_rows(lineup_labels, roster)

    if not _lineup_has_rotation_minutes(matched_rows):
        fallback_names = _fallback_top_minutes(roster)
        matched_rows = _matched_roster_rows(fallback_names, roster)
        lineup_mpg = None

    if len(matched_rows) < 5:
        return None

    lineup_df = pd.DataFrame(matched_rows).head(5).reset_index(drop=True)
    roles = assign_player_roles(lineup_df)

    starters = []
    for _, row in lineup_df.iterrows():
        pid = int(row["PLAYER_ID"])
        starters.append(
            {
                "player_id": pid,
                "name": str(row["PLAYER_NAME"]),
                "roles": roles.get(pid, ["Secondary Creator"]),
                "gp": int(_num(row.get("GP", 0))),
                "mpg": round(_num(row.get("MIN", 0)), 1),
            }
        )

    source = (
        "lineup_minutes"
        if api_lineup is not None and lineup_mpg is not None
        else "minutes_fallback"
    )
    out: dict = {
        "team_id": team_id,
        "season": season,
        "source": source,
        "starters": starters,
    }
    if lineup_mpg is not None:
        out["lineup_mpg"] = round(lineup_mpg, 1)
    return out

