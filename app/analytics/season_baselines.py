from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from analytics.lineup_synergy import compute_lineup_collective_stats, assign_player_roles_absolute

METRIC_KEYS: List[str] = [
    "pace",
    "fg3a",
    "space",
    "paint_fga",
    "def_score",
    "ast",
    "ast_pct",
    "reb",
    "blk",
]


def _team_pace(teams_df: pd.DataFrame, team_id: Any, season: str) -> float:
    if team_id is None:
        return 98.0
    row = teams_df[(teams_df["TEAM_ID"] == team_id) & (teams_df["SEASON"] == season)]
    if row.empty:
        return 98.0
    try:
        return float(row.iloc[0].get("PACE", 98.0) or 98.0)
    except (TypeError, ValueError):
        return 98.0


def compute_lineup_metrics(
    lineup_rows: pd.DataFrame,
    player_roles: Dict[int, List[str]],
    teams_df: pd.DataFrame,
    team_abbr_to_id: Dict[str, int],
    season: str,
) -> Dict[str, float]:
    player_rows_list = [lineup_rows.iloc[i] for i in range(len(lineup_rows))]
    collective = compute_lineup_collective_stats(player_rows_list, player_roles)

    paces: List[float] = []
    mpgs: List[float] = []
    for _, r in lineup_rows.iterrows():
        abbr = r.get("TEAM_ABBREVIATION", "")
        gp = float(r.get("GP", 1.0) or 1.0) or 1.0
        mpg = float(r.get("MIN", 0.0) or 0.0) / gp
        team_id = team_abbr_to_id.get(abbr)
        p_season = r.get("SEASON", season)
        paces.append(_team_pace(teams_df, team_id, p_season))
        mpgs.append(mpg)
    total_mpg = sum(mpgs)
    pace = sum(p * m for p, m in zip(paces, mpgs)) / total_mpg if total_mpg > 0 else 98.0

    return {
        "pace": float(pace),
        "fg3a": float(collective["fg3a"]),
        "space": float(collective["space"]),
        "paint_fga": float(collective["paint_fga"]),
        "def_score": float(collective["def_score"]),
        "ast": float(collective["ast"]),
        "ast_pct": float(collective["ast_pct"]),
        "reb": float(collective["reb"]),
        "blk": float(collective["blk"]),
    }


def _summarise(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(sorted(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [round(float(v), 4) for v in arr],
    }


def build_season_metric_samples(
    season: str,
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_abbr_to_id: Dict[str, int],
) -> Dict[str, List[float]]:
    season_players = player_season_df[player_season_df["SEASON"] == season]
    samples: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for key, lineup in starting_lineups.items():
        if not key.endswith(f":{season}"):
            continue
        starter_ids = [s["player_id"] for s in lineup["starters"]]
        rows = season_players[season_players["PLAYER_ID"].isin(starter_ids)]
        if len(rows) < 5:
            continue
        roles = assign_player_roles_absolute(rows)
        metrics = compute_lineup_metrics(rows, roles, teams_df, team_abbr_to_id, season)
        for k in METRIC_KEYS:
            samples[k].append(metrics[k])
    return samples


def build_season_baseline(
    season: str,
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_abbr_to_id: Dict[str, int],
) -> Dict[str, Any] | None:
    samples = build_season_metric_samples(
        season, player_season_df, teams_df, starting_lineups, team_abbr_to_id
    )
    n = len(samples[METRIC_KEYS[0]])
    if n < 5:
        return None
    return {
        "season": season,
        "n_lineups": n,
        "metrics": {k: _summarise(v) for k, v in samples.items()},
    }


def build_all_season_baselines(
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_metadata: Dict[str, Any],
    seasons: List[str] | None = None,
) -> Dict[str, Any]:
    team_abbr_to_id = {v["abbreviation"]: int(k) for k, v in team_metadata.items()}
    if seasons is None:
        seasons = sorted({k.split(":")[1] for k in starting_lineups.keys() if ":" in k})
    out: Dict[str, Any] = {}
    for season in seasons:
        baseline = build_season_baseline(
            season, player_season_df, teams_df, starting_lineups, team_abbr_to_id
        )
        if baseline is not None:
            out[season] = baseline
    return out


def percentile_in(metric_baseline: Dict[str, Any], value: float) -> float:
    values = metric_baseline.get("values") if metric_baseline else None
    if not values:
        return 50.0
    arr = np.asarray(values, dtype=float)
    left = float(np.searchsorted(arr, value, side="left"))
    right = float(np.searchsorted(arr, value, side="right"))
    return float(min(100.0, max(0.0, 100.0 * (left + right) / (2.0 * arr.size))))


def get_season_baseline(
    season: str,
    season_baselines: Dict[str, Any] | None,
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_abbr_to_id: Dict[str, int],
) -> Dict[str, Any] | None:
    if season_baselines and season in season_baselines:
        return season_baselines[season]
    return build_season_baseline(
        season, player_season_df, teams_df, starting_lineups, team_abbr_to_id
    )
