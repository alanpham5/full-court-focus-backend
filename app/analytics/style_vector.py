"""
Within-season league percentiles for stylistic axes (0–100, higher = more of that trait).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# (column, higher_raw_value_is_better). For defense, lower DEF_RATING is better → invert.
_STYLE_AXES: tuple[tuple[str, str, bool], ...] = (
    ("pace", "PACE", True),
    ("three_point_volume", "FG3A", True),
    ("paint", "PAINT_FGA", True),
    ("defense", "DEF_RATING", False),
    ("playmaking", "AST_PCT", True),
    ("rebounding", "REB_PCT", True),
)


def _style_vector_for_season_row(season_df: pd.DataFrame, team_row: pd.Series) -> dict[str, float]:
    """League percentiles for one team-season vs that season's `season_df` pool."""
    out: dict[str, float] = {}

    for key, col, higher_is_better in _STYLE_AXES:
        if col not in season_df.columns:
            out[key] = 50.0
            continue

        raw = pd.to_numeric(season_df[col], errors="coerce").astype(float)
        score = float(pd.to_numeric(team_row.get(col), errors="coerce"))
        if not np.isfinite(score) or raw.notna().sum() < 2:
            out[key] = 50.0
            continue

        oriented = raw.copy()
        if not higher_is_better:
            oriented = -oriented
            score = -score

        pool = oriented.to_numpy(dtype=float)
        pool = pool[np.isfinite(pool)]
        if pool.size < 2:
            out[key] = 50.0
            continue

        pct = float(percentileofscore(pool, score, kind="rank"))
        out[key] = round(min(100.0, max(0.0, pct)), 1)

    return out


def compute_style_vector(df: pd.DataFrame, team_id: int, season: str) -> dict[str, float]:
    """
    Percentile of this team vs all teams that season (0–100).
    Uses scipy rank percentile: share of league at or below this team's oriented score.
    """
    season_df = df[df["SEASON"] == season]
    if season_df.empty:
        raise ValueError(f"No rows for season {season!r}")

    team_rows = season_df[season_df["TEAM_ID"] == team_id]
    if team_rows.empty:
        raise ValueError(f"No team {team_id} in season {season!r}")

    team_row = team_rows.iloc[0]
    return _style_vector_for_season_row(season_df, team_row)


def build_style_vector_lookup(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Precompute style vectors for every team-season (keys "{team_id}:{season}").
    One groupby per season instead of filtering the full history per request.
    """
    out: dict[str, dict[str, float]] = {}
    for season, season_df in df.groupby("SEASON", sort=False):
        if season_df.empty:
            continue
        for _, team_row in season_df.iterrows():
            tid = int(team_row["TEAM_ID"])
            key = f"{tid}:{season}"
            out[key] = _style_vector_for_season_row(season_df, team_row)
    return out
