"""
k-NN similarity search across all team-seasons using normalized vectors.

Similar picks exclude the same franchise and any team from the query season
(cross-season, cross-team comparisons only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

Z_COLS = [
    "PACE_Z",
    "OFF_RATING_Z",
    "DEF_RATING_Z",
    "AST_PCT_Z",
    "TM_TOV_PCT_Z",
    "OREB_PCT_Z",
    "DREB_PCT_Z",
    "TS_PCT_Z",
    "FG3A_Z",
    "PAINT_FGA_Z",
]


def _z_feature_matrix(pool: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Finite float64 matrix for distances (avoids inf/NaN blowups in cdist)."""
    X = pool[cols].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -80.0, 80.0, out=X)
    return X


def _similar_candidate_mask(pool: pd.DataFrame, team_id: int, season: str) -> np.ndarray:
    """Exclude same franchise and any team-season from the query season."""
    s = pool["SEASON"].astype(str)
    bad = (pool["TEAM_ID"].to_numpy() == team_id) | (s.to_numpy() == str(season))
    return ~bad


def _row_dict_from_pool(pool: pd.DataFrame, row_idx: int, dist: float) -> dict:
    row = pool.iloc[int(row_idx)]
    return {
        "team_id": int(row["TEAM_ID"]),
        "team_name": row["TEAM_NAME"],
        "season": row["SEASON"],
        "similarity_pct": round(100 / (1 + float(dist)), 1),
        "record": f"{int(row['W'])}-{int(row['L'])}",
    }


def find_similar_teams(
    team_id: int,
    season: str,
    normalized_df: pd.DataFrame,
    k: int = 6,
) -> list[dict]:
    available_z = [c for c in Z_COLS if c in normalized_df.columns]

    pool = normalized_df.reset_index(drop=True)
    is_self = (pool["TEAM_ID"] == team_id) & (pool["SEASON"].astype(str) == str(season))
    if not is_self.any() or not available_z:
        return []

    cand = _similar_candidate_mask(pool, team_id, str(season))
    cand_idx = np.flatnonzero(np.asarray(cand))
    if cand_idx.size == 0:
        return []

    X = _z_feature_matrix(pool, available_z)
    target = X[np.flatnonzero(is_self.to_numpy())[0]]
    diff = X[cand_idx] - target
    dists = np.linalg.norm(diff, axis=1)
    k_eff = min(k, dists.size)
    order = np.argsort(dists)[:k_eff]

    out: list[dict] = []
    for o in order:
        j = int(cand_idx[int(o)])
        out.append(_row_dict_from_pool(pool, j, float(dists[int(o)])))
    return out


def build_similar_teams_index(
    normalized_df: pd.DataFrame,
    k: int = 6,
) -> dict[str, list[dict]]:
    """
    Precompute find_similar_teams for every team-season.
    Keys are "{team_id}:{season}" (strings) for JSON lookup.

    Neighbors are the closest rows in z-space among candidates that are **not**
    the same franchise and **not** from the query season (so no in-season peers
    and no other years of the same team).
    """
    available_z = [c for c in Z_COLS if c in normalized_df.columns]
    pool = normalized_df.reset_index(drop=True)
    n_samples = len(pool)
    index: dict[str, list[dict]] = {}

    if n_samples < 2 or not available_z:
        for i in range(n_samples):
            tid = int(pool.iloc[i]["TEAM_ID"])
            season = str(pool.iloc[i]["SEASON"])
            index[f"{tid}:{season}"] = []
        return index

    X = _z_feature_matrix(pool, available_z)
    team_ids = pool["TEAM_ID"].to_numpy()
    seasons = pool["SEASON"].astype(str).to_numpy()

    # One O(n²) distance matrix; per-query masking enforces franchise + season rules.
    dists_all = cdist(X, X, metric="euclidean")

    for i in range(n_samples):
        tid_i = int(team_ids[i])
        season_i = str(seasons[i])
        key = f"{tid_i}:{season_i}"
        mask = (np.arange(n_samples) != i) & (team_ids != tid_i) & (seasons != season_i)
        d = dists_all[i].copy()
        d[~mask] = np.inf
        finite = np.isfinite(d)
        if not finite.any():
            index[key] = []
            continue
        k_eff = min(k, int(finite.sum()))
        neigh_local = np.argpartition(d, k_eff - 1)[:k_eff]
        neigh = neigh_local[np.argsort(d[neigh_local])]
        out: list[dict] = []
        for j in neigh:
            out.append(_row_dict_from_pool(pool, int(j), float(dists_all[i, int(j)])))
        index[key] = out

    return index
