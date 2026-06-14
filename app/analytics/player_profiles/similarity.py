from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from analytics.similarity_scoring import similarity_pct


def _nba_nba_display_score(composite: float) -> float:
    x = max(0.0, min(float(composite) / NBA_NBA_DISPLAY_NORMALIZER, 1.0))
    return round(NBA_NBA_DISPLAY_CEILING * (x ** NBA_NBA_DISPLAY_GAMMA), 1)

EXPLANATION_FEATURES = {
    "scoring volume": ["pts_per36_z", "mpg_z"],
    "creation": ["ast_per36_z", "ast_pct_z", "tov_per36_z"],
    "scoring efficiency": ["ts_pct_z", "efg_pct_z"],
    "shot mix": ["fg3a_rate_z", "fta_rate_z"],
    "defensive box score": ["stl_per36_z", "blk_per36_z"],
    "rebounding": ["reb_per36_z"],
}


NBA_NBA_PLAYSTYLE_FEATURES = [
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "tov_per36",
    "fg3a_rate",
    "fta_rate",
    "ts_pct",
    "efg_pct",
    "ast_pct",
    "mpg",
]

NBA_NBA_PLAYSTYLE_WEIGHTS = np.array([
    1.0,
    1.9,
    1.7,
    1.6,
    1.2,
    0.5,
    1.9,
    1.1,
    0.7,
    0.6,
    2.1,
    0.6,
])

NBA_NBA_PLAYSTYLE_BANDWIDTH = 2.2
NBA_NBA_ARCHETYPE_BONUS_PER_OVERLAP = 0.18
NBA_NBA_SIZE_SIGMA = 12.0
NBA_NBA_CALIBER_SIGMA = 0.35
NBA_NBA_ROLE_BONUS = 1.07
NBA_NBA_ERA_REWARD = 0.10
NBA_NBA_ERA_TIMESCALE = 14.0

NBA_NBA_DISPLAY_CEILING = 78.0
NBA_NBA_DISPLAY_NORMALIZER = 1.45
NBA_NBA_DISPLAY_GAMMA = 2.0


def _percentile_matrix(career_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols = []
    for f in features:
        pct_col = f"{f}_global_pctile"
        if pct_col in career_df.columns:
            cols.append(career_df[pct_col].to_numpy(dtype=float) / 100.0)
        elif f in career_df.columns:
            vals = career_df[f].to_numpy(dtype=float)
            order = np.argsort(np.argsort(vals))
            cols.append((order + 1) / max(len(vals), 1))
        else:
            cols.append(np.full(len(career_df), 0.5))
    return np.column_stack(cols)


def _height_inches_array(career_df: pd.DataFrame) -> np.ndarray:
    from analytics.player_profiles.features import height_to_inches
    h = career_df["height"].apply(height_to_inches).to_numpy(dtype=float, copy=True)
    mean_h = h[h > 0].mean() if (h > 0).any() else 78.0
    h[h == 0] = mean_h
    return h


def _weight_lbs_array(career_df: pd.DataFrame) -> np.ndarray:
    from analytics.player_profiles.features import clean_weight_to_float
    w = career_df["weight"].apply(lambda v: clean_weight_to_float(v, 215.0)).to_numpy(dtype=float, copy=True)
    mean_w = w[w > 0].mean() if (w > 0).any() else 215.0
    w[w == 0] = mean_w
    return w


def _caliber_array(career_df: pd.DataFrame) -> np.ndarray:
    from analytics.player_profiles.archetypes import (
        calculate_apfv_batch_by_height, calculate_adjusted_pfv, height_bucket,
    )
    apfvs: list[float] = []
    buckets: list[str] = []
    for _, row in career_df.iterrows():
        metrics = {}
        for col in ["pts_per36", "reb_per36", "ast_per36", "blk_per36",
                    "stl_per36", "ts_pct", "mpg"]:
            pct_col = f"{col}_global_pctile"
            val = float(row.get(col, 0.0) or 0.0)
            pct = float(row.get(pct_col, 50.0) or 50.0)
            metrics[col] = {"value": val, "percentile": pct}
        apfvs.append(calculate_adjusted_pfv(metrics, is_prospect=False))
        buckets.append(height_bucket(row.get("height")))
    return np.array(calculate_apfv_batch_by_height(apfvs, buckets), dtype=float)


def build_similarity_embeddings(
    career_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    n_components: int = 12,
) -> tuple[pd.DataFrame, PCA | None]:
    features = [c for c in feature_df.columns if c != "player_id"]
    ids = feature_df["player_id"].astype(int).to_numpy()
    X = feature_df[features].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[0] < 3 or X.shape[1] < 2:
        emb = pd.DataFrame({"player_id": ids})
        for i in range(min(X.shape[1], n_components)):
            emb[f"component_{i + 1}"] = X[:, i]
        return emb, None

    comps = min(n_components, X.shape[0] - 1, X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=comps, svd_solver="full", random_state=7)
        Z = pca.fit_transform(X)
    emb = pd.DataFrame(Z, columns=[f"component_{i + 1}" for i in range(comps)])
    emb.insert(0, "player_id", ids)
    return emb, pca


def _parse_start_year(season_str: Any) -> int:
    if pd.isna(season_str) or not season_str:
        return 2010
    try:
        return int(str(season_str).split("-", 1)[0])
    except Exception:
        return 2010


def build_similarity_index(
    career_df: pd.DataFrame,
    embeddings: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    if career_df.empty:
        return {}

    feature_by_id = feature_df.set_index("player_id") if not feature_df.empty else None
    career = career_df.reset_index(drop=True)

    player_ids = career["player_id"].astype(int).to_numpy()
    n = len(player_ids)
    if n < 2:
        return {str(int(pid)): [] for pid in player_ids}

    P = _percentile_matrix(career, NBA_NBA_PLAYSTYLE_FEATURES)
    P = P * NBA_NBA_PLAYSTYLE_WEIGHTS

    heights = _height_inches_array(career)
    weights = _weight_lbs_array(career)
    caliber = _caliber_array(career)
    years = np.array(
        [_parse_start_year(s) for s in career["first_season"].tolist()],
        dtype=float,
    )
    roles = career.get("role", pd.Series([""] * n)).astype(str).to_numpy()

    sq = (
        (P ** 2).sum(axis=1)[:, None]
        + (P ** 2).sum(axis=1)[None, :]
        - 2.0 * P @ P.T
    )
    sq = np.maximum(sq, 0.0)
    dists = np.sqrt(sq)
    playstyle_aff = np.exp(-dists / NBA_NBA_PLAYSTYLE_BANDWIDTH)

    size_aff = np.exp(-((heights[:, None] - heights[None, :]) ** 2) / (2.0 * NBA_NBA_SIZE_SIGMA ** 2))
    weight_aff = np.exp(-((weights[:, None] - weights[None, :]) ** 2) / (2.0 * 36.0 ** 2))
    size_aff = (size_aff * weight_aff) ** 0.25

    caliber_aff = np.exp(-((caliber[:, None] - caliber[None, :]) ** 2) / (2.0 * NBA_NBA_CALIBER_SIGMA ** 2))

    dt = np.abs(years[:, None] - years[None, :])
    era_aff = 1.0 + NBA_NBA_ERA_REWARD * (1.0 - np.exp(-dt / NBA_NBA_ERA_TIMESCALE))

    role_match = (roles[:, None] == roles[None, :]).astype(float)
    role_factor = 1.0 + (NBA_NBA_ROLE_BONUS - 1.0) * role_match

    arch_sets: list[set[str]] = []
    for a in career.get("archetypes", pd.Series([[]] * n)).tolist():
        if a is None:
            arch_sets.append(set())
        elif isinstance(a, (list, tuple)):
            arch_sets.append({str(x) for x in a})
        else:
            arch_sets.append({str(a)})
    arch_overlap = np.zeros((n, n), dtype=float)
    for i_ in range(n):
        si = arch_sets[i_]
        if not si:
            continue
        for j_ in range(n):
            if i_ == j_:
                continue
            sj = arch_sets[j_]
            if sj:
                arch_overlap[i_, j_] = len(si & sj)
    arch_factor = 1.0 + NBA_NBA_ARCHETYPE_BONUS_PER_OVERLAP * arch_overlap

    composite = playstyle_aff * size_aff * caliber_aff * era_aff * role_factor * arch_factor

    np.fill_diagonal(composite, -np.inf)

    out: dict[str, list[dict[str, Any]]] = {}
    career_by_id = career.set_index("player_id")
    for i, pid in enumerate(player_ids):
        order = np.argsort(composite[i])[::-1][:k]
        payloads: list[dict[str, Any]] = []
        for j in order:
            j = int(j)
            other_id = int(player_ids[j])
            if other_id == int(pid):
                continue
            other = career_by_id.loc[other_id]
            score = float(composite[i, j])
            payloads.append(
                {
                    "player_id": other_id,
                    "player_name": str(other["player_name"]),
                    "similarity_score": _nba_nba_display_score(score),
                    "career_span": str(other.get("career_span", "")),
                    "explanation": explain_similarity(
                        feature_by_id.loc[int(pid)] if feature_by_id is not None and int(pid) in feature_by_id.index else pd.Series(dtype=float),
                        feature_by_id.loc[other_id] if feature_by_id is not None and other_id in feature_by_id.index else pd.Series(dtype=float),
                    ),
                }
            )
            if len(payloads) >= k:
                break
        out[str(int(pid))] = payloads
    return out


def explain_similarity(a: pd.Series, b: pd.Series) -> str:
    parts: list[tuple[str, float]] = []
    for label, cols in EXPLANATION_FEATURES.items():
        present = [c for c in cols if c in a.index and c in b.index]
        if not present:
            continue
        diff = float(np.mean([abs(float(a[c]) - float(b[c])) for c in present]))
        parts.append((label, diff))
    parts.sort(key=lambda x: x[1])
    labels = [label for label, _ in parts[:3]]
    if not labels:
        return "Similar career feature vector."
    if len(labels) == 1:
        return f"Closest match in {labels[0]}."
    return "Closest match in " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def similarity_for_player(player_id: int, feature_df: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series | None:
    row = embeddings[embeddings["player_id"].astype(int) == int(player_id)]
    if row.empty:
        return None
    return row.iloc[0]
