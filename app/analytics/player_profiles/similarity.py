from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from analytics.similarity_scoring import similarity_pct

EXPLANATION_FEATURES = {
    "scoring volume": ["pts_per36_z", "mpg_z"],
    "creation": ["ast_per36_z", "ast_pct_z", "tov_per36_z"],
    "scoring efficiency": ["ts_pct_z", "efg_pct_z"],
    "shot mix": ["fg3a_rate_z", "fta_rate_z"],
    "defensive box score": ["stl_per36_z", "blk_per36_z"],
    "rebounding": ["reb_per36_z"],
}


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
    if embeddings.empty:
        return {}

    emb_cols = [c for c in embeddings.columns if c.startswith("component_")]
    if not emb_cols:
        emb_cols = [c for c in feature_df.columns if c != "player_id"]
        matrix_source = feature_df[["player_id", *emb_cols]].copy()
    else:
        matrix_source = embeddings[["player_id", *emb_cols]].copy()

    player_ids = matrix_source["player_id"].astype(int).to_numpy()
    X = matrix_source[emb_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_neighbors = min(max(k * 3, k + 1), len(player_ids))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(X)
        dists, indices = nn.kneighbors(X)

    career_by_id = career_df.set_index("player_id")
    feature_by_id = feature_df.set_index("player_id")
    out: dict[str, list[dict[str, Any]]] = {}
    for row_idx, pid in enumerate(player_ids):
        query_player = career_by_id.loc[int(pid)]
        query_year = _parse_start_year(query_player.get("first_season"))
        candidates: list[tuple[float, dict[str, Any]]] = []
        for dist, idx in zip(dists[row_idx], indices[row_idx]):
            other_id = int(player_ids[int(idx)])
            if other_id == int(pid):
                continue
            other = career_by_id.loc[other_id]
            other_year = _parse_start_year(other.get("first_season"))
            
            # Apply same-era decay penalty: c = 0.05, tau = 10.0
            diff = abs(query_year - other_year)
            era_penalty = 1.0 - 0.05 * np.exp(-diff / 10.0)
            
            raw_similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            adj_similarity = raw_similarity * era_penalty
            
            candidates.append(
                (
                    adj_similarity,
                    {
                        "player_id": other_id,
                        "player_name": str(other["player_name"]),
                        "similarity_score": similarity_pct(adj_similarity),
                        "career_span": str(other.get("career_span", "")),
                        "explanation": explain_similarity(
                            feature_by_id.loc[int(pid)],
                            feature_by_id.loc[other_id],
                        ),
                    },
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        out[str(int(pid))] = [payload for _, payload in candidates[:k]]
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
