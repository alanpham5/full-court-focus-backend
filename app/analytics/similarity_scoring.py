from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DEFAULT_SIMILARITY_GAMMA = 5.5


def cross_pool_cosine_similarity(
    query_features: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    query_scaled = StandardScaler().fit_transform(query_features)
    candidate_scaled = StandardScaler().fit_transform(candidate_features)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return cosine_similarity(query_scaled, candidate_scaled)


def similarity_pct(cos_sim: float, *, gamma: float = DEFAULT_SIMILARITY_GAMMA) -> float:
    x = max(0.0, min(1.0, float(cos_sim)))
    return round(100.0 * (x**gamma), 1)
