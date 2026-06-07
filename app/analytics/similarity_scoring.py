from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DEFAULT_SIMILARITY_GAMMA = 5.5


def cross_pool_cosine_similarity(
    query_features: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    """Cosine similarity with independent z-score normalization per pool.

    Each matrix is standardized using only its own rows so comparisons reflect
    relative profile shape (e.g. high-scoring prospect vs high-scoring NBA
    player) rather than absolute stat levels across pools.
    """
    query_scaled = StandardScaler().fit_transform(query_features)
    candidate_scaled = StandardScaler().fit_transform(candidate_features)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return cosine_similarity(query_scaled, candidate_scaled)


def similarity_pct(cos_sim: float, *, gamma: float = DEFAULT_SIMILARITY_GAMMA) -> float:
    """Map cosine similarity in [0, 1] to a 0–100 display score.

    Applies a monotonic power transform so strong matches read lower than raw
    cosine × 100 without additive penalties.
    """
    x = max(0.0, min(1.0, float(cos_sim)))
    return round(100.0 * (x**gamma), 1)
