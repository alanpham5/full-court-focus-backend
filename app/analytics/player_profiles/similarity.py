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

# ---------------------------------------------------------------------------
# NBA -> NBA similarity (peer-percentile composite)
# ---------------------------------------------------------------------------
#
# A player's comps should: (1) play like them, (2) be about their size,
# (3) be about as good. We build the score as a product of three soft
# affinity terms so any of the three can veto an otherwise close match:
#
#     composite = playstyle_aff * size_aff * caliber_aff * era_aff
#
# Each term lives in [0, 1] and is 1.0 for a perfect match.

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

# Higher weight = bigger discriminator. Calibrated so high-signal stylistic
# axes (shot mix, creation, rim protection) drive the comp; volume/efficiency
# stats nudge it. Sum^2 ~ 12, so playstyle bandwidth ~1.8 gives smooth scores.
NBA_NBA_PLAYSTYLE_WEIGHTS = np.array([
    1.0,   # pts_per36
    1.1,   # reb_per36
    1.4,   # ast_per36   -- creation is a strong style signal
    1.5,   # blk_per36   -- rim protection is unique
    1.0,   # stl_per36
    0.6,   # tov_per36   -- noisy, downweight
    1.6,   # fg3a_rate   -- 3pt volume is a defining style axis
    1.0,   # fta_rate
    0.8,   # ts_pct
    0.7,   # efg_pct     -- correlated with ts_pct
    1.2,   # ast_pct
    0.7,   # mpg         -- partial role proxy
])

NBA_NBA_PLAYSTYLE_BANDWIDTH = 1.8
NBA_NBA_SIZE_SIGMA = 5.0          # inches: ~5" mismatch ≈ 1 sigma (soft prior, not gate)
NBA_NBA_CALIBER_SIGMA = 0.18      # APFV / PFV diff in [0,1]; ~0.18 ≈ 1 sigma
NBA_NBA_ROLE_BONUS = 1.07         # multiply by 1.07 if same role (capped)
NBA_NBA_ERA_HALFLIFE = 30.0       # seasons: very long half-life so the era
                                  # term is a gentle nudge rather than a gate
NBA_NBA_ERA_FLOOR = 0.65          # max era penalty: cross-era pairs always
                                  # keep at least 65% of their composite

# Post-rank era diversification: after scoring, walk the ranked candidate
# list and cap how many of the top-K can share an era bucket. Without this,
# the playstyle features (which are era-correlated -- 3pt rate, pace etc.)
# cause same-era pile-ups even with a generous era floor.
NBA_NBA_ERA_BUCKET_YEARS = 5      # width of each era bucket in seasons
NBA_NBA_ERA_MAX_PER_BUCKET = 2    # at most this many comps per 5-yr bucket
NBA_NBA_ERA_POOL_MULT = 8         # search 8*k candidates for era diversity


def _percentile_matrix(career_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return (n_players, n_features) of career percentile / 100 in [0,1]."""
    cols = []
    for f in features:
        pct_col = f"{f}_career_pctile"
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
    h = career_df["height"].apply(height_to_inches).to_numpy(dtype=float)
    mean_h = h[h > 0].mean() if (h > 0).any() else 78.0
    h[h == 0] = mean_h
    return h


def _weight_lbs_array(career_df: pd.DataFrame) -> np.ndarray:
    from analytics.player_profiles.features import clean_weight_to_float
    w = career_df["weight"].apply(lambda v: clean_weight_to_float(v, 215.0)).to_numpy(dtype=float)
    mean_w = w[w > 0].mean() if (w > 0).any() else 215.0
    w[w == 0] = mean_w
    return w


def _caliber_array(career_df: pd.DataFrame) -> np.ndarray:
    """Return per-player caliber in [0, 1]. Prefer APFV if present, else PFV."""
    from analytics.player_profiles.archetypes import (
        calculate_apfv_batch_by_height, calculate_adjusted_pfv, height_bucket,
    )
    apfvs: list[float] = []
    buckets: list[str] = []
    for _, row in career_df.iterrows():
        metrics = {}
        for col in ["pts_per36", "reb_per36", "ast_per36", "blk_per36",
                    "stl_per36", "ts_pct", "mpg"]:
            pct_col = f"{col}_career_pctile"
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
    """Composite NBA -> NBA similarity.

    score(i, j) = playstyle_aff * size_aff * caliber_aff * era_aff

    where each affinity is a Gaussian-kernel in its own metric:
      - playstyle: weighted Euclidean in career-percentile space [0,1]^d
      - size:      |height_i - height_j| in inches
      - caliber:   |apfv_i - apfv_j| in [0,1]
      - era:       |start_year_i - start_year_j| with a long half-life so
                   matches across decades aren't suppressed too hard

    Same-role players also get a small (≤7%) bonus so a "Playmaker" doesn't
    surface as the top comp for a "Designated Scorer".
    """
    if career_df.empty:
        return {}

    feature_by_id = feature_df.set_index("player_id") if not feature_df.empty else None
    career = career_df.reset_index(drop=True)

    player_ids = career["player_id"].astype(int).to_numpy()
    n = len(player_ids)
    if n < 2:
        return {str(int(pid)): [] for pid in player_ids}

    # ---- feature matrices
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

    # ---- pairwise playstyle Euclidean -> affinity
    sq = (
        (P ** 2).sum(axis=1)[:, None]
        + (P ** 2).sum(axis=1)[None, :]
        - 2.0 * P @ P.T
    )
    sq = np.maximum(sq, 0.0)
    dists = np.sqrt(sq)
    playstyle_aff = np.exp(-dists / NBA_NBA_PLAYSTYLE_BANDWIDTH)

    # ---- pairwise affinities
    size_aff = np.exp(-((heights[:, None] - heights[None, :]) ** 2) / (2.0 * NBA_NBA_SIZE_SIGMA ** 2))
    # add a soft weight term too (lbs differ ~4x more than inches; sigma=36 ≈ ~5"-equivalent)
    weight_aff = np.exp(-((weights[:, None] - weights[None, :]) ** 2) / (2.0 * 36.0 ** 2))
    # 4th root (instead of sqrt of product) softens size further so playstyle dominates
    size_aff = (size_aff * weight_aff) ** 0.25

    caliber_aff = np.exp(-((caliber[:, None] - caliber[None, :]) ** 2) / (2.0 * NBA_NBA_CALIBER_SIGMA ** 2))

    # era half-life: exp(-ln2 * |dt|/halflife)
    era_aff = np.exp(-np.log(2.0) * np.abs(years[:, None] - years[None, :]) / NBA_NBA_ERA_HALFLIFE)
    # Clamp so even very-cross-era pairs keep most of the composite score.
    era_aff = np.maximum(era_aff, NBA_NBA_ERA_FLOOR)

    # role match bonus (multiplicative, capped)
    role_match = (roles[:, None] == roles[None, :]).astype(float)
    role_factor = 1.0 + (NBA_NBA_ROLE_BONUS - 1.0) * role_match

    composite = playstyle_aff * size_aff * caliber_aff * era_aff * role_factor

    # zero out self-pairs
    np.fill_diagonal(composite, -np.inf)

    # Precompute era buckets for diversification.
    era_buckets = (years // NBA_NBA_ERA_BUCKET_YEARS).astype(int)

    out: dict[str, list[dict[str, Any]]] = {}
    career_by_id = career.set_index("player_id")
    pool_size = min(k * NBA_NBA_ERA_POOL_MULT, len(player_ids))
    for i, pid in enumerate(player_ids):
        # Pull a large ranked pool; then pick top-K with era-bucket caps so
        # the same 5-year window cannot fill more than ~2 of the K spots
        # (until we exhaust diverse candidates).
        ranked = np.argsort(composite[i])[::-1][:pool_size]
        bucket_counts: dict[int, int] = {}
        chosen: list[int] = []
        leftovers: list[int] = []
        for j in ranked:
            j = int(j)
            if int(player_ids[j]) == int(pid):
                continue
            b = int(era_buckets[j])
            if bucket_counts.get(b, 0) < NBA_NBA_ERA_MAX_PER_BUCKET:
                chosen.append(j)
                bucket_counts[b] = bucket_counts.get(b, 0) + 1
                if len(chosen) >= k:
                    break
            else:
                leftovers.append(j)
        # Top up from leftovers if cap left us short of k.
        for j in leftovers:
            if len(chosen) >= k:
                break
            chosen.append(j)

        payloads: list[dict[str, Any]] = []
        for j in chosen:
            other_id = int(player_ids[j])
            other = career_by_id.loc[other_id]
            score = float(composite[i, j])
            payloads.append(
                {
                    "player_id": other_id,
                    "player_name": str(other["player_name"]),
                    "similarity_score": similarity_pct(max(0.0, min(1.0, score))),
                    "career_span": str(other.get("career_span", "")),
                    "explanation": explain_similarity(
                        feature_by_id.loc[int(pid)] if feature_by_id is not None and int(pid) in feature_by_id.index else pd.Series(dtype=float),
                        feature_by_id.loc[other_id] if feature_by_id is not None and other_id in feature_by_id.index else pd.Series(dtype=float),
                    ),
                }
            )
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
