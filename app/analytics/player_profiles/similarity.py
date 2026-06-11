from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from analytics.similarity_scoring import similarity_pct


def _nba_nba_display_score(composite: float) -> float:
    """Map a raw composite score (~[0, 1.4]) to a calibrated display %.

    The previous code passed the composite straight through similarity_pct
    which clipped to [0,1] then raised to the 5.5 power -- so any composite
    >= 1.0 read as 100%. With the era reward and role bonus, this happens
    often and produces too many 100s. This function:
      - normalizes composite by NBA_NBA_DISPLAY_NORMALIZER (so a "twin"
        match saturates without ever hitting 100)
      - applies a soft power curve with NBA_NBA_DISPLAY_GAMMA
      - caps at NBA_NBA_DISPLAY_CEILING (so the very best read in the
        high 70s, strong matches 50-70, typical 30-50)
    """
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
    1.9,   # reb_per36   -- separates scoring wing from scoring big
    1.7,   # ast_per36   -- creation signal
    1.6,   # blk_per36   -- rim protection
    1.2,   # stl_per36
    0.5,   # tov_per36   -- noisy
    1.9,   # fg3a_rate   -- 3pt volume is a defining axis
    1.1,   # fta_rate
    0.7,   # ts_pct
    0.6,   # efg_pct     -- correlated with ts_pct
    2.1,   # ast_pct     -- "passing big" + "playmaking guard" signal
    0.6,   # mpg
])

NBA_NBA_PLAYSTYLE_BANDWIDTH = 2.2          # more forgiving playstyle metric so
                                           # moderately-similar pairs still
                                           # rank inside top-10
NBA_NBA_ARCHETYPE_BONUS_PER_OVERLAP = 0.18 # archetype overlap factor: shared
                                           # box-score archetype is a strong
                                           # signal (e.g. "rebounding
                                           # defender" for passing bigs,
                                           # "high-volume creator" for lead
                                           # guards). Each overlap multiplies
                                           # composite by 1 + 0.18.
NBA_NBA_SIZE_SIGMA = 12.0         # inches: very soft size prior so playstyle
                                  # similarity drives the comp; a 6" mismatch
                                  # only costs ~12% (keeps "long guards"
                                  # category from being driven apart by minor
                                  # size differences between 6'3" and 6'5")
NBA_NBA_CALIBER_SIGMA = 0.35      # broader so role-player and star versions
                                  # of the same archetype still match (e.g.
                                  # Jokic <-> Sengun, despite caliber gap)
NBA_NBA_ROLE_BONUS = 1.07         # multiply by 1.07 if same role (capped)
#
# Era handling: instead of a decay penalty (which clusters same-era pairs)
# we apply a continuous CROSS-ERA REWARD. Two players with the same start
# year get factor 1.0; far-apart pairs get up to (1 + NBA_NBA_ERA_REWARD).
# The reward saturates on a soft exponential — NBA_NBA_ERA_TIMESCALE
# controls how fast it ramps up.
#
NBA_NBA_ERA_REWARD = 0.10         # gentle cross-era nudge -- keeps a few
                                  # cross-era comps showing up but doesn't
                                  # push same-era network peers (which are
                                  # usually the right answer) off top-10
NBA_NBA_ERA_TIMESCALE = 14.0      # seasons: distance at which boost reaches
                                  # ~63% of its asymptotic value

# Display calibration: composite scores fall roughly in [0, 1.4] (size /
# caliber affinities can hit 1.0, era reward adds up to 22%, role bonus
# 7%). similarity_pct() clips to [0,1] then x^5.5 saturates at 100 — so
# any composite ≥ 1.0 read as 100%. We replace it with a soft ceiling at
# ~78% and a gentler curve so most top-10 entries land in 40-70%.
NBA_NBA_DISPLAY_CEILING = 78.0    # top end of displayed % range
NBA_NBA_DISPLAY_NORMALIZER = 1.45 # composite value that saturates the cap
                                  # (era 1.10 × role 1.07 × archetype-overlap
                                  # up to 1.36 → max ≈ 1.6); set just below
                                  # the max so only true-twin pairs read at
                                  # the 78% ceiling
NBA_NBA_DISPLAY_GAMMA = 2.0       # curvature; 2.0 is a gentle parabola


def _percentile_matrix(career_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return (n_players, n_features) of career percentile / 100 in [0,1]."""
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
    # Cross-era REWARD: same-year pairs get 1.0, larger |dt| approaches
    # (1 + NBA_NBA_ERA_REWARD). This actively boosts cross-era candidates
    # so same-era playstyle-correlation pile-ups get diversified.
    dt = np.abs(years[:, None] - years[None, :])
    era_aff = 1.0 + NBA_NBA_ERA_REWARD * (1.0 - np.exp(-dt / NBA_NBA_ERA_TIMESCALE))

    # role match bonus (multiplicative, capped)
    role_match = (roles[:, None] == roles[None, :]).astype(float)
    role_factor = 1.0 + (NBA_NBA_ROLE_BONUS - 1.0) * role_match

    # Archetype-overlap bonus: each shared box-score archetype multiplies
    # the composite by (1 + NBA_NBA_ARCHETYPE_BONUS_PER_OVERLAP).
    # This naturally clusters concept groups: passing centers all share
    # "rebounding defender"; lead guards share "high-volume creator"; etc.
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

    # zero out self-pairs
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
