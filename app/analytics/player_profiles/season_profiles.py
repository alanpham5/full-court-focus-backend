"""Per-season player profiles.

Builds the same payload shape as the career player endpoint, but scoped to a
single season, from `player_season_features.parquet`. Everything is recomputed
within that season's player pool:

- playstyle metrics (per-36 / rate values + within-season percentile),
- archetypes and role (from within-season z-scores),
- PFV/APFV (impact + versatility, §1.1-1.2, ranked within the season),
- similar players (a season-specific similarity index over that season's pool).

Bio fields (height, weight, draft) are season-invariant and are taken from the
career profile when available, falling back to the season row.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from analytics.player_profiles.archetypes import (
    assign_player_role,
    calculate_impact_apfv_batch,
    calculate_impact_pfv,
    display_percentile,
    get_player_archetypes,
)
from analytics.player_profiles.features import PLAYSTYLE_METRIC_KEYS, STYLE_FEATURES
from analytics.player_profiles.similarity import build_similarity_index

SEASON_CREDIBILITY_MINUTES = 1500.0


def normalize_season(season: str) -> str | None:
    """Accept "2024-25" or "2024" and return the canonical "YYYY-YY" form."""
    s = str(season).strip()
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}", s):
        start = int(s)
        return f"{start}-{str(start + 1)[-2:]}"
    return None


def _season_credibility(minutes: float) -> float:
    return min(1.0, (max(float(minutes or 0.0), 0.0) / SEASON_CREDIBILITY_MINUTES) ** 0.5)


def _season_apfvs(frame: pd.DataFrame) -> tuple[list[str], list[float], list[float]]:
    """Per-player (pid, apfv, minutes) for one season frame.

    Mirrors the APFV the season-profile endpoint serves: within-season impact PFV
    (season credibility) and workload, ranked within that season's pool.
    """
    frame = frame.reset_index(drop=True)
    pfvs: list[float] = []
    mpg_pcts: list[float] = []
    minutes: list[float] = []
    pids: list[str] = []
    for _, row in frame.iterrows():
        metrics = {
            key: {
                "value": float(row.get(key, 0.0) or 0.0),
                "percentile": float(row.get(f"{key}_pctile", 50.0) or 50.0),
            }
            for key in PLAYSTYLE_METRIC_KEYS
        }
        pfvs.append(calculate_impact_pfv(metrics, credibility=_season_credibility(row.get("MIN", 0.0))))
        mpg_pcts.append(float(row.get("mpg_pctile", 0.0) or 0.0))
        minutes.append(float(row.get("MIN", 0.0) or 0.0))
        pids.append(str(int(row["PLAYER_ID"])))
    apfvs = calculate_impact_apfv_batch(pfvs, mpg_pcts, minutes)
    return pids, apfvs, minutes


def career_apfv_from_seasons(season_features: pd.DataFrame) -> dict[str, float]:
    """Career APFV as the minutes-weighted mean of a player's per-season APFVs.

    Each season APFV is computed exactly as the season-profile endpoint serves it
    (§ `build_season_bundle`), then blended across the player's seasons weighted by
    minutes played. Because the result is a weighted average of the season values,
    a career APFV stays within the range of the player's actual seasons — it can
    never exceed their best season — and lives on the same scale as the season
    selector. This replaces ranking career-*aggregate* rates against the all-career
    pool, which inflated efficient low-usage role players (e.g. a career APFV above
    every individual season) and disagreed with both the season APFVs and the
    position-relative radar.
    """
    if season_features.empty or "SEASON" not in season_features.columns:
        return {}
    weighted_sum: dict[str, float] = {}
    weight_total: dict[str, float] = {}
    for _season, frame in season_features.groupby("SEASON"):
        pids, apfvs, minutes = _season_apfvs(frame)
        for pid, apfv, mins in zip(pids, apfvs, minutes):
            w = max(float(mins), 0.0)
            if w <= 0.0:
                continue
            weighted_sum[pid] = weighted_sum.get(pid, 0.0) + apfv * w
            weight_total[pid] = weight_total.get(pid, 0.0) + w
    return {
        pid: round(weighted_sum[pid] / weight_total[pid], 4)
        for pid in weighted_sum
        if weight_total[pid] > 0.0
    }


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_season_bundle(
    season_frame: pd.DataFrame,
    career_profiles: dict[str, dict],
) -> dict[str, dict]:
    """Build per-player season payloads for one season's frame.

    `season_frame` is all rows of `player_season_features` for a single SEASON.
    Returns a dict keyed by player_id (str) of PlayerProfileResponse-shaped dicts.
    """
    frame = season_frame.reset_index(drop=True).copy()
    if frame.empty:
        return {}

    season = str(frame["SEASON"].iloc[0])

    frame["player_id"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce").fillna(0).astype(int)
    frame["player_name"] = frame["PLAYER_NAME"].astype(str)
    frame["height"] = frame["HEIGHT"]
    frame["weight"] = frame["WEIGHT"]

    for key in PLAYSTYLE_METRIC_KEYS:
        src = f"{key}_pctile"
        if src in frame.columns:
            frame[f"{key}_global_pctile"] = frame[src]

    frame["role"] = frame.apply(assign_player_role, axis=1)
    frame["archetypes"] = frame.apply(get_player_archetypes, axis=1)
    frame["first_season"] = season
    frame["career_span"] = season

    pfvs: list[float] = []
    for _, row in frame.iterrows():
        metrics = {
            key: {
                "value": float(row.get(key, 0.0) or 0.0),
                "percentile": float(row.get(f"{key}_pctile", 50.0) or 50.0),
            }
            for key in PLAYSTYLE_METRIC_KEYS
        }
        pfvs.append(calculate_impact_pfv(metrics, credibility=_season_credibility(row.get("MIN", 0.0))))

    mpg_pctiles = [float(v or 0.0) for v in frame.get("mpg_pctile", pd.Series([0.0] * len(frame)))]
    minutes = [float(v or 0.0) for v in frame.get("MIN", pd.Series([0.0] * len(frame)))]
    apfvs = calculate_impact_apfv_batch(pfvs, mpg_pctiles, minutes)

    feature_cols = [c for c in STYLE_FEATURES if c in frame.columns]
    feature_df = frame[["player_id", *feature_cols]].copy()
    similar = build_similarity_index(frame, pd.DataFrame(), feature_df, k=10)

    bundle: dict[str, dict] = {}
    for i, row in frame.iterrows():
        pid = str(int(row["player_id"]))
        career = career_profiles.get(pid, {})

        credibility = _season_credibility(row.get("MIN", 0.0))
        playstyle_metrics = {}
        for key in PLAYSTYLE_METRIC_KEYS:
            playstyle_metrics[key] = {
                "value": round(float(row.get(key, 0.0) or 0.0), 4),
                "percentile": display_percentile(
                    key, float(row.get(f"{key}_pctile", 50.0) or 50.0), credibility
                ),
            }

        team = str(row.get("TEAM_ABBREVIATION", "") or "").strip()
        bundle[pid] = {
            "player_id": int(row["player_id"]),
            "player_name": str(career.get("player_name") or row.get("PLAYER_NAME", "")),
            "height": _clean(career.get("height", row.get("HEIGHT"))),
            "weight": _clean(career.get("weight", row.get("WEIGHT"))),
            "draft_year": _clean(career.get("draft_year", row.get("DRAFT_YEAR"))),
            "draft_position": _clean(career.get("draft_position", row.get("DRAFT_NUMBER"))),
            "role": str(row["role"]),
            "career_teams": [team] if team else [],
            "career_span": season,
            "career_games": int(row.get("GP", 0) or 0),
            "archetypes": list(row["archetypes"]),
            "playstyle_metrics": playstyle_metrics,
            "pfv": pfvs[i],
            "apfv": apfvs[i],
            "similar_players": similar.get(pid, []),
        }

    return bundle
