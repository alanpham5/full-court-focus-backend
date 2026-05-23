from __future__ import annotations

import math
from typing import Any

import pandas as pd

from analytics.playstyle import ALL_BADGES, BADGE_MAP, assign_badges, exemplar_score


def _badge_payload(badge_id: str) -> dict[str, str]:
    b = BADGE_MAP[badge_id]
    return {
        "id": b.id,
        "label": b.label,
        "emoji": b.emoji,
        "description": b.description,
    }


def _team_entry(
    team_id: int,
    row: pd.Series,
    *,
    team_metadata: dict[str, dict],
    score: float,
) -> dict[str, Any]:
    meta = team_metadata.get(str(team_id), {})
    return {
        "team_id": team_id,
        "team_name": str(row["TEAM_NAME"]),
        "abbreviation": str(meta.get("abbreviation", "")),
        "record": f"{int(row['W'])}-{int(row['L'])}",
        "exemplar_score": round(score, 3),
    }


def _score_out_of_100(score: float) -> float:
    return max(0.0, min(100.0, 100.0 / (1.0 + math.exp(-score))))


def build_badge_leaders_index(
    df: pd.DataFrame,
    norm_df: pd.DataFrame,
    team_metadata: dict[str, dict],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per season, per badge: top 2 badge holders ranked by exemplar_score."""
    hist_ix = df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first").set_index(
        ["TEAM_ID", "SEASON"]
    )
    norm_ix = norm_df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first").set_index(
        ["TEAM_ID", "SEASON"]
    )

    seasons: dict[str, list[tuple[int, pd.Series, pd.Series]]] = {}
    for key_tuple in hist_ix.index:
        if key_tuple not in norm_ix.index:
            continue
        tid, season = int(key_tuple[0]), str(key_tuple[1])
        hist_row = hist_ix.loc[key_tuple]
        if isinstance(hist_row, pd.DataFrame):
            hist_row = hist_row.iloc[0]
        norm_row = norm_ix.loc[key_tuple]
        if isinstance(norm_row, pd.DataFrame):
            norm_row = norm_row.iloc[0]
        seasons.setdefault(season, []).append((tid, hist_row, norm_row))

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for season_key, pairs in seasons.items():
        out[season_key] = {}
        earned_by_team = [
            ({b.id for b in assign_badges(norm_row)}, tid, hist_row, norm_row)
            for tid, hist_row, norm_row in pairs
        ]
        for badge in ALL_BADGES:
            scored = [
                (exemplar_score(badge.id, norm_row), tid, hist_row)
                for earned, tid, hist_row, norm_row in earned_by_team
                if badge.id in earned
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            entry: dict[str, Any] = {"badge": _badge_payload(badge.id)}
            if scored:
                top_score, top_tid, top_row = scored[0]
                entry["top"] = _team_entry(
                    top_tid,
                    top_row,
                    team_metadata=team_metadata,
                    score=_score_out_of_100(top_score),
                )
            else:
                entry["top"] = None
            if len(scored) >= 2:
                runner_score, runner_tid, runner_row = scored[1]
                entry["runner_up"] = _team_entry(
                    runner_tid,
                    runner_row,
                    team_metadata=team_metadata,
                    score=_score_out_of_100(runner_score),
                )
            else:
                entry["runner_up"] = None
            out[season_key][badge.id] = entry
    return out
