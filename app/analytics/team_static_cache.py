from __future__ import annotations

import pandas as pd

from analytics.playstyle import assign_badges
from analytics.style_vector import build_style_vector_lookup


def merge_similar_teams_with_abbreviations(
    similar_index: dict[str, list[dict]],
    team_metadata: dict[str, dict],
) -> dict[str, list[dict]]:
    meta = team_metadata or {}
    out: dict[str, list[dict]] = {}
    for key, lst in (similar_index or {}).items():
        merged: list[dict] = []
        for item in lst:
            tid = str(item["team_id"])
            abbr = str(meta.get(tid, {}).get("abbreviation", ""))
            merged.append({**item, "abbreviation": abbr})
        out[key] = merged
    return out


def build_team_profile_static_cache(
    df: pd.DataFrame,
    norm_df: pd.DataFrame,
    similar_with_abbr: dict[str, list[dict]],
) -> dict[str, dict]:
    style_by_key = build_style_vector_lookup(df)
    hist_ix = df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first").set_index(
        ["TEAM_ID", "SEASON"]
    )
    norm_ix = norm_df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first").set_index(
        ["TEAM_ID", "SEASON"]
    )

    cache: dict[str, dict] = {}
    for key_tuple in hist_ix.index:
        if key_tuple not in norm_ix.index:
            continue
        tid, season = int(key_tuple[0]), str(key_tuple[1])
        sk = f"{tid}:{season}"
        row = hist_ix.loc[key_tuple]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        norm_row = norm_ix.loc[key_tuple]
        if isinstance(norm_row, pd.DataFrame):
            norm_row = norm_row.iloc[0]

        badges = assign_badges(norm_row)
        cache[sk] = {
            "team_name": str(row["TEAM_NAME"]),
            "record": f"{int(row['W'])}-{int(row['L'])}",
            "win_pct": round(float(row["W_PCT"]), 3),
            "style_vector": style_by_key.get(sk, {}),
            "badges": [
                {"id": b.id, "label": b.label, "emoji": b.emoji, "description": b.description}
                for b in badges
            ],
            "similar_teams": similar_with_abbr.get(sk),
        }
    return cache
