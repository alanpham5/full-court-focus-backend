from __future__ import annotations

from typing import Any

import pandas as pd

from analytics.player_profiles.features import PLAYSTYLE_METRIC_KEYS


def _v(row: pd.Series, key: str) -> float:
    val = pd.to_numeric(row.get(key, 0.0), errors="coerce")
    return float(val) if pd.notna(val) else 0.0


def get_player_archetypes(row: pd.Series) -> list[str]:
    pts = _v(row, "pts_per36_z")
    ast = _v(row, "ast_per36_z")
    fg3 = _v(row, "fg3a_rate_z")
    fta = _v(row, "fta_rate_z")
    efg = _v(row, "efg_pct_z")
    tov = _v(row, "tov_per36_z")
    blk = _v(row, "blk_per36_z")
    stl = _v(row, "stl_per36_z")
    reb = _v(row, "reb_per36_z")

    matches = []

    # 1. High-Volume Creator
    if pts > 0.9 and ast > 0.7:
        matches.append("high-volume creator")
    # 2. Perimeter Scorer
    if fg3 > 0.8 and efg > 0.2:
        matches.append("perimeter scorer")
    # 3. Free-Throw Pressure Scorer
    if fta > 0.8 and pts > 0.4:
        matches.append("free-throw pressure scorer")
    # 4. Table Setter
    if ast > 0.8 and tov < 0.5:
        matches.append("table setter")
    # 5. Efficient Finisher
    if efg > 0.8 and pts < 0.6:
        matches.append("efficient finisher")
    # 6. Rim Protection
    if blk > 0.9 and reb > 0.4:
        matches.append("rim protection")
    # 7. 3-and-D Profile
    if stl > 0.8 and fg3 > 0.2:
        matches.append("3-and-D profile")
    # 8. Rebounding Defender
    if reb > 0.9:
        matches.append("rebounding defender")
    # 9. Event Creator
    if stl > 0.75 or blk > 0.75:
        matches.append("event creator")

    # Limit to up to 3 archetypes
    if matches:
        return matches[:3]

    # Fallbacks if none matched
    fallbacks = []
    if pts > 0.0 or ast > 0.0:
        fallbacks.append("balanced scorer")
    if reb > 0.0 or blk > 0.0 or stl > 0.0:
        fallbacks.append("box-score defender")

    if not fallbacks:
        fallbacks = ["balanced scorer", "box-score defender"]
    return fallbacks[:3]


def assign_player_role(row: pd.Series) -> str:
    pts = _v(row, "pts_per36_z")
    ast = _v(row, "ast_per36_z")
    reb = _v(row, "reb_per36_z")
    stl = _v(row, "stl_per36_z")
    blk = _v(row, "blk_per36_z")
    ts = _v(row, "ts_pct_z")
    efg = _v(row, "efg_pct_z")
    ast_pct = _v(row, "ast_pct_z")
    fg3 = _v(row, "fg3a_rate_z")
    fta = _v(row, "fta_rate_z")
    pos_group = str(row.get("position_group", "")).upper()

    # Rule 1: Playmaker
    if ast > 1.0 and ast_pct > 0.8:
        return "Playmaker"

    # Rule 2: Interior Presence
    if pos_group == "B" and fg3 < -0.2:
        return "Interior Presence"
    if blk > 1.0 and reb > 0.8 and fg3 < 0.0:
        return "Interior Presence"

    # Rule 3: Designated Scorer
    if pts > 1.0 and ast_pct < 0.6:
        return "Designated Scorer"

    # Rule 4: Secondary Creator
    if ast > 0.3 and (pts > 0.2 or ast_pct > 0.3):
        return "Secondary Creator"

    # Rule 5: Perimeter Specialist
    if fg3 > 0.8 and efg > -0.5:
        return "Perimeter Specialist"

    # Rule 6: Rim Attacker
    if fta > 0.5 and pts > 0.0:
        return "Rim Attacker"

    # Rule 7: Defensive Specialist
    if (stl > 0.5 or blk > 0.5) and pts < -0.2:
        return "Defensive Specialist"

    # Fallbacks based on position group/stats
    if pos_group == "B":
        return "Interior Presence"
    if fg3 > 0.3:
        return "Perimeter Specialist"
    if ast > 0.0:
        return "Secondary Creator"
    if pts > 0.0:
        return "Rim Attacker"
    return "Defensive Specialist"


def add_archetypes(career_df: pd.DataFrame) -> pd.DataFrame:
    out = career_df.copy()
    out["archetypes"] = out.apply(get_player_archetypes, axis=1)
    out["role"] = out.apply(assign_player_role, axis=1)
    return out


def style_summary(row: pd.Series) -> dict[str, dict[str, float]]:
    return {
        k: {
            "value": round(_v(row, k), 4),
            "percentile": round(_v(row, f"{k}_career_pctile"), 1),
        }
        for k in PLAYSTYLE_METRIC_KEYS
    }


def profile_payload(row: pd.Series, similar: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "player_id": int(row["player_id"]),
        "player_name": str(row["player_name"]),
        "height": _clean(row.get("height")),
        "weight": _clean(row.get("weight")),
        "draft_year": _clean(row.get("draft_year")),
        "draft_position": _clean(row.get("draft_position")),
        "role": _clean(row.get("role")),
        "career_teams": row.get("career_teams", []),
        "career_span": str(row.get("career_span", "")),
        "career_games": int(row.get("career_games", 0) or 0),
        "archetypes": list(row.get("archetypes", [])),
        "playstyle_metrics": style_summary(row),
        "similar_players": similar or [],
    }


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    return value
