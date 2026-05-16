"""Assign up to two lineup roles per player from per-game box-score stats."""

from __future__ import annotations

import pandas as pd

ALL_ROLES: tuple[str, ...] = (
    "Playmaker",
    "Secondary Creator",
    "Designated Scorer",
    "Perimeter Specialist",
    "Rim Attacker",
    "Interior Presence",
    "Defensive Specialist",
)

ROLE_DESCRIPTIONS: dict[str, str] = {
    "Playmaker": (
        "High-usage initiator who runs offense through passing, pick-and-rolls, "
        "and creating advantages for others."
    ),
    "Secondary Creator": (
        "Off-ball or second-side handler who can pass into actions, punish rotations, "
        "and create offense without being the main engine."
    ),
    "Designated Scorer": (
        "Player whose value is self-generated scoring: pull-ups, isolations, "
        "tough shot conversion, and late-clock offense."
    ),
    "Perimeter Specialist": (
        "Spacing-focused scorer who primarily provides catch-and-shoot gravity "
        "and punishes help defense from range."
    ),
    "Rim Attacker": (
        "Downhill scorer who pressures the paint through drives, cuts, "
        "transition, and rim finishing."
    ),
    "Interior Presence": (
        "Paint-focused big skill set: screening, rolling, post scoring, "
        "offensive rebounding, rim protection, and paint deterrence."
    ),
    "Defensive Specialist": (
        "Primary defensive role player defined by matchup coverage, switching, "
        "on-ball containment, help defense, or rim/wing disruption."
    ),
}

MIN_TOP_ROLE_SCORE = 0.45
MIN_SECOND_ROLE_SCORE = 0.50
# Second role only when nearly as strong a fit as the first (clear specialists keep 1).
MAX_SCORE_GAP_FOR_SECOND = 0.18
MAX_ROLES_PER_PLAYER = 2
FALLBACK_ROLE = "Secondary Creator"
PLAYMAKER_AST_MIN = 4.0
PLAYMAKER_AST_NORM_MIN = 0.60
PLAYMAKER_SCORE_MIN = 0.55


def _norm_col(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _role_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    pts = _norm_col(df["PTS"])
    ast = _norm_col(df["AST"])
    fg3a = _norm_col(df.get("FG3A", 0))
    fg3p = _norm_col(df.get("FG3_PCT", 0))
    reb = _norm_col(df["REB"])
    blk = _norm_col(df.get("BLK", 0))
    stl = _norm_col(df.get("STL", 0))
    paint_focus = (1.0 - fg3a)

    return pd.DataFrame(
        {
            "Playmaker": ast * 1.15 + pts * 0.08,
            "Secondary Creator": ast * 0.7 + pts * 0.28 + stl * 0.08,
            "Designated Scorer": pts * 1.05 + ast * 0.05,
            "Perimeter Specialist": fg3a * 0.75 + fg3p * 0.85 + pts * 0.1,
            "Rim Attacker": pts * 0.65 + paint_focus * 0.4 + ast * 0.05,
            "Interior Presence": reb * 0.55 + blk * 0.85 + paint_focus * 0.35,
            "Defensive Specialist": stl * 0.85 + blk * 0.45 + (1.0 - fg3a) * 0.15,
        },
        index=df.index,
    )


def assign_player_roles(players_df: pd.DataFrame) -> dict[int, list[str]]:
    """Return player_id → one or two roles they best exemplify (by relative fit)."""
    if players_df.empty:
        return {}

    df = players_df.copy().reset_index(drop=True)
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype(int)
    scores = _role_score_frame(df)
    ast = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    ast_norm = _norm_col(ast)
    assignments: dict[int, list[str]] = {}

    for idx in range(len(df)):
        pid = int(df.at[idx, "PLAYER_ID"])
        row_scores = scores.iloc[idx].sort_values(ascending=False)
        ranked = [(str(role), float(score)) for role, score in row_scores.items()]
        playmaker_score = float(scores.at[idx, "Playmaker"])
        playmaker_by_usage = (
            float(ast.iat[idx]) >= PLAYMAKER_AST_MIN
            and float(ast_norm.iat[idx]) >= PLAYMAKER_AST_NORM_MIN
            and playmaker_score >= PLAYMAKER_SCORE_MIN
        )

        if not ranked or ranked[0][1] < MIN_TOP_ROLE_SCORE:
            assignments[pid] = ["Playmaker"] if playmaker_by_usage else [FALLBACK_ROLE]
            continue

        top_score = ranked[0][1]
        roles = [ranked[0][0]]
        if playmaker_by_usage and "Playmaker" not in roles:
            roles.append("Playmaker")
        if len(ranked) > 1:
            second_score = ranked[1][1]
            gap = top_score - second_score
            if (
                len(roles) < MAX_ROLES_PER_PLAYER
                and second_score >= MIN_SECOND_ROLE_SCORE
                and gap <= MAX_SCORE_GAP_FOR_SECOND
            ):
                roles.append(ranked[1][0])

        assignments[pid] = roles

    return assignments
