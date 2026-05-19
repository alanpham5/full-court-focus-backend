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
        "Primary offensive engine who organizes possessions through passing, "
        "advantage creation, and ball distribution."
    ),
    "Secondary Creator": (
        "Score-first creator who can handle secondary initiation, attack tilted "
        "defenses, and create offense for teammates."
    ),
    "Designated Scorer": (
        "High-volume self-creator whose primary value comes from shot creation, "
        "difficult scoring, and late-clock offense."
    ),
    "Perimeter Specialist": (
        "Floor-spacing offensive player who provides shooting gravity and "
        "efficient perimeter scoring."
    ),
    "Rim Attacker": (
        "Downhill offensive threat who pressures the paint through drives, cuts, "
        "transition, and rim finishing."
    ),
    "Interior Presence": (
        "Paint-focused interior player providing rebounding, rim protection, "
        "screening, rolling, and interior scoring."
    ),
    "Defensive Specialist": (
        "Defense-oriented player whose primary value comes from disruption, "
        "containment, switching, or rim deterrence."
    ),
}

MIN_TOP_ROLE_SCORE = 0.45
MIN_SECOND_ROLE_SCORE = 0.55
MAX_SCORE_GAP_FOR_SECOND = 0.16
MAX_ROLES_PER_PLAYER = 2
FALLBACK_ROLE = "Secondary Creator"

PLAYMAKER_AST_MIN = 6.0
PLAYMAKER_AST_NORM_MIN = 0.78
PLAYMAKER_SCORE_MIN = 0.75


def _norm_col(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())

    if hi <= lo:
        return pd.Series(0.5, index=s.index)

    return (s - lo) / (hi - lo)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, 1)
    return numerator / denominator


def _role_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    pts = _norm_col(df["PTS"])
    ast = _norm_col(df["AST"])
    reb = _norm_col(df["REB"])

    fg3a = _norm_col(df.get("FG3A", 0))
    fg3p = _norm_col(df.get("FG3_PCT", 0))

    blk = _norm_col(df.get("BLK", 0))
    stl = _norm_col(df.get("STL", 0))

    fga = pd.to_numeric(df.get("FGA", 1), errors="coerce").fillna(1.0)

    ast_raw = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    playmaking_ratio = _norm_col(_safe_ratio(ast_raw, fga))

    paint_focus = 1.0 - fg3a

    return pd.DataFrame(
        {
            "Playmaker": (
                ast * 1.15
                + playmaking_ratio * 0.55
                - pts * 0.08
            ),
            "Secondary Creator": (
                ast * 0.9
                + pts * 0.45
                + stl * 0.05
            ),
            "Designated Scorer": (
                pts * 1.15
                - ast * 0.05
            ),
            "Perimeter Specialist": (
                fg3a * 0.75
                + fg3p * 0.9
                + pts * 0.1
            ),
            "Rim Attacker": (
                pts * 0.7
                + paint_focus * 0.4
                + ast * 0.05
            ),
            "Interior Presence": (
                reb * 0.6
                + blk * 0.9
                + paint_focus * 0.35
            ),
            "Defensive Specialist": (
                stl * 0.9
                + blk * 0.5
                + paint_focus * 0.1
            ),
        },
        index=df.index,
    )


def assign_player_roles(players_df: pd.DataFrame) -> dict[int, list[str]]:
    if players_df.empty:
        return {}

    df = players_df.copy().reset_index(drop=True)

    df["PLAYER_ID"] = (
        pd.to_numeric(df["PLAYER_ID"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    scores = _role_score_frame(df)

    ast = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    ast_norm = _norm_col(ast)

    assignments: dict[int, list[str]] = {}

    for idx in range(len(df)):
        pid = int(df.at[idx, "PLAYER_ID"])

        row_scores = scores.iloc[idx].sort_values(ascending=False)

        ranked = [
            (str(role), float(score))
            for role, score in row_scores.items()
        ]

        if not ranked or ranked[0][1] < MIN_TOP_ROLE_SCORE:
            fallback = (
                "Playmaker"
                if (
                    float(ast.iat[idx]) >= PLAYMAKER_AST_MIN
                    and float(ast_norm.iat[idx]) >= PLAYMAKER_AST_NORM_MIN
                    and float(scores.at[idx, "Playmaker"]) >= PLAYMAKER_SCORE_MIN
                )
                else FALLBACK_ROLE
            )

            assignments[pid] = [fallback]
            continue

        top_role, top_score = ranked[0]

        roles = [top_role]

        if len(ranked) > 1:
            second_role, second_score = ranked[1]

            if (
                second_score >= MIN_SECOND_ROLE_SCORE
                and (top_score - second_score) <= MAX_SCORE_GAP_FOR_SECOND
                and second_role not in roles
                and len(roles) < MAX_ROLES_PER_PLAYER
            ):
                roles.append(second_role)

        assignments[pid] = roles

    return assignments