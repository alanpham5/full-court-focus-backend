from dataclasses import dataclass

import pandas as pd


@dataclass
class Badge:
    id: str
    label: str
    emoji: str
    description: str


ALL_BADGES = [
    Badge(
        "run_and_gun",
        "Run-and-Gun",
        "🚀",
        "Elite pace and transition scoring",
    ),
    Badge(
        "deep_threat",
        "Deep Threat",
        "🎯",
        "High three-point volume, elite deep shooting",
    ),
    Badge(
        "paint_punishers",
        "Paint Punishers",
        "🏰",
        "Interior-focused offense built around post play and rim scoring",
    ),
    Badge(
        "defensive_wall",
        "Defensive Wall",
        "🧱",
        "Elite team defense that consistently suppresses scoring",
    ),
    Badge(
        "playmakers",
        "Playmakers",
        "🎪",
        "High assist rates with strong ball security",
    ),
    Badge(
        "grind_it_out",
        "Grind-It-Out",
        "⚙️",
        "Slow pace, disciplined execution, and half-court efficiency",
    ),
    Badge(
        "glass_eaters",
        "Glass Eaters",
        "💪",
        "Dominates the boards and creates extra possessions",
    ),
    Badge(
        "foul_magnets",
        "Foul Magnets",
        "🆓",
        "Relentlessly attacks to generate free throws",
    ),
    Badge(
        "iso_kings",
        "Isolation Kings",
        "👑",
        "Relies on one-on-one shot creation from elite scorers",
    ),
    Badge(
        "ball_movement",
        "Ball Movement",
        "🌀",
        "Creates open looks through constant passing and player movement",
    ),
]

BADGE_MAP = {b.id: b for b in ALL_BADGES}

PACE_RUN = 0.65
PACE_GRIND = -0.65
DEF_WALL = 0.65

FG3A_IDENTITY = 0.55
FG3_PCT_IDENTITY_FLOOR = -0.20
FG3A_STRONG = 0.40
FG3_PCT_STRONG = 0.08
FG3A_SNIPER = 0.25
FG3_PCT_SNIPER = 0.40
FG3_VOL_FALLBACK_NO_PCT = 0.60

PAINT_VOL = 0.55
PAINT_LOW_3 = 0.35
PLAY_AST = 0.58
PLAY_TOV = 0.28
PLAY_HUB_AST = 0.90
PLAY_HUB_AST_TO = 0.85
PLAY_HUB_TOV_CAP = 0.75
OREB_GLASS = 0.70
FTA_FOUL = 0.70
AST_MOVE = 0.50
AST_TO_MOVE = 0.20
BALL_MOVE_ALT_AST_TO = 0.85
BALL_MOVE_ALT_AST_FLOOR = -0.15
ISO_AST_CAP = -0.45
ISO_OFF_FLOOR = 0.35
ISO_AST_TO_MAX = 0.4


def _z(row: pd.Series, col: str) -> float:
    v = pd.to_numeric(row.get(col), errors="coerce")
    return float(v) if pd.notna(v) else 0.0


def assign_badges(row: pd.Series) -> list[Badge]:
    badges: list[Badge] = []

    pace_z = _z(row, "PACE_Z")
    fg3a_z = _z(row, "FG3A_Z")
    fg3_pct_z = _z(row, "FG3_PCT_Z")
    paint_z = _z(row, "PAINT_FGA_Z")
    def_z = _z(row, "DEF_RATING_Z")
    ast_z = _z(row, "AST_PCT_Z")
    tov_z = _z(row, "TM_TOV_PCT_Z")
    oreb_z = _z(row, "OREB_PCT_Z")
    fta_z = _z(row, "FTA_RATE_Z")
    off_z = _z(row, "OFF_RATING_Z")
    ast_to_z = _z(row, "AST_TO_Z")

    if pace_z > PACE_RUN:
        badges.append(BADGE_MAP["run_and_gun"])

    has_fg3_pct = "FG3_PCT_Z" in row.index and pd.notna(row.get("FG3_PCT_Z"))
    if has_fg3_pct:
        volume_identity = fg3a_z >= FG3A_IDENTITY and fg3_pct_z > FG3_PCT_IDENTITY_FLOOR
        volume_and_pct = fg3a_z >= FG3A_STRONG and fg3_pct_z > FG3_PCT_STRONG
        sniper = fg3a_z >= FG3A_SNIPER and fg3_pct_z > FG3_PCT_SNIPER
        deep = volume_identity or volume_and_pct or sniper
    else:
        deep = fg3a_z >= FG3_VOL_FALLBACK_NO_PCT
    if deep:
        badges.append(BADGE_MAP["deep_threat"])

    if paint_z > PAINT_VOL and fg3a_z < PAINT_LOW_3:
        badges.append(BADGE_MAP["paint_punishers"])

    if -def_z > DEF_WALL:
        badges.append(BADGE_MAP["defensive_wall"])

    traditional_playmaking = ast_z > PLAY_AST and -tov_z > PLAY_TOV
    hub_playmaking = (
        ast_z > PLAY_HUB_AST
        and ast_to_z > PLAY_HUB_AST_TO
        and tov_z < PLAY_HUB_TOV_CAP
    )
    if traditional_playmaking or hub_playmaking:
        badges.append(BADGE_MAP["playmakers"])

    if pace_z < PACE_GRIND:
        badges.append(BADGE_MAP["grind_it_out"])

    if oreb_z > OREB_GLASS:
        badges.append(BADGE_MAP["glass_eaters"])

    if fta_z > FTA_FOUL:
        badges.append(BADGE_MAP["foul_magnets"])

    ball_movement = (ast_z > AST_MOVE and ast_to_z > AST_TO_MOVE) or (
        ast_to_z >= BALL_MOVE_ALT_AST_TO and ast_z > BALL_MOVE_ALT_AST_FLOOR
    )
    if ball_movement:
        badges.append(BADGE_MAP["ball_movement"])

    if (
        ast_z < ISO_AST_CAP
        and off_z > ISO_OFF_FLOOR
        and ast_to_z < ISO_AST_TO_MAX
        and not ball_movement
    ):
        badges.append(BADGE_MAP["iso_kings"])

    return badges


def exemplar_score(badge_id: str, row: pd.Series) -> float:
    pace_z = _z(row, "PACE_Z")
    fg3a_z = _z(row, "FG3A_Z")
    fg3_pct_z = _z(row, "FG3_PCT_Z")
    paint_z = _z(row, "PAINT_FGA_Z")
    def_z = _z(row, "DEF_RATING_Z")
    ast_z = _z(row, "AST_PCT_Z")
    tov_z = _z(row, "TM_TOV_PCT_Z")
    oreb_z = _z(row, "OREB_PCT_Z")
    fta_z = _z(row, "FTA_RATE_Z")
    off_z = _z(row, "OFF_RATING_Z")
    ast_to_z = _z(row, "AST_TO_Z")

    if badge_id == "run_and_gun":
        return pace_z
    if badge_id == "deep_threat":
        has_fg3_pct = "FG3_PCT_Z" in row.index and pd.notna(row.get("FG3_PCT_Z"))
        if has_fg3_pct:
            vol_identity = fg3a_z + max(0.0, fg3_pct_z - FG3_PCT_IDENTITY_FLOOR)
            vol_and_pct = fg3a_z + fg3_pct_z
            sniper = fg3a_z * 0.5 + fg3_pct_z * 1.5
            return max(vol_identity, vol_and_pct, sniper)
        return fg3a_z
    if badge_id == "paint_punishers":
        return paint_z - fg3a_z
    if badge_id == "defensive_wall":
        return -def_z
    if badge_id == "playmakers":
        return ast_z - tov_z
    if badge_id == "grind_it_out":
        return -pace_z
    if badge_id == "glass_eaters":
        return oreb_z
    if badge_id == "foul_magnets":
        return fta_z
    if badge_id == "iso_kings":
        return off_z - ast_z
    if badge_id == "ball_movement":
        return ast_z + ast_to_z
    return 0.0
